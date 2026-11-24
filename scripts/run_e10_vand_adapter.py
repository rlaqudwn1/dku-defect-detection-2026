"""E10: VAND / AnomalyCLIP Multi-Layer Patch-to-Text Residual Linear Adapter (5-Fold x 15 Epochs).

Strictly engineered for 5.50 GiB VRAM / 8.0 GiB cgroup RAM (Swap 0B) / 64 GB Disk containers AND Kaggle/Colab GPUs:
  - Pretrained weights: OpenCLIP `ViT-L-14-336` (`openai`, MIT license, published 2021 <= 2026-11-15)
    (supports `--model ViT-L-14 --pretrained laion2b_s32b_b82k` as well).
  - Full-FOV 336x336 resize (24x24 = 576 patches, ZERO Center-Crop border defect loss).
  - Processes ONE layer at a time (L6, L12, L18, L24) via temporary disk `np.memmap` (<13 GB peak disk, <2.5 GB RAM,
    <2.1 GiB VRAM), saving per-layer checkpoints (`ckpt_e10_L*.npz`) and immediately deleting each memmap.
  - Strictly NO Oracle per-fold epoch cherry-picking: records full 5-Fold Unseen OOF AUROC curve across Epochs 1..15
    and picks ONE consensus optimal epoch per layer + optimal multi-layer fusion + Whole-Core-Cluster Robust Z-Score.
  - ZERO Test image viewing or visualization (pure black-box Test inference once).
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import shutil
import tempfile
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

SEED = 42

NORMAL_PROMPTS = [
    "a photo of a flawless object",
    "a photo of an intact product without any defect",
    "a photo of a clean undamaged industrial part",
    "a photo of a perfect manufactured item",
    "a close-up photo of a normal good quality product",
    "a photo of an unblemished surface with no scratches or cracks",
    "a photo of intact identical items with no damage",
    "a photo of a smooth clean industrial component in good condition",
]

ANOMALY_PROMPTS = [
    "a photo of a damaged object",
    "a photo of a defective product with a crack",
    "a photo of a broken chipped industrial part",
    "a photo of a scratched or dented manufactured item",
    "a close-up photo of a flawed object with a stain or hole",
    "a photo of a bent deformed or broken component",
    "a photo of a product with a missing piece or cut",
    "a photo of a contaminated dirty damaged surface",
    "a photo of a burnt discolored or corroded defect on an object",
    "a photo of an anomaly or manufacturing defect on an object",
]


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_tables_and_folds(
    root: Path, folds_json_path: str | None
) -> tuple[list[str], np.ndarray, list[str], np.ndarray, dict[str, str]]:
    train_rows: list[tuple[str, int]] = []
    with (root / "train.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            train_rows.append((r["image_path"], int(r["label"])))
    test_ids: list[str] = []
    with (root / "sample_submission.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            test_ids.append(r["image_path"])

    train_ids = [r[0] for r in train_rows]
    y_tr = np.array([r[1] for r in train_rows], dtype=np.int64)
    id2idx = {i: k for k, i in enumerate(train_ids)}

    if folds_json_path and Path(folds_json_path).is_file():
        payload = json.loads(Path(folds_json_path).read_text(encoding="utf-8"))
        fold_arr = np.full(len(train_ids), -1, dtype=np.int64)
        for f_str, f_data in payload["folds"].items():
            f_i = int(f_str)
            for img_id in f_data["val_ids"]:
                if img_id in id2idx:
                    fold_arr[id2idx[img_id]] = f_i
        groups = dict(payload["groups"])
        assert (fold_arr >= 0).all(), "Some train_ids missing from folds_unseen.json"
        print(f"[CV] Loaded exact 5-Fold Unseen split from {folds_json_path}", flush=True)
        return train_ids, y_tr, test_ids, fold_arr, groups

    print("[CV] Reconstructing deterministic 5-Fold Unseen-Product GroupKFold...", flush=True)

    def _read_wh(item: tuple[str, int]) -> tuple[str, int, str]:
        img_id, lbl = item
        p = root / "train" / img_id if not (root / img_id).is_file() else root / img_id
        with Image.open(p) as im:
            w, h = ImageOps.exif_transpose(im).size
        return img_id, lbl, f"{w}x{h}"

    by_g: dict[str, list[tuple[str, int]]] = defaultdict(list)
    groups = {}
    with ThreadPoolExecutor(max_workers=4) as pool:
        for img_id, lbl, g in pool.map(_read_wh, train_rows):
            groups[img_id] = g
            by_g[g].append((img_id, lbl))

    items = sorted(
        [(g, [x[0] for x in v], sum(x[1] for x in v), len(v)) for g, v in by_g.items()],
        key=lambda x: (-x[2], -x[3], x[0]),
    )
    fold_arr = np.full(len(train_ids), -1, dtype=np.int64)
    fold_pos, fold_tot = [0] * 5, [0] * 5
    for _, ids_g, pos, tot in items:
        f_idx = min(range(5), key=lambda k: (fold_pos[k], fold_tot[k], k))
        for img_id in ids_g:
            fold_arr[id2idx[img_id]] = f_idx
        fold_pos[f_idx] += pos
        fold_tot[f_idx] += tot
    return train_ids, y_tr, test_ids, fold_arr, groups


class FullFOVClipDataset(Dataset):
    """Full-FOV resize preserving 100% of image borders (NO Center-Crop defect truncation!)."""

    def __init__(self, root: Path, role: str, ids: list[str], img_size: int):
        self.root = root
        self.role = role
        self.ids = ids
        self.img_size = img_size
        self.mean = torch.tensor([0.48145466, 0.4578275, 0.40821073], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([0.26862954, 0.26130258, 0.27577711], dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_id = self.ids[idx]
        p = self.root / self.role / img_id if not (self.root / img_id).is_file() else self.root / img_id
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            rgb = rgb.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return (t - self.mean) / self.std, idx


# ---------------------------------------------------------------------------
# VAND / AnomalyCLIP Patch-to-Text Residual Linear Adapter
# ---------------------------------------------------------------------------
class PatchTextAdapter(nn.Module):
    """Maps frozen CLIP intermediate patch tokens -> frozen text space [normal, anomaly]."""

    def __init__(self, patch_dim: int, text_dim: int, top_k_ratio: float = 0.025, dropout: float = 0.30):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.ln = nn.LayerNorm(patch_dim)
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Linear(patch_dim, text_dim, bias=True)
        # Learnable blend between raw patch text logit and intra-image residual patch logit
        self.res_alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.temp = nn.Parameter(torch.tensor(12.0, dtype=torch.float32))

    def forward(self, patches: torch.Tensor, text_diff: torch.Tensor) -> torch.Tensor:
        """patches: (B, P, D_patch), text_diff: (D_text,) normalized (t_anom - t_norm) -> img_logit (B,)."""
        h = self.proj(self.drop(self.ln(patches)))  # (B, P, D_text)
        h = F.normalize(h, dim=-1)
        patch_logits = self.temp.clamp(2.0, 40.0) * torch.matmul(h, text_diff)  # (B, P)
        # Intra-image residual suppresses product-level baseline texture offset!
        patch_med = patch_logits.median(dim=1, keepdim=True).values
        alpha = torch.sigmoid(self.res_alpha)
        eff_logits = (1.0 - alpha) * patch_logits + alpha * (patch_logits - patch_med)
        k = max(1, int(round(eff_logits.shape[1] * self.top_k_ratio)))
        return torch.topk(eff_logits, k=k, dim=1).values.mean(dim=1)


def within_group_ranking_loss(logits: torch.Tensor, labels: torch.Tensor, group_ids: torch.Tensor, margin: float = 0.5) -> torch.Tensor:
    """Pairwise ranking margin loss strictly between (defect, normal) pairs from the SAME product group."""
    pos_m = labels > 0.5
    neg_m = ~pos_m
    if not pos_m.any() or not neg_m.any():
        return logits.new_tensor(0.0)
    # Pairwise matrix: (n_pos, n_neg)
    pos_l, pos_g = logits[pos_m], group_ids[pos_m]
    neg_l, neg_g = logits[neg_m], group_ids[neg_m]
    same_g = pos_g[:, None] == neg_g[None, :]
    if not same_g.any():
        # Fallback to global pairwise if a small mini-batch has no same-group pair
        diff = pos_l[:, None] - neg_l[None, :]
        return F.softplus(margin - diff).mean()
    diff = pos_l[:, None] - neg_l[None, :]
    return F.softplus(margin - diff)[same_g].mean()


def discover_core_clusters_and_calibrate(
    raw_logits: np.ndarray,
    cls_vecs: np.ndarray,
    dist_threshold: float = 0.22,
    core_sim_floor: float = 0.70,
) -> np.ndarray:
    """Whole-Core-Cluster Robust Z-Score directly on raw logits (NO destructive Min-Max distortion!)."""
    n = len(raw_logits)
    g_med = float(np.median(raw_logits))
    g_iqr = float(np.percentile(raw_logits, 75) - np.percentile(raw_logits, 25)) + 1e-4
    z = (raw_logits - g_med) / g_iqr
    if n < 12:
        return z

    sim_mat = cls_vecs @ cls_vecs.T
    np.fill_diagonal(sim_mat, -1.0)
    k_den = min(20, max(1, n - 1))
    density = np.sort(sim_mat, axis=1)[:, -k_den:].mean(axis=1)
    core_mask = density >= np.percentile(density, 25)
    if core_mask.sum() < 4:
        core_mask = np.ones(n, dtype=bool)

    ac = AgglomerativeClustering(n_clusters=None, metric="cosine", linkage="average", distance_threshold=dist_threshold)
    core_lbls = ac.fit_predict(cls_vecs[core_mask])
    uniq, cnts = np.unique(core_lbls, return_counts=True)
    valid_u = uniq[cnts >= max(5, int(0.008 * n))]
    if len(valid_u) == 0:
        return z

    centroids = np.stack([cls_vecs[core_mask][core_lbls == u].mean(axis=0) for u in valid_u], axis=0)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    sim_to_c = cls_vecs @ centroids.T
    best_c = np.argmax(sim_to_c, axis=1)
    max_sim = sim_to_c[np.arange(n), best_c]
    best_c[max_sim < core_sim_floor] = -1

    for cid in range(len(valid_u)):
        m = best_c == cid
        n_c = int(m.sum())
        if n_c < 4:
            continue
        w = max(0.0, n_c - 3.0) / (n_c + 10.0)
        c_med = float(np.median(raw_logits[m]))
        c_iqr = float(np.percentile(raw_logits[m], 75) - np.percentile(raw_logits[m], 25)) + 1e-4
        eff_med = w * c_med + (1.0 - w) * g_med
        eff_iqr = 0.85 * (w * c_iqr + (1.0 - w) * g_iqr) + 0.15 * g_iqr
        z[m] = (raw_logits[m] - eff_med) / eff_iqr
    return z


@torch.inference_mode()
def extract_single_layer_memmap(
    model: nn.Module,
    layer_idx_0: int,
    root: Path,
    role: str,
    ids: list[str],
    img_size: int,
    batch_size: int,
    device: str,
    mm_path: Path,
    capture_cls_final: bool = False,
) -> tuple[np.memmap, np.ndarray | None]:
    """Stream ONE OpenCLIP visual resblock's patch tokens directly to fp16 disk memmap (<13 GB peak disk)."""
    n = len(ids)
    captured: dict[str, torch.Tensor] = {}

    def _hook(_mod, _inp, outp):
        captured["act"] = outp.detach()

    handle = model.visual.transformer.resblocks[layer_idx_0].register_forward_hook(_hook)
    ds = FullFOVClipDataset(root, role, ids, img_size=img_size)
    # Strict num_workers=0 for 8.0 GiB cgroup RAM (Swap 0B) safety!
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=(device == "cuda"))
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    mm: np.memmap | None = None
    cls_arr: np.ndarray | None = None
    try:
        for x, batch_idxs in loader:
            b_idxs = batch_idxs.numpy()
            b = len(b_idxs)
            x = x.to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                img_emb = F.normalize(model.encode_image(x).float(), dim=-1)
            act = captured["act"]
            # Handle both (B, L, D) and (L, B, D) OpenCLIP tensor layouts
            if act.shape[0] != b and act.shape[1] == b:
                act = act.transpose(0, 1)
            patch_tok = F.normalize(act[:, 1:, :].float(), dim=-1).half().cpu().numpy()  # (B, P, D)

            if mm is None:
                _, n_p, dim_p = patch_tok.shape
                mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(n, n_p, dim_p))
            mm[b_idxs] = patch_tok

            if capture_cls_final:
                emb_np = img_emb.cpu().numpy()
                if cls_arr is None:
                    cls_arr = np.empty((n, emb_np.shape[-1]), dtype=np.float32)
                cls_arr[b_idxs] = emb_np
    finally:
        handle.remove()

    assert mm is not None
    mm.flush()
    return mm, cls_arr


def train_5fold_layer_adapter(
    tr_mm: np.memmap,
    te_mm: np.memmap,
    y_tr: np.ndarray,
    fold_arr: np.ndarray,
    group_int_arr: np.ndarray,
    text_diff: torch.Tensor,
    epochs: int,
    lr: float,
    weight_decay: float,
    batch_size: int,
    device: str,
    layer_name: str,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Train 5-Fold Unseen PatchTextAdapter for 1..epochs; return FULL (epochs, N_tr) OOF & (epochs, N_te) Test logits."""
    n_tr, _, patch_dim = tr_mm.shape
    n_te = te_mm.shape[0]
    text_dim = int(text_diff.shape[0])

    oof_by_ep = np.zeros((epochs, n_tr), dtype=np.float32)
    te_by_ep = np.zeros((epochs, n_te), dtype=np.float32)

    for fold in range(5):
        val_m = fold_arr == fold
        tr_idxs = np.where(~val_m)[0]
        val_idxs = np.where(val_m)[0]

        y_fold = y_tr[tr_idxs]
        pos_w = torch.tensor([(y_fold == 0).sum() / max(1, (y_fold == 1).sum())], device=device, dtype=torch.float32)

        seed_everything(SEED + fold)
        adapter = PatchTextAdapter(patch_dim=patch_dim, text_dim=text_dim, top_k_ratio=0.025, dropout=0.30).to(device)
        opt = torch.optim.AdamW(adapter.parameters(), lr=lr, weight_decay=weight_decay)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
        rng = np.random.default_rng(SEED + fold)

        for ep in range(epochs):
            adapter.train()
            perm = tr_idxs[rng.permutation(len(tr_idxs))]
            for start in range(0, len(perm), batch_size):
                sel = perm[start : start + batch_size]
                bx = torch.from_numpy(np.array(tr_mm[sel], copy=True)).to(device, dtype=torch.float32)
                by = torch.from_numpy(y_tr[sel]).to(device, dtype=torch.float32)
                bg = torch.from_numpy(group_int_arr[sel]).to(device, dtype=torch.long)

                opt.zero_grad(set_to_none=True)
                logits = adapter(bx, text_diff)
                loss_bce = F.binary_cross_entropy_with_logits(logits, by, pos_weight=pos_w)
                loss_rank = within_group_ranking_loss(logits, by, bg, margin=0.5)
                loss = 0.65 * loss_bce + 0.35 * loss_rank
                loss.backward()
                torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
                opt.step()
            sched.step()

            # Evaluate Val Fold & Test at EVERY epoch (zero Oracle per-fold cherry-picking!)
            adapter.eval()
            with torch.inference_mode():
                val_out = []
                for start in range(0, len(val_idxs), batch_size):
                    sel = val_idxs[start : start + batch_size]
                    bx = torch.from_numpy(np.array(tr_mm[sel], copy=True)).to(device, dtype=torch.float32)
                    val_out.append(adapter(bx, text_diff).cpu().numpy())
                oof_by_ep[ep, val_idxs] = np.concatenate(val_out, axis=0)

                te_out = []
                for start in range(0, n_te, batch_size):
                    bx = torch.from_numpy(np.array(te_mm[start : start + batch_size], copy=True)).to(
                        device, dtype=torch.float32
                    )
                    te_out.append(adapter(bx, text_diff).cpu().numpy())
                te_by_ep[ep] += np.concatenate(te_out, axis=0) / 5.0

        del adapter, opt, sched
        if device == "cuda":
            torch.cuda.empty_cache()

    ep_aucs = [round(float(roc_auc_score(y_tr, oof_by_ep[ep])), 4) for ep in range(epochs)]
    best_ep = int(np.argmax(ep_aucs)) + 1
    print(f"  [{layer_name}] 5-Fold Raw OOF Curve (Ep 1..{epochs}): {ep_aucs} | Best Consensus Ep={best_ep} ({max(ep_aucs):.4f})", flush=True)
    return oof_by_ep, te_by_ep, ep_aucs


def export_submission(path: Path, test_ids: list[str], z_scores: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    probs = 1.0 / (1.0 + np.exp(-np.clip(z_scores * 0.8, -20.0, 20.0)))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_path", "anomaly_score"])
        for img_id, sc in zip(test_ids, probs):
            w.writerow([img_id, f"{float(sc):.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="E10: VAND/AnomalyCLIP Multi-Layer Patch-to-Text Adapter (5-Fold x 15ep)")
    parser.add_argument("--data-root", type=str, default="data/raw/open")
    parser.add_argument("--folds-json", type=str, default="runs/folds_unseen.json")
    parser.add_argument("--out-dir", type=str, default="runs/e10_vand_adapter")
    parser.add_argument("--model", type=str, default="ViT-L-14-336")
    parser.add_argument("--pretrained", type=str, default="openai")
    parser.add_argument("--img-size", type=int, default=336)
    parser.add_argument("--extract-batch-size", type=int, default=16)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=2e-2)
    parser.add_argument("--smoke-test", action="store_true", help="Run tiny 2-epoch 60-image smoke test")
    args = parser.parse_args()

    t0 = time.monotonic()
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    root = Path(args.data_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_ids, y_tr, test_ids, fold_arr, groups = load_tables_and_folds(root, args.folds_json)
    if args.smoke_test:
        print("[SMOKE TEST] Subsampling 12 images/fold (6 pos + 6 neg) & epochs=2...", flush=True)
        keep_idx = []
        for f in range(5):
            f_idx = np.where(fold_arr == f)[0]
            pos = f_idx[y_tr[f_idx] == 1][:6]
            neg = f_idx[y_tr[f_idx] == 0][:6]
            keep_idx.extend(pos.tolist() + neg.tolist())
        keep_idx = sorted(keep_idx)
        train_ids = [train_ids[i] for i in keep_idx]
        y_tr = y_tr[keep_idx]
        fold_arr = fold_arr[keep_idx]
        test_ids = test_ids[:12]
        args.epochs = 2

    uniq_g = sorted(set(groups[i] for i in train_ids))
    g2int = {g: k for k, g in enumerate(uniq_g)}
    group_int_arr = np.array([g2int[groups[i]] for i in train_ids], dtype=np.int64)

    print(
        f"[E10] Device={device} | Model={args.model} ({args.pretrained}) @ {args.img_size}px | "
        f"Train={len(train_ids)}, Test={len(test_ids)}, Epochs={args.epochs}",
        flush=True,
    )
    clip_model, _, _ = open_clip.create_model_and_transforms(args.model, pretrained=args.pretrained)
    tokenizer = open_clip.get_tokenizer(args.model)
    clip_model = clip_model.to(device).eval()

    with torch.no_grad():
        t_norm = F.normalize(clip_model.encode_text(tokenizer(NORMAL_PROMPTS).to(device)).float(), dim=-1).mean(dim=0)
        t_anom = F.normalize(clip_model.encode_text(tokenizer(ANOMALY_PROMPTS).to(device)).float(), dim=-1).mean(dim=0)
        text_diff = F.normalize(t_anom - t_norm, dim=-1).detach().clone().to(device)  # (D_text,)

    n_blocks = len(clip_model.visual.transformer.resblocks)
    layer_idx_0 = [n_blocks // 4 - 1, n_blocks // 2 - 1, (3 * n_blocks) // 4 - 1, n_blocks - 1]
    layer_names = [f"L{idx + 1}" for idx in layer_idx_0]
    print(f"[E10] Target Layers={layer_names} (0-idx {layer_idx_0})", flush=True)

    cls_cache_path = out_dir / "ckpt_cls_final.npz"
    tr_cls_final: np.ndarray | None = None
    te_cls_final: np.ndarray | None = None
    if cls_cache_path.is_file() and not args.smoke_test:
        loaded_cls = np.load(cls_cache_path)
        if len(loaded_cls["tr_cls"]) == len(train_ids):
            tr_cls_final = loaded_cls["tr_cls"]
            te_cls_final = loaded_cls["te_cls"]

    layer_oof_ep: dict[str, np.ndarray] = {}
    layer_te_ep: dict[str, np.ndarray] = {}
    layer_reports: dict[str, dict] = {}

    for li, (l0, lname) in enumerate(zip(layer_idx_0, layer_names)):
        ckpt_path = out_dir / f"ckpt_e10_{lname}.npz"
        if ckpt_path.is_file() and not args.smoke_test and tr_cls_final is not None:
            ck = np.load(ckpt_path)
            if ck["oof_by_ep"].shape == (args.epochs, len(train_ids)):
                print(f"  [{lname}] Resumed from checkpoint -> {ckpt_path}", flush=True)
                layer_oof_ep[lname] = ck["oof_by_ep"]
                layer_te_ep[lname] = ck["te_by_ep"]
                continue

        t_l = time.monotonic()
        tmp_dir = Path(tempfile.mkdtemp(prefix=f"e10_{lname}_"))
        try:
            need_cls = tr_cls_final is None
            print(f"\n--- [{lname}] Streaming patch tokens to temp disk memmap ({tmp_dir}) ---", flush=True)
            tr_mm, tr_c = extract_single_layer_memmap(
                clip_model, l0, root, "train", train_ids, args.img_size, args.extract_batch_size,
                device, tmp_dir / "tr.fp16", capture_cls_final=need_cls,
            )
            te_mm, te_c = extract_single_layer_memmap(
                clip_model, l0, root, "test", test_ids, args.img_size, args.extract_batch_size,
                device, tmp_dir / "te.fp16", capture_cls_final=need_cls,
            )
            if need_cls and tr_c is not None and te_c is not None:
                tr_cls_final, te_cls_final = tr_c, te_c
                if not args.smoke_test:
                    np.savez_compressed(cls_cache_path, tr_cls=tr_cls_final, te_cls=te_cls_final)

            print(f"  [{lname}] Extracted patches {tr_mm.shape} in {time.monotonic() - t_l:.1f}s. Training 5-Fold x {args.epochs}ep...", flush=True)
            oof_ep, te_ep, ep_aucs = train_5fold_layer_adapter(
                tr_mm, te_mm, y_tr, fold_arr, group_int_arr, text_diff,
                epochs=args.epochs, lr=args.lr, weight_decay=args.weight_decay,
                batch_size=args.train_batch_size, device=device, layer_name=lname,
            )
            layer_oof_ep[lname] = oof_ep
            layer_te_ep[lname] = te_ep
            if not args.smoke_test:
                np.savez_compressed(ckpt_path, oof_by_ep=oof_ep, te_by_ep=te_ep)
        finally:
            del tr_mm, te_mm
            shutil.rmtree(tmp_dir, ignore_errors=True)
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()

    assert tr_cls_final is not None and te_cls_final is not None

    # Evaluate per-layer consensus best epoch (raw vs Whole-Core-Cluster Robust Z-Score strictly per unseen fold!)
    print("\n=== [E10 AUDIT] PER-LAYER CONSENSUS EPOCH & CALIBRATION COMPARISON (5-FOLD UNSEEN OOF) ===", flush=True)
    cal_oof_per_layer: dict[str, np.ndarray] = {}
    cal_te_per_layer: dict[str, np.ndarray] = {}

    for lname in layer_names:
        oof_ep = layer_oof_ep[lname]
        te_ep = layer_te_ep[lname]
        raw_curve = [round(float(roc_auc_score(y_tr, oof_ep[ep])), 4) for ep in range(args.epochs)]

        cal_curve = []
        cal_oof_all_ep = np.zeros_like(oof_ep)
        for ep in range(args.epochs):
            for f in range(5):
                m = fold_arr == f
                cal_oof_all_ep[ep, m] = discover_core_clusters_and_calibrate(oof_ep[ep, m], tr_cls_final[m])
            cal_curve.append(round(float(roc_auc_score(y_tr, cal_oof_all_ep[ep])), 4))

        best_ep_idx = int(np.argmax(cal_curve))
        best_oof_cal = cal_oof_all_ep[best_ep_idx]
        best_te_cal = discover_core_clusters_and_calibrate(te_ep[best_ep_idx], te_cls_final)

        cal_oof_per_layer[lname] = best_oof_cal
        cal_te_per_layer[lname] = best_te_cal

        per_fold = {f"fold_{f}": round(float(roc_auc_score(y_tr[fold_arr == f], best_oof_cal[fold_arr == f])), 4) for f in range(5)}
        layer_reports[lname] = {
            "best_consensus_epoch": best_ep_idx + 1,
            "raw_oof_at_best_ep": raw_curve[best_ep_idx],
            "cal_oof_at_best_ep": cal_curve[best_ep_idx],
            "per_fold_cal_auroc": per_fold,
            "raw_epoch_curve": raw_curve,
            "cal_epoch_curve": cal_curve,
        }
        print(
            f"  {lname:4s} | Best Ep={best_ep_idx + 1:2d} | Raw OOF={raw_curve[best_ep_idx]:.4f} -> Cal OOF={cal_curve[best_ep_idx]:.4f} | {per_fold}",
            flush=True,
        )

    # Optimize Multi-Layer Weights on 5-Fold Unseen OOF (with Nested 5-Fold leakage-free verification)
    weight_candidates = [
        (0.25, 0.25, 0.25, 0.25),
        (0.10, 0.30, 0.40, 0.20),
        (0.00, 0.30, 0.45, 0.25),
        (0.00, 0.00, 0.50, 0.50),
        (0.15, 0.35, 0.35, 0.15),
        (0.00, 0.50, 0.50, 0.00),
        (0.00, 0.00, 0.00, 1.00),
        (0.00, 0.00, 1.00, 0.00),
        (0.00, 1.00, 0.00, 0.00),
    ]
    oof_stack = np.stack([cal_oof_per_layer[ln] for ln in layer_names], axis=0)  # (4, N_tr)
    te_stack = np.stack([cal_te_per_layer[ln] for ln in layer_names], axis=0)    # (4, N_te)

    nested_oof = np.zeros(len(y_tr), dtype=np.float32)
    for held_f in range(5):
        tr_m = fold_arr != held_f
        val_m = ~tr_m
        best_w = max(weight_candidates, key=lambda w: roc_auc_score(y_tr[tr_m], np.tensordot(w, oof_stack[:, tr_m], axes=1)))
        nested_oof[val_m] = np.tensordot(best_w, oof_stack[:, val_m], axes=1)
    nested_auc = float(roc_auc_score(y_tr, nested_oof))

    best_w_global = max(weight_candidates, key=lambda w: roc_auc_score(y_tr, np.tensordot(w, oof_stack, axes=1)))
    best_oof_z = np.tensordot(best_w_global, oof_stack, axes=1).astype(np.float32)
    best_te_z = np.tensordot(best_w_global, te_stack, axes=1).astype(np.float32)
    best_global_auc = float(roc_auc_score(y_tr, best_oof_z))
    best_per_fold = {f"fold_{f}": round(float(roc_auc_score(y_tr[fold_arr == f], best_oof_z[fold_arr == f])), 4) for f in range(5)}

    print(f"\n==> [E10 MULTI-LAYER FUSION] Nested 5-Fold OOF AUROC = {nested_auc:.4f} | Consensus OOF = {best_global_auc:.4f}", flush=True)
    print(f"    Best Layer Weights ({layer_names}) = {best_w_global} | Per-Fold = {best_per_fold}", flush=True)

    # Save submission, OOF validation JSON (for `defectlab errors`), and compact NPZ cache for E09+E10 blending
    sub_path = out_dir / "sub_e10_vand_patch_adapter.csv"
    export_submission(sub_path, test_ids, best_te_z)

    oof_probs = 1.0 / (1.0 + np.exp(-np.clip(best_oof_z * 0.8, -20.0, 20.0)))
    val_payload = {
        "schema_version": 1,
        "kind": "validation",
        "auroc": round(best_global_auc, 6),
        "nested_oof_auroc": round(nested_auc, 6),
        "rows": [
            {"image_path": i, "label": int(y), "anomaly_score": float(p), "z_score": float(z)}
            for i, y, p, z in zip(train_ids, y_tr, oof_probs, best_oof_z)
        ],
    }
    (out_dir / "val_oof_e10.json").write_text(json.dumps(val_payload, indent=2), encoding="utf-8")

    np.savez_compressed(
        out_dir / "e10_oof_and_test_cache.npz",
        oof_ids=np.array(train_ids),
        oof_y=y_tr,
        oof_z=best_oof_z,
        oof_layer_stack=oof_stack.astype(np.float32),
        test_ids=np.array(test_ids),
        test_z=best_te_z,
        test_layer_stack=te_stack.astype(np.float32),
    )

    summary = {
        "schema_version": 1,
        "experiment": "E10_vand_patch_text_adapter",
        "model": f"open_clip:{args.model}:{args.pretrained}",
        "license": "MIT (OpenAI CLIP, published 2021 <= 2026-11-15)",
        "img_size": args.img_size,
        "epochs": args.epochs,
        "nested_5fold_oof_auroc": round(nested_auc, 4),
        "best_consensus_5fold_oof_auroc": round(best_global_auc, 4),
        "best_layer_weights": dict(zip(layer_names, best_w_global)),
        "best_per_fold_auroc": best_per_fold,
        "layer_reports": layer_reports,
        "elapsed_seconds": round(time.monotonic() - t0, 1),
    }
    (out_dir / "e10_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved submission -> {sub_path}", flush=True)
    print(f"[DONE] Saved 5-Fold OOF & summary -> {out_dir / 'e10_summary.json'} ({summary['elapsed_seconds']}s)", flush=True)


if __name__ == "__main__":
    main()
