"""Kaggle GPU Kernel — E06: 3-Paradigm Zero-Shot Anomaly Pipeline + Post-Experiment Feature Analysis.

Paradigms (all <= 2026-11-15 open-source pretrained weights):
  - E06a: Frozen DINOv2-Base (518px, 37x37 patches, multi-layer) + Learned Patch-MIL Head (5-Fold Unseen CV)
  - E06b: DINOv2-Base Intra-Image Self-Comparison + High-Sim Peer (cos > 0.88) Patch MuSc (NO K-Means!)
  - E06c: OpenCLIP ViT-L-14-336 WinCLIP (Global + 3x3 Grid Crops)
  - E06-Ens: Rank-Averaged Ensemble of E06a + E06b + E06c

Built-in Feature Research Outputs:
  1. Patch Anomaly Heatmap Overlay on Test Top-20 (`e06_test_heatmaps_top20.jpg`)
  2. Test Peer Cosine-Similarity Distribution (singleton vs multi-instance grid ratio)
  3. Cross-Paradigm Spearman Rank Correlation Matrix
  4. Unseen 5-Fold OOF AUROC per paradigm & per product group
"""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageEnhance, ImageOps
from scipy.stats import rankdata, spearmanr
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset


def _ensure_deps():
    for pkg, imp in (("timm", "timm"), ("open-clip-torch", "open_clip")):
        try:
            __import__(imp)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_ROOT = Path("/kaggle/working/e06_output") if Path("/kaggle/working").exists() else Path("runs/e06_gpu")
OUT_ROOT.mkdir(parents=True, exist_ok=True)


def seed_everything(seed: int = SEED):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_data_root() -> Path:
    candidates = [
        Path("/kaggle/input/dku-defect-detection-2026-open-private/open"),
        Path("/kaggle/input/dku-defect-detection-2026-open-private"),
        Path("data/raw/open"),
    ]
    for c in candidates:
        if (c / "train.csv").is_file() and (c / "sample_submission.csv").is_file():
            return c
    # Fallback recursive search in /kaggle/input
    ki = Path("/kaggle/input")
    if ki.exists():
        for p in ki.rglob("train.csv"):
            if (p.parent / "sample_submission.csv").is_file():
                return p.parent
    raise FileNotFoundError("Could not locate train.csv and sample_submission.csv")


def load_tables(root: Path):
    train_rows = []
    with (root / "train.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            train_rows.append((r["image_path"], int(r["label"])))
    test_ids = []
    with (root / "sample_submission.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for r in csv.DictReader(f):
            test_ids.append(r["image_path"])
    return train_rows, test_ids


def build_unseen_product_folds(root: Path, train_rows: list[tuple[str, int]], n_splits: int = 5):
    """Group Train by (width, height) -> 5-Fold Unseen-Product StratifiedGroupKFold (zero leakage)."""
    def _read_size(item):
        img_id, lbl = item
        p = root / "train" / img_id if not (root / img_id).is_file() else root / img_id
        with Image.open(p) as im:
            w, h = im.size
        return img_id, lbl, f"{w}x{h}"

    groups = {}
    by_g = defaultdict(list)
    with ThreadPoolExecutor(max_workers=16) as pool:
        for img_id, lbl, g in pool.map(_read_size, train_rows):
            groups[img_id] = g
            by_g[g].append((img_id, lbl))

    items = sorted(
        [(g, [x[0] for x in v], sum(x[1] for x in v), len(v)) for g, v in by_g.items()],
        key=lambda x: (-x[2], -x[3]),
    )
    fold_ids = [[] for _ in range(n_splits)]
    fold_pos = [0] * n_splits
    fold_tot = [0] * n_splits
    for _, ids, pos, tot in items:
        f = min(range(n_splits), key=lambda k: (fold_pos[k], fold_tot[k], k))
        fold_ids[f].extend(ids)
        fold_pos[f] += pos
        fold_tot[f] += tot

    id_to_fold = {}
    for k, ids in enumerate(fold_ids):
        for i in ids:
            id_to_fold[i] = k
    return id_to_fold, groups


# ---------------------------------------------------------------------------
# DINOv2-Base 518px Multi-Layer Patch Feature Extractor (37x37 = 1,369 patches)
# ---------------------------------------------------------------------------
class _ImgDS(Dataset):
    def __init__(self, root: Path, role: str, ids: list[str], img_size: int = 518, domain_aug: bool = False):
        self.root, self.role, self.ids, self.img_size, self.domain_aug = root, role, ids, img_size, domain_aug
        self.mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225])[:, None, None]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        p = self.root / self.role / img_id if not (self.root / img_id).is_file() else self.root / img_id
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            if self.domain_aug:
                # Break Train->Test domain shift (rectangular/green-black -> square/neutral-gray)
                w, h = rgb.size
                s = min(w, h)
                left = (w - s) // 2
                top = (h - s) // 2
                rgb = rgb.crop((left, top, left + s, top + s))
                if np.random.rand() < 0.5:
                    rgb = ImageOps.mirror(rgb)
                if np.random.rand() < 0.5:
                    rgb = ImageOps.flip(rgb)
                if np.random.rand() < 0.35:
                    rgb = ImageOps.grayscale(rgb).convert("RGB")
                else:
                    rgb = ImageEnhance.Color(rgb).enhance(float(np.random.uniform(0.5, 1.3)))
                    rgb = ImageEnhance.Brightness(rgb).enhance(float(np.random.uniform(0.8, 1.3)))
            rgb = rgb.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return (t - self.mean) / self.std, img_id


@torch.inference_mode()
def extract_dinov2_patches(root: Path, role: str, ids: list[str], model, img_size: int = 518, domain_aug: bool = False, pool_grid: int = 14):
    """Extract multi-layer DINOv2 patch tokens (mid+last) pooled to (pool_grid x pool_grid) + CLS global vec."""
    loader = DataLoader(_ImgDS(root, role, ids, img_size=img_size, domain_aug=domain_aug),
                        batch_size=32, shuffle=False, num_workers=4, pin_memory=(DEVICE == "cuda"))
    patch_list, cls_list, out_ids = [], [], []
    for x, batch_ids in loader:
        x = x.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            # Extract intermediate layers (block 7 = local texture, block 11 = semantic)
            inter = model.get_intermediate_layers(x, n=[7, 11], reshape=True, return_prefix_tokens=True, norm=True)
            f_mid, _ = inter[0]
            f_last, prefix_last = inter[1]
            # Pool 37x37 -> 14x14 (196 fine patches/img) for fast GPU/RAM efficiency
            p_mid = F.adaptive_avg_pool2d(f_mid.float(), (pool_grid, pool_grid))
            p_last = F.adaptive_avg_pool2d(f_last.float(), (pool_grid, pool_grid))
            p_cat = torch.cat([F.normalize(p_mid, dim=1), F.normalize(p_last, dim=1)], dim=1)  # (B, 1536, G, G)
            p_tok = F.normalize(p_cat.flatten(2).transpose(1, 2), dim=-1)  # (B, G*G, 1536)
            cls_v = F.normalize(prefix_last[:, 0].float(), dim=-1)
        patch_list.append(p_tok.half().cpu())
        cls_list.append(cls_v.float().cpu())
        out_ids.extend(batch_ids)
    return torch.cat(patch_list, dim=0), torch.cat(cls_list, dim=0), out_ids


# ---------------------------------------------------------------------------
# E06a: Learned Object-Agnostic Patch-MIL Head (5-Fold Unseen-Product CV)
# ---------------------------------------------------------------------------
class PatchMILHead(nn.Module):
    def __init__(self, dim: int = 1536, top_k_ratio: float = 0.05):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
        )

    def forward(self, patches: torch.Tensor):
        # patches: (B, N, D) -> patch_logits: (B, N)
        patch_logits = self.net(patches).squeeze(-1)
        k = max(1, int(round(patch_logits.shape[1] * self.top_k_ratio)))
        topk_vals = torch.topk(patch_logits, k=k, dim=1).values
        img_logit = topk_vals.mean(dim=1)
        return img_logit, patch_logits


def run_e06a_patch_mil(X_tr_orig, X_tr_aug, y_tr, fold_arr, X_te, epochs: int = 12):
    n_tr = len(y_tr)
    oof = np.zeros(n_tr, dtype=np.float32)
    te_preds = np.zeros(len(X_te), dtype=np.float32)
    te_maps = np.zeros((len(X_te), X_te.shape[1]), dtype=np.float32)

    for fold in range(5):
        val_m = fold_arr == fold
        tr_idxs = np.where(~val_m)[0]
        n_fold_tr = len(tr_idxs)
        y_fold = y_tr[tr_idxs]
        pos_w = torch.tensor([(y_fold == 0).sum() / max(1, (y_fold == 1).sum())], device=DEVICE, dtype=torch.float32)

        head = PatchMILHead(dim=X_tr_orig.shape[-1], top_k_ratio=0.05).to(DEVICE)
        opt = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=1e-2)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)

        best_auc, best_val, best_state = -1.0, None, None
        rng = np.random.default_rng(SEED + fold)
        # Zero-copy indexing across [orig, aug] (avoids duplicating >10GB tensor in RAM!)
        for _ in range(epochs):
            head.train()
            perm = rng.permutation(n_fold_tr * 2)
            for start in range(0, len(perm), 64):
                p_slice = perm[start:start + 64]
                orig_sel = tr_idxs[p_slice[p_slice < n_fold_tr]]
                aug_sel = tr_idxs[p_slice[p_slice >= n_fold_tr] - n_fold_tr]
                parts_x, parts_y = [], []
                if len(orig_sel):
                    parts_x.append(X_tr_orig[orig_sel].float())
                    parts_y.append(y_tr[orig_sel])
                if len(aug_sel):
                    parts_x.append(X_tr_aug[aug_sel].float())
                    parts_y.append(y_tr[aug_sel])
                bx = torch.cat(parts_x, dim=0).to(DEVICE, non_blocking=True)
                by = torch.from_numpy(np.concatenate(parts_y, axis=0)).float().to(DEVICE, non_blocking=True)
                opt.zero_grad(set_to_none=True)
                logit, _ = head(bx)
                loss = F.binary_cross_entropy_with_logits(logit, by, pos_weight=pos_w)
                loss.backward()
                opt.step()
            sched.step()

            head.eval()
            val_scores = []
            with torch.inference_mode():
                val_idxs = np.where(val_m)[0]
                for start in range(0, len(val_idxs), 64):
                    bx = X_tr_orig[val_idxs[start:start + 64]].float().to(DEVICE)
                    logit, _ = head(bx)
                    val_scores.append(logit.sigmoid().cpu().numpy())
            val_scores = np.concatenate(val_scores)
            auc = float(roc_auc_score(y_tr[val_m], val_scores))
            if auc > best_auc:
                best_auc, best_val = auc, val_scores
                best_state = {k: v.cpu().clone() for k, v in head.state_dict().items()}

        oof[val_m] = best_val
        print(f"  [E06a Patch-MIL] Fold {fold} Unseen AUROC = {best_auc:.4f}")
        head.load_state_dict({k: v.to(DEVICE) for k, v in best_state.items()})
        head.eval()
        with torch.inference_mode():
            for start in range(0, len(X_te), 64):
                bx = X_te[start:start + 64].float().to(DEVICE)
                logit, pmap = head(bx)
                te_preds[start:start + 64] += logit.sigmoid().cpu().numpy() / 5.0
                te_maps[start:start + 64] += pmap.sigmoid().cpu().numpy() / 5.0

    oof_auc = float(roc_auc_score(y_tr, oof))
    print(f"==> [E06a Patch-MIL] 5-Fold Unseen OOF AUROC = {oof_auc:.4f}")
    return oof, te_preds, te_maps, oof_auc


# ---------------------------------------------------------------------------
# E06b: Intra-Image + High-Sim Peer (cos > 0.88) Patch MuSc (NO K-Means!)
# ---------------------------------------------------------------------------
@torch.inference_mode()
def run_e06b_intra_and_highsim_musc(patches: torch.Tensor, cls_vecs: torch.Tensor, grid: int = 18,
                                    peer_thresh: float = 0.88, max_peers: int = 8, top_p: float = 0.03):
    """Score every image via (1) Intra-Image distant-patch outlier + (2) High-Sim Peer patch distance (only if cos > peer_thresh)."""
    n, npatch, dim = patches.shape
    sim_mat = (cls_vecs @ cls_vecs.T).numpy()
    np.fill_diagonal(sim_mat, -1.0)
    max_peer_sim = sim_mat.max(axis=1)

    # Precompute spatial mask for Intra-Image comparison (ignore patches within Manhattan distance <= 3)
    coords = torch.tensor([(r, c) for r in range(grid) for c in range(grid)], device=DEVICE)
    manhattan = (coords[:, None, :] - coords[None, :, :]).abs().sum(dim=-1)
    near_mask = manhattan <= 3  # (N_p, N_p)

    top_k = max(1, int(round(npatch * top_p)))
    intra_scores = np.zeros(n, dtype=np.float32)
    peer_scores = np.zeros(n, dtype=np.float32)
    has_peer = np.zeros(n, dtype=bool)
    patch_maps = np.zeros((n, npatch), dtype=np.float32)

    for i in range(n):
        pi = patches[i].float().to(DEVICE)  # (N_p, D)
        # 1) Intra-Image Self-Comparison (catches 1 defective item in a 3/6/9-object grid!)
        self_sim = pi @ pi.T
        self_sim.masked_fill_(near_mask, -2.0)
        intra_dist = 1.0 - self_sim.max(dim=1).values
        intra_scores[i] = float(torch.topk(intra_dist, k=top_k).values.mean().item())

        # 2) High-Sim Peer Comparison (ONLY when truly identical-layout peers exist!)
        peers = np.where(sim_mat[i] >= peer_thresh)[0]
        if len(peers) >= 2:
            if len(peers) > max_peers:
                peers = peers[np.argsort(-sim_mat[i, peers])[:max_peers]]
            bank = patches[peers].float().to(DEVICE).reshape(-1, dim)  # (P*N_p, D)
            p_sim = pi @ bank.T
            peer_dist = 1.0 - p_sim.max(dim=1).values
            peer_scores[i] = float(torch.topk(peer_dist, k=top_k).values.mean().item())
            has_peer[i] = True
            patch_maps[i] = peer_dist.cpu().numpy()
        else:
            patch_maps[i] = intra_dist.cpu().numpy()

    # Combine gracefully: where high-sim peers exist, blend rank(peer) + rank(intra); else use intra
    intra_r = rankdata(intra_scores) / n
    combined = intra_r.copy()
    if has_peer.sum() >= 5:
        peer_r = np.zeros(n, dtype=np.float32)
        peer_r[has_peer] = rankdata(peer_scores[has_peer]) / has_peer.sum()
        combined[has_peer] = 0.75 * peer_r[has_peer] + 0.25 * intra_r[has_peer]

    return combined, intra_scores, peer_scores, has_peer, max_peer_sim, patch_maps


# ---------------------------------------------------------------------------
# E06c: OpenCLIP ViT-L-14-336 WinCLIP (Global + 3x3 Grid = 10 Views)
# ---------------------------------------------------------------------------
NORMAL_PROMPTS = [
    "a photo of a flawless object", "a photo of an intact product without any defect",
    "a photo of a clean undamaged industrial part", "a photo of a perfect manufactured item",
    "a close-up photo of a normal good quality product", "a photo of an unblemished object in good condition",
    "a photo of intact identical items with no damage", "a photo of a smooth clean surface with no scratches or cracks",
]
ANOMALY_PROMPTS = [
    "a photo of a damaged object", "a photo of a defective product with a crack",
    "a photo of a broken chipped industrial part", "a photo of a scratched or dented manufactured item",
    "a close-up photo of a flawed object with a stain or hole", "a photo of a bent deformed or broken component",
    "a photo of a product with a missing piece or cut", "a photo of a contaminated dirty damaged surface",
    "a photo of a rusted corroded metal part", "a photo of an anomaly or manufacturing defect on an object",
]


def _views_3x3(rgb: Image.Image) -> list[Image.Image]:
    w, h = rgb.size
    views = [rgb]
    cw, ch = max(1, int(w * 0.42)), max(1, int(h * 0.42))
    xs = [0, (w - cw) // 2, w - cw]
    ys = [0, (h - ch) // 2, h - ch]
    for y in ys:
        for x in xs:
            views.append(rgb.crop((x, y, x + cw, y + ch)))
    return views  # 10 views: 1 global + 9 overlapping 3x3 tiles


class _ClipDS(Dataset):
    def __init__(self, root: Path, role: str, ids: list[str], preprocess):
        self.root, self.role, self.ids, self.preprocess = root, role, ids, preprocess

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        p = self.root / self.role / img_id if not (self.root / img_id).is_file() else self.root / img_id
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            views = torch.stack([self.preprocess(v) for v in _views_3x3(rgb)], dim=0)
        return views, img_id


@torch.inference_mode()
def run_e06c_winclip(root: Path, role: str, ids: list[str], model, preprocess, text_feats, logit_scale):
    loader = DataLoader(_ClipDS(root, role, ids, preprocess), batch_size=12, shuffle=False, num_workers=4,
                        pin_memory=(DEVICE == "cuda"))
    scores = []
    for views, _ in loader:
        b, v, c, h, w = views.shape
        flat = views.view(b * v, c, h, w).to(DEVICE, non_blocking=True)
        with torch.autocast(device_type=DEVICE, enabled=(DEVICE == "cuda")):
            img_f = F.normalize(model.encode_image(flat).float(), dim=-1)
            logits = (logit_scale * (img_f @ text_feats.T)).view(b, v, 2)
            probs = F.softmax(logits, dim=-1)[:, :, 1]
        g_sc = probs[:, 0]
        l_sc = probs[:, 1:].max(dim=1).values
        scores.append((0.35 * g_sc + 0.65 * l_sc).cpu().numpy())
    return np.concatenate(scores)


# ---------------------------------------------------------------------------
# Post-Experiment Feature Visualization: Patch Anomaly Heatmap Overlay
# ---------------------------------------------------------------------------
def save_heatmap_sheet(root: Path, role: str, ids: list[str], scores: np.ndarray, patch_maps: np.ndarray,
                       grid: int, out_path: Path, title: str, top_k: int = 20):
    order = np.argsort(-scores)[:top_k]
    cols, tw, th, hh = 5, 220, 240, 32
    canvas = Image.new("RGB", (cols * tw, hh + 4 * th), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    for rank, idx in enumerate(order):
        r, c = divmod(rank, cols)
        x0, y0 = c * tw, hh + r * th
        img_id = ids[idx]
        p = root / role / img_id if not (root / img_id).is_file() else root / img_id
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB").resize((tw - 10, tw - 10), Image.Resampling.BILINEAR)
        hmap = patch_maps[idx].reshape(grid, grid)
        hmap = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        hm_img = Image.fromarray((hmap * 255).astype(np.uint8)).resize(rgb.size, Image.Resampling.BILINEAR)
        hm_arr = np.asarray(hm_img, dtype=np.float32) / 255.0
        base = np.asarray(rgb, dtype=np.float32) / 255.0
        # Red overlay where anomaly heatmap is high
        overlay = base.copy()
        overlay[:, :, 0] = np.clip(base[:, :, 0] * 0.6 + hm_arr * 0.8, 0, 1)
        overlay[:, :, 1] = np.clip(base[:, :, 1] * (1.0 - 0.5 * hm_arr), 0, 1)
        overlay[:, :, 2] = np.clip(base[:, :, 2] * (1.0 - 0.6 * hm_arr), 0, 1)
        canvas.paste(Image.fromarray((overlay * 255).astype(np.uint8)), (x0 + 5, y0 + 2))
        draw.text((x0 + 6, y0 + th - 26), f"#{rank + 1} {img_id[:13]} p={scores[idx]:.3f}", fill="black")
    canvas.save(out_path, quality=88)


def write_sub(path: Path, test_ids: list[str], scores: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    s = np.asarray(scores, dtype=np.float64)
    s = (s - s.min()) / (s.max() - s.min() + 1e-12)
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_path", "anomaly_score"])
        for img_id, sc in zip(test_ids, s):
            w.writerow([img_id, f"{sc:.6f}"])


def main():
    _ensure_deps()
    import open_clip
    import timm

    t0 = time.monotonic()
    seed_everything(SEED)
    root = find_data_root()
    train_rows, test_ids = load_tables(root)
    train_ids = [r[0] for r in train_rows]
    y_tr = np.array([r[1] for r in train_rows], dtype=np.int64)
    id_to_fold, groups = build_unseen_product_folds(root, train_rows, n_splits=5)
    fold_arr = np.array([id_to_fold[i] for i in train_ids], dtype=np.int64)
    print(f"[E06] Device={DEVICE} | Train={len(train_ids)} | Test={len(test_ids)} | Data={root}")

    # 1. Extract DINOv2-Base 518px multi-layer patches (pool_grid=12 -> 144 patches/img, <10GB total RAM)
    print("[1/3] Extracting DINOv2-Base 518px multi-layer patches (orig + domain-aug train + test)...")
    dino = timm.create_model("vit_base_patch14_dinov2.lvd142m", pretrained=True, img_size=518).to(DEVICE).eval()
    X_tr_orig, cls_tr, _ = extract_dinov2_patches(root, "train", train_ids, dino, img_size=518, domain_aug=False, pool_grid=12)
    X_tr_aug, _, _ = extract_dinov2_patches(root, "train", train_ids, dino, img_size=518, domain_aug=True, pool_grid=12)
    X_te, cls_te, _ = extract_dinov2_patches(root, "test", test_ids, dino, img_size=518, domain_aug=False, pool_grid=12)
    del dino
    if DEVICE == "cuda":
        torch.cuda.empty_cache()

    # 2. E06a: 5-Fold Unseen Patch-MIL
    print("[2/3] Training E06a Patch-MIL (5-Fold Unseen CV)...")
    oof_a, te_a, te_maps_a, auc_a = run_e06a_patch_mil(X_tr_orig, X_tr_aug, y_tr, fold_arr, X_te, epochs=12)
    del X_tr_aug

    # 3. E06b: Intra-Image + High-Sim Peer (cos > 0.88) MuSc (evaluated per unseen fold & on Test)
    print("[2b/3] Running E06b Intra-Image + High-Sim Peer MuSc...")
    oof_b = np.zeros(len(y_tr), dtype=np.float32)
    for f in range(5):
        m = fold_arr == f
        sc_f, _, _, _, _, _ = run_e06b_intra_and_highsim_musc(X_tr_orig[m], cls_tr[m], grid=12)
        oof_b[m] = sc_f
    auc_b = float(roc_auc_score(y_tr, oof_b))
    te_b, te_b_intra, te_b_peer, te_has_peer, te_max_sim, te_maps_b = run_e06b_intra_and_highsim_musc(X_te, cls_te, grid=12)
    del X_tr_orig, X_te
    print(f"==> [E06b Intra+HighSim MuSc] 5-Fold Unseen OOF AUROC = {auc_b:.4f} | Test high-sim peer ratio = {te_has_peer.mean():.3f}")

    # 4. E06c: OpenCLIP ViT-L-14-336 WinCLIP (Global + 3x3 = 10 views)
    print("[3/3] Running E06c OpenCLIP ViT-L-14-336 WinCLIP (10 views) on Test + Fold-4...")
    clip_model, _, clip_tf = open_clip.create_model_and_transforms("ViT-L-14-336", pretrained="openai")
    clip_tok = open_clip.get_tokenizer("ViT-L-14-336")
    clip_model = clip_model.to(DEVICE).eval()
    with torch.inference_mode():
        t_n = F.normalize(clip_model.encode_text(clip_tok(NORMAL_PROMPTS).to(DEVICE)).float(), dim=-1).mean(dim=0)
        t_a = F.normalize(clip_model.encode_text(clip_tok(ANOMALY_PROMPTS).to(DEVICE)).float(), dim=-1).mean(dim=0)
        text_f = F.normalize(torch.stack([t_n, t_a], dim=0), dim=-1)
        lscale = float(clip_model.logit_scale.exp().clamp(max=100.0).item())
    f4_idx = np.where(fold_arr == 4)[0]
    f4_c = run_e06c_winclip(root, "train", [train_ids[i] for i in f4_idx], clip_model, clip_tf, text_f, lscale)
    auc_c_f4 = float(roc_auc_score(y_tr[f4_idx], f4_c))
    te_c = run_e06c_winclip(root, "test", test_ids, clip_model, clip_tf, text_f, lscale)
    print(f"==> [E06c WinCLIP ViT-L-14-336] Fold-4 Unseen AUROC = {auc_c_f4:.4f}")

    # 5. Rank-Averaged 3-Paradigm Ensemble & Feature Research Diagnostics
    def to_rank(v):
        return rankdata(v) / len(v)

    oof_ab = 0.55 * to_rank(oof_a) + 0.45 * to_rank(oof_b)
    auc_ab = float(roc_auc_score(y_tr, oof_ab))
    f4_abc = 0.40 * to_rank(oof_a[f4_idx]) + 0.30 * to_rank(oof_b[f4_idx]) + 0.30 * to_rank(f4_c)
    auc_abc_f4 = float(roc_auc_score(y_tr[f4_idx], f4_abc))
    te_ens = 0.40 * to_rank(te_a) + 0.30 * to_rank(te_b) + 0.30 * to_rank(te_c)

    corr_ab = float(spearmanr(te_a, te_b).statistic)
    corr_ac = float(spearmanr(te_a, te_c).statistic)
    corr_bc = float(spearmanr(te_b, te_c).statistic)
    print(f"==> [E06 Ensemble] OOF(A+B)={auc_ab:.4f} | Fold-4(A+B+C)={auc_abc_f4:.4f}")
    print(f"==> [Feature Research] Test Spearman Corr: A-B={corr_ab:.3f}, A-C={corr_ac:.3f}, B-C={corr_bc:.3f}")

    # Save submissions + Heatmap Overlays + JSON Summary
    sub_dir = OUT_ROOT / "submissions"
    write_sub(sub_dir / "sub_e06a_dinov2_patch_mil.csv", test_ids, te_a)
    write_sub(sub_dir / "sub_e06b_dinov2_intra_peer_musc.csv", test_ids, te_b)
    write_sub(sub_dir / "sub_e06c_winclip_vitl14_336.csv", test_ids, te_c)
    write_sub(sub_dir / "sub_e06_rank_ensemble.csv", test_ids, te_ens)

    save_heatmap_sheet(root, "test", test_ids, te_ens, 0.5 * te_maps_a + 0.5 * te_maps_b, 12,
                       OUT_ROOT / "e06_test_heatmaps_top20.jpg", "E06 Ensemble Test Top-20 + Patch Anomaly Heatmap")

    summary = {
        "oof_auroc_e06a_patch_mil": auc_a,
        "oof_auroc_e06b_intra_highsim_musc": auc_b,
        "oof_auroc_ab_rank": auc_ab,
        "fold4_auroc_e06c_winclip_336": auc_c_f4,
        "fold4_auroc_abc_rank": auc_abc_f4,
        "test_highsim_peer_ratio": float(te_has_peer.mean()),
        "test_max_peer_sim_quantiles": {str(q): float(np.percentile(te_max_sim, q)) for q in (10, 25, 50, 75, 90)},
        "test_spearman_corr": {"A_B": corr_ab, "A_C": corr_ac, "B_C": corr_bc},
        "elapsed_seconds": time.monotonic() - t0,
    }
    (OUT_ROOT / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    # Optional W&B logging if WANDB_API_KEY is available in Kaggle Secrets
    if os.environ.get("WANDB_API_KEY"):
        import wandb
        run = wandb.init(entity="rlaqudwn123456-dankook-university", project="CV",
                         name="E06-gpu-dinov2-patchmil-musc-winclip336", group="unseen-product-loop")
        wandb.log({
            **{k: v for k, v in summary.items() if isinstance(v, (int, float))},
            "heatmaps/test_top20": wandb.Image(str(OUT_ROOT / "e06_test_heatmaps_top20.jpg")),
        })
        run.finish()


if __name__ == "__main__":
    main()
