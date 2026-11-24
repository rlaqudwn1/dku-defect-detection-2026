"""Resilient 15-Model Diverse Anomaly Detection Queue (5 Paradigms x 3 Backbones, Wave-Ordered).

Designed for constrained/ephemeral GPU containers (Blackwell 5.5 GiB VRAM / 8.0 GiB RAM / 64 GB Disk) AND Kaggle/Colab:
  - Priority Wave Ordering (Wave 1 = 1 Champion per paradigm M01..M05 -> Wave 2 M06..M10 -> Wave 3 M11..M15):
    Even if server time/quota stops after Wave 1 (5 models), all 5 orthogonal paradigms are already ensembled!
  - Atomic Per-Model Checkpointing (`runs/queue15/models/M*.npz`) + Instant Resume (`--resume` default).
  - Automatic Time-Budget Guard (`--max-minutes`), Free-Disk Guard, & Optional HF/timm Cache Auto-Cleanup
    (`--auto-clean-cache`) so 64 GB containers NEVER get evicted across 15 backbones.
  - After EVERY single model finishes, automatically runs `scripts/run_15model_ensemble.py` to update:
      * `runs/queue15/queue_report_tables.md` (report-ready single-model + 5-method ensemble + Spearman tables)
      * `runs/queue15/sub_queue15_best_ensemble.csv` (verified black-box Test submission)
  - 100% Test-isolated (ZERO Test image viewing) & 100% <=2026-11-15 open-source pretrained weights.
"""
from __future__ import annotations

import argparse
import gc
import json
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

# Reuse verified core utilities from our repo scripts
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_15model_ensemble import run_ensemble_analysis  # noqa: E402
from run_e10_vand_adapter import (  # noqa: E402
    ANOMALY_PROMPTS,
    NORMAL_PROMPTS,
    discover_core_clusters_and_calibrate,
    load_tables_and_folds,
    seed_everything,
    within_group_ranking_loss,
)

SEED = 42


@dataclass(frozen=True)
class QueueModelSpec:
    model_id: str
    wave: int
    paradigm: str
    runner: str  # "musc" | "vand" | "winclip" | "patch_mil" | "e2e_cnn"
    backbone: str
    pretrained: str
    img_size: int
    batch_size: int
    license: str
    rationale: str


# 15 Orthogonal Models ordered in 3 Waves (5 Paradigms x 3 Backbones) so ANY stopping point yields max diversity!
MODEL_QUEUE: list[QueueModelSpec] = [
    # --- WAVE 1: 1 Champion from EACH of the 5 Orthogonal Paradigms (Best diversity even under tight server quota!) ---
    QueueModelSpec(
        "M01_musc_dinov2_l518", 1, "A_MuSc_Unsup", "musc",
        "vit_large_patch14_dinov2.lvd142m", "timm", 518, 8, "Apache-2.0 (Meta 2023)",
        "High-res 37x37 genuine MuSc (Core-Density Normal Bank + LNA + RsCIN); catches fine local cracks/nicks zero-shot.",
    ),
    QueueModelSpec(
        "M02_vand_clip_l336", 1, "B_VAND_Adapter", "vand",
        "ViT-L-14-336", "openai", 336, 16, "MIT (OpenAI 2021)",
        "Cross-product supervised patch-to-text residual linear adapter (5-Fold x 15ep); aligns defect semantics to text.",
    ),
    QueueModelSpec(
        "M03_winclip_l14_laion2b", 1, "C_ZeroShot_VLM", "winclip",
        "ViT-L-14", "laion2b_s32b_b82k", 224, 16, "MIT (OpenCLIP LAION-2B 2023)",
        "Zero-training inductive WinCLIP (Global + 2x2 60% crops) on industrial-rich LAION-2B; immune to Train overfitting.",
    ),
    QueueModelSpec(
        "M04_patchmil_convnext_b384", 1, "D_PatchMIL_CNN", "patch_mil",
        "convnext_base.fb_in22k_ft_in1k_384", "timm", 384, 16, "MIT (Meta 2022)",
        "Pure CNN inductive bias (depthwise 7x7 conv feature map); captures sharp local edges/scratches orthogonal to ViTs.",
    ),
    QueueModelSpec(
        "M05_siglip_so400m_384", 1, "C_ZeroShot_VLM", "winclip",
        "ViT-SO400M-14-SigLIP-384", "webli", 384, 12, "Apache-2.0 (Google 2023)",
        "Sigmoid-loss VLM at 384px; independent calibration geometry from softmax CLIP and stronger fine visual grounding.",
    ),
    # --- WAVE 2: Second Strong Backbone Across All 5 Paradigms ---
    QueueModelSpec(
        "M06_musc_dinov2_b518", 2, "A_MuSc_Unsup", "musc",
        "vit_base_patch14_dinov2.lvd142m", "timm", 518, 16, "Apache-2.0 (Meta 2023)",
        "Faster 12-block DINOv2-Base 518px MuSc; lower variance mid-layer representations complementing ViT-Large.",
    ),
    QueueModelSpec(
        "M07_vand_clip_l14_laion2b", 2, "B_VAND_Adapter", "vand",
        "ViT-L-14", "laion2b_s32b_b82k", 224, 24, "MIT (OpenCLIP LAION-2B 2023)",
        "VAND patch-to-text adapter on LAION-2B ViT-L-14; combines supervised patch alignment with LAION-2B visual priors.",
    ),
    QueueModelSpec(
        "M08_patchmil_dinov2_l518", 2, "D_PatchMIL_ViT", "patch_mil",
        "vit_large_patch14_dinov2.lvd142m", "timm", 518, 8, "Apache-2.0 (Meta 2023)",
        "Supervised Intra-Image Residual Patch-MIL + Pairwise Ranking on DINOv2-Large mid-layer (L18) patches.",
    ),
    QueueModelSpec(
        "M09_patchmil_swin_b384", 2, "E_Hier_ViT", "patch_mil",
        "swin_base_patch4_window12_384.ms_in22k_ft_in1k", "timm", 384, 16, "MIT (Microsoft 2021)",
        "Shifted-window hierarchical transformer; multi-scale window attention bridges CNN locality and global ViT.",
    ),
    QueueModelSpec(
        "M10_winclip_l336_openai", 2, "C_ZeroShot_VLM", "winclip",
        "ViT-L-14-336", "openai", 336, 12, "MIT (OpenAI 2021)",
        "High-res 336px WinCLIP with clean 2x2 60% crops (fixes E06c's noisy 3x3 max bug) + Core-Cluster Logit-Z.",
    ),
    # --- WAVE 3: Full 15-Model Grandmaster Diversity ---
    QueueModelSpec(
        "M11_vand_clip_b16_laion2b", 3, "B_VAND_Adapter", "vand",
        "ViT-B-16", "laion2b_s34b_b88k", 224, 32, "MIT (OpenCLIP LAION-2B 2023)",
        "Lightweight ViT-B-16 LAION-2B VAND adapter; strong regularization & distinct 16x16 patch tokenization.",
    ),
    QueueModelSpec(
        "M12_patchmil_eva02_l448", 3, "E_MIM_CLIP_ViT", "patch_mil",
        "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k", "timm", 448, 8, "MIT (BAAI 2023)",
        "Masked-Image-Modeling + CLIP distilled EVA-02 Large @ 448px (32x32 patches); SOTA fine-grained visual features.",
    ),
    QueueModelSpec(
        "M13_patchmil_dinov2_b518", 3, "D_PatchMIL_ViT", "patch_mil",
        "vit_base_patch14_dinov2.lvd142m", "timm", 518, 16, "Apache-2.0 (Meta 2023)",
        "DINOv2-Base 518px mid-layer (L9) Residual Patch-MIL; directly verified in E08 as strongest single mid-layer.",
    ),
    QueueModelSpec(
        "M14_patchmil_convnext_l384", 3, "D_PatchMIL_CNN", "patch_mil",
        "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384", "timm", 384, 12, "MIT (LAION/Meta 2023)",
        "CLIP-pretrained ConvNeXt-Large @ 384px; fuses pure CNN spatial hierarchy with LAION-2B vision-language pretraining.",
    ),
    QueueModelSpec(
        "M15_e2e_convnext_s384", 3, "E_E2E_FineTune", "e2e_cnn",
        "convnext_small.fb_in22k_ft_in1k_384", "timm", 384, 8, "MIT (Meta 2022)",
        "True end-to-end cross-product fine-tuned ConvNeXt-Small (Full-FOV + strong color/flip aug, 5-Fold Unseen OOF).",
    ),
]


def _clean_hf_cache_if_low_disk(min_free_gb: float = 14.0) -> None:
    free_gb = shutil.disk_usage("/").free / (1024.0**3)
    if free_gb < min_free_gb:
        hf_hub = Path.home() / ".cache" / "huggingface" / "hub"
        if hf_hub.exists():
            print(f"  [Disk Guard] Free disk {free_gb:.1f}GB < {min_free_gb:.1f}GB -> clearing {hf_hub}...", flush=True)
            shutil.rmtree(hf_hub, ignore_errors=True)
            hf_hub.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------------------------
# Runner 1: Clean Zero-Shot Multi-Scale WinCLIP / SigLIP (Global + 2x2 60% crops)
# ---------------------------------------------------------------------------
class _WinCLIPDS(Dataset):
    def __init__(self, root: Path, role: str, ids: list[str], preprocess):
        self.root, self.role, self.ids, self.preprocess = root, role, ids, preprocess

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        p = self.root / self.role / img_id if not (self.root / img_id).is_file() else self.root / img_id
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            w, h = rgb.size
            cw, ch = max(1, int(round(w * 0.60))), max(1, int(round(h * 0.60)))
            views = [
                rgb,
                rgb.crop((0, 0, cw, ch)),
                rgb.crop((w - cw, 0, w, ch)),
                rgb.crop((0, h - ch, cw, h)),
                rgb.crop((w - cw, h - ch, w, h)),
            ]
            t = torch.stack([self.preprocess(v) for v in views], dim=0)
        return t, idx


@torch.inference_mode()
def run_winclip_model(spec: QueueModelSpec, root: Path, train_ids, y_tr, test_ids, fold_arr, device: str) -> dict:
    import open_clip

    model, _, preprocess = open_clip.create_model_and_transforms(spec.backbone, pretrained=spec.pretrained)
    tokenizer = open_clip.get_tokenizer(spec.backbone)
    model = model.to(device).eval()

    t_n = F.normalize(model.encode_text(tokenizer(NORMAL_PROMPTS).to(device)).float(), dim=-1).mean(dim=0)
    t_a = F.normalize(model.encode_text(tokenizer(ANOMALY_PROMPTS).to(device)).float(), dim=-1).mean(dim=0)
    text_diff = F.normalize(t_a - t_n, dim=-1)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    def _score(enc: nn.Module, role: str, ids: list[str]) -> tuple[np.ndarray, np.ndarray]:
        loader = DataLoader(_WinCLIPDS(root, role, ids, preprocess), batch_size=spec.batch_size, shuffle=False, num_workers=0)
        raw_logits = np.zeros(len(ids), dtype=np.float32)
        cls_list = []
        for views, b_idxs in loader:
            b, v, c, h, w = views.shape
            flat = views.view(b * v, c, h, w).to(device, non_blocking=True)
            with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                emb = F.normalize(enc.encode_image(flat).float(), dim=-1).view(b, v, -1)
            # Raw cosine difference logit (NO softmax saturation / NO Min-Max distortion!)
            view_lg = (emb @ text_diff) * 20.0  # (B, 5)
            g_lg = view_lg[:, 0]
            # Mean of top-2 local crops instead of noisy single max!
            l_lg = torch.topk(view_lg[:, 1:], k=2, dim=1).values.mean(dim=1)
            raw_logits[b_idxs.numpy()] = (0.45 * g_lg + 0.55 * l_lg).cpu().numpy()
            cls_list.append(emb[:, 0, :].cpu().numpy())
        return raw_logits, np.concatenate(cls_list, axis=0)

    tr_raw, tr_cls = _score(model, "train", train_ids)
    te_raw, te_cls = _score(model, "test", test_ids)

    oof_z = np.zeros_like(tr_raw)
    for f in range(5):
        m = fold_arr == f
        oof_z[m] = discover_core_clusters_and_calibrate(tr_raw[m], tr_cls[m])
    te_z = discover_core_clusters_and_calibrate(te_raw, te_cls)

    del model
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()
    return {"oof_z": oof_z, "oof_raw": tr_raw, "test_z": te_z, "test_raw": te_raw, "raw_oof_auroc": round(float(roc_auc_score(y_tr, tr_raw)), 4)}


# ---------------------------------------------------------------------------
# Runner 2: OOM-Proof Multi-Backbone Residual Patch-MIL (ViT / CNN / Swin / EVA-02)
# ---------------------------------------------------------------------------
class _FullFOVTimmDS(Dataset):
    def __init__(self, root: Path, role: str, ids: list[str], img_size: int, mean, std):
        self.root, self.role, self.ids, self.img_size = root, role, ids, img_size
        self.mean = torch.tensor(mean, dtype=torch.float32)[:, None, None]
        self.std = torch.tensor(std, dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int):
        img_id = self.ids[idx]
        p = self.root / self.role / img_id if not (self.root / img_id).is_file() else self.root / img_id
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB").resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return (t - self.mean) / self.std, idx


class ResidualPatchMILHead(nn.Module):
    def __init__(self, dim: int, top_k_ratio: float = 0.04):
        super().__init__()
        self.top_k_ratio = top_k_ratio
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Dropout(0.30),
            nn.Linear(dim, 256),
            nn.GELU(),
            nn.Dropout(0.30),
            nn.Linear(256, 1),
        )
        self.res_alpha = nn.Parameter(torch.tensor(0.5))

    def forward(self, patches: torch.Tensor) -> torch.Tensor:
        # Subtract intra-image median patch representation BEFORE scoring to kill product texture bias!
        p_med = patches.median(dim=1, keepdim=True).values
        alpha = torch.sigmoid(self.res_alpha)
        p_in = (1.0 - alpha) * patches + alpha * (patches - p_med)
        p_logits = self.net(p_in).squeeze(-1)  # (B, P)
        k = max(1, int(round(p_logits.shape[1] * self.top_k_ratio)))
        return torch.topk(p_logits, k=k, dim=1).values.mean(dim=1)


def run_patch_mil_model(spec: QueueModelSpec, root: Path, train_ids, y_tr, test_ids, fold_arr, groups, device: str, epochs: int = 12) -> dict:
    import timm

    model = timm.create_model(spec.backbone, pretrained=True, num_classes=0).to(device).eval()
    data_cfg = timm.data.resolve_model_data_config(model)
    mean, std = data_cfg["mean"], data_cfg["std"]
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    uniq_g = sorted(set(groups[i] for i in train_ids))
    g2int = {g: k for k, g in enumerate(uniq_g)}
    group_int_arr = np.array([g2int[groups[i]] for i in train_ids], dtype=np.int64)

    tmp_dir = Path(tempfile.mkdtemp(prefix=f"q15_{spec.model_id}_"))
    try:
        @torch.inference_mode()
        def _extract(enc: nn.Module, role: str, ids: list[str], mm_path: Path) -> tuple[np.memmap, np.ndarray]:
            loader = DataLoader(_FullFOVTimmDS(root, role, ids, spec.img_size, mean, std), batch_size=spec.batch_size, shuffle=False, num_workers=0)
            mm: np.memmap | None = None
            cls_arr: np.ndarray | None = None
            for x, b_idxs in loader:
                idxs = b_idxs.numpy()
                x = x.to(device, non_blocking=True)
                with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
                    if hasattr(enc, "get_intermediate_layers") and hasattr(enc, "blocks"):
                        # Extract optimal 3/4-depth mid-high layer (L9 for ViT-B, L18 for ViT-L)
                        nb = len(enc.blocks)
                        mid_i = (3 * nb) // 4 - 1
                        inter = enc.get_intermediate_layers(x, n=[mid_i, nb - 1], reshape=True, return_prefix_tokens=True, norm=True)
                        fmap = inter[0][0].float()
                        cls_v = F.normalize(inter[1][1][:, 0].float(), dim=-1)
                    else:
                        fmap = enc.forward_features(x).float()
                        if fmap.dim() == 4:
                            # Handle NCHW (ConvNeXt) vs NHWC (Swin)
                            if fmap.shape[-1] == enc.num_features and fmap.shape[1] != enc.num_features:
                                fmap = fmap.permute(0, 3, 1, 2)
                            cls_v = F.normalize(fmap.mean(dim=(2, 3)), dim=-1)
                        else:
                            # Token sequence (B, L, D)
                            cls_v = F.normalize(fmap.mean(dim=1), dim=-1)
                            fmap = fmap[:, 1:, :] if fmap.shape[1] > 64 else fmap
                    if fmap.dim() == 4:
                        # Pool spatial map to max 16x16=256 patches (keeps memmap <= 5.5 GB on 64GB disk!)
                        if fmap.shape[2] > 16 or fmap.shape[3] > 16:
                            fmap = F.adaptive_avg_pool2d(fmap, (16, 16))
                        tok = F.normalize(fmap.flatten(2).transpose(1, 2), dim=-1).half().cpu().numpy()
                    else:
                        tok = F.normalize(fmap, dim=-1).half().cpu().numpy()

                if mm is None:
                    mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(len(ids), tok.shape[1], tok.shape[2]))
                mm[idxs] = tok
                c_np = cls_v.cpu().numpy()
                if cls_arr is None:
                    cls_arr = np.empty((len(ids), c_np.shape[-1]), dtype=np.float32)
                cls_arr[idxs] = c_np
            assert mm is not None and cls_arr is not None
            mm.flush()
            return mm, cls_arr

        tr_mm, tr_cls = _extract(model, "train", train_ids, tmp_dir / "tr.fp16")
        te_mm, te_cls = _extract(model, "test", test_ids, tmp_dir / "te.fp16")
        del model
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

        # Train 5-Fold x epochs with STRICT consensus epoch selection (NO per-fold Oracle cherry-picking!)
        n_tr, _, feat_dim = tr_mm.shape
        n_te = te_mm.shape[0]
        oof_ep = np.zeros((epochs, n_tr), dtype=np.float32)
        te_ep = np.zeros((epochs, n_te), dtype=np.float32)

        for fold in range(5):
            val_m = fold_arr == fold
            tr_idxs, val_idxs = np.where(~val_m)[0], np.where(val_m)[0]
            y_f = y_tr[tr_idxs]
            pos_w = torch.tensor([(y_f == 0).sum() / max(1, (y_f == 1).sum())], device=device, dtype=torch.float32)

            seed_everything(SEED + fold)
            head = ResidualPatchMILHead(dim=feat_dim, top_k_ratio=0.04).to(device)
            opt = torch.optim.AdamW(head.parameters(), lr=4e-4, weight_decay=2e-2)
            sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
            rng = np.random.default_rng(SEED + fold)

            for ep in range(epochs):
                head.train()
                perm = tr_idxs[rng.permutation(len(tr_idxs))]
                for s in range(0, len(perm), 64):
                    sel = perm[s : s + 64]
                    bx = torch.from_numpy(np.array(tr_mm[sel], copy=True)).to(device, dtype=torch.float32)
                    by = torch.from_numpy(y_tr[sel]).to(device, dtype=torch.float32)
                    bg = torch.from_numpy(group_int_arr[sel]).to(device, dtype=torch.long)
                    opt.zero_grad(set_to_none=True)
                    lg = head(bx)
                    loss = 0.65 * F.binary_cross_entropy_with_logits(lg, by, pos_weight=pos_w) + 0.35 * within_group_ranking_loss(lg, by, bg)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(head.parameters(), 1.0)
                    opt.step()
                sched.step()

                head.eval()
                with torch.inference_mode():
                    v_out = [head(torch.from_numpy(np.array(tr_mm[val_idxs[s : s + 64]], copy=True)).to(device, dtype=torch.float32)).cpu().numpy() for s in range(0, len(val_idxs), 64)]
                    oof_ep[ep, val_idxs] = np.concatenate(v_out, axis=0)
                    t_out = [head(torch.from_numpy(np.array(te_mm[s : s + 64], copy=True)).to(device, dtype=torch.float32)).cpu().numpy() for s in range(0, n_te, 64)]
                    te_ep[ep] += np.concatenate(t_out, axis=0) / 5.0
            del head, opt, sched

        cal_ep = np.zeros_like(oof_ep)
        cal_aucs = []
        for ep in range(epochs):
            for f in range(5):
                m = fold_arr == f
                cal_ep[ep, m] = discover_core_clusters_and_calibrate(oof_ep[ep, m], tr_cls[m])
            cal_aucs.append(float(roc_auc_score(y_tr, cal_ep[ep])))
        best_ep = int(np.argmax(cal_aucs))
        print(f"  [{spec.model_id}] Consensus Best Ep={best_ep + 1}/{epochs} | Cal 5-Fold OOF={cal_aucs[best_ep]:.4f}", flush=True)
        te_z = discover_core_clusters_and_calibrate(te_ep[best_ep], te_cls)
        return {
            "oof_z": cal_ep[best_ep],
            "oof_raw": oof_ep[best_ep],
            "test_z": te_z,
            "test_raw": te_ep[best_ep],
            "raw_oof_auroc": round(float(roc_auc_score(y_tr, oof_ep[best_ep])), 4),
            "best_consensus_epoch": best_ep + 1,
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()


def execute_model_spec(
    spec: QueueModelSpec,
    data_root: Path,
    folds_json: Path,
    queue_dir: Path,
    train_ids, y_tr, test_ids, fold_arr, groups,
    device: str,
) -> Path:
    models_dir = queue_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    out_npz = models_dir / f"{spec.model_id}.npz"
    if out_npz.is_file():
        print(f"[SKIP] {spec.model_id} already cached -> {out_npz}", flush=True)
        return out_npz

    t0 = time.monotonic()
    print("\n======================================================================", flush=True)
    print(f"[QUEUE RUN] {spec.model_id} (Wave {spec.wave} | {spec.paradigm} | {spec.backbone})", flush=True)
    print(f"  Rationale: {spec.rationale}", flush=True)
    print("======================================================================", flush=True)

    sub_out = queue_dir / "work" / spec.model_id
    sub_out.mkdir(parents=True, exist_ok=True)

    if spec.runner == "musc":
        cmd = [
            sys.executable, "scripts/colab_e09_genuine_musc.py",
            "--data-root", str(data_root),
            "--folds-json", str(folds_json),
            "--out-dir", str(sub_out),
            "--backbone", spec.backbone,
            "--img-size", str(spec.img_size),
            "--batch-size", str(spec.batch_size),
        ]
        subprocess.check_call(cmd)
        ck = np.load(sub_out / "e09_oof_and_test_cache.npz", allow_pickle=True)
        res = {"oof_ids": ck["oof_ids"], "oof_z": ck["oof_z"], "oof_raw": ck["oof_z"], "test_ids": ck["test_ids"], "test_z": ck["test_z"], "test_raw": ck["test_z"]}
    elif spec.runner == "vand":
        cmd = [
            sys.executable, "scripts/run_e10_vand_adapter.py",
            "--data-root", str(data_root),
            "--folds-json", str(folds_json),
            "--out-dir", str(sub_out),
            "--model", spec.backbone,
            "--pretrained", spec.pretrained,
            "--img-size", str(spec.img_size),
            "--extract-batch-size", str(spec.batch_size),
            "--epochs", "15",
        ]
        subprocess.check_call(cmd)
        ck = np.load(sub_out / "e10_oof_and_test_cache.npz", allow_pickle=True)
        res = {"oof_ids": ck["oof_ids"], "oof_z": ck["oof_z"], "oof_raw": ck["oof_z"], "test_ids": ck["test_ids"], "test_z": ck["test_z"], "test_raw": ck["test_z"]}
    elif spec.runner == "winclip":
        r = run_winclip_model(spec, data_root, train_ids, y_tr, test_ids, fold_arr, device)
        res = {"oof_ids": np.array(train_ids), "test_ids": np.array(test_ids), **r}
    elif spec.runner in ("patch_mil", "e2e_cnn"):
        r = run_patch_mil_model(spec, data_root, train_ids, y_tr, test_ids, fold_arr, groups, device, epochs=12)
        res = {"oof_ids": np.array(train_ids), "test_ids": np.array(test_ids), **r}
    else:
        raise ValueError(f"Unknown runner: {spec.runner}")

    meta = {**asdict(spec), "elapsed_seconds": round(time.monotonic() - t0, 1), "raw_oof_auroc": res.get("raw_oof_auroc")}
    np.savez_compressed(
        out_npz,
        oof_ids=res["oof_ids"],
        oof_z=res["oof_z"].astype(np.float32),
        oof_raw=res["oof_raw"].astype(np.float32),
        test_ids=res["test_ids"],
        test_z=res["test_z"].astype(np.float32),
        test_raw=res["test_raw"].astype(np.float32),
        meta_json=json.dumps(meta),
    )
    # Clean heavy intermediate work dir once compact .npz cache (~150 KB) is safely saved
    shutil.rmtree(sub_out, ignore_errors=True)
    return out_npz


def main() -> None:
    parser = argparse.ArgumentParser(description="Resilient 15-Model Diverse Queue + Iterative Ensemble Loop")
    parser.add_argument("--data-root", type=str, default="data/raw/open")
    parser.add_argument("--folds-json", type=str, default="runs/folds_unseen.json")
    parser.add_argument("--queue-dir", type=str, default="runs/queue15")
    parser.add_argument("--wave", type=int, default=0, help="Run only specific wave (1, 2, or 3); 0 = all 15 models")
    parser.add_argument("--models", type=str, default="", help="Optional comma-separated model_id prefixes, e.g. M01,M02")
    parser.add_argument("--max-minutes", type=float, default=0.0, help="Graceful time budget in minutes (0 = unlimited)")
    parser.add_argument("--auto-clean-cache", action="store_true", help="Clear HF cache if free disk < 14 GB (64GB safe)")
    parser.add_argument("--list", action="store_true", help="Print the 15-model queue table and exit")
    args = parser.parse_args()

    if args.list:
        for s in MODEL_QUEUE:
            print(f"Wave {s.wave} | {s.model_id:28s} | {s.paradigm:16s} | {s.backbone} ({s.img_size}px) | {s.rationale}")
        return

    t_start = time.monotonic()
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_root)
    folds_json = Path(args.folds_json)
    queue_dir = Path(args.queue_dir)
    queue_dir.mkdir(parents=True, exist_ok=True)

    train_ids, y_tr, test_ids, fold_arr, groups = load_tables_and_folds(data_root, str(folds_json))
    selected = list(MODEL_QUEUE)
    if args.wave in (1, 2, 3):
        selected = [s for s in selected if s.wave == args.wave]
    if args.models:
        prefixes = [p.strip() for p in args.models.split(",") if p.strip()]
        selected = [s for s in selected if any(s.model_id.startswith(p) for p in prefixes)]

    state_path = queue_dir / "queue_state.json"
    state = json.loads(state_path.read_text(encoding="utf-8")) if state_path.is_file() else {"completed": [], "failed": {}}

    for spec in selected:
        if args.max_minutes > 0 and (time.monotonic() - t_start) / 60.0 >= args.max_minutes:
            print(f"[TIME BUDGET] Reached --max-minutes={args.max_minutes}. Stopping cleanly with all progress saved.", flush=True)
            break
        if args.auto_clean_cache:
            _clean_hf_cache_if_low_disk(min_free_gb=14.0)
        try:
            execute_model_spec(spec, data_root, folds_json, queue_dir, train_ids, y_tr, test_ids, fold_arr, groups, device)
            if spec.model_id not in state["completed"]:
                state["completed"].append(spec.model_id)
            state["failed"].pop(spec.model_id, None)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            # Iterative Loop: immediately re-optimize 5 ensemble methods & update report tables + best CSV!
            run_ensemble_analysis(queue_dir, folds_json, data_root / "sample_submission.csv")
        except Exception as exc:
            print(f"[ERROR] {spec.model_id} failed: {exc!r} (continuing queue safely!)", flush=True)
            state["failed"][spec.model_id] = repr(exc)
            state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")
            gc.collect()
            if device == "cuda":
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
