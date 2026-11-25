"""Track A: MuSc-v2 Automated Blackwell & Kaggle Runner (M16~M19 Orthogonal Backbones).

Audited & hardened for strict Blackwell cgroup limits (VRAM <= 5.5 GiB, cgroup RAM <= 8.0 GiB,
Swap 0B, Disk <= 64 GB, CPU quota throttling):
  - Caps BLAS/OpenMP/PyTorch CPU threads to 2 BEFORE importing numpy/torch (prevents CPU overload)
  - Zero-copy disk-backed memmap streaming (NEVER loads full cluster into RAM) + periodic
    `mm.flush()` & `POSIX_FADV_DONTNEED` every 128 images (prevents cgroup page-cache OOMKill!)
  - Single-pass multi-layer extraction (2x faster!) with `output_fmt="NCHW"` + `bfloat16` autocast
  - Immediately frees unused 450M text tower on M18 (SigLIP-SO400M)
  - `ImageOps.exif_transpose` + genuine `Native-AR` (`672x448` / `448x672` / `518x518`) on M16
  - `LNA (r=1 + r=3)` + `min(orig, flip)` unique-peer `Bank-TTA` (fixes duplicate-slot lo_i bug)
  - VLM-Guided Bank Purification (uses M05/M03 normal confidence if available to prevent >50% defect bank pollution)
  - Multi-Granular 3-Channel + Global Logical Channel + Cluster-SVD Subspace + `32x32` 2D Anomaly Map saving
  - Private-Safe: verifies `test_ids` match before using `canonical_cids.npz`, else computes Cluster-v2 on-the-fly!

ZERO Test images are ever rendered, saved, or visually inspected.
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import subprocess
import sys
import time
from pathlib import Path

# CRITICAL: Cap CPU threads BEFORE importing numpy/torch to prevent cgroup CPU throttling & pod reset!
for _env in (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
):
    os.environ.setdefault(_env, "2")

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image, ImageOps  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402
from torchvision import transforms  # noqa: E402

from cluster_v2_and_recal import cluster_v2_zero_singleton, left_tail_robust_z  # noqa: E402

torch.set_num_threads(2)
try:
    torch.set_num_interop_threads(1)
except RuntimeError:
    pass


MODEL_REGISTRY = {
    "M16_musc_dinov2_reg4_l": {
        "paradigm": "A_MuSc_v2_Reg4",
        "backend": "timm_vit",
        "backbone": "vit_large_patch14_reg4_dinov2.lvd142m",
        "img_size": 518,
        "native_ar": True,  # (448, 672) for W/H>=1.25, (672, 448) for H/W>=1.25, else (518, 518)
        "layers": [17, 23],
        "layer_weights": [0.6, 0.4],
        "channel_weights": [0.72, 0.18, 0.10],
        "batch_size": 6,
        "license": "Apache-2.0 (Meta 2023)",
    },
    "M17_musc_eva02_l448": {
        "paradigm": "A_MuSc_v2_MIM",
        "backend": "timm_vit",
        "backbone": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "img_size": 448,
        "native_ar": False,
        "layers": [17, 23],
        "layer_weights": [0.6, 0.4],
        "channel_weights": [0.40, 0.60, 0.00],  # One-Change from M16 audit: drop noisy logical(0.7545), boost orthogonal svd(0.9191)
        "batch_size": 6,
        "license": "MIT (BAAI 2023)",
    },
    "M18_musc_siglip_so400m_384": {
        "paradigm": "A_MuSc_v2_VLM",
        "backend": "openclip_vit",
        "backbone": "ViT-SO400M-14-SigLIP-384",
        "pretrained": "webli",
        "img_size": 384,
        "native_ar": False,
        "layers": [18, 25],
        "layer_weights": [0.6, 0.4],
        "channel_weights": [0.40, 0.60, 0.00],
        "batch_size": 4,
        "license": "Apache-2.0 (Google 2023)",
    },
    "M19_musc_convnext_l384": {
        "paradigm": "A_MuSc_v2_CNN",
        "backend": "timm_cnn",
        "backbone": "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
        "img_size": 384,
        "native_ar": False,
        "layers": [1, 2],  # Stage-2 (48x48 kept native!) & Stage-3 (24x24)
        "layer_weights": [0.45, 0.55],
        "channel_weights": [0.40, 0.60, 0.00],
        "batch_size": 6,
        "license": "MIT (LAION/Meta 2023)",
    },
}


def fadvise_dontneed(path: Path) -> None:
    if hasattr(os, "posix_fadvise") and hasattr(os, "POSIX_FADV_DONTNEED") and path.exists():
        try:
            fd = os.open(str(path), os.O_RDONLY)
            try:
                os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            finally:
                os.close(fd)
        except OSError:
            pass


def get_mem_stats() -> tuple[float, float]:
    vram_gb = round(torch.cuda.memory_allocated() / (1024 ** 3), 2) if torch.cuda.is_available() else 0.0
    ram_gb = 0.0
    try:
        with open("/proc/self/status", "r", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    ram_gb = round(int(line.split()[1]) / (1024 ** 2), 2)
                    break
    except OSError:
        pass
    return vram_gb, ram_gb


def write_progress(prog_path: Path, payload: dict) -> None:
    prog_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = prog_path.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    tmp.replace(prog_path)


def init_wandb_safe(enabled: bool, model_id: str, cfg: dict):
    if not enabled:
        return None
    try:
        import wandb

        mode = "online" if (os.environ.get("WANDB_API_KEY") or (Path.home() / ".netrc").exists()) else "offline"
        run = wandb.init(
            entity=os.environ.get("WANDB_ENTITY", "rlaqudwn123456-dankook-university"),
            project=os.environ.get("WANDB_PROJECT", "CV"),
            name=f"trackA_{model_id}",
            group="track-a-musc-v2",
            tags=["track-a", "musc-v2", model_id],
            config=cfg,
            mode=mode,
        )
        print(f"[W&B] Initialized ({mode}) -> {getattr(run, 'url', 'offline')}", flush=True)
        return run
    except Exception as exc:
        print(f"[W&B] Skipped ({exc})", flush=True)
        return None


def resolve_img_path(data_root: Path, img_id: str, split: str) -> Path:
    p = data_root / split / img_id
    if p.exists():
        return p
    p2 = data_root / img_id
    if p2.exists():
        return p2
    raise FileNotFoundError(f"Image not found: {img_id} in {data_root}/{split}")


def pick_target_hw(data_root: Path, img_ids: list[str], split: str, cfg: dict) -> tuple[int, int]:
    """Determine Native-AR (H, W) from median aspect ratio of the split (only when native_ar=True)."""
    sz = cfg["img_size"]
    if not cfg.get("native_ar", False):
        return (sz, sz)
    ratios = []
    for img_id in img_ids[: min(64, len(img_ids))]:
        with Image.open(resolve_img_path(data_root, img_id, split)) as im:
            im = ImageOps.exif_transpose(im)
            ratios.append(im.width / max(1, im.height))
    med_ar = float(np.median(ratios)) if ratios else 1.0
    if med_ar >= 1.25:
        return (448, 672)  # 32x48 = 1,536 patches (+30% horizontal res, 0% distortion!)
    if med_ar <= 0.80:
        return (672, 448)
    return (sz, sz)


def load_backbone(cfg: dict, device: torch.device, pretrained: bool = True):
    backend = cfg["backend"]
    gc.collect()

    if backend == "timm_vit":
        import timm

        model = timm.create_model(
            cfg["backbone"],
            pretrained=pretrained,
            img_size=cfg["img_size"],
            dynamic_img_size=True,
        )
        model.eval().to(device=device)
        data_cfg = timm.data.resolve_model_data_config(model)
        norm_params = (data_cfg["mean"], data_cfg["std"])
        return model, norm_params

    if backend == "timm_cnn":
        import timm

        model = timm.create_model(
            cfg["backbone"],
            pretrained=pretrained,
            features_only=True,
            out_indices=cfg["layers"],
        )
        model.eval().to(device=device)
        data_cfg = timm.data.resolve_model_data_config(model)
        norm_params = (data_cfg["mean"], data_cfg["std"])
        return model, norm_params

    if backend == "openclip_vit":
        import open_clip

        # Load on CPU first -> immediately drop unused 450M text tower -> move ONLY visual to GPU!
        full_model, _, _ = open_clip.create_model_and_transforms(
            cfg["backbone"],
            pretrained=cfg["pretrained"] if pretrained else None,
            device="cpu",
        )
        visual = full_model.visual
        del full_model
        gc.collect()
        visual.eval().to(device=device)
        norm_params = ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        return visual, norm_params

    raise ValueError(f"Unknown backend: {backend}")


def make_transform(hw: tuple[int, int], norm_params: tuple) -> transforms.Compose:
    mean, std = norm_params
    return transforms.Compose([
        transforms.Resize(hw, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


@torch.no_grad()
def extract_multi_layer_tokens(
    model,
    backend: str,
    x: torch.Tensor,
    layers: list[int],
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Single-pass multi-layer extraction with bfloat16 autocast and output_fmt='NCHW'.

    Returns: (cls_emb [B, D_cls], list of patch_hwc [B, H, W, D] per layer)
    """
    use_amp = x.device.type == "cuda"
    amp_ctx = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else torch.no_grad()

    with amp_ctx:
        if backend == "timm_vit":
            _, inters = model.forward_intermediates(
                x, indices=layers, return_prefix_tokens=True, norm=True, output_fmt="NCHW"
            )
            patch_list = []
            cls_emb = None
            for item in inters:
                if isinstance(item, tuple):
                    p_nchw, prefix = item
                    if prefix is not None and prefix.shape[1] > 0:
                        cls_emb = prefix[:, 0, :].float()
                else:
                    p_nchw = item
                p_hwc = F.normalize(p_nchw.float().permute(0, 2, 3, 1).contiguous(), dim=-1)
                patch_list.append(p_hwc)
            if cls_emb is None:
                cls_emb = patch_list[-1].mean(dim=(1, 2))
            cls_emb = F.normalize(cls_emb.float(), dim=-1)
            return cls_emb, patch_list

        if backend == "timm_cnn":
            fmaps = model(x)  # returns list matching out_indices=[1, 2] -> Stage2 (48x48) & Stage3 (24x24)
            patch_list = []
            for fmap in fmaps:
                p_hwc = F.normalize(fmap.float().permute(0, 2, 3, 1).contiguous(), dim=-1)
                patch_list.append(p_hwc)
            cls_emb = F.normalize(fmaps[-1].float().mean(dim=(2, 3)), dim=-1)
            return cls_emb, patch_list

        if backend == "openclip_vit":
            # `model` is already `full_model.visual` (text tower freed!)
            trunk = model.trunk
            _, inters = trunk.forward_intermediates(
                x, indices=layers, return_prefix_tokens=True, norm=True, output_fmt="NCHW"
            )
            patch_list = []
            for item in inters:
                p_nchw = item[0] if isinstance(item, tuple) else item
                p_hwc = F.normalize(p_nchw.float().permute(0, 2, 3, 1).contiguous(), dim=-1)
                patch_list.append(p_hwc)
            # SigLIP has no CLS token -> global mean of final extracted layer
            cls_emb = F.normalize(patch_list[-1].mean(dim=(1, 2)), dim=-1)
            return cls_emb, patch_list

    raise ValueError(backend)


def load_vlm_normal_prior(models_dir: Path, img_ids: list[str], split: str) -> np.ndarray | None:
    """Load existing Zero-Shot VLM (M05 SigLIP + M03 WinCLIP) anomaly z-scores for Bank Purification."""
    z_list = []
    key_id = "oof_ids" if split == "train" else "test_ids"
    key_z = "oof_z" if split == "train" else "test_z"
    for prefix in ("M05", "M03"):
        matches = sorted(models_dir.glob(f"{prefix}_*.npz"))
        if not matches:
            continue
        d = np.load(matches[0], allow_pickle=True)
        mp = dict(zip([str(x) for x in d[key_id]], d[key_z].astype(np.float64)))
        if all(i in mp for i in img_ids):
            z_list.append(np.array([mp[i] for i in img_ids], dtype=np.float64))
    if not z_list:
        return None
    return np.mean(z_list, axis=0)


def apply_lna_r3(patch_hwc: torch.Tensor) -> torch.Tensor:
    """Local Neighborhood Aggregation (r=3) on GPU: (..., H, W, D) -> normalized (..., H, W, D)."""
    orig_ndim = patch_hwc.ndim
    if orig_ndim == 3:
        x = patch_hwc.unsqueeze(0)
    else:
        x = patch_hwc
    x_nchw = x.permute(0, 3, 1, 2).float()
    pooled = F.avg_pool2d(x_nchw, kernel_size=3, stride=1, padding=1)
    out = F.normalize(pooled.permute(0, 2, 3, 1).contiguous(), dim=-1).to(patch_hwc.dtype)
    return out.squeeze(0) if orig_ndim == 3 else out


def score_cluster_streaming(
    mm: np.memmap,
    idx: np.ndarray,
    cls_all: np.ndarray,
    vlm_prior: np.ndarray | None,
    device: torch.device,
    svd_rank: int = 48,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Zero-copy streaming MuSc-v2 + LNA(r=1,3) + min-merged Bank-TTA + Cluster-SVD + 32x32 Maps.

    NEVER loads the full cluster into CPU RAM (`mm[idx]`)! Only loads bank (<=128 imgs) + 1 query at a time.
    Returns: (musc_raw [N_c], svd_raw [N_c], logical_raw [N_c], map_32x32 [N_c, 32, 32] fp16)
    """
    n_c = len(idx)
    _, h, w, d = mm.shape
    p_cnt = h * w

    if n_c <= 1:
        pt = torch.from_numpy(np.array(mm[idx[0]], copy=True)).to(device=device, dtype=torch.float32)
        diff = 1.0 - (pt * torch.flip(pt, dims=[1])).sum(dim=-1)
        sc = float(torch.topk(diff.reshape(-1), k=max(1, int(p_cnt * 0.01))).values.mean().item())
        m32 = F.interpolate(diff.reshape(1, 1, h, w), size=(32, 32), mode="bilinear", align_corners=False)
        return (
            np.array([sc], dtype=np.float64),
            np.array([sc], dtype=np.float64),
            np.array([0.0], dtype=np.float64),
            m32.squeeze(0).squeeze(0).cpu().numpy().astype(np.float16)[None, :, :],
        )

    # 1. Core-Density + VLM-Guided Bank Purification (prevents >50% defect bank pollution!)
    cls_c = cls_all[idx]  # (N_c, D_cls)
    sim_cls = cls_c @ cls_c.T
    np.fill_diagonal(sim_cls, -1.0)
    k_c = min(max(2, n_c // 3), n_c - 1)
    den = np.partition(sim_cls, -k_c, axis=1)[:, -k_c:].mean(axis=1)
    den_z = (den - np.median(den)) / (np.std(den) + 1e-6)

    if vlm_prior is not None:
        vp = vlm_prior[idx]
        vp_z = (vp - np.median(vp)) / (np.std(vp) + 1e-6)
        # Higher density AND lower VLM anomaly score = purest normal core!
        purity = den_z - 0.50 * vp_z
    else:
        purity = den_z

    # Select top 45% purest normal pool, capped at 120 peers via greedy farthest-point diversity
    pool_sz = max(2, min(n_c, int(round(0.50 * n_c))))
    pure_order = np.argsort(purity)[::-1][:pool_sz]
    max_bank = 120
    if len(pure_order) > max_bank:
        # Farthest-point coreset inside pure normal pool (Scale-Invariant!)
        selected = [int(pure_order[0])]
        cand = cls_c[pure_order]
        min_d = 1.0 - (cand @ cand[0])
        for _ in range(max_bank - 1):
            nxt = int(np.argmax(min_d))
            selected.append(int(pure_order[nxt]))
            min_d = np.minimum(min_d, 1.0 - (cand @ cand[nxt]))
        bank_local = np.array(selected, dtype=np.int64)
    else:
        bank_local = pure_order

    # Load ONLY bank_local images (<= 120 imgs -> <= 330 MB RAM/VRAM!)
    bank_np = np.stack([np.array(mm[idx[bi]], copy=True) for bi in bank_local], axis=0)  # (B, H, W, D)
    bank_r1 = torch.from_numpy(bank_np).to(device=device, dtype=torch.float16)
    del bank_np
    bank_r1_flip = torch.flip(bank_r1, dims=[2])
    bank_r3 = apply_lna_r3(bank_r1)
    bank_r3_flip = torch.flip(bank_r3, dims=[2])

    # Global Logical Channel: normal bank mean CLS + mean patch vector
    bank_cls_mu = cls_c[bank_local].mean(axis=0)
    bank_cls_mu /= np.linalg.norm(bank_cls_mu) + 1e-8
    bank_patch_mu = F.normalize(bank_r1.float().mean(dim=(0, 1, 2)), dim=-1)  # (D,)

    # 2. Cluster-SVD Subspace (adaptive rank prevents small-cluster self-memorization!)
    flat_bank = bank_r1.reshape(-1, d).float()
    if flat_bank.shape[0] > 12288:
        sel = torch.linspace(0, flat_bank.shape[0] - 1, 12288, device=device).long()
        flat_sub = flat_bank[sel]
    else:
        flat_sub = flat_bank
    mu_b = flat_sub.mean(dim=0, keepdim=True)
    centered = flat_sub - mu_b
    q_eff = min(svd_rank, max(6, n_c * 2), d // 4, centered.shape[0] - 1)
    V_svd = torch.svd_lowrank(centered, q=q_eff, niter=2)[2] if q_eff >= 6 else None
    del flat_bank, flat_sub, centered

    B = bank_r1.shape[0]
    b1_f = bank_r1.reshape(B, p_cnt, d)
    b1_ff = bank_r1_flip.reshape(B, p_cnt, d)
    b3_f = bank_r3.reshape(B, p_cnt, d)
    b3_ff = bank_r3_flip.reshape(B, p_cnt, d)

    musc_scores = np.zeros(n_c, dtype=np.float64)
    svd_scores = np.zeros(n_c, dtype=np.float64)
    logical_scores = np.zeros(n_c, dtype=np.float64)
    maps_32 = np.zeros((n_c, 32, 32), dtype=np.float16)

    k_micro = max(1, int(round(p_cnt * 0.005)))
    k_meso = max(2, int(round(p_cnt * 0.05)))
    chunk = 8 if p_cnt > 1600 else 16

    for local_i in range(n_c):
        # Stream ONE query image at a time from disk memmap (2.8 MB!)
        q_r1 = torch.from_numpy(np.array(mm[idx[local_i]], copy=True)).to(device=device, dtype=torch.float16)
        q_r3 = apply_lna_r3(q_r1)
        q1_flat = q_r1.reshape(p_cnt, d)
        q3_flat = q_r3.reshape(p_cnt, d)

        valid_mask = bank_local != local_i
        if not valid_mask.any():
            valid_mask = np.ones(B, dtype=bool)
        vm_t = torch.from_numpy(valid_mask).to(device=device)
        m_uniq = int(valid_mask.sum())

        def _min_merged_pmap(q_fl, b_orig, b_flp):
            bo = b_orig[vm_t]
            bf = b_flp[vm_t]
            dists = torch.empty((p_cnt, m_uniq), device=device, dtype=torch.float32)
            for s0 in range(0, m_uniq, chunk):
                s1 = min(m_uniq, s0 + chunk)
                s_o = torch.matmul(bo[s0:s1], q_fl.T).max(dim=1).values.float()
                s_f = torch.matmul(bf[s0:s1], q_fl.T).max(dim=1).values.float()
                # Merge orig & flip per unique peer BEFORE sorting -> fixes duplicate-slot lo_i bug!
                dists[:, s0:s1] = (1.0 - torch.maximum(s_o, s_f)).T
            d_sorted, _ = torch.sort(dists, dim=1)
            lo_i = max(1, int(m_uniq * 0.08)) if m_uniq >= 4 else 0
            hi_i = min(m_uniq, max(lo_i + 1, int(m_uniq * 0.25)))
            return d_sorted[:, lo_i:hi_i].mean(dim=1)

        pmap_r1 = _min_merged_pmap(q1_flat, b1_f, b1_ff)
        pmap_r3 = _min_merged_pmap(q3_flat, b3_f, b3_ff)
        pmap = 0.65 * pmap_r1 + 0.35 * pmap_r3

        sc_micro = float(torch.topk(pmap, k=k_micro).values.mean().item())
        sc_meso = float(torch.topk(pmap, k=k_meso).values.mean().item())
        pmap_2d = pmap.reshape(1, 1, h, w)
        sc_spatial = float(F.avg_pool2d(pmap_2d, kernel_size=3, stride=1, padding=1).max().item())
        musc_scores[local_i] = 0.50 * sc_micro + 0.30 * sc_spatial + 0.20 * sc_meso

        # Save 32x32 2D Anomaly Map for cross-backbone Mid-Level Spatial Fusion!
        m32 = F.interpolate(pmap_2d, size=(32, 32), mode="bilinear", align_corners=False)
        maps_32[local_i] = m32.squeeze(0).squeeze(0).cpu().numpy().astype(np.float16)

        # Global Logical Anomaly Channel (catches missing components / count errors!)
        d_cls = float(1.0 - np.dot(cls_c[local_i], bank_cls_mu))
        q_pmu = F.normalize(q_r1.float().mean(dim=(0, 1)), dim=-1)
        d_pmu = float(1.0 - torch.dot(q_pmu, bank_patch_mu).item())
        logical_scores[local_i] = 0.5 * d_cls + 0.5 * d_pmu

        # Cluster-SVD Subspace Reconstruction Error
        if V_svd is not None:
            qc = q1_flat.float() - mu_b
            proj = (qc @ V_svd) @ V_svd.T
            recon_err = ((qc - proj) ** 2).sum(dim=-1)
            svd_scores[local_i] = float(torch.topk(recon_err, k=k_micro).values.mean().item())
        else:
            svd_scores[local_i] = musc_scores[local_i]

    del bank_r1, bank_r1_flip, bank_r3, bank_r3_flip, b1_f, b1_ff, b3_f, b3_ff, V_svd
    torch.cuda.empty_cache()
    return musc_scores, svd_scores, logical_scores, maps_32


def run_split_single_pass(
    model,
    norm_params: tuple,
    cfg: dict,
    data_root: Path,
    img_ids: list[str],
    split: str,
    cids: np.ndarray | None,
    vlm_prior: np.ndarray | None,
    tmp_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Single-pass multi-layer extraction + streaming scoring + immediate page-cache eviction."""
    n = len(img_ids)
    bs = cfg["batch_size"]
    backend = cfg["backend"]
    layers = cfg["layers"]
    lweights = cfg["layer_weights"]

    hw = pick_target_hw(data_root, img_ids, split, cfg)
    tf = make_transform(hw, norm_params)

    # Probe exact layer shapes with 1st image
    with Image.open(resolve_img_path(data_root, img_ids[0], split)) as im0:
        x0 = tf(ImageOps.exif_transpose(im0).convert("RGB")).unsqueeze(0).to(device)
    cls0, p0_list = extract_multi_layer_tokens(model, backend, x0, layers)
    cls_d = cls0.shape[1]

    mm_paths, mms = [], []
    for li, p0 in enumerate(p0_list):
        h, w, d = p0.shape[1], p0.shape[2], p0.shape[3]
        mp = tmp_dir / f"patches_{split}_L{li}.mmap"
        mm_paths.append(mp)
        mms.append(np.memmap(mp, dtype=np.float16, mode="w+", shape=(n, h, w, d)))

    cls_all = np.zeros((n, cls_d), dtype=np.float32)

    for s0 in range(0, n, bs):
        s1 = min(n, s0 + bs)
        imgs = []
        for img_id in img_ids[s0:s1]:
            with Image.open(resolve_img_path(data_root, img_id, split)) as im:
                imgs.append(tf(ImageOps.exif_transpose(im).convert("RGB")))
        xb = torch.stack(imgs, dim=0).to(device)
        cb, pb_list = extract_multi_layer_tokens(model, backend, xb, layers)
        cls_all[s0:s1] = cb.cpu().numpy()
        for li, pb in enumerate(pb_list):
            mms[li][s0:s1] = pb.cpu().numpy().astype(np.float16)

        # Periodic flush + page-cache eviction EVERY 128 images -> keeps cgroup RAM < 2.5 GiB!
        if (s1 % 128) < bs or s1 == n:
            for li, mm in enumerate(mms):
                mm.flush()
                fadvise_dontneed(mm_paths[li])

    # Compute on-the-fly Cluster-v2 if canonical_cids unavailable or mismatched (Private-Safe!)
    if cids is None:
        cids_fine = cluster_v2_zero_singleton(cls_all, agg_dist_thr=0.28)
        cids_coarse = cluster_v2_zero_singleton(cls_all, agg_dist_thr=0.36)
    else:
        cids_fine = cids
        cids_coarse = cids

    musc_tot = np.zeros(n, dtype=np.float64)
    svd_tot = np.zeros(n, dtype=np.float64)
    log_tot = np.zeros(n, dtype=np.float64)
    maps_tot = np.zeros((n, 32, 32), dtype=np.float32)

    for li, (mm, mp, lw) in enumerate(zip(mms, mm_paths, lweights)):
        m_out = np.zeros(n, dtype=np.float64)
        s_out = np.zeros(n, dtype=np.float64)
        l_out = np.zeros(n, dtype=np.float64)
        mp_out = np.zeros((n, 32, 32), dtype=np.float16)

        for c in np.unique(cids_fine):
            idx = np.where(cids_fine == c)[0]
            if c < 0:
                for si in idx:
                    ms, ss, ls, m32 = score_cluster_streaming(mm, np.array([si]), cls_all, vlm_prior, device)
                    m_out[si], s_out[si], l_out[si], mp_out[si] = ms[0], ss[0], ls[0], m32[0]
            else:
                ms, ss, ls, m32 = score_cluster_streaming(mm, idx, cls_all, vlm_prior, device)
                m_out[idx], s_out[idx], l_out[idx], mp_out[idx] = ms, ss, ls, m32
            fadvise_dontneed(mp)

        musc_tot += lw * m_out
        svd_tot += lw * s_out
        log_tot += lw * l_out
        maps_tot += lw * mp_out.astype(np.float32)

        # Immediately delete memmap after scoring this layer -> frees disk & page cache!
        del mm
        try:
            mp.unlink(missing_ok=True)
        except OSError:
            pass
        gc.collect()

    return musc_tot, svd_tot, log_tot, maps_tot.astype(np.float16), cids_coarse


def main() -> None:
    parser = argparse.ArgumentParser(description="Track A MuSc-v2 Hardened Blackwell/Kaggle Runner")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data-root", type=str, default="data/raw/open")
    parser.add_argument("--folds-json", type=str, default="runs/folds_unseen.json")
    parser.add_argument("--sample-sub", type=str, default="data/raw/open/sample_submission.csv")
    parser.add_argument("--canonical-cids", type=str, default="runs/day0_recal/canonical_cids.npz")
    parser.add_argument("--out-dir", type=str, default="runs/musc_v2")
    parser.add_argument("--queue-models-dir", type=str, default="runs/queue15/models")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--auto-sync", action="store_true")
    args = parser.parse_args()

    t_start = time.time()
    cfg = MODEL_REGISTRY[args.model]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    work_dir = Path(args.out_dir) / args.model
    tmp_dir = work_dir / "tmp_mmap"
    work_dir.mkdir(parents=True, exist_ok=True)
    tmp_dir.mkdir(parents=True, exist_ok=True)
    prog_path = work_dir / "progress.json"

    folds_payload = json.loads(Path(args.folds_json).read_text(encoding="utf-8"))
    labels_map = folds_payload["labels"]
    fold_list = [int(x.strip()) for x in args.folds.split(",") if x.strip()]

    # Load canonical_cids ONLY if IDs match (Private-Safe for arbitrary Private test sizes!)
    tr_cid_map, te_cid_cache_ids, te_cid_cache_vals = {}, [], None
    if Path(args.canonical_cids).exists():
        cd = np.load(args.canonical_cids, allow_pickle=True)
        tr_cid_map = dict(zip([str(x) for x in cd["train_ids"]], cd["train_cids"].astype(np.int64)))
        te_cid_cache_ids = [str(x) for x in cd["test_ids"]]
        te_cid_cache_vals = cd["test_cids"].astype(np.int64)
        print(f"[CLUSTER-V2] Loaded shared canonical_cids from {args.canonical_cids}", flush=True)

    wb_run = init_wandb_safe(args.wandb, args.model, cfg)
    model, norm_params = load_backbone(cfg, device, pretrained=True)

    total_stages = len(fold_list) + (0 if args.skip_test else 1)
    done_stages = 0

    oof_ids_all, oof_raw_all, oof_z_all, oof_y_all = [], [], [], []
    oof_musc_all, oof_svd_all, oof_log_all, oof_maps_all = [], [], [], []
    per_fold_aucs = {}

    for f in fold_list:
        ckpt_p = work_dir / f"ckpt_fold{f}.npz"
        val_ids = sorted(folds_payload["folds"][str(f)]["val_ids"])
        y_f = np.array([labels_map[i] for i in val_ids], dtype=np.int64)

        if ckpt_p.exists():
            ck = np.load(ckpt_p, allow_pickle=True)
            raw_f, z_f = ck["raw"].astype(np.float64), ck["z"].astype(np.float64)
            musc_f, svd_f, log_f = ck["musc"].astype(np.float64), ck["svd"].astype(np.float64), ck["logical"].astype(np.float64)
            maps_f = ck["maps_32"].astype(np.float16)
            print(f"[RESUME] Fold {f} loaded from {ckpt_p.name}", flush=True)
        else:
            cids_f = np.array([tr_cid_map[i] for i in val_ids], dtype=np.int64) if all(i in tr_cid_map for i in val_ids) else None
            vlm_f = load_vlm_normal_prior(Path(args.queue_models_dir), val_ids, "train")
            musc_f, svd_f, log_f, maps_f, cids_cal = run_split_single_pass(
                model, norm_params, cfg, Path(args.data_root), val_ids, "train", cids_f, vlm_f, tmp_dir, device
            )
            z_m = left_tail_robust_z(musc_f, cids_cal)
            z_s = left_tail_robust_z(svd_f, cids_cal)
            z_l = left_tail_robust_z(log_f, cids_cal)
            wm, ws, wl = cfg.get("channel_weights", [0.72, 0.18, 0.10])
            raw_f = wm * musc_f + ws * svd_f + wl * log_f
            z_f = wm * z_m + ws * z_s + wl * z_l
            np.savez_compressed(
                ckpt_p, ids=np.array(val_ids), raw=raw_f, z=z_f,
                musc=musc_f, svd=svd_f, logical=log_f, maps_32=maps_f,
            )

        auc_f = float(roc_auc_score(y_f, z_f))
        per_fold_aucs[f"fold_{f}"] = round(auc_f, 4)
        oof_ids_all.extend(val_ids)
        oof_raw_all.append(raw_f)
        oof_z_all.append(z_f)
        oof_y_all.append(y_f)
        oof_musc_all.append(musc_f)
        oof_svd_all.append(svd_f)
        oof_log_all.append(log_f)
        oof_maps_all.append(maps_f)

        done_stages += 1
        elapsed = time.time() - t_start
        eta = (elapsed / done_stages) * (total_stages - done_stages)
        vram_gb, ram_gb = get_mem_stats()
        prog = {
            "model": args.model,
            "percent": round(100.0 * done_stages / total_stages, 1),
            "stage": f"fold_{f}_done",
            "elapsed_sec": round(elapsed, 1),
            "eta_sec": round(eta, 1),
            "vram_gb": vram_gb,
            "ram_gb": ram_gb,
            "per_fold_auroc": per_fold_aucs,
        }
        write_progress(prog_path, prog)
        print(f"[{args.model}] Fold {f} AUROC={auc_f:.4f} | {prog['percent']}% | ETA={eta/60:.1f}m | VRAM={vram_gb}G RAM={ram_gb}G", flush=True)
        if wb_run is not None:
            wb_run.log({f"fold_auroc/fold_{f}": auc_f, "progress/percent": prog["percent"], "sys/ram_gb": ram_gb, "sys/vram_gb": vram_gb})

    oof_raw = np.concatenate(oof_raw_all)
    oof_z = np.concatenate(oof_z_all)
    oof_y = np.concatenate(oof_y_all)
    overall_oof = float(roc_auc_score(oof_y, oof_z))
    print(f"\n[{args.model}] Completed Folds {fold_list} -> Combined OOF AUROC = {overall_oof:.4f}", flush=True)

    if not args.skip_test and len(fold_list) == 5:
        with Path(args.sample_sub).open("r", encoding="utf-8-sig", newline="") as f:
            test_ids = [r["image_path"] for r in csv.DictReader(f)]
        ckpt_te = work_dir / "ckpt_test.npz"
        if ckpt_te.exists():
            ck = np.load(ckpt_te, allow_pickle=True)
            test_raw, test_z = ck["raw"].astype(np.float64), ck["z"].astype(np.float64)
            test_musc, test_svd, test_log = ck["musc"].astype(np.float64), ck["svd"].astype(np.float64), ck["logical"].astype(np.float64)
            test_maps = ck["maps_32"].astype(np.float16)
        else:
            # Strict ID-match guard for Private-Safe execution!
            te_cids = te_cid_cache_vals if (te_cid_cache_vals is not None and te_cid_cache_ids == test_ids) else None
            if te_cids is None:
                print("[CLUSTER-V2] test_ids differ from cache (Private set?) -> computing Cluster-v2 on-the-fly!", flush=True)
            vlm_te = load_vlm_normal_prior(Path(args.queue_models_dir), test_ids, "test")
            test_musc, test_svd, test_log, test_maps, cids_te_cal = run_split_single_pass(
                model, norm_params, cfg, Path(args.data_root), test_ids, "test", te_cids, vlm_te, tmp_dir, device
            )
            wm, ws, wl = cfg.get("channel_weights", [0.72, 0.18, 0.10])
            test_raw = wm * test_musc + ws * test_svd + wl * test_log
            test_z = (
                wm * left_tail_robust_z(test_musc, cids_te_cal)
                + ws * left_tail_robust_z(test_svd, cids_te_cal)
                + wl * left_tail_robust_z(test_log, cids_te_cal)
            )
            np.savez_compressed(
                ckpt_te, ids=np.array(test_ids), raw=test_raw, z=test_z,
                musc=test_musc, svd=test_svd, logical=test_log, maps_32=test_maps,
            )

        # Save lightweight .npz into queue15/models/ (~120 KB) + 32x32 2D maps separately (~23 MB)
        out_npz = Path(args.queue_models_dir) / f"{args.model}.npz"
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_id": args.model,
            "paradigm": cfg["paradigm"],
            "backbone": cfg["backbone"],
            "license": cfg["license"],
            "channel_weights": cfg.get("channel_weights", [0.72, 0.18, 0.10]),
            "raw_oof_auroc": round(overall_oof, 4),
            "per_fold": per_fold_aucs,
        }
        np.savez_compressed(
            out_npz,
            oof_ids=np.array(oof_ids_all),
            oof_raw=oof_raw,
            oof_z=oof_z,
            oof_musc=np.concatenate(oof_musc_all),
            oof_svd=np.concatenate(oof_svd_all),
            oof_logical=np.concatenate(oof_log_all),
            test_ids=np.array(test_ids),
            test_raw=test_raw,
            test_z=test_z,
            test_musc=test_musc,
            test_svd=test_svd,
            test_logical=test_log,
            meta_json=json.dumps(meta),
        )
        np.savez_compressed(
            work_dir / f"{args.model}_maps32.npz",
            oof_ids=np.array(oof_ids_all),
            oof_maps_32=np.concatenate(oof_maps_all, axis=0),
            test_ids=np.array(test_ids),
            test_maps_32=test_maps,
        )
        print(f"[SAVED QUEUE CACHE] {out_npz} (OOF={overall_oof:.4f}) + 32x32 2D maps", flush=True)

    write_progress(
        prog_path,
        {
            "model": args.model,
            "percent": 100.0,
            "stage": "complete",
            "elapsed_sec": round(time.time() - t_start, 1),
            "eta_sec": 0.0,
            "overall_oof_auroc": round(overall_oof, 4),
            "per_fold_auroc": per_fold_aucs,
        },
    )
    if wb_run is not None:
        wb_run.log({"summary/overall_oof_auroc": overall_oof, "progress/percent": 100.0})
        wb_run.finish()

    if args.auto_sync and not args.skip_test and len(fold_list) == 5:
        print("[AUTO-SYNC] Triggering scripts/auto_sync_and_report.py ...", flush=True)
        subprocess.run([sys.executable, "scripts/auto_sync_and_report.py", "--run-name", f"auto_{args.model}"], check=False)


if __name__ == "__main__":
    main()
