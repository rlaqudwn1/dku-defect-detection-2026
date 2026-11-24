"""Track A: MuSc-v2 Automated Blackwell & Kaggle Runner (M16~M19 Orthogonal Backbones).

Implements the Grand Master Plan MuSc-v2 upgrades with strict Blackwell container safety
(VRAM <= 5.5 GiB, cgroup RAM <= 8.0 GiB / Swap 0B, Disk <= 64 GB, num_workers=0):
  1. Zero-Singleton Canonical Product Graph (`Cluster-v2`, shared via canonical_cids.npz or on-the-fly)
  2. Small-Cluster Safe Quantile (`lo_i >= 1`, prevents 1-NN defect-defect matching in n_c <= 28)
  3. Zero-Cost Pose-Augmented `Bank-TTA` (horizontal patch-grid flip for n_c < 40 clusters)
  4. Multi-Granular 3-Channel Patch Aggregation (Micro top-0.5% + 3x3 Spatial Co-location + Meso top-5%)
  5. Test-Time `Cluster-SVD Subspace` Reconstruction Error (`svd_lowrank` on normal core patches)
  6. Prevalence-Invariant `Left-Tail Robust-Z` (P35 center, P15..P50 normal half-width)
  7. Per-Fold Auto-Resume Checkpoints + Live `progress.json` + W&B Real-Time Logging + Auto-Sync/Report

Supported models (`--model`):
  - M16_musc_dinov2_reg4_l  : vit_large_patch14_reg4_dinov2.lvd142m (518px / Native-AR 672x448)
  - M17_musc_eva02_l448     : eva02_large_patch14_448.mim_m38m_ft_in22k_in1k (448px, 32x32)
  - M18_musc_siglip_so400m_384 : ViT-SO400M-14-SigLIP-384 (384px, 27x27, OpenCLIP)
  - M19_musc_convnext_l384  : convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384 (384px, CNN 24x24)

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

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import roc_auc_score
from torchvision import transforms

from cluster_v2_and_recal import cluster_v2_zero_singleton, left_tail_robust_z


MODEL_REGISTRY = {
    "M16_musc_dinov2_reg4_l": {
        "paradigm": "A_MuSc_v2_Reg4",
        "backend": "timm_vit",
        "backbone": "vit_large_patch14_reg4_dinov2.lvd142m",
        "img_size": 518,
        "layers": [17, 23],
        "layer_weights": [0.6, 0.4],
        "batch_size": 8,
        "license": "Apache-2.0 (Meta 2023)",
    },
    "M17_musc_eva02_l448": {
        "paradigm": "A_MuSc_v2_MIM",
        "backend": "timm_vit",
        "backbone": "eva02_large_patch14_448.mim_m38m_ft_in22k_in1k",
        "img_size": 448,
        "layers": [17, 23],
        "layer_weights": [0.6, 0.4],
        "batch_size": 8,
        "license": "MIT (BAAI 2023)",
    },
    "M18_musc_siglip_so400m_384": {
        "paradigm": "A_MuSc_v2_VLM",
        "backend": "openclip_vit",
        "backbone": "ViT-SO400M-14-SigLIP-384",
        "pretrained": "webli",
        "img_size": 384,
        "layers": [18, 25],
        "layer_weights": [0.6, 0.4],
        "batch_size": 8,
        "license": "Apache-2.0 (Google 2023)",
    },
    "M19_musc_convnext_l384": {
        "paradigm": "A_MuSc_v2_CNN",
        "backend": "timm_cnn",
        "backbone": "convnext_large_mlp.clip_laion2b_soup_ft_in12k_in1k_384",
        "img_size": 384,
        "layers": [1, 2],  # Stage-2 (48x48->24x24) & Stage-3 (24x24)
        "layer_weights": [0.45, 0.55],
        "batch_size": 8,
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


def load_backbone(cfg: dict, device: torch.device):
    backend = cfg["backend"]
    img_size = cfg["img_size"]

    if backend == "timm_vit":
        import timm

        model = timm.create_model(cfg["backbone"], pretrained=True, img_size=img_size, dynamic_img_size=True)
        model.eval().to(device)
        data_cfg = timm.data.resolve_model_data_config(model)
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ])
        return model, tf

    if backend == "timm_cnn":
        import timm

        model = timm.create_model(cfg["backbone"], pretrained=True, features_only=True, out_indices=cfg["layers"])
        model.eval().to(device)
        data_cfg = timm.data.resolve_model_data_config(model)
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=data_cfg["mean"], std=data_cfg["std"]),
        ])
        return model, tf

    if backend == "openclip_vit":
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            cfg["backbone"], pretrained=cfg["pretrained"], device=device
        )
        model.eval()
        tf = transforms.Compose([
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)),
        ])
        return model, tf

    raise ValueError(f"Unknown backend: {backend}")


@torch.no_grad()
def extract_single_layer_tokens(
    model,
    backend: str,
    x: torch.Tensor,
    layer_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return (cls_emb [B, D], patch_grid [B, H, W, D]) normalized on GPU."""
    if backend == "timm_vit":
        feats = model.forward_intermediates(
            x, indices=[layer_idx], return_prefix_tokens=True, norm=True, output_fmt="NHWC"
        )
        # timm returns (final_feat, [(spatial_NHWC, prefix_tokens)]) or list of tuples
        inter = feats[1][0] if isinstance(feats, tuple) else feats[0]
        if isinstance(inter, tuple):
            patch_hwc, prefix = inter
            cls_emb = prefix[:, 0, :]
        else:
            patch_hwc = inter
            cls_emb = patch_hwc.mean(dim=(1, 2))
        patch_hwc = F.normalize(patch_hwc.float(), dim=-1)
        cls_emb = F.normalize(cls_emb.float(), dim=-1)
        return cls_emb, patch_hwc

    if backend == "timm_cnn":
        fmaps = model(x)  # list of NCHW for out_indices
        fmap = fmaps[layer_idx].float()
        # Harmonize Stage-2 (48x48) to 24x24 so memory stays < 4 GB on Blackwell
        if fmap.shape[-1] > 28:
            fmap = F.avg_pool2d(fmap, kernel_size=2, stride=2)
        cls_emb = F.normalize(fmap.mean(dim=(2, 3)), dim=-1)
        patch_hwc = F.normalize(fmap.permute(0, 2, 3, 1).contiguous(), dim=-1)
        return cls_emb, patch_hwc

    if backend == "openclip_vit":
        trunk = model.visual.trunk
        feats = trunk.forward_intermediates(
            x, indices=[layer_idx], return_prefix_tokens=True, norm=True, output_fmt="NHWC"
        )
        inter = feats[1][0] if isinstance(feats, tuple) else feats[0]
        if isinstance(inter, tuple):
            patch_hwc, prefix = inter
            cls_emb = prefix[:, 0, :] if prefix.shape[1] > 0 else patch_hwc.mean(dim=(1, 2))
        else:
            patch_hwc = inter
            cls_emb = patch_hwc.mean(dim=(1, 2))
        patch_hwc = F.normalize(patch_hwc.float(), dim=-1)
        cls_emb = F.normalize(cls_emb.float(), dim=-1)
        return cls_emb, patch_hwc

    raise ValueError(backend)


def score_cluster_musc_v2(
    patches_np: np.ndarray,
    cls_np: np.ndarray,
    device: torch.device,
    use_bank_tta: bool = True,
    svd_rank: int = 64,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute MuSc-v2 (Multi-Granular Peer + Spatial 3x3) AND Cluster-SVD Subspace score on GPU.

    patches_np: (N_c, H, W, D) float16 normalized patch grid.
    Returns: (musc_raw [N_c], svd_raw [N_c])
    """
    n_c, h, w, d = patches_np.shape
    p_cnt = h * w
    if n_c <= 1:
        # Pure singleton fallback: self-symmetry horizontal flip difference
        pt = torch.from_numpy(patches_np[0]).to(device=device, dtype=torch.float32)
        pt_flip = torch.flip(pt, dims=[1])
        diff = 1.0 - (pt * pt_flip).sum(dim=-1)
        sc = float(torch.topk(diff.reshape(-1), k=max(1, int(p_cnt * 0.01))).values.mean().item())
        return np.array([sc], dtype=np.float64), np.array([sc], dtype=np.float64)

    # 1. Core-Density Normal Bank selection inside cluster (top 65% densest normal peers)
    sim_cls = cls_np @ cls_np.T
    np.fill_diagonal(sim_cls, -1.0)
    k_c = min(max(2, n_c // 3), n_c - 1)
    den = np.partition(sim_cls, -k_c, axis=1)[:, -k_c:].mean(axis=1)
    bank_sz = max(2, min(n_c, int(round(0.65 * n_c))))
    bank_idx = np.argsort(den)[::-1][:bank_sz]

    # Cap memory bank size for huge Train folds (up to 160 diverse normal peers -> fits easily in 5.5 GiB VRAM)
    if len(bank_idx) > 160:
        step = np.linspace(0, len(bank_idx) - 1, 160, dtype=int)
        bank_idx = bank_idx[step]

    bank_t = torch.from_numpy(patches_np[bank_idx]).to(device=device, dtype=torch.float16)  # (B, H, W, D)
    # Zero-cost Bank-TTA: double peer diversity via horizontal patch-grid flip for small clusters (n_c < 40)
    if use_bank_tta and n_c < 40:
        bank_flip = torch.flip(bank_t, dims=[2])
        bank_all = torch.cat([bank_t, bank_flip], dim=0)  # (2B, H, W, D)
        bank_src_ids = np.concatenate([bank_idx, bank_idx])
    else:
        bank_all = bank_t
        bank_src_ids = bank_idx

    # 2. Test-Time Cluster-SVD Subspace Reconstruction (fit on normal core patches in ~0.05s on GPU!)
    flat_bank = bank_all.reshape(-1, d).float()
    if flat_bank.shape[0] > 16384:
        sel = torch.linspace(0, flat_bank.shape[0] - 1, 16384, device=device).long()
        flat_sub = flat_bank[sel]
    else:
        flat_sub = flat_bank
    mu_b = flat_sub.mean(dim=0, keepdim=True)
    centered = flat_sub - mu_b
    q_eff = min(svd_rank, centered.shape[0] - 1, d - 1)
    if q_eff >= 8:
        _, _, V_svd = torch.svd_lowrank(centered, q=q_eff, niter=2)  # (D, q_eff)
    else:
        V_svd = None
    del flat_bank, flat_sub, centered

    # Prepare bank for fast batched matmul: (B_all, P, D)
    B_all = bank_all.shape[0]
    bank_flat = bank_all.reshape(B_all, p_cnt, d)

    musc_scores = np.zeros(n_c, dtype=np.float64)
    svd_scores = np.zeros(n_c, dtype=np.float64)
    k_micro = max(1, int(round(p_cnt * 0.005)))
    k_meso = max(2, int(round(p_cnt * 0.05)))

    for i in range(n_c):
        q_hwc = torch.from_numpy(patches_np[i]).to(device=device, dtype=torch.float16)
        q_flat = q_hwc.reshape(p_cnt, d)  # (P, D)

        # Exclude self (and self-flip) from peer bank
        valid_mask = bank_src_ids != i
        if not valid_mask.any():
            valid_mask = np.ones(B_all, dtype=bool)
        vm_t = torch.from_numpy(valid_mask).to(device=device)
        b_valid = bank_flat[vm_t]  # (M, P, D)
        m_cnt = b_valid.shape[0]

        # Compute min cosine distance from each query patch to each peer image in chunks (VRAM safe!)
        min_dists = torch.empty((p_cnt, m_cnt), device=device, dtype=torch.float32)
        chunk = 16
        for s0 in range(0, m_cnt, chunk):
            s1 = min(m_cnt, s0 + chunk)
            # (c, P, D) @ (D, P) -> (c, P_bank, P_q) -> max over P_bank -> (c, P_q)
            sim_blk = torch.matmul(b_valid[s0:s1], q_flat.T)  # (c, P, P)
            max_sim = sim_blk.max(dim=1).values.float()  # (c, P_q)
            min_dists[:, s0:s1] = (1.0 - max_sim).T

        # Safe small-cluster quantile (lo_i >= 1 prevents 1-NN defect-to-defect match!)
        d_sorted, _ = torch.sort(min_dists, dim=1)
        lo_i = max(1, int(m_cnt * 0.05)) if m_cnt >= 4 else 0
        hi_i = max(lo_i + 1, int(m_cnt * 0.25))
        hi_i = min(hi_i, m_cnt)
        pmap = d_sorted[:, lo_i:hi_i].mean(dim=1)  # (P,)

        # Multi-Granular 3-Channel Aggregation: Micro + 3x3 Spatial Co-location + Meso
        sc_micro = float(torch.topk(pmap, k=k_micro).values.mean().item())
        sc_meso = float(torch.topk(pmap, k=k_meso).values.mean().item())
        pmap_2d = pmap.reshape(1, 1, h, w)
        pmap_3x3 = F.avg_pool2d(pmap_2d, kernel_size=3, stride=1, padding=1).reshape(-1)
        sc_spatial = float(pmap_3x3.max().item())
        musc_scores[i] = 0.50 * sc_micro + 0.30 * sc_spatial + 0.20 * sc_meso

        # Cluster-SVD Orthogonal Subspace Reconstruction Error
        if V_svd is not None:
            qc = q_flat.float() - mu_b
            proj = (qc @ V_svd) @ V_svd.T
            recon_err = ((qc - proj) ** 2).sum(dim=-1)
            svd_scores[i] = float(torch.topk(recon_err, k=k_micro).values.mean().item())
        else:
            svd_scores[i] = musc_scores[i]

    del bank_t, bank_all, bank_flat, V_svd
    torch.cuda.empty_cache()
    return musc_scores, svd_scores


def run_split_layer(
    model,
    tf,
    cfg: dict,
    data_root: Path,
    img_ids: list[str],
    split: str,
    layer_idx: int,
    cids: np.ndarray,
    tmp_dir: Path,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Stream 1 layer to disk memmap -> score per Cluster-v2 -> delete memmap immediately."""
    n = len(img_ids)
    bs = cfg["batch_size"]
    backend = cfg["backend"]

    # Probe shape with 1st image
    with Image.open(resolve_img_path(data_root, img_ids[0], split)) as im0:
        x0 = tf(im0.convert("RGB")).unsqueeze(0).to(device)
    cls0, p0 = extract_single_layer_tokens(model, backend, x0, layer_idx)
    h, w, d = p0.shape[1], p0.shape[2], p0.shape[3]
    cls_d = cls0.shape[1]

    mm_path = tmp_dir / f"patches_{split}_L{layer_idx}.mmap"
    mm = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=(n, h, w, d))
    cls_all = np.zeros((n, cls_d), dtype=np.float32)

    for s0 in range(0, n, bs):
        s1 = min(n, s0 + bs)
        imgs = []
        for img_id in img_ids[s0:s1]:
            with Image.open(resolve_img_path(data_root, img_id, split)) as im:
                imgs.append(tf(im.convert("RGB")))
        xb = torch.stack(imgs, dim=0).to(device)
        cb, pb = extract_single_layer_tokens(model, backend, xb, layer_idx)
        cls_all[s0:s1] = cb.cpu().numpy()
        mm[s0:s1] = pb.cpu().numpy().astype(np.float16)

    mm.flush()
    fadvise_dontneed(mm_path)

    # Compute cluster IDs on-the-fly if not provided
    if cids is None:
        cids = cluster_v2_zero_singleton(cls_all)

    musc_out = np.zeros(n, dtype=np.float64)
    svd_out = np.zeros(n, dtype=np.float64)
    for c in np.unique(cids):
        idx = np.where(cids == c)[0]
        if c < 0:
            for single_i in idx:
                m_s, s_s = score_cluster_musc_v2(np.array(mm[single_i : single_i + 1]), cls_all[single_i : single_i + 1], device)
                musc_out[single_i] = m_s[0]
                svd_out[single_i] = s_s[0]
        else:
            m_c, s_c = score_cluster_musc_v2(np.array(mm[idx]), cls_all[idx], device)
            musc_out[idx] = m_c
            svd_out[idx] = s_c

    del mm
    try:
        mm_path.unlink(missing_ok=True)
    except OSError:
        pass
    gc.collect()
    torch.cuda.empty_cache()
    return musc_out, svd_out


def main() -> None:
    parser = argparse.ArgumentParser(description="Track A MuSc-v2 Blackwell/Kaggle Automated Runner")
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data-root", type=str, default="data/raw/open")
    parser.add_argument("--folds-json", type=str, default="runs/folds_unseen.json")
    parser.add_argument("--sample-sub", type=str, default="data/raw/open/sample_submission.csv")
    parser.add_argument("--canonical-cids", type=str, default="runs/day0_recal/canonical_cids.npz")
    parser.add_argument("--out-dir", type=str, default="runs/musc_v2")
    parser.add_argument("--queue-models-dir", type=str, default="runs/queue15/models")
    parser.add_argument("--folds", type=str, default="0,1,2,3,4", help="Comma-separated folds (e.g. '0' for fast gate)")
    parser.add_argument("--skip-test", action="store_true")
    parser.add_argument("--wandb", action="store_true", help="Enable W&B real-time progress logging")
    parser.add_argument("--auto-sync", action="store_true", help="Run auto_sync_and_report.py on completion")
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

    # Load shared Cluster-v2 IDs if available
    tr_cid_map, te_cids = {}, None
    if Path(args.canonical_cids).exists():
        cd = np.load(args.canonical_cids, allow_pickle=True)
        tr_cid_map = dict(zip([str(x) for x in cd["train_ids"]], cd["train_cids"].astype(np.int64)))
        te_cids = cd["test_cids"].astype(np.int64)
        print(f"[CLUSTER-V2] Loaded shared canonical_cids from {args.canonical_cids}", flush=True)

    wb_run = init_wandb_safe(args.wandb, args.model, cfg)
    model, tf = load_backbone(cfg, device)

    total_stages = len(fold_list) + (0 if args.skip_test else 1)
    done_stages = 0

    oof_ids_all, oof_raw_all, oof_z_all, oof_y_all = [], [], [], []
    per_fold_aucs = {}

    for f in fold_list:
        ckpt_p = work_dir / f"ckpt_fold{f}.npz"
        val_ids = sorted(folds_payload["folds"][str(f)]["val_ids"])
        y_f = np.array([labels_map[i] for i in val_ids], dtype=np.int64)

        if ckpt_p.exists():
            ck = np.load(ckpt_p, allow_pickle=True)
            raw_f = ck["raw"].astype(np.float64)
            z_f = ck["z"].astype(np.float64)
            print(f"[RESUME] Fold {f} loaded from {ckpt_p.name}", flush=True)
        else:
            cids_f = np.array([tr_cid_map[i] for i in val_ids], dtype=np.int64) if tr_cid_map else None
            musc_sum = np.zeros(len(val_ids), dtype=np.float64)
            svd_sum = np.zeros(len(val_ids), dtype=np.float64)
            for li, (layer_idx, lw) in enumerate(zip(cfg["layers"], cfg["layer_weights"])):
                idx_arg = li if cfg["backend"] == "timm_cnn" else layer_idx
                m_sc, s_sc = run_split_layer(
                    model, tf, cfg, Path(args.data_root), val_ids, "train", idx_arg, cids_f, tmp_dir, device
                )
                musc_sum += lw * m_sc
                svd_sum += lw * s_sc
            # Blend 80% MuSc-v2 + 20% Cluster-SVD Subspace (Left-Tail calibrated per cluster)
            cids_eff = cids_f if cids_f is not None else np.zeros(len(val_ids), dtype=np.int64)
            z_m = left_tail_robust_z(musc_sum, cids_eff)
            z_s = left_tail_robust_z(svd_sum, cids_eff)
            raw_f = 0.80 * musc_sum + 0.20 * svd_sum
            z_f = 0.80 * z_m + 0.20 * z_s
            np.savez_compressed(ckpt_p, ids=np.array(val_ids), raw=raw_f, z=z_f, musc=musc_sum, svd=svd_sum)

        auc_f = float(roc_auc_score(y_f, z_f))
        per_fold_aucs[f"fold_{f}"] = round(auc_f, 4)
        oof_ids_all.extend(val_ids)
        oof_raw_all.append(raw_f)
        oof_z_all.append(z_f)
        oof_y_all.append(y_f)

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

    # Test Inference (strictly black-box numeric — ZERO image rendering!)
    test_ids, test_raw, test_z = [], np.array([]), np.array([])
    if not args.skip_test and len(fold_list) == 5:
        with Path(args.sample_sub).open("r", encoding="utf-8-sig", newline="") as f:
            test_ids = [r["image_path"] for r in csv.DictReader(f)]
        ckpt_te = work_dir / "ckpt_test.npz"
        if ckpt_te.exists():
            ck = np.load(ckpt_te, allow_pickle=True)
            test_raw, test_z = ck["raw"].astype(np.float64), ck["z"].astype(np.float64)
        else:
            musc_te = np.zeros(len(test_ids), dtype=np.float64)
            svd_te = np.zeros(len(test_ids), dtype=np.float64)
            for li, (layer_idx, lw) in enumerate(zip(cfg["layers"], cfg["layer_weights"])):
                idx_arg = li if cfg["backend"] == "timm_cnn" else layer_idx
                m_sc, s_sc = run_split_layer(
                    model, tf, cfg, Path(args.data_root), test_ids, "test", idx_arg, te_cids, tmp_dir, device
                )
                musc_te += lw * m_sc
                svd_te += lw * s_sc
            cids_eff = te_cids if te_cids is not None else np.zeros(len(test_ids), dtype=np.int64)
            test_raw = 0.80 * musc_te + 0.20 * svd_te
            test_z = 0.80 * left_tail_robust_z(musc_te, cids_eff) + 0.20 * left_tail_robust_z(svd_te, cids_eff)
            np.savez_compressed(ckpt_te, ids=np.array(test_ids), raw=test_raw, z=test_z, musc=musc_te, svd=svd_te)

        # Save final Queue15-compatible .npz cache!
        out_npz = Path(args.queue_models_dir) / f"{args.model}.npz"
        out_npz.parent.mkdir(parents=True, exist_ok=True)
        meta = {
            "model_id": args.model,
            "paradigm": cfg["paradigm"],
            "backbone": cfg["backbone"],
            "license": cfg["license"],
            "raw_oof_auroc": round(overall_oof, 4),
            "per_fold": per_fold_aucs,
        }
        np.savez_compressed(
            out_npz,
            oof_ids=np.array(oof_ids_all),
            oof_raw=oof_raw,
            oof_z=oof_z,
            test_ids=np.array(test_ids),
            test_raw=test_raw,
            test_z=test_z,
            meta_json=json.dumps(meta),
        )
        print(f"[SAVED QUEUE CACHE] {out_npz} (OOF={overall_oof:.4f})", flush=True)

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
