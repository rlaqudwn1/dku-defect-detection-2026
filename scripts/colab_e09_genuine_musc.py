"""E09: Genuine High-Resolution Multi-Layer MuSc (ICLR 2024) + Unseen 5-Fold Layer/Hyperparam Optimization.

Designed for Google Colab GPU (A100 40GB / L4 24GB / T4 16GB) & Local Linux GPU.
Strictly adheres to Competition Rules & Test Isolation:
  - Pretrained weights: `timm` `vit_large_patch14_dinov2.lvd142m` (Meta AI, Apache-2.0, published April 2023 <= 2026-11-15).
  - ZERO Test image viewing, heatmaps, or visual galleries. Test is scored purely once in black-box mode.
  - Fixes all 4 bottlenecks identified in E06/E07:
      1) Preserves FULL 37x37 = 1,369 high-res patch grid at 518px (NO 12x12 pooling loss) + LNA (r=1, 3).
      2) Replaces buggy top-1 max peer matching with Genuine MuSc Interval Average (MSM quantiles) over an
         unsupervised High-Density Normal Core Reference Bank (~97% normal purity) + Singleton fallback.
      3) Systematic Multi-Layer ([6, 12, 18, 24]) + Hyperparameter optimization via Nested 5-Fold Unseen OOF.
      4) Eliminates destructive Min-Max -> Logit double-transform; uses Whole-Core-Cluster Robust Z-Score + RsCIN.

Colab Quick Run:
  !python scripts/colab_e09_genuine_musc.py \
      --data-root /content/open \
      --folds-json runs/folds_unseen.json \
      --out-dir /content/drive/MyDrive/dku_e09_output
"""
from __future__ import annotations

import argparse
import csv
import gc
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset


def _ensure_deps() -> None:
    for pkg, imp in (("timm", "timm"), ("scikit-learn", "sklearn"), ("scipy", "scipy")):
        try:
            __import__(imp)
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


SEED = 42


def seed_everything(seed: int = SEED) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_available_ram_gb() -> float:
    try:
        with open("/proc/meminfo", encoding="utf-8") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return float(line.split()[1]) / (1024.0 * 1024.0)
    except OSError:
        pass
    return 12.0


# ---------------------------------------------------------------------------
# Data Discovery & Fast Local SSD Staging for Google Colab
# ---------------------------------------------------------------------------
def prepare_data_root(data_root_arg: str | None, zip_path_arg: str | None) -> Path:
    if data_root_arg:
        p = Path(data_root_arg)
        if (p / "train.csv").is_file() and (p / "sample_submission.csv").is_file():
            return p

    candidates = [
        Path("/content/open_fast/open"),
        Path("/content/open_fast"),
        Path("/content/open"),
        Path("data/raw/open"),
        Path("/kaggle/input/dku-defect-detection-2026-open-private/open"),
        Path("/kaggle/input/dku-defect-detection-2026-open-private"),
    ]
    for c in candidates:
        if (c / "train.csv").is_file() and (c / "sample_submission.csv").is_file():
            return c

    # Check if a zip archive exists (explicitly passed or in Google Drive) -> stage to fast local SSD!
    zip_candidates = []
    if zip_path_arg:
        zip_candidates.append(Path(zip_path_arg))
    zip_candidates.extend([
        Path("runs/open.zip"),
        Path("/content/open.zip"),
        Path("/content/drive/MyDrive/open.zip"),
        Path("/content/drive/MyDrive/dku/open.zip"),
    ])
    drive_root = Path("/content/drive/MyDrive")
    if drive_root.exists() and not any(z.is_file() for z in zip_candidates):
        for found in drive_root.rglob("open.zip"):
            zip_candidates.append(found)
            break

    for zp in zip_candidates:
        if zp.is_file():
            target = Path("/content/open_fast") if Path("/content").exists() else Path("runs/open_fast")
            print(f"[I/O] Extracting {zp} -> fast local disk {target} ...")
            t0 = time.monotonic()
            target.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zp, "r") as zf:
                zf.extractall(target)
            print(f"[I/O] Extracted in {time.monotonic() - t0:.1f}s")
            for sub in (target, target / "open"):
                if (sub / "train.csv").is_file():
                    return sub

    # Last resort search on Google Drive
    if drive_root.exists():
        for tc in drive_root.rglob("train.csv"):
            if (tc.parent / "sample_submission.csv").is_file():
                print(f"[I/O] Found dataset on Drive at {tc.parent} (Tip: pass --zip-path open.zip for 10x faster SSD I/O)")
                return tc.parent

    raise FileNotFoundError(
        "Could not find train.csv & sample_submission.csv. Pass --data-root /path/to/open or --zip-path /path/to/open.zip"
    )


def load_tables_and_folds(root: Path, folds_json_path: str | None) -> tuple[list[str], np.ndarray, list[str], dict[int, list[str]], dict[str, str]]:
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

    if folds_json_path and Path(folds_json_path).is_file():
        payload = json.loads(Path(folds_json_path).read_text(encoding="utf-8"))
        fold_val_ids = {int(k): list(v["val_ids"]) for k, v in payload["folds"].items()}
        groups = dict(payload["groups"])
        print(f"[CV] Loaded exact 5-Fold Unseen split from {folds_json_path} ({sum(len(v) for v in fold_val_ids.values())} val imgs)")
        return train_ids, y_tr, test_ids, fold_val_ids, groups

    # Deterministic fallback identical to runs/folds_unseen.json (Stratified GroupKFold by exact WxH resolution group)
    print("[CV] folds_unseen.json not provided; reconstructing deterministic 5-Fold Unseen-Product GroupKFold...")

    def _read_wh(item: tuple[str, int]) -> tuple[str, int, str]:
        img_id, lbl = item
        p = root / "train" / img_id if not (root / img_id).is_file() else root / img_id
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im)
            w, h = rgb.size
        return img_id, lbl, f"{w}x{h}"

    by_g: dict[str, list[tuple[str, int]]] = defaultdict(list)
    groups = {}
    with ThreadPoolExecutor(max_workers=16) as pool:
        for img_id, lbl, g in pool.map(_read_wh, train_rows):
            groups[img_id] = g
            by_g[g].append((img_id, lbl))

    items = sorted(
        [(g, [x[0] for x in v], sum(x[1] for x in v), len(v)) for g, v in by_g.items()],
        key=lambda x: (-x[2], -x[3], x[0]),
    )
    fold_val_ids = {k: [] for k in range(5)}
    fold_pos = [0] * 5
    fold_tot = [0] * 5
    for _, ids_g, pos, tot in items:
        f_idx = min(range(5), key=lambda k: (fold_pos[k], fold_tot[k], k))
        fold_val_ids[f_idx].extend(ids_g)
        fold_pos[f_idx] += pos
        fold_tot[f_idx] += tot
    for k in range(5):
        fold_val_ids[k] = sorted(fold_val_ids[k])
    return train_ids, y_tr, test_ids, fold_val_ids, groups


# ---------------------------------------------------------------------------
# High-Resolution Full-FOV Dataset (NO Edge-Cropping!)
# ---------------------------------------------------------------------------
class FullFOVDataset(Dataset):
    def __init__(self, root: Path, role: str, ids: list[str], img_size: int = 518):
        self.root = root
        self.role = role
        self.ids = ids
        self.img_size = img_size
        self.mean = torch.tensor([0.485, 0.456, 0.406], dtype=torch.float32)[:, None, None]
        self.std = torch.tensor([0.229, 0.224, 0.225], dtype=torch.float32)[:, None, None]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        img_id = self.ids[idx]
        p = self.root / self.role / img_id if not (self.root / img_id).is_file() else self.root / img_id
        with Image.open(p) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            # Full-FOV resize: preserves 100% of border defects without adding artificial letterbox seams
            rgb = rgb.resize((self.img_size, self.img_size), Image.Resampling.BICUBIC)
            arr = np.asarray(rgb, dtype=np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1)
        return (t - self.mean) / self.std, idx


# ---------------------------------------------------------------------------
# Core MuSc Math: LNA + Core-Density Normal Bank + MSM Quantile Scoring + RsCIN
# ---------------------------------------------------------------------------
QUANTILE_INTERVALS: list[tuple[float, float]] = [
    (0.05, 0.25),
    (0.10, 0.30),
    (0.15, 0.40),
]
TOPK_RATIOS: list[float] = [0.005, 0.01, 0.02, 0.05]  # out of 1,369 patches: k = 7, 14, 27, 68


def discover_core_product_clusters(
    cls_vecs: np.ndarray,
    dist_threshold: float = 0.22,
    core_sim_floor: float = 0.70,
) -> tuple[np.ndarray, np.ndarray]:
    """Discover clean product clusters on high-density normal cores; assign outliers safely (or -1 if singleton)."""
    from sklearn.cluster import AgglomerativeClustering

    n = len(cls_vecs)
    sim_mat = cls_vecs @ cls_vecs.T
    np.fill_diagonal(sim_mat, -1.0)
    k_den = min(20, max(1, n - 1))
    density = np.sort(sim_mat, axis=1)[:, -k_den:].mean(axis=1)

    if n < 12:
        return np.zeros(n, dtype=np.int64), density

    # Cluster top 75% densest (normal-dominated) images so broken-in-half defects never form tiny clusters
    core_mask = density >= np.percentile(density, 25)
    if core_mask.sum() < 4:
        core_mask = np.ones(n, dtype=bool)

    ac = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=dist_threshold,
    )
    core_lbls = ac.fit_predict(cls_vecs[core_mask])
    uniq, cnts = np.unique(core_lbls, return_counts=True)
    min_cluster_sz = max(5, int(0.008 * n))
    valid_u = uniq[cnts >= min_cluster_sz]
    if len(valid_u) == 0:
        return np.full(n, -1, dtype=np.int64), density

    centroids = np.stack([cls_vecs[core_mask][core_lbls == u].mean(axis=0) for u in valid_u], axis=0)
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12

    sim_to_c = cls_vecs @ centroids.T
    best_c = np.argmax(sim_to_c, axis=1).astype(np.int64)
    max_c_sim = sim_to_c[np.arange(n), best_c]
    # True rare/singleton items in Test that don't match any core product cluster get cid = -1 (safe Intra fallback!)
    best_c[max_c_sim < core_sim_floor] = -1
    return best_c, density


@torch.inference_mode()
def apply_lna(patch_grid: torch.Tensor, radius: int) -> torch.Tensor:
    """MuSc Local Neighborhood Aggregation (LNA) on (B, C, H, W) feature map -> normalized (B, H*W, C) fp16."""
    if radius > 1:
        pooled = F.avg_pool2d(patch_grid, kernel_size=radius, stride=1, padding=radius // 2)
    else:
        pooled = patch_grid
    flat = pooled.flatten(2).transpose(1, 2)  # (B, P, C)
    return F.normalize(flat.float(), dim=-1).half()


@torch.inference_mode()
def _to_lna_tokens(raw_patches_np: np.ndarray, radius: int, grid_hw: int, device: str, compute_dtype: torch.dtype) -> torch.Tensor:
    """Convert raw (B, H*W, C) normalized patches -> on-the-fly LNA (r=1 or r=3) normalized (B, P, C) on GPU."""
    t = torch.from_numpy(np.array(raw_patches_np, copy=True)).to(device, dtype=compute_dtype)
    if radius <= 1:
        return t
    b, _, c = t.shape
    fmap = t.float().transpose(1, 2).reshape(b, c, grid_hw, grid_hw)
    pooled = F.avg_pool2d(fmap, kernel_size=radius, stride=1, padding=radius // 2)
    flat = pooled.flatten(2).transpose(1, 2)
    return F.normalize(flat, dim=-1).to(dtype=compute_dtype)


@torch.inference_mode()
def score_split_layer_scale(
    patches_np_or_mm: np.ndarray,
    cids: np.ndarray,
    density: np.ndarray,
    grid_hw: int,
    device: str,
    radius: int = 1,
    max_bank: int = 64,
    query_batch: int = 16,
    peer_chunk: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute Genuine MuSc peer scores (N, n_q, n_k) AND Intra-Image self-mutual scores (N,) on GPU.

    Applies LNA (radius=1 or 3) on-the-fly on GPU so disk/RAM storage is cut in half!
    """
    n, n_patches, _ = patches_np_or_mm.shape
    n_q = len(QUANTILE_INTERVALS)
    n_k = len(TOPK_RATIOS)
    k_list = [max(1, int(round(n_patches * r))) for r in TOPK_RATIOS]
    k_intra = max(1, int(round(n_patches * 0.01)))

    # Spatial mask for Intra-Image self-comparison (ignore nearby patches within Manhattan distance <= 4 on 37x37)
    coords = torch.tensor([(r, c) for r in range(grid_hw) for c in range(grid_hw)], device=device)
    near_mask = (coords[:, None, :] - coords[None, :, :]).abs().sum(dim=-1) <= max(2, grid_hw // 9)

    peer_scores = np.zeros((n, n_q, n_k), dtype=np.float32)
    intra_scores = np.zeros(n, dtype=np.float32)
    compute_dtype = torch.float16 if device == "cuda" else torch.float32

    # First compute Intra-Image Self-Mutual Score for all images in fast batches
    for start in range(0, n, query_batch):
        end = min(n, start + query_batch)
        q_gpu = _to_lna_tokens(patches_np_or_mm[start:end], radius, grid_hw, device, compute_dtype)
        self_sim = torch.bmm(q_gpu, q_gpu.transpose(1, 2))
        self_sim.masked_fill_(near_mask.unsqueeze(0), -2.0)
        top5_self = torch.topk(self_sim, k=min(6, n_patches), dim=-1).values[:, :, -1]
        intra_dist = 1.0 - top5_self.float()
        intra_scores[start:end] = torch.topk(intra_dist, k=k_intra, dim=-1).values.mean(dim=-1).cpu().numpy()

    # Second, per Core Product Cluster: keep 97%-pure Normal Core Bank ON GPU ONCE & score all cluster members!
    unique_cids = sorted(set(int(c) for c in cids if c >= 0))
    for cid in unique_cids:
        m_idxs = np.where(cids == cid)[0]
        if len(m_idxs) < 4:
            peer_scores[m_idxs] = intra_scores[m_idxs, None, None]
            continue

        bank_sz = min(max_bank, max(4, int(round(len(m_idxs) * 0.70))))
        bank_idxs = m_idxs[np.argsort(-density[m_idxs])[:bank_sz]]
        bank_gpu = _to_lna_tokens(patches_np_or_mm[bank_idxs], radius, grid_hw, device, compute_dtype)
        bank_t = bank_gpu.transpose(1, 2).contiguous()  # (M, D, P)

        for start in range(0, len(m_idxs), query_batch):
            q_idxs = m_idxs[start : start + query_batch]
            q_gpu = _to_lna_tokens(patches_np_or_mm[q_idxs], radius, grid_hw, device, compute_dtype)

            max_sims_chunks = []
            for p_start in range(0, bank_sz, peer_chunk):
                p_end = min(bank_sz, p_start + peer_chunk)
                sub_bt = bank_t[p_start:p_end]  # (mc, D, P)
                sim_4d = torch.matmul(q_gpu.unsqueeze(1), sub_bt.unsqueeze(0))
                max_sims_chunks.append(sim_4d.amax(dim=-1))

            max_sims = torch.cat(max_sims_chunks, dim=1)  # (B, M, P)
            peer_dists = 1.0 - max_sims.float()  # (B, M, P)

            for bi, img_idx in enumerate(q_idxs):
                matches = np.where(bank_idxs == img_idx)[0]
                if len(matches):
                    peer_dists[bi, matches[0], :] = float("inf")

            d_sorted, _ = torch.sort(peer_dists, dim=1)  # (B, M, P)
            valid_m = bank_sz - 1 if bank_sz > 4 else bank_sz

            for qi, (q_lo, q_hi) in enumerate(QUANTILE_INTERVALS):
                lo_i = min(valid_m - 1, max(0, int(valid_m * q_lo)))
                hi_i = min(valid_m, max(lo_i + 1, int(round(valid_m * q_hi))))
                patch_anom_map = d_sorted[:, lo_i:hi_i, :].mean(dim=1)  # (B, P)
                for ki, k_val in enumerate(k_list):
                    sc = torch.topk(patch_anom_map, k=k_val, dim=-1).values.mean(dim=-1)
                    peer_scores[q_idxs, qi, ki] = sc.cpu().numpy()

        del bank_gpu, bank_t

    singleton_m = np.where(cids < 0)[0]
    if len(singleton_m):
        peer_scores[singleton_m] = intra_scores[singleton_m, None, None]

    return peer_scores, intra_scores


def rscin_and_cluster_calibrate(
    raw_scores: np.ndarray,
    intra_scores: np.ndarray,
    cls_vecs: np.ndarray,
    cids: np.ndarray,
    intra_weight: float = 0.15,
    k_cin: int = 2,
) -> np.ndarray:
    """Apply Whole-Core-Cluster Robust Z-Score + MuSc RsCIN (Constrained Image-level Neighborhood) smoothing."""
    n = len(raw_scores)

    def _cluster_z(arr: np.ndarray, only_nonneg_cid: bool = False) -> np.ndarray:
        g_med = float(np.median(arr))
        g_iqr = float(np.percentile(arr, 75) - np.percentile(arr, 25)) + 1e-5
        z = (arr - g_med) / g_iqr
        for cid in sorted(set(int(c) for c in cids if c >= 0)):
            m = cids == cid
            n_c = int(m.sum())
            if n_c < 4:
                continue
            w = max(0.0, n_c - 3.0) / (n_c + 10.0)
            c_med = float(np.median(arr[m]))
            c_iqr = float(np.percentile(arr[m], 75) - np.percentile(arr[m], 25)) + 1e-5
            eff_med = w * c_med + (1.0 - w) * g_med
            eff_iqr = 0.85 * (w * c_iqr + (1.0 - w) * g_iqr) + 0.15 * g_iqr
            z[m] = (arr[m] - eff_med) / eff_iqr
        return z

    z_peer = _cluster_z(raw_scores)
    z_intra = _cluster_z(intra_scores)
    # Singletons (cids < 0) rely purely on standardized intra score
    z_comb = (1.0 - intra_weight) * z_peer + intra_weight * z_intra
    singleton_m = cids < 0
    if singleton_m.any():
        z_comb[singleton_m] = z_intra[singleton_m]

    # MuSc RsCIN: subtle graph smoothing ONLY among ultra-high global-CLS twins (cos >= 0.96)
    if k_cin > 0 and n > k_cin + 1:
        sim = cls_vecs @ cls_vecs.T
        np.fill_diagonal(sim, -1.0)
        z_out = z_comb.copy()
        for i in range(n):
            nbrs = np.argsort(-sim[i])[:k_cin]
            nbrs = nbrs[sim[i, nbrs] >= 0.96]
            if len(nbrs):
                z_out[i] = 0.80 * z_comb[i] + 0.20 * float(z_comb[nbrs].mean())
        return z_out
    return z_comb


# ---------------------------------------------------------------------------
# Full Split Extraction + Multi-Layer Multi-Scale Scoring Pipeline
# ---------------------------------------------------------------------------
@torch.inference_mode()
def process_split(
    split_name: str,
    root: Path,
    role: str,
    ids: list[str],
    model: torch.nn.Module,
    layer_indices_0based: list[int],
    lna_radii: list[int],
    img_size: int,
    batch_size: int,
    device: str,
    use_memmap: bool,
    ckpt_dir: Path | None = None,
) -> dict[str, np.ndarray]:
    """Extract DINOv2 multi-layer 37x37 patches for ONE split with automatic checkpoint resume & 2x-faster LNA."""
    ckpt_path = (ckpt_dir / f"ckpt_{split_name}.npz") if ckpt_dir is not None else None
    if ckpt_path is not None and ckpt_path.is_file():
        loaded = np.load(ckpt_path)
        if len(loaded["cids"]) == len(ids):
            print(f"  [{split_name}] Resumed instantly from checkpoint -> {ckpt_path}", flush=True)
            return {k: loaded[k] for k in ("cls", "cids", "density", "peer_grid", "intra_grid")}

    t0 = time.monotonic()
    n = len(ids)
    grid_hw = img_size // 14
    n_patches = grid_hw * grid_hw
    feat_dim = model.embed_dim
    n_layers = len(layer_indices_0based)
    n_r = len(lna_radii)

    ds = FullFOVDataset(root, role, ids, img_size=img_size)
    loader = DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(2, os.cpu_count() or 1),
        pin_memory=(device == "cuda"),
    )

    # Store ONLY r=1 raw normalized fp16 patches (4 files instead of 8 -> 2x faster disk I/O & half space!)
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"e09_{split_name}_")) if use_memmap else None
    storage: dict[int, np.ndarray] = {}
    for li in range(n_layers):
        shape = (n, n_patches, feat_dim)
        if tmp_dir is not None:
            mm_path = tmp_dir / f"L{li}.fp16"
            storage[li] = np.memmap(mm_path, dtype=np.float16, mode="w+", shape=shape)
        else:
            storage[li] = np.empty(shape, dtype=np.float16)

    cls_last = np.empty((n, feat_dim), dtype=np.float32)
    amp_dtype = torch.bfloat16 if (device == "cuda" and torch.cuda.is_bf16_supported()) else torch.float16

    for x, batch_idxs in loader:
        b_idxs = batch_idxs.numpy()
        x = x.to(device, non_blocking=True)
        with torch.autocast(device_type=device, dtype=amp_dtype, enabled=(device == "cuda")):
            inter = model.get_intermediate_layers(
                x, n=layer_indices_0based, reshape=True, return_prefix_tokens=True, norm=True
            )
        _, prefix_last = inter[-1]
        cls_last[b_idxs] = F.normalize(prefix_last[:, 0].float(), dim=-1).cpu().numpy()

        for li, (fmap, _) in enumerate(inter):
            lna_r1 = apply_lna(fmap.float(), radius=1)  # (B, P, D) fp16
            storage[li][b_idxs] = lna_r1.cpu().numpy()

    if tmp_dir is not None:
        for arr in storage.values():
            arr.flush()

    t_ext = time.monotonic() - t0
    cids, density = discover_core_product_clusters(cls_last)
    n_clusters = len(set(int(c) for c in cids if c >= 0))
    n_single = int((cids < 0).sum())
    print(
        f"  [{split_name}] Extracted {n} imgs ({grid_hw}x{grid_hw}={n_patches} patches x {n_layers} layers) in {t_ext:.1f}s "
        f"| Core clusters={n_clusters}, singletons={n_single}",
        flush=True,
    )

    t1 = time.monotonic()
    n_q = len(QUANTILE_INTERVALS)
    n_k = len(TOPK_RATIOS)
    peer_grid = np.zeros((n, n_layers, n_r, n_q, n_k), dtype=np.float32)
    intra_grid = np.zeros((n, n_layers, n_r), dtype=np.float32)

    for li in range(n_layers):
        for ri, rad in enumerate(lna_radii):
            p_sc, i_sc = score_split_layer_scale(
                storage[li], cids=cids, density=density, grid_hw=grid_hw, device=device, radius=rad
            )
            peer_grid[:, li, ri, :, :] = p_sc
            intra_grid[:, li, ri] = i_sc

    storage.clear()
    if tmp_dir is not None and tmp_dir.exists():
        shutil.rmtree(tmp_dir, ignore_errors=True)
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    res = {
        "cls": cls_last,
        "cids": cids,
        "density": density,
        "peer_grid": peer_grid,
        "intra_grid": intra_grid,
    }
    if ckpt_path is not None:
        np.savez_compressed(ckpt_path, **res)
    print(f"  [{split_name}] Genuine MuSc scoring & checkpoint saved in {time.monotonic() - t1:.1f}s", flush=True)
    return res


# ---------------------------------------------------------------------------
# Systematic Layer & Hyperparameter Optimization (Nested 5-Fold Unseen CV)
# ---------------------------------------------------------------------------
def evaluate_config_on_split(
    split_dict: dict[str, np.ndarray],
    layer_weights: tuple[float, ...],
    r_weights: tuple[float, ...],
    qi: int,
    ki: int,
    intra_w: float,
    k_cin: int,
) -> np.ndarray:
    lw = np.asarray(layer_weights, dtype=np.float32)[None, :, None]  # (1, L, 1)
    rw = np.asarray(r_weights, dtype=np.float32)[None, None, :]      # (1, 1, R)
    w_lr = lw * rw
    w_lr /= w_lr.sum() + 1e-12

    # Calibrate each (layer, radius) first so layers with different cosine norms blend cleanly!
    n, n_l, n_r = split_dict["intra_grid"].shape
    z_accum = np.zeros(n, dtype=np.float32)
    for li in range(n_l):
        for ri in range(n_r):
            wt = float(w_lr[0, li, ri])
            if wt <= 1e-6:
                continue
            raw_s = split_dict["peer_grid"][:, li, ri, qi, ki]
            int_s = split_dict["intra_grid"][:, li, ri]
            z_lr = rscin_and_cluster_calibrate(
                raw_s, int_s, split_dict["cls"], split_dict["cids"], intra_weight=intra_w, k_cin=k_cin
            )
            z_accum += wt * z_lr
    return z_accum


def optimize_and_audit_5fold(
    fold_results: dict[int, dict[str, np.ndarray]],
    fold_val_ids: dict[int, list[str]],
    labels_map: dict[str, int],
    layer_names: list[str],
    lna_radii: list[int],
) -> tuple[dict, np.ndarray, list[str], np.ndarray]:
    from sklearn.metrics import roc_auc_score

    n_l = len(layer_names)
    n_r = len(lna_radii)

    oof_ids: list[str] = []
    oof_y_list: list[np.ndarray] = []
    fold_slices: dict[int, slice] = {}
    cursor = 0
    for f in range(5):
        v_ids = fold_val_ids[f]
        oof_ids.extend(v_ids)
        oof_y_list.append(np.array([labels_map[i] for i in v_ids], dtype=np.int64))
        fold_slices[f] = slice(cursor, cursor + len(v_ids))
        cursor += len(v_ids)
    oof_y = np.concatenate(oof_y_list, axis=0)

    # 1. Single-Layer Audit Table (fixes qi=1 [10%,30%], ki=1 [top 1%], intra_w=0.15, k_cin=2)
    print("\n=== [1/3] SINGLE-LAYER & LNA-RADIUS 5-FOLD UNSEEN OOF AUDIT ===")
    single_layer_report: dict[str, dict[str, float]] = {}
    for li, lname in enumerate(layer_names):
        lw = tuple(1.0 if j == li else 0.0 for j in range(n_l))
        for ri, rad in enumerate(lna_radii):
            rw = tuple(1.0 if j == ri else 0.0 for j in range(n_r))
            oof_sc = np.zeros(len(oof_y), dtype=np.float32)
            f_aucs = {}
            for f in range(5):
                sc_f = evaluate_config_on_split(fold_results[f], lw, rw, qi=1, ki=1, intra_w=0.15, k_cin=2)
                oof_sc[fold_slices[f]] = sc_f
                f_aucs[f"fold_{f}"] = round(float(roc_auc_score(oof_y[fold_slices[f]], sc_f)), 4)
            full_auc = float(roc_auc_score(oof_y, oof_sc))
            key = f"{lname}_r{rad}"
            single_layer_report[key] = {"oof_auroc": round(full_auc, 4), **f_aucs}
            print(f"  {key:14s} | 5-Fold OOF AUROC = {full_auc:.4f} | Folds: {f_aucs}")

    # 2. Candidate Multi-Layer Weightings x LNA x Quantile x Top-K Grid
    if n_l == 4:
        candidate_lw = [
            (0.0, 0.35, 0.45, 0.20),   # Mid-high heavy (L12 + L18 + L24)
            (0.10, 0.30, 0.40, 0.20),  # Balanced 4-layer MuSc
            (0.15, 0.35, 0.35, 0.15),  # Texture + mid-structure focus
            (0.0, 0.50, 0.50, 0.0),    # Pure mid layers (L12 + L18)
            (0.0, 0.0, 0.60, 0.40),    # Pure deep layers (L18 + L24)
            (0.25, 0.25, 0.25, 0.25),  # Uniform 4-layer MuSc
        ]
    else:
        candidate_lw = [tuple(1.0 / n_l for _ in range(n_l))]

    candidate_rw = [(0.5, 0.5), (0.35, 0.65), (1.0, 0.0), (0.0, 1.0)] if n_r == 2 else [(1.0,)]
    candidate_intra = [0.0, 0.12, 0.25]
    candidate_kcin = [0, 2]

    configs = list(product(
        candidate_lw,
        candidate_rw,
        range(len(QUANTILE_INTERVALS)),
        range(len(TOPK_RATIOS)),
        candidate_intra,
        candidate_kcin,
    ))
    print(f"\n=== [2/3] OPTIMIZING {len(configs)} LAYER/HYPERPARAM CONFIGS VIA NESTED 5-FOLD UNSEEN CV ===")

    # Precompute per-fold score vectors for all candidate configs (fast vectorized numpy!)
    n_cfg = len(configs)
    fold_cfg_scores = {f: np.zeros((n_cfg, len(fold_val_ids[f])), dtype=np.float32) for f in range(5)}
    fold_cfg_aucs = np.zeros((5, n_cfg), dtype=np.float64)

    for ci, (lw, rw, qi, ki, iw, kc) in enumerate(configs):
        for f in range(5):
            sc_f = evaluate_config_on_split(fold_results[f], lw, rw, qi, ki, iw, kc)
            fold_cfg_scores[f][ci] = sc_f
            fold_cfg_aucs[f, ci] = roc_auc_score(oof_y[fold_slices[f]], sc_f)

    # Strict Nested 5-Fold OOF: pick config maximizing OOF AUROC on the OTHER 4 folds, evaluate on held-out fold!
    nested_oof = np.zeros(len(oof_y), dtype=np.float32)
    nested_selected = {}
    for held_f in range(5):
        train_folds = [f for f in range(5) if f != held_f]
        y_4 = np.concatenate([oof_y[fold_slices[f]] for f in train_folds], axis=0)
        best_ci, best_4_auc = 0, -1.0
        for ci in range(n_cfg):
            s_4 = np.concatenate([fold_cfg_scores[f][ci] for f in train_folds], axis=0)
            auc_4 = float(roc_auc_score(y_4, s_4))
            if auc_4 > best_4_auc:
                best_4_auc = auc_4
                best_ci = ci
        nested_oof[fold_slices[held_f]] = fold_cfg_scores[held_f][best_ci]
        lw, rw, qi, ki, iw, kc = configs[best_ci]
        nested_selected[f"held_out_fold_{held_f}"] = {
            "train4_oof_auc": round(best_4_auc, 4),
            "val_fold_auc": round(float(fold_cfg_aucs[held_f, best_ci]), 4),
            "layer_weights": lw,
            "r_weights": rw,
            "quantile": QUANTILE_INTERVALS[qi],
            "topk_ratio": TOPK_RATIOS[ki],
            "intra_weight": iw,
            "k_cin": kc,
        }
    nested_oof_auc = float(roc_auc_score(oof_y, nested_oof))

    # Select final consensus config across all 5 Train folds for Test inference
    full_oof_aucs = np.zeros(n_cfg, dtype=np.float64)
    for ci in range(n_cfg):
        s_all = np.concatenate([fold_cfg_scores[f][ci] for f in range(5)], axis=0)
        full_oof_aucs[ci] = roc_auc_score(oof_y, s_all)

    best_global_ci = int(np.argmax(full_oof_aucs))
    best_lw, best_rw, best_qi, best_ki, best_iw, best_kc = configs[best_global_ci]
    best_oof_scores = np.concatenate([fold_cfg_scores[f][best_global_ci] for f in range(5)], axis=0)
    best_global_auc = float(full_oof_aucs[best_global_ci])

    print(f"  ==> Strict Nested 5-Fold Unseen OOF AUROC (zero hyperparam leakage) = {nested_oof_auc:.4f}")
    print(f"  ==> Best Consensus Config 5-Fold Unseen OOF AUROC                   = {best_global_auc:.4f}")
    print(
        f"      Selected: layers={best_lw}, LNA_r={best_rw}, quantile={QUANTILE_INTERVALS[best_qi]}, "
        f"topk={TOPK_RATIOS[best_ki]}, intra_w={best_iw}, k_cin={best_kc}"
    )

    opt_summary = {
        "single_layer_audit": single_layer_report,
        "nested_5fold_oof_auroc": round(nested_oof_auc, 4),
        "best_consensus_5fold_oof_auroc": round(best_global_auc, 4),
        "best_per_fold_auroc": {
            f"fold_{f}": round(float(fold_cfg_aucs[f, best_global_ci]), 4) for f in range(5)
        },
        "selected_hyperparams": {
            "layer_names": layer_names,
            "layer_weights": list(best_lw),
            "lna_radii": lna_radii,
            "r_weights": list(best_rw),
            "quantile_interval": list(QUANTILE_INTERVALS[best_qi]),
            "quantile_idx": best_qi,
            "topk_ratio": TOPK_RATIOS[best_ki],
            "topk_idx": best_ki,
            "intra_weight": best_iw,
            "k_cin": best_kc,
        },
        "nested_fold_selections": nested_selected,
    }
    return opt_summary, best_oof_scores, oof_ids, oof_y


def export_submission(path: Path, test_ids: list[str], z_scores: np.ndarray) -> None:
    """Export calibrated Z-scores monotonically into (0, 1) via logistic sigmoid (NO min-max clipping distortion!)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    # Smooth temperature-scaled sigmoid preserves exact AUROC ordering AND well-behaved probabilities for future blending
    probs = 1.0 / (1.0 + np.exp(-np.clip(z_scores * 0.8, -20.0, 20.0)))
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_path", "anomaly_score"])
        for img_id, sc in zip(test_ids, probs):
            w.writerow([img_id, f"{float(sc):.6f}"])


def main() -> None:
    parser = argparse.ArgumentParser(description="E09: Genuine High-Res Multi-Layer MuSc + 5-Fold Unseen Optimization")
    parser.add_argument("--data-root", type=str, default=None, help="Path to open/ directory containing train.csv")
    parser.add_argument("--zip-path", type=str, default=None, help="Optional path to open.zip for fast SSD extraction")
    parser.add_argument("--folds-json", type=str, default="runs/folds_unseen.json", help="Path to folds_unseen.json")
    parser.add_argument("--out-dir", type=str, default="runs/e09_genuine_musc", help="Output directory for OOF & CSV")
    parser.add_argument("--backbone", type=str, default="vit_large_patch14_dinov2.lvd142m", help="timm DINOv2 model")
    parser.add_argument("--img-size", type=int, default=518, help="Input resolution (518 -> 37x37 = 1,369 patches)")
    parser.add_argument("--batch-size", type=int, default=32, help="Inference batch size (32-64 on L4/A100, 8 on 5.5GB)")
    parser.add_argument("--smoke-test", action="store_true", help="Run fast 30-image smoke test on Train only")
    args = parser.parse_args()

    _ensure_deps()
    import timm

    t_start = time.monotonic()
    seed_everything(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    avail_ram_gb = get_available_ram_gb()
    use_memmap = avail_ram_gb < 38.0 and not args.smoke_test
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    root = prepare_data_root(args.data_root, args.zip_path)
    train_ids, y_tr, test_ids, fold_val_ids, _ = load_tables_and_folds(root, args.folds_json)
    labels_map = dict(zip(train_ids, y_tr.tolist()))

    if args.smoke_test:
        print("[SMOKE TEST] Subsampling 12 images per fold (6 pos + 6 neg) on Train only...")
        for f in range(5):
            v = fold_val_ids[f]
            pos = [i for i in v if labels_map[i] == 1][:6]
            neg = [i for i in v if labels_map[i] == 0][:6]
            fold_val_ids[f] = pos + neg
        test_ids = test_ids[:12]

    print(
        f"[E09] Device={device} | RAM_avail={avail_ram_gb:.1f}GB (memmap={use_memmap}) | "
        f"Backbone={args.backbone} @ {args.img_size}px"
    )
    model = timm.create_model(args.backbone, pretrained=True, img_size=args.img_size).to(device).eval()
    n_blocks = len(model.blocks)
    # 4 evenly-spaced multi-scale blocks: e.g. [5, 11, 17, 23] (1-indexed L6, L12, L18, L24) for 24-block ViT-L,
    # or [2, 5, 8, 11] (1-indexed L3, L6, L9, L12) for 12-block ViT-B
    layer_idx_0 = [n_blocks // 4 - 1, n_blocks // 2 - 1, (3 * n_blocks) // 4 - 1, n_blocks - 1]
    layer_names = [f"L{idx + 1}" for idx in layer_idx_0]
    lna_radii = [1, 3]
    print(f"[E09] Probing Multi-Scale Layers={layer_names} (0-idx {layer_idx_0}) x LNA_radii={lna_radii}")

    # 1. Process all 5 Unseen Train Folds strictly independently (with automatic per-fold checkpointing!)
    fold_results: dict[int, dict[str, np.ndarray]] = {}
    ckpt_dir = None if args.smoke_test else out_dir
    for f in range(5):
        fold_results[f] = process_split(
            split_name=f"train_fold{f}",
            root=root,
            role="train",
            ids=fold_val_ids[f],
            model=model,
            layer_indices_0based=layer_idx_0,
            lna_radii=lna_radii,
            img_size=args.img_size,
            batch_size=args.batch_size,
            device=device,
            use_memmap=use_memmap,
            ckpt_dir=ckpt_dir,
        )

    # 2. Optimize Layer Weights & MuSc Hyperparameters on Train 5-Fold Unseen OOF
    opt_summary, best_oof_z, oof_ids, oof_y = optimize_and_audit_5fold(
        fold_results=fold_results,
        fold_val_ids=fold_val_ids,
        labels_map=labels_map,
        layer_names=layer_names,
        lna_radii=lna_radii,
    )

    # 3. Pure Black-Box Test Inference (1 single pass using Train-selected hyperparameters; ZERO Test inspection!)
    print("\n=== [3/3] BLACK-BOX TEST INFERENCE (800 IMAGES, ZERO VISUAL ACCESS) ===", flush=True)
    test_split = process_split(
        split_name="test_blackbox",
        root=root,
        role="test",
        ids=test_ids,
        model=model,
        layer_indices_0based=layer_idx_0,
        lna_radii=lna_radii,
        img_size=args.img_size,
        batch_size=args.batch_size,
        device=device,
        use_memmap=use_memmap,
        ckpt_dir=ckpt_dir,
    )
    sel = opt_summary["selected_hyperparams"]
    test_z = evaluate_config_on_split(
        test_split,
        layer_weights=tuple(sel["layer_weights"]),
        r_weights=tuple(sel["r_weights"]),
        qi=int(sel["quantile_idx"]),
        ki=int(sel["topk_idx"]),
        intra_w=float(sel["intra_weight"]),
        k_cin=int(sel["k_cin"]),
    )

    sub_path = out_dir / "sub_e09_genuine_musc_dinov2_518.csv"
    export_submission(sub_path, test_ids, test_z)

    # Save Train OOF JSON (compatible with `defectlab errors`) + compact NPZ caches for future E10/E11 ensembling
    oof_probs = 1.0 / (1.0 + np.exp(-np.clip(best_oof_z * 0.8, -20.0, 20.0)))
    val_payload = {
        "schema_version": 1,
        "kind": "validation",
        "auroc": opt_summary["best_consensus_5fold_oof_auroc"],
        "nested_oof_auroc": opt_summary["nested_5fold_oof_auroc"],
        "rows": [
            {"image_path": i, "label": int(y), "anomaly_score": float(p), "z_score": float(z)}
            for i, y, p, z in zip(oof_ids, oof_y, oof_probs, best_oof_z)
        ],
    }
    (out_dir / "val_oof_e09.json").write_text(json.dumps(val_payload, indent=2), encoding="utf-8")

    np.savez_compressed(
        out_dir / "e09_oof_and_test_cache.npz",
        oof_ids=np.array(oof_ids),
        oof_y=oof_y,
        oof_z=best_oof_z,
        oof_cls=np.concatenate([fold_results[f]["cls"] for f in range(5)], axis=0).astype(np.float16),
        oof_peer_grid=np.concatenate([fold_results[f]["peer_grid"] for f in range(5)], axis=0),
        oof_intra_grid=np.concatenate([fold_results[f]["intra_grid"] for f in range(5)], axis=0),
        oof_cids=np.concatenate([fold_results[f]["cids"] for f in range(5)], axis=0),
        test_ids=np.array(test_ids),
        test_z=test_z,
        test_cls=test_split["cls"].astype(np.float16),
        test_peer_grid=test_split["peer_grid"],
        test_intra_grid=test_split["intra_grid"],
        test_cids=test_split["cids"],
    )

    full_summary = {
        "schema_version": 1,
        "experiment": "E09_genuine_musc",
        "backbone": args.backbone,
        "license": "Apache-2.0 (Meta AI DINOv2, published 2023-04 <= 2026-11-15)",
        "img_size": args.img_size,
        "patch_grid": f"{args.img_size // 14}x{args.img_size // 14}",
        "smoke_test": args.smoke_test,
        "elapsed_seconds": round(time.monotonic() - t_start, 1),
        **opt_summary,
    }
    (out_dir / "e09_summary.json").write_text(json.dumps(full_summary, indent=2), encoding="utf-8")
    print(f"\n[DONE] Saved submission -> {sub_path}")
    print(f"[DONE] Saved 5-Fold OOF & audit -> {out_dir / 'e09_summary.json'} ({full_summary['elapsed_seconds']}s)")


if __name__ == "__main__":
    main()
