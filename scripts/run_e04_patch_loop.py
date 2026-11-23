"""E04 Loop: Patch-level Multi-Scale kNN (MuSc/PatchCore) + Unsupervised Clustering Calibration + W&B logging."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from sklearn.cluster import KMeans
from sklearn.metrics import roc_auc_score, silhouette_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet18_Weights, resnet18

from defectlab.common import write_json
from defectlab.errors import analyze_errors


class _ImageDS(Dataset):
    def __init__(self, root: Path, ids: list[str], tf):
        self.root, self.ids, self.tf = root, ids, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        with Image.open(self.root / img_id) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            return self.tf(rgb), img_id


def extract_multiscale_patches(root: Path, ids: list[str], batch_size: int = 64):
    weights = ResNet18_Weights.IMAGENET1K_V1
    model = resnet18(weights=weights).eval()
    tf = weights.transforms()

    feats_l2, feats_global = [], []
    hook_out = {}

    def _hook(name):
        def fn(_m, _inp, out):
            hook_out[name] = out
        return fn

    h2 = model.layer2.register_forward_hook(_hook("l2"))
    h3 = model.layer3.register_forward_hook(_hook("l3"))
    h4 = model.avgpool.register_forward_hook(_hook("avg"))

    loader = DataLoader(_ImageDS(root, ids, tf), batch_size=batch_size, shuffle=False, num_workers=4)
    order = []
    with torch.inference_mode():
        for x, batch_ids in loader:
            _ = model(x)
            # Downsample layer2 (28x28) to match layer3 (14x14), pool to 7x7 (49 patches/img) for fast CPU scoring
            f2 = F.adaptive_avg_pool2d(hook_out["l2"], (7, 7))
            f3 = F.adaptive_avg_pool2d(hook_out["l3"], (7, 7))
            patch = torch.cat([f2, f3], dim=1)  # (B, 128+256=384, 7, 7)
            patch = F.normalize(patch.flatten(2).transpose(1, 2), dim=-1)  # (B, 49, 384)
            glob = F.normalize(hook_out["avg"].flatten(1), dim=-1)  # (B, 512)
            feats_l2.append(patch.numpy().astype(np.float32))
            feats_global.append(glob.numpy().astype(np.float32))
            order.extend(batch_ids)
    h2.remove()
    h3.remove()
    h4.remove()
    return np.concatenate(feats_l2, axis=0), np.concatenate(feats_global, axis=0), order


def unsupervised_cluster(X_glob: np.ndarray, k_range=(2, 3, 4, 5, 6), seed: int = 42):
    best_k, best_sil, best_labels = 2, -1.0, None
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=10)
        lbls = km.fit_predict(X_glob)
        sil = float(silhouette_score(X_glob, lbls, metric="cosine", sample_size=min(1000, len(X_glob)), random_state=seed))
        if sil > best_sil:
            best_k, best_sil, best_labels = k, sil, lbls
    return best_labels, best_k, best_sil


def patch_knn_scores_within_cluster(patches: np.ndarray, cluster_ids: np.ndarray, top_p: float = 0.08, ref_limit: int = 120, seed: int = 42):
    """For each unsupervised cluster, score every image's patches against peer images in the same cluster."""
    rng = np.random.default_rng(seed)
    n, n_patches, dim = patches.shape
    raw_scores = np.zeros(n, dtype=np.float32)
    z_scores = np.zeros(n, dtype=np.float32)
    top_k_patches = max(1, int(round(n_patches * top_p)))

    for cid in np.unique(cluster_ids):
        idxs = np.where(cluster_ids == cid)[0]
        # Subsample peer reference images (purely unsupervised!) for fast matrix multiply on CPU
        if len(idxs) > ref_limit:
            ref_idxs = rng.choice(idxs, ref_limit, replace=False)
        else:
            ref_idxs = idxs
        ref_bank = patches[ref_idxs].reshape(-1, dim)  # (R*49, D)
        ref_img_owner = np.repeat(ref_idxs, n_patches)

        cluster_raw = np.zeros(len(idxs), dtype=np.float32)
        for pos, i in enumerate(idxs):
            # Cosine similarity (49, R*49) -> mask out self image patches
            sim = patches[i] @ ref_bank.T
            self_mask = ref_img_owner == i
            if self_mask.any():
                sim[:, self_mask] = -2.0
            # For each patch, distance to its nearest peer patch (1 - max_sim)
            min_dist = 1.0 - sim.max(axis=1)
            # Image anomaly score = mean of top-p% most anomalous patches
            cluster_raw[pos] = float(np.sort(min_dist)[-top_k_patches:].mean())

        raw_scores[idxs] = cluster_raw
        med = float(np.median(cluster_raw))
        iqr = float(np.percentile(cluster_raw, 75) - np.percentile(cluster_raw, 25)) + 1e-8
        z_scores[idxs] = (cluster_raw - med) / iqr

    return raw_scores, z_scores


def main():
    t0 = time.monotonic()
    torch.set_num_threads(8)
    folds = json.loads(Path("runs/folds_unseen.json").read_text(encoding="utf-8"))
    f4 = folds["folds"]["4"]
    labels = folds["labels"]
    true_groups = folds["groups"]
    val_ids = f4["val_ids"]

    print(f"[E04] Extracting multi-scale patch features (layer2+layer3, 7x7) for Unseen Fold-4 (n={len(val_ids)})...")
    patches, X_glob, order = extract_multiscale_patches(Path("data/raw/open/train"), val_ids)
    y = np.array([labels[i] for i in order])
    t_feat = time.monotonic() - t0
    print(f"  Feature extraction done in {t_feat:.1f}s | patches={patches.shape}, glob={X_glob.shape}")

    c_ids, best_k, best_sil = unsupervised_cluster(X_glob)
    # Check unsupervised cluster purity against true resolution groups (for diagnostics only, never used in scoring!)
    g_arr = np.array([true_groups[i] for i in order])
    purity = sum(
        max(int((g_arr[c_ids == c] == g).sum()) for g in set(g_arr))
        for c in np.unique(c_ids)
    ) / len(order)
    print(f"  Unsupervised K-Means selected K={best_k} (silhouette={best_sil:.4f}, purity={purity:.4f})")

    raw_s, z_s = patch_knn_scores_within_cluster(patches, c_ids, top_p=0.08, ref_limit=150)
    auroc_raw = float(roc_auc_score(y, raw_s))
    auroc_z = float(roc_auc_score(y, z_s))
    per_group = {}
    for g in sorted(set(g_arr)):
        m = g_arr == g
        per_group[g] = float(roc_auc_score(y[m], z_s[m]))

    total_sec = time.monotonic() - t0
    print(f"[E04] Patch-kNN Raw AUROC = {auroc_raw:.4f}")
    print(f"[E04] Patch-kNN + Unsupervised Cluster Robust Z-Score AUROC = {auroc_z:.4f} (total {total_sec:.1f}s)")
    for g, score in per_group.items():
        print(f"  Group {g}: AUROC = {score:.4f}")

    # Save artifacts in defectlab format
    out_dir = Path("runs/e04_patch_knn_f4")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Min-max map z_s to [0, 1] (strictly monotone -> identical AUROC)
    prob_s = (z_s - z_s.min()) / (z_s.max() - z_s.min() + 1e-12)
    rows = [{"image_path": i, "label": int(labels[i]), "anomaly_score": float(p)} for i, p in zip(order, prob_s)]
    val_path = out_dir / "val_fold4.json"
    write_json(val_path, {"schema_version": 1, "kind": "validation", "fold": 4, "auroc": auroc_z, "rows": rows})

    run_id = uuid.uuid4().hex
    write_json(out_dir / "run.json", {
        "schema_version": 1, "status": "completed", "run_id": run_id, "fold": 4,
        "best_auroc": auroc_z, "completed_epochs": 0,
        "metrics": {"auroc": auroc_z, "raw_auroc": auroc_raw, "cluster_k": best_k,
                    "cluster_silhouette": best_sil, "cluster_purity": purity, "elapsed_seconds": total_sec},
    })

    # Run defectlab errors for E04
    err_dir = out_dir / "errors"
    err_summary = analyze_errors("data/raw/open", val_path, err_dir, top_k=20)
    print(f"  Generated error sheets: {err_summary['fp_sheet']}, {err_summary['fn_sheet']}")

    # Log complete loop progression (E00 -> E04) + error sheets to W&B!
    import wandb
    run = wandb.init(
        entity="rlaqudwn123456-dankook-university",
        project="CV",
        name="E04-patch-knn-unsupervised-cluster-f4",
        group="unseen-product-loop",
        tags=["unseen-fold4", "patch-knn", "unsupervised", "resnet18", "senior-loop"],
        config={
            "exp_id": "E04",
            "backbone": "torchvision.resnet18.IMAGENET1K_V1",
            "layers": "layer2+layer3",
            "patch_grid": "7x7",
            "top_p_patches": 0.08,
            "ref_limit": 150,
            "unsupervised_k": best_k,
            "unsupervised_silhouette": round(best_sil, 4),
            "unsupervised_purity": round(purity, 4),
            "cv_split": "runs/folds_unseen.json",
            "fold": 4,
            "labels_used": 0,
        },
    )
    # Log progression curve across loop iterations so W&B shows the clear jump!
    for step, (exp_name, score) in enumerate([
        ("E00_scratch_holdout", 0.7034),
        ("E01_supervised_linear_f4", 0.7408),
        ("E02_global_knn_f4", 0.7826),
        ("E03_global_knn_oracle_z_f4", 0.9004),
        ("E04_patch_knn_unsup_cluster_z_f4", auroc_z),
    ], start=1):
        wandb.log({"loop_auroc": score, "loop_step": step, "exp": exp_name}, step=step)

    wandb.log({
        "val_auroc": auroc_z,
        "val_auroc_raw": auroc_raw,
        "cluster_k": best_k,
        "cluster_silhouette": best_sil,
        "cluster_purity": purity,
        "elapsed_seconds": total_sec,
        **{f"group_auroc/{g}": sc for g, sc in per_group.items()},
        "errors/fp_top20": wandb.Image(err_summary["fp_sheet"], caption=f"E04 FP Top-20 (AUROC={auroc_z:.4f})"),
        "errors/fn_top20": wandb.Image(err_summary["fn_sheet"], caption=f"E04 FN Top-20 (AUROC={auroc_z:.4f})"),
    })
    print(f"W&B Run URL: {run.url}")
    run.finish()


if __name__ == "__main__":
    main()
