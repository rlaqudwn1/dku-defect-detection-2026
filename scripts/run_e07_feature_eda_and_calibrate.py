"""E07: Feature EDA (OpenCLIP semantic product clusters + gallery) & Whole-Cluster Logit-Z-Score Calibration."""
from __future__ import annotations

import csv
import json
import time
from pathlib import Path

import numpy as np
import open_clip
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageOps
from scipy.stats import rankdata, spearmanr
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from torch.utils.data import DataLoader, Dataset


def load_sub(path: str):
    with Path(path).open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    return [r["image_path"] for r in rows], np.array([float(r["anomaly_score"]) for r in rows], dtype=np.float64)


class _SingleViewDS(Dataset):
    def __init__(self, root: Path, ids: list[str], tf):
        self.root, self.ids, self.tf = root, ids, tf

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, i):
        img_id = self.ids[i]
        with Image.open(self.root / img_id) as im:
            return self.tf(ImageOps.exif_transpose(im).convert("RGB")), img_id


def logit_z_calibrate(scores: np.ndarray, cluster_ids: np.ndarray, shrinkage: float = 0.15) -> np.ndarray:
    """Whole-cluster Logit-Z-Score with mild global shrinkage for small-cluster stability."""
    eps = 1e-5
    p = np.clip(scores, eps, 1.0 - eps)
    lg = np.log(p / (1.0 - p))
    g_med = float(np.median(lg))
    g_iqr = float(np.percentile(lg, 75) - np.percentile(lg, 25)) + 1e-4

    out = np.zeros_like(lg)
    for c in np.unique(cluster_ids):
        m = cluster_ids == c
        c_med = float(np.median(lg[m]))
        c_iqr = float(np.percentile(lg[m], 75) - np.percentile(lg[m], 25)) + 1e-4
        # Shrink small clusters gently toward global IQR to prevent variance explosion
        w = len(lg[m]) / (len(lg[m]) + 10.0)
        eff_iqr = (1.0 - shrinkage) * (w * c_iqr + (1.0 - w) * g_iqr) + shrinkage * g_iqr
        eff_med = w * c_med + (1.0 - w) * g_med
        out[m] = (lg[m] - eff_med) / eff_iqr
    return out


def render_cluster_gallery(root: Path, ids: list[str], cluster_ids: np.ndarray, out_path: Path):
    """Feature EDA: render 4 sample images per cluster to visually verify cluster purity."""
    uniq = sorted(np.unique(cluster_ids))
    cols = 4
    tw, th = 140, 140
    row_h = th + 24
    canvas = Image.new("RGB", (120 + cols * tw, 36 + len(uniq) * row_h), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 10), f"Feature EDA: Test Semantic Clusters (K={len(uniq)}, 4 samples/cluster)", fill="black")
    for r_idx, c in enumerate(uniq):
        idxs = np.where(cluster_ids == c)[0]
        y0 = 36 + r_idx * row_h
        draw.text((10, y0 + 50), f"C{c:02d} (n={len(idxs)})", fill="black")
        for j, idx in enumerate(idxs[:cols]):
            with Image.open(root / ids[idx]) as im:
                rgb = ImageOps.exif_transpose(im).convert("RGB")
                thumb = ImageOps.contain(rgb, (tw - 8, th - 8))
                canvas.paste(thumb, (120 + j * tw + 4, y0 + 4))
    canvas.save(out_path, quality=85)


def render_top20(root: Path, ids: list[str], scores: np.ndarray, cluster_ids: np.ndarray, out_path: Path, title: str, highest: bool = True):
    order = np.argsort(-scores)[:20] if highest else np.argsort(scores)[:20]
    cols, tw, th, hh = 5, 220, 240, 32
    canvas = Image.new("RGB", (cols * tw, hh + 4 * th), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), title, fill="black")
    for rank, idx in enumerate(order):
        r, c = divmod(rank, cols)
        x0, y0 = c * tw, hh + r * th
        with Image.open(root / ids[idx]) as im:
            rgb = ImageOps.exif_transpose(im).convert("RGB")
            thumb = ImageOps.contain(rgb, (tw - 10, th - 36))
            canvas.paste(thumb, (x0 + (tw - thumb.width) // 2, y0 + (th - 36 - thumb.height) // 2))
        draw.text((x0 + 6, y0 + th - 30), f"#{rank + 1} C{cluster_ids[idx]:02d} {ids[idx][:10]} p={scores[idx]:.3f}", fill="black")
    canvas.save(out_path, quality=88)


def main():
    t0 = time.monotonic()
    torch.set_num_threads(8)
    root = Path("data/raw/open/test")
    out_dir = Path("runs/e07_feature_eda")
    out_dir.mkdir(parents=True, exist_ok=True)

    ids, s_e05 = load_sub("runs/submissions/e05_winclip_submission.csv")
    _, s_a = load_sub("runs/kaggle_e06_output/e06_output/submissions/sub_e06a_dinov2_patch_mil.csv")
    _, s_c = load_sub("runs/kaggle_e06_output/e06_output/submissions/sub_e06c_winclip_vitl14_336.csv")

    print("[E07] Extracting OpenCLIP ViT-B-16 single-view semantic embeddings for Test 800...")
    model, _, tf = open_clip.create_model_and_transforms("ViT-B-16", pretrained="laion2b_s34b_b88k")
    model.eval()
    loader = DataLoader(_SingleViewDS(root, ids, tf), batch_size=64, shuffle=False, num_workers=4)
    embs = []
    with torch.inference_mode():
        for x, _ in loader:
            embs.append(F.normalize(model.encode_image(x), dim=-1).numpy())
    X = np.concatenate(embs, axis=0)
    print(f"  Embeddings extracted in {time.monotonic() - t0:.1f}s | shape={X.shape}")

    # Sweep K in [12..24] to find clean semantic product clusters
    best_k, best_sil, cids = 18, -1.0, None
    for k in range(12, 25):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X)
        sil = float(silhouette_score(X, lbls, metric="cosine", random_state=42))
        if sil > best_sil:
            best_k, best_sil, cids = k, sil, lbls
    sizes = [int((cids == c).sum()) for c in range(best_k)]
    print(f"  Semantic K-Means selected K={best_k} (silhouette={best_sil:.4f}, sizes min={min(sizes)} med={int(np.median(sizes))} max={max(sizes)})")

    # Apply Whole-Cluster Logit-Z-Score to each single-image inductive model
    z05 = logit_z_calibrate(s_e05, cids)
    za = logit_z_calibrate(s_a, cids)
    zc = logit_z_calibrate(s_c, cids)

    # Blend calibrated scores (equal-balanced across all 3 inductive models: E05 + E06a + E06c)
    def to_r(v):
        return rankdata(v) / len(v)

    z_ens = 0.35 * to_r(z05) + 0.35 * to_r(za) + 0.30 * to_r(zc)
    # Also map to smooth [0, 1]
    sub_scores = (z_ens - z_ens.min()) / (z_ens.max() - z_ens.min() + 1e-12)

    # Feature EDA Audit: Cluster Bias BEFORE vs AFTER calibration!
    print("\n=== FEATURE EDA: CLUSTER BIAS STD (BEFORE -> AFTER CALIBRATION) ===")
    audit = {}
    for name, raw_s, cal_s in [("E05", s_e05, z05), ("E06a", s_a, za), ("E06c", s_c, zc), ("E07_Ens", 0.35*to_r(s_e05)+0.35*to_r(s_a)+0.30*to_r(s_c), sub_scores)]:
        r_raw, r_cal = to_r(raw_s), to_r(cal_s)
        std_raw = float(np.std([r_raw[cids == c].mean() for c in range(best_k)]))
        std_cal = float(np.std([r_cal[cids == c].mean() for c in range(best_k)]))
        audit[name] = {"bias_std_before": round(std_raw, 4), "bias_std_after": round(std_cal, 4)}
        print(f"  {name:8s}: cluster_mean_rank std {std_raw:.4f} -> {std_cal:.4f}")

    # Resolution bias check BEFORE vs AFTER
    res_arr = np.array([f"{Image.open(root / i).size[0]}x{Image.open(root / i).size[1]}" for i in ids])
    print("\n=== FEATURE EDA: RESOLUTION MEAN RANK (E05 RAW vs E07 CALIBRATED) ===")
    for name, arr in [("E05 Raw", s_e05), ("E07 Calibrated", sub_scores)]:
        r = to_r(arr)
        parts = [f"{sz}: {r[res_arr == sz].mean():.3f}" for sz in sorted(set(res_arr))]
        print(f"  {name:16s} | " + " | ".join(parts))

    print("\n=== FEATURE EDA: SPEARMAN CORR AFTER CALIBRATION ===")
    print(f"  corr(z05, za)={spearmanr(z05, za).statistic:.3f}, corr(z05, zc)={spearmanr(z05, zc).statistic:.3f}, corr(za, zc)={spearmanr(za, zc).statistic:.3f}")

    # Save submission CSV & Feature EDA visual sheets
    sub_path = Path("runs/submissions/sub_e07_cluster_logit_calibrated_ensemble.csv")
    sub_path.parent.mkdir(parents=True, exist_ok=True)
    with sub_path.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f, lineterminator="\n")
        w.writerow(["image_path", "anomaly_score"])
        for img_id, sc in zip(ids, sub_scores):
            w.writerow([img_id, f"{float(sc):.6f}"])

    render_cluster_gallery(root, ids, cids, out_dir / "test_clusters_gallery.jpg")
    render_top20(root, ids, sub_scores, cids, out_dir / "e07_top20_high.jpg", "E07 Calibrated Ensemble Test Top-20 HIGHEST", highest=True)
    render_top20(root, ids, sub_scores, cids, out_dir / "e07_bot20_low.jpg", "E07 Calibrated Ensemble Test Top-20 LOWEST", highest=False)

    (out_dir / "feature_eda_summary.json").write_text(json.dumps({
        "k": best_k, "silhouette": best_sil, "cluster_sizes": sizes, "bias_audit": audit,
        "elapsed_seconds": time.monotonic() - t0,
    }, indent=2), encoding="utf-8")
    print(f"\nSaved {sub_path} and Feature EDA sheets in {out_dir} ({time.monotonic() - t0:.1f}s)")


if __name__ == "__main__":
    main()
