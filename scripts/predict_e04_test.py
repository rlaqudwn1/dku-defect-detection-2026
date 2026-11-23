"""Generate Test submission CSV using Champion E04 (ResNet18 multi-scale patch kNN + unsupervised cluster Z-score)."""
from __future__ import annotations

import csv
import time
from pathlib import Path

import numpy as np
import torch
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from run_e04_patch_loop import extract_multiscale_patches, patch_knn_scores_within_cluster


def main():
    t0 = time.monotonic()
    torch.set_num_threads(8)
    data_root = Path("data/raw/open")
    template_path = data_root / "sample_submission.csv"

    with template_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != ["image_path", "anomaly_score"]:
            raise ValueError(f"unexpected columns: {reader.fieldnames}")
        test_ids = [row["image_path"] for row in reader]

    print(f"[E04-Test] Extracting multi-scale patch features for {len(test_ids)} test images...")
    patches, X_glob, order = extract_multiscale_patches(data_root / "test", test_ids, batch_size=64)
    assert order == test_ids, "order mismatch with sample_submission.csv"

    # Sweep K for Test (which contains many unseen product categories) using cosine Silhouette score
    best_k, best_sil, best_cids = 8, -1.0, None
    for k in range(5, 25):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        lbls = km.fit_predict(X_glob)
        sil = float(silhouette_score(X_glob, lbls, metric="cosine", random_state=42))
        if sil > best_sil:
            best_k, best_sil, best_cids = k, sil, lbls

    counts = {int(c): int((best_cids == c).sum()) for c in np.unique(best_cids)}
    print(f"  Unsupervised K-Means selected K={best_k} (silhouette={best_sil:.4f})")
    print(f"  Cluster sizes: min={min(counts.values())}, median={int(np.median(list(counts.values())))}, max={max(counts.values())}")

    # Score every test image's patches against peer images in the same unsupervised product cluster
    raw_s, z_s = patch_knn_scores_within_cluster(patches, best_cids, top_p=0.08, ref_limit=200, seed=42)

    # Map z_s smoothly to (0, 1) via sigmoid(z_s) (strictly monotone -> preserves exact AUROC ranking!)
    clipped = np.clip(z_s, -15.0, 15.0)
    scores = 1.0 / (1.0 + np.exp(-clipped))
    assert np.isfinite(scores).all() and (scores >= 0.0).all() and (scores <= 1.0).all()

    out_path = Path("runs/submissions/e04_patch_knn_submission.csv")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, lineterminator="\n")
        writer.writerow(["image_path", "anomaly_score"])
        for img_id, sc in zip(test_ids, scores):
            writer.writerow([img_id, f"{float(sc):.6f}"])

    print(f"  Saved {out_path} ({len(test_ids)} rows) in {time.monotonic() - t0:.1f}s")
    print(f"  Score stats: min={scores.min():.4f}, p25={np.percentile(scores, 25):.4f}, "
          f"med={np.median(scores):.4f}, p75={np.percentile(scores, 75):.4f}, max={scores.max():.4f}")


if __name__ == "__main__":
    main()
