"""Sim-LB-800: Train-Only Monte-Carlo Simulator for Fragmented Multi-Product Test Topology.

Reproduces the real Test 800-image regime strictly inside Train (`runs/folds_unseen.json`):
  - 22~30 fine-grained visual sub-clusters (small clusters n_c in [1, 55])
  - Variable per-cluster defect prevalence pi_c in [0.08, 0.50] (stress-tests IQR collapse)
  - 200 deterministic mini-worlds (SEED=42)

ZERO Test images are ever accessed.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import roc_auc_score


def load_train_canonical(folds_json: Path, e09_cache: Path):
    folds_payload = json.loads(folds_json.read_text(encoding="utf-8"))
    labels_map = folds_payload["labels"]
    fold_map = {
        img_id: int(f_str)
        for f_str, f_data in folds_payload["folds"].items()
        for img_id in f_data["val_ids"]
    }
    canon_ids = sorted(labels_map.keys())
    y = np.array([labels_map[i] for i in canon_ids], dtype=np.int64)
    folds = np.array([fold_map[i] for i in canon_ids], dtype=np.int64)

    d = np.load(e09_cache, allow_pickle=True)
    cls_map = dict(zip([str(x) for x in d["oof_ids"]], d["oof_cls"].astype(np.float32)))
    X_cls = np.stack([cls_map[i] for i in canon_ids], axis=0)
    X_cls = X_cls / (np.linalg.norm(X_cls, axis=1, keepdims=True) + 1e-8)
    return canon_ids, y, folds, X_cls


def build_sim_worlds(
    X_cls: np.ndarray,
    y: np.ndarray,
    n_worlds: int = 200,
    world_size: int = 800,
    n_subclusters: int = 36,
    seed: int = 42,
) -> list[np.ndarray]:
    """Create 200 deterministic 800-image fragmented multi-product variable-prevalence worlds."""
    km = MiniBatchKMeans(n_clusters=n_subclusters, random_state=seed, batch_size=1024, n_init=5)
    sub_cids = km.fit_predict(X_cls)

    # Pre-split pos/neg indices per visual sub-cluster
    cluster_pools: list[tuple[np.ndarray, np.ndarray]] = []
    for c in range(n_subclusters):
        idx = np.where(sub_cids == c)[0]
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        if len(pos) >= 3 and len(neg) >= 10:
            cluster_pools.append((neg, pos))

    rng = np.random.default_rng(seed)
    worlds: list[np.ndarray] = []
    n_pools = len(cluster_pools)

    for _ in range(n_worlds):
        # Pick 22~30 active sub-clusters with heavy-tailed sizes (including small n_c=3~15)
        k_active = int(rng.integers(min(22, n_pools), min(31, n_pools + 1)))
        chosen = rng.choice(n_pools, size=k_active, replace=False)

        # Heavy-tailed cluster sizes summing to world_size (800)
        raw_sizes = rng.lognormal(mean=2.8, sigma=0.85, size=k_active)
        sizes = np.maximum(2, np.round(raw_sizes / raw_sizes.sum() * world_size).astype(int))
        # Adjust to hit exact world_size
        diff = world_size - int(sizes.sum())
        if diff > 0:
            sizes[np.argmax(sizes)] += diff
        elif diff < 0:
            order = np.argsort(sizes)[::-1]
            for oi in order:
                take = min(-diff, max(0, sizes[oi] - 2))
                sizes[oi] -= take
                diff += take
                if diff == 0:
                    break

        world_idx = []
        for pool_i, sz in zip(chosen, sizes):
            neg_pool, pos_pool = cluster_pools[pool_i]
            # Variable defect prevalence pi_c in [0.08, 0.50] (stress-tests IQR P75 collapse!)
            pi_c = float(rng.uniform(0.08, 0.50))
            n_pos = max(1, min(int(round(sz * pi_c)), sz - 1))
            n_neg = sz - n_pos
            p_sel = rng.choice(pos_pool, size=n_pos, replace=(n_pos > len(pos_pool)))
            n_sel = rng.choice(neg_pool, size=n_neg, replace=(n_neg > len(neg_pool)))
            world_idx.append(n_sel)
            world_idx.append(p_sel)

        worlds.append(np.concatenate(world_idx, axis=0))
    return worlds


def legacy_cluster_ids(X: np.ndarray, k_den: int = 20, sim_thr: float = 0.70) -> np.ndarray:
    """Reproduce the legacy P25-density-pruned clustering from colab_e09 / run_e10."""
    n = X.shape[0]
    sim = X @ X.T
    np.fill_diagonal(sim, -1.0)
    k_eff = min(k_den, max(1, n - 1))
    topk = np.partition(sim, -k_eff, axis=1)[:, -k_eff:]
    density = topk.mean(axis=1)
    core_mask = density >= np.percentile(density, 25)
    core_idx = np.where(core_mask)[0]

    # Greedy centroid clustering on core points (mirrors legacy behavior)
    from sklearn.cluster import AgglomerativeClustering

    if len(core_idx) < 4:
        return np.full(n, -1, dtype=np.int64)
    dist = np.clip(1.0 - (X[core_idx] @ X[core_idx].T), 0.0, 2.0)
    agg = AgglomerativeClustering(n_clusters=None, metric="precomputed", linkage="average", distance_threshold=1.0 - sim_thr)
    core_labels = agg.fit_predict(dist)

    centroids = []
    valid_cids = []
    for c in np.unique(core_labels):
        pts = core_idx[core_labels == c]
        if len(pts) >= 5:
            mu = X[pts].mean(axis=0)
            mu /= np.linalg.norm(mu) + 1e-8
            centroids.append(mu)
            valid_cids.append(len(valid_cids))

    cids = np.full(n, -1, dtype=np.int64)
    if not centroids:
        return cids
    C = np.stack(centroids, axis=0)
    sims_c = X @ C.T
    best_c = np.argmax(sims_c, axis=1)
    best_s = sims_c[np.arange(n), best_c]
    cids[best_s >= sim_thr] = best_c[best_s >= sim_thr]
    return cids


def legacy_iqr_recal(raw: np.ndarray, cids: np.ndarray) -> np.ndarray:
    """Reproduce legacy IQR (P75-P25) calibration with shrinkage (n_c-3)/(n_c+10) and cid=-1 global Z."""
    g_med = float(np.median(raw))
    g_iqr = float(np.percentile(raw, 75) - np.percentile(raw, 25) + 1e-6)
    out = (raw - g_med) / g_iqr
    for c in np.unique(cids):
        if c < 0:
            continue
        m = cids == c
        nc = int(m.sum())
        if nc < 4:
            continue
        v = raw[m]
        c_med = float(np.median(v))
        c_iqr = max(float(np.percentile(v, 75) - np.percentile(v, 25)), 0.25 * g_iqr, 1e-6)
        w = max(0.0, (nc - 3.0) / (nc + 10.0))
        eff_med = w * c_med + (1.0 - w) * g_med
        eff_iqr = w * c_iqr + (1.0 - w) * g_iqr
        out[m] = (v - eff_med) / eff_iqr
    return out


def summarize_world_aucs(aucs: list[float], singleton_rates: list[float]) -> dict:
    arr = np.array(aucs, dtype=np.float64)
    return {
        "mean_auroc": round(float(arr.mean()), 4),
        "std_auroc": round(float(arr.std()), 4),
        "p10_worst_auroc": round(float(np.percentile(arr, 10)), 4),
        "p50_median_auroc": round(float(np.percentile(arr, 50)), 4),
        "mean_singleton_rate": round(float(np.mean(singleton_rates)), 4),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Train-only Sim-LB-800 baseline diagnostic")
    parser.add_argument("--folds-json", type=str, default="runs/folds_unseen.json")
    parser.add_argument("--e09-cache", type=str, default="runs/kaggle_m01_output/e09_output/e09_oof_and_test_cache.npz")
    parser.add_argument("--models-dir", type=str, default="runs/queue15/models")
    parser.add_argument("--out-dir", type=str, default="runs/day0_sim_lb")
    parser.add_argument("--n-worlds", type=int, default=200)
    args = parser.parse_args()

    canon_ids, y, _, X_cls = load_train_canonical(Path(args.folds_json), Path(args.e09_cache))
    worlds = build_sim_worlds(X_cls, y, n_worlds=args.n_worlds, seed=42)

    # Load Elite-9 raw scores
    elite_keys = ["M01", "M06", "M07", "M05", "M03", "M08", "M14", "M04", "M09"]
    raw_list = []
    for k in elite_keys:
        matches = list(Path(args.models_dir).glob(f"{k}_*.npz"))
        if not matches:
            continue
        d = np.load(matches[0], allow_pickle=True)
        m = dict(zip([str(x) for x in d["oof_ids"]], d["oof_raw"].astype(np.float64)))
        raw_list.append(np.array([m[i] for i in canon_ids], dtype=np.float64))
    R = np.stack(raw_list, axis=0)

    aucs_legacy, sing_legacy = [], []
    for w_idx in worlds:
        cids = legacy_cluster_ids(X_cls[w_idx])
        sing_legacy.append(float((cids < 0).mean()))
        z_models = np.stack([legacy_iqr_recal(R[m_i, w_idx], cids) for m_i in range(R.shape[0])], axis=0)
        aucs_legacy.append(float(roc_auc_score(y[w_idx], z_models.mean(axis=0))))

    res = {"n_worlds": args.n_worlds, "legacy_iqr_p25": summarize_world_aucs(aucs_legacy, sing_legacy)}
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "sim_lb_baseline.json").write_text(json.dumps(res, indent=2), encoding="utf-8")
    print("[Sim-LB-800 Baseline]", json.dumps(res["legacy_iqr_p25"]))


if __name__ == "__main__":
    main()
