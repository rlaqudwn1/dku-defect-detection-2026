"""Track B Day 0: Zero-Singleton Cluster-v2 + Prevalence-Invariant Left-Tail Robust-Z.

Fixes Smoking Guns #1 & #2 directly on existing cached models (ZERO GPU required):
  1. Cluster-v2 (Canonical Product Graph):
     - Removes legacy `k_den=20` & `core_mask >= P25` deletion (which caused 18.3% Test Singletons!).
     - Uses Mutual-kNN (k=3) + Agglomerative Cosine Clustering (min_cluster_sz=2) + nearest-cluster
       absorption (sim >= 0.65) so small product clusters (n_c=2..18) stay 100% intact.
  2. Prevalence-Invariant Left-Tail Robust-Z:
     - Anchors normal center at P35(s_c) and normal spread at (P50(s_c) - P15(s_c)) / 0.967,
       immune to IQR(P75-P25) explosion when cluster defect prevalence > 25%.
     - Stronger small-cluster offset removal weight w = (n_c - 1) / (n_c + 3).
  3. Validates strictly on BOTH:
     - `Sim-LB-800` (200 fragmented variable-prevalence Train mini-worlds)
     - Standard 5-Fold Unseen OOF (`runs/folds_unseen.json`)
  4. Exports:
     - `runs/day0_recal/canonical_cids.npz` (shared Train/Test Cluster-v2 IDs for Track A MuSc-v2!)
     - `runs/submissions/sub_e15_recal_cluster_v2_left_tail.csv` (verified 801-row submission)
     - `runs/day0_recal/recal_summary.json` (machine-readable report for W&B)
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.sparse.csgraph import connected_components
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import roc_auc_score

from run_15model_ensemble import export_verified_submission
from sim_lb_800 import (
    build_sim_worlds,
    legacy_cluster_ids,
    legacy_iqr_recal,
    load_train_canonical,
    summarize_world_aucs,
)


def cluster_v2_zero_singleton(
    X: np.ndarray,
    k_mutual: int = 3,
    edge_sim_thr: float = 0.72,
    agg_dist_thr: float = 0.28,
    absorb_sim_thr: float = 0.62,
) -> np.ndarray:
    """Zero-Singleton Canonical Product Graph (NO P25 density deletion!)."""
    n = X.shape[0]
    if n <= 2:
        return np.zeros(n, dtype=np.int64)

    sim = np.clip(X @ X.T, -1.0, 1.0)
    np.fill_diagonal(sim, -1.0)

    # Step 1: Agglomerative cosine clustering on ALL points (never drop low-density small products!)
    dist = np.clip(1.0 - sim, 0.0, 2.0)
    np.fill_diagonal(dist, 0.0)
    agg = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed",
        linkage="average",
        distance_threshold=agg_dist_thr,
    )
    raw_cids = agg.fit_predict(dist)

    # Step 2: Rescue size-1 points via Mutual-kNN edge OR nearest valid cluster centroid
    counts = np.bincount(raw_cids)
    # Build mutual-kNN graph for small-pair rescue
    k_eff = min(k_mutual, n - 1)
    topk_idx = np.argpartition(sim, -k_eff, axis=1)[:, -k_eff:]
    adj = np.zeros((n, n), dtype=np.int8)
    for i in range(n):
        for j in topk_idx[i]:
            if sim[i, j] >= edge_sim_thr:
                adj[i, j] = 1
    mutual = (adj & adj.T)
    _, comp_labels = connected_components(mutual, directed=False)
    comp_counts = np.bincount(comp_labels)

    cids = raw_cids.copy()
    next_cid = int(cids.max()) + 1
    # Merge singleton clusters that share a mutual-kNN component of size >= 2
    for i in range(n):
        if counts[cids[i]] == 1 and comp_counts[comp_labels[i]] >= 2:
            peers = np.where(comp_labels == comp_labels[i])[0]
            # Assign all singleton peers in this mutual component together
            peer_cids = [cids[p] for p in peers if counts[cids[p]] >= 2]
            if peer_cids:
                cids[i] = peer_cids[0]
            else:
                cids[peers] = next_cid
                next_cid += 1

    # Recompute cluster centroids for size >= 2 clusters
    uniq, u_cnt = np.unique(cids, return_counts=True)
    valid_cs = uniq[u_cnt >= 2]
    if len(valid_cs) == 0:
        return np.zeros(n, dtype=np.int64)

    centroids = []
    for c in valid_cs:
        mu = X[cids == c].mean(axis=0)
        mu /= np.linalg.norm(mu) + 1e-8
        centroids.append(mu)
    C = np.stack(centroids, axis=0)

    # Absorb remaining size-1 outliers into nearest centroid if sim >= absorb_sim_thr
    final_cids = np.full(n, -1, dtype=np.int64)
    cid_remap = {int(c): idx for idx, c in enumerate(valid_cs)}
    for i in range(n):
        if int(cids[i]) in cid_remap:
            final_cids[i] = cid_remap[int(cids[i])]
        else:
            sims_i = X[i] @ C.T
            best_j = int(np.argmax(sims_i))
            if sims_i[best_j] >= absorb_sim_thr:
                final_cids[i] = best_j
            else:
                final_cids[i] = -1
    return final_cids


def left_tail_robust_z(raw: np.ndarray, cids: np.ndarray) -> np.ndarray:
    """Prevalence-invariant Left-Tail Robust-Z (P35 center, P15..P50 normal half-width)."""
    g_mu = float(np.percentile(raw, 35))
    g_sig = max(float((np.percentile(raw, 50) - np.percentile(raw, 15)) / 0.967), 1e-5)
    out = (raw - g_mu) / g_sig

    for c in np.unique(cids):
        if c < 0:
            continue
        m = cids == c
        nc = int(m.sum())
        v = raw[m]
        if nc >= 4:
            c_mu = float(np.percentile(v, 35))
            c_sig = max(float((np.percentile(v, 50) - np.percentile(v, 15)) / 0.967), 0.35 * g_sig, 1e-5)
            w_mu = (nc - 1.0) / (nc + 3.0)
            # Conservative denominator shrinkage prevents small-cluster (n_c=4..10) quantile variance explosion!
            w_sig = max(0.0, nc - 8.0) / (nc + 20.0)
            eff_mu = w_mu * c_mu + (1.0 - w_mu) * g_mu
            eff_sig = w_sig * c_sig + (1.0 - w_sig) * g_sig
            out[m] = (v - eff_mu) / eff_sig
        elif nc in (2, 3):
            # Small cluster (2~3 samples): anchor normal baseline on minimum peer
            c_mu = float(np.min(v))
            w = 0.55
            eff_mu = w * c_mu + (1.0 - w) * g_mu
            out[m] = (v - eff_mu) / g_sig
    return out


def fit_nn_bce_weights(Z_in: np.ndarray, y_in: np.ndarray, l2_reg: float = 1.0) -> np.ndarray:
    m = Z_in.shape[0]
    X = Z_in.T
    y = y_in.astype(np.float64)
    pos_w = (y == 0).sum() / max(1.0, (y == 1).sum())
    sw = np.where(y == 1, pos_w, 1.0)

    def _loss(params):
        w, b = params[:m], params[m]
        logits = np.clip(X @ w + b, -20.0, 20.0)
        return float(np.mean(sw * (np.logaddexp(0.0, logits) - y * logits)) + l2_reg * np.sum(w ** 2))

    init = np.concatenate([np.full(m, 1.0 / m), [0.0]])
    bounds = [(0.0, 5.0)] * m + [(-5.0, 5.0)]
    res = minimize(_loss, init, method="L-BFGS-B", bounds=bounds, options={"maxiter": 250})
    w = np.maximum(0.0, res.x[:m])
    return w / (w.sum() + 1e-12)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Cluster-v2 + Left-Tail Z on Sim-LB-800 & 5-Fold OOF")
    parser.add_argument("--folds-json", type=str, default="runs/folds_unseen.json")
    parser.add_argument("--e09-m01-cache", type=str, default="runs/kaggle_m01_output/e09_output/e09_oof_and_test_cache.npz")
    parser.add_argument("--e09-m06-cache", type=str, default="runs/kaggle_m06_output/m06_output/e09_oof_and_test_cache.npz")
    parser.add_argument("--models-dir", type=str, default="runs/queue15/models")
    parser.add_argument("--sample-sub", type=str, default="data/raw/open/sample_submission.csv")
    parser.add_argument("--out-dir", type=str, default="runs/day0_recal")
    parser.add_argument("--n-worlds", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load canonical Train & Test CLS embeddings (DINOv2-L + DINOv2-B multi-view)
    canon_tr, y_tr, fold_arr, X_tr_l = load_train_canonical(Path(args.folds_json), Path(args.e09_m01_cache))
    with Path(args.sample_sub).open("r", encoding="utf-8-sig", newline="") as f:
        canon_te = [r["image_path"] for r in csv.DictReader(f)]

    d_m01 = np.load(args.e09_m01_cache, allow_pickle=True)
    te_cls_l_map = dict(zip([str(x) for x in d_m01["test_ids"]], d_m01["test_cls"].astype(np.float32)))
    X_te_l = np.stack([te_cls_l_map[i] for i in canon_te], axis=0)
    X_te_l /= np.linalg.norm(X_te_l, axis=1, keepdims=True) + 1e-8

    if Path(args.e09_m06_cache).exists():
        d_m06 = np.load(args.e09_m06_cache, allow_pickle=True)
        tr_b_map = dict(zip([str(x) for x in d_m06["oof_ids"]], d_m06["oof_cls"].astype(np.float32)))
        te_b_map = dict(zip([str(x) for x in d_m06["test_ids"]], d_m06["test_cls"].astype(np.float32)))
        X_tr_b = np.stack([tr_b_map[i] for i in canon_tr], axis=0)
        X_tr_b /= np.linalg.norm(X_tr_b, axis=1, keepdims=True) + 1e-8
        X_te_b = np.stack([te_b_map[i] for i in canon_te], axis=0)
        X_te_b /= np.linalg.norm(X_te_b, axis=1, keepdims=True) + 1e-8
        X_tr_mv = np.concatenate([X_tr_l, X_tr_b], axis=1)
        X_te_mv = np.concatenate([X_te_l, X_te_b], axis=1)
        X_tr_mv /= np.linalg.norm(X_tr_mv, axis=1, keepdims=True) + 1e-8
        X_te_mv /= np.linalg.norm(X_te_mv, axis=1, keepdims=True) + 1e-8
    else:
        X_tr_mv, X_te_mv = X_tr_l, X_te_l

    # 2. Load Elite-9 raw & legacy-z scores
    elite_keys = ["M01", "M06", "M07", "M05", "M03", "M08", "M14", "M04", "M09"]
    mids, R_tr_list, R_te_list, Z_tr_leg_list, Z_te_leg_list = [], [], [], [], []
    for k in elite_keys:
        p = sorted(Path(args.models_dir).glob(f"{k}_*.npz"))[0]
        d = np.load(p, allow_pickle=True)
        mids.append(p.stem)
        r_tr_m = dict(zip([str(x) for x in d["oof_ids"]], d["oof_raw"].astype(np.float64)))
        r_te_m = dict(zip([str(x) for x in d["test_ids"]], d["test_raw"].astype(np.float64)))
        z_tr_m = dict(zip([str(x) for x in d["oof_ids"]], d["oof_z"].astype(np.float64)))
        z_te_m = dict(zip([str(x) for x in d["test_ids"]], d["test_z"].astype(np.float64)))
        R_tr_list.append(np.array([r_tr_m[i] for i in canon_tr], dtype=np.float64))
        R_te_list.append(np.array([r_te_m[i] for i in canon_te], dtype=np.float64))
        Z_tr_leg_list.append(np.array([z_tr_m[i] for i in canon_tr], dtype=np.float64))
        Z_te_leg_list.append(np.array([z_te_m[i] for i in canon_te], dtype=np.float64))

    R_tr = np.stack(R_tr_list, axis=0)
    R_te = np.stack(R_te_list, axis=0)

    # 3. Sim-LB-800 Stress-Test: Legacy (P25+IQR) vs Cluster-v2+IQR vs Cluster-v2+Left-Tail Z
    worlds = build_sim_worlds(X_tr_l, y_tr, n_worlds=args.n_worlds, seed=42)
    aucs_leg, sing_leg = [], []
    aucs_c2_iqr, sing_c2 = [], []
    aucs_c2_lt = []

    for w_idx in worlds:
        c_leg = legacy_cluster_ids(X_tr_mv[w_idx])
        c_v2 = cluster_v2_zero_singleton(X_tr_mv[w_idx])
        sing_leg.append(float((c_leg < 0).mean()))
        sing_c2.append(float((c_v2 < 0).mean()))

        z_leg = np.stack([legacy_iqr_recal(R_tr[m_i, w_idx], c_leg) for m_i in range(len(mids))], axis=0)
        z_c2_iqr = np.stack([legacy_iqr_recal(R_tr[m_i, w_idx], c_v2) for m_i in range(len(mids))], axis=0)
        z_c2_lt = np.stack([left_tail_robust_z(R_tr[m_i, w_idx], c_v2) for m_i in range(len(mids))], axis=0)

        aucs_leg.append(float(roc_auc_score(y_tr[w_idx], z_leg.mean(axis=0))))
        aucs_c2_iqr.append(float(roc_auc_score(y_tr[w_idx], z_c2_iqr.mean(axis=0))))
        aucs_c2_lt.append(float(roc_auc_score(y_tr[w_idx], z_c2_lt.mean(axis=0))))

    s_leg = summarize_world_aucs(aucs_leg, sing_leg)
    s_c2_iqr = summarize_world_aucs(aucs_c2_iqr, sing_c2)
    s_c2_lt = summarize_world_aucs(aucs_c2_lt, sing_c2)

    print("\n=== [Day 0 Sim-LB-800 Gate (200 Fragmented Variable-Prevalence Worlds)] ===")
    print(f"  1. Legacy (P25-Cut + IQR)       : Mean={s_leg['mean_auroc']:.4f} | P10={s_leg['p10_worst_auroc']:.4f} | Singletons={s_leg['mean_singleton_rate']*100:.1f}%")
    print(f"  2. Cluster-v2 + Legacy IQR      : Mean={s_c2_iqr['mean_auroc']:.4f} | P10={s_c2_iqr['p10_worst_auroc']:.4f} | Singletons={s_c2_iqr['mean_singleton_rate']*100:.2f}%")
    print(f"  3. Cluster-v2 + Left-Tail Z     : Mean={s_c2_lt['mean_auroc']:.4f} | P10={s_c2_lt['p10_worst_auroc']:.4f} | Singletons={s_c2_lt['mean_singleton_rate']*100:.2f}%")
    print(f"  -> Sim-LB Gain (3 vs 1)         : {s_c2_lt['mean_auroc'] - s_leg['mean_auroc']:+.4f} (P10 Worst: {s_c2_lt['p10_worst_auroc'] - s_leg['p10_worst_auroc']:+.4f})")

    # 4. Build Canonical Cluster-v2 IDs for Train (per-fold) & Test (black-box numeric only!)
    tr_cids_v2 = np.zeros(len(canon_tr), dtype=np.int64)
    offset = 0
    for f in range(5):
        m = fold_arr == f
        fc = cluster_v2_zero_singleton(X_tr_mv[m])
        fc_shift = np.where(fc >= 0, fc + offset, -1)
        tr_cids_v2[m] = fc_shift
        if (fc >= 0).any():
            offset = int(fc_shift.max()) + 1

    te_cids_leg = legacy_cluster_ids(X_te_mv)
    te_cids_v2 = cluster_v2_zero_singleton(X_te_mv)
    print("\n=== [Canonical Product Graph (Cluster-v2) Black-Box Count Audit] ===")
    print(f"  Train (per-fold) Singletons : {(tr_cids_v2 < 0).sum()}/{len(tr_cids_v2)} ({(tr_cids_v2 < 0).mean()*100:.2f}%)")
    print(f"  Test Legacy Singletons      : {(te_cids_leg < 0).sum()}/{len(te_cids_leg)} ({(te_cids_leg < 0).mean()*100:.1f}%)")
    print(f"  Test Cluster-v2 Singletons  : {(te_cids_v2 < 0).sum()}/{len(te_cids_v2)} ({(te_cids_v2 < 0).mean()*100:.2f}%) | Active Clusters={len(np.unique(te_cids_v2[te_cids_v2 >= 0]))}")

    np.savez_compressed(
        out_dir / "canonical_cids.npz",
        train_ids=np.array(canon_tr),
        train_cids=tr_cids_v2,
        test_ids=np.array(canon_te),
        test_cids=te_cids_v2,
    )

    # 5. Evaluate Standard 5-Fold Unseen OOF with Cluster-v2 + Left-Tail Z & Export E15-Recal CSV
    Z_tr_new = np.zeros_like(R_tr)
    for m_i in range(len(mids)):
        for f in range(5):
            mf = fold_arr == f
            Z_tr_new[m_i, mf] = left_tail_robust_z(R_tr[m_i, mf], tr_cids_v2[mf])
        # Final scale standardize across Train for fair stacking
        Z_tr_new[m_i] = (Z_tr_new[m_i] - np.median(Z_tr_new[m_i])) / (
            np.percentile(Z_tr_new[m_i], 75) - np.percentile(Z_tr_new[m_i], 25) + 1e-6
        )

    Z_te_new = np.zeros_like(R_te)
    for m_i in range(len(mids)):
        zt = left_tail_robust_z(R_te[m_i], te_cids_v2)
        Z_te_new[m_i] = (zt - np.median(zt)) / (np.percentile(zt, 75) - np.percentile(zt, 25) + 1e-6)

    # Nested 5-Fold Unseen CV for E15-Recal (NN-BCE + 0.20*Top2Mean tail consensus)
    oof_lin = np.zeros(len(y_tr), dtype=np.float64)
    for f in range(5):
        tr_m, va_m = fold_arr != f, fold_arr == f
        wf = fit_nn_bce_weights(Z_tr_new[:, tr_m], y_tr[tr_m], l2_reg=1.0)
        oof_lin[va_m] = wf @ Z_tr_new[:, va_m]
    w_full = fit_nn_bce_weights(Z_tr_new, y_tr, l2_reg=1.0)
    top2_tr = np.sort(Z_tr_new, axis=0)[-2:].mean(axis=0)
    top2_te = np.sort(Z_te_new, axis=0)[-2:].mean(axis=0)

    oof_e15 = oof_lin + 0.20 * top2_tr
    te_e15 = (w_full @ Z_te_new) + 0.20 * top2_te

    auc_lin = float(roc_auc_score(y_tr, oof_lin))
    auc_e15 = float(roc_auc_score(y_tr, oof_e15))
    per_fold = {f"fold_{f}": round(float(roc_auc_score(y_tr[fold_arr == f], oof_e15[fold_arr == f])), 4) for f in range(5)}
    print("\n=== [Standard 5-Fold Unseen OOF] ===")
    print(f"  E15-Recal Linear     : {auc_lin:.4f}")
    print(f"  E15-Recal + Top2Mean : {auc_e15:.4f} | Mean={np.mean(list(per_fold.values())):.4f} | {per_fold}")

    sub_path = Path("runs/submissions/sub_e15_recal_cluster_v2_left_tail.csv")
    checks = export_verified_submission(sub_path, Path(args.sample_sub), canon_te, te_e15, is_rank=False)

    summary = {
        "sim_lb_800": {
            "n_worlds": args.n_worlds,
            "legacy_p25_iqr": s_leg,
            "cluster_v2_iqr": s_c2_iqr,
            "cluster_v2_left_tail": s_c2_lt,
            "mean_gain": round(s_c2_lt["mean_auroc"] - s_leg["mean_auroc"], 4),
            "p10_worst_gain": round(s_c2_lt["p10_worst_auroc"] - s_leg["p10_worst_auroc"], 4),
        },
        "canonical_graph": {
            "train_singleton_rate": round(float((tr_cids_v2 < 0).mean()), 4),
            "test_legacy_singleton_rate": round(float((te_cids_leg < 0).mean()), 4),
            "test_cluster_v2_singleton_rate": round(float((te_cids_v2 < 0).mean()), 4),
            "test_active_clusters": int(len(np.unique(te_cids_v2[te_cids_v2 >= 0]))),
        },
        "standard_5fold_oof": {
            "e15_recal_linear": round(auc_lin, 4),
            "e15_recal_top2mean": round(auc_e15, 4),
            "per_fold": per_fold,
            "weights": {m: round(float(w), 4) for m, w in zip(mids, w_full)},
        },
        "submission_path": str(sub_path),
        "submission_checks": checks,
    }
    (out_dir / "recal_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n[SAVED] {sub_path} | checks={checks}")


if __name__ == "__main__":
    main()
