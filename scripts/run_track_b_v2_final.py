"""Track B v2 Final Integration: M01~M19 3-Channel Pruning + 32x32 Spatial Fusion + Cluster-v2 Left-Tail Recal.

Integrates Track A (`M16~M19` MuSc-v2 orthogonal backbones + 32x32 spatial anomaly maps)
with Track B (`Cluster-v2` Zero-Singleton + Prevalence-Invariant Left-Tail Robust-Z):
  1. Computes pixel-aligned 4-backbone Mid-Level 32x32 Spatial Consensus (`SP4_top2`,
     2nd-highest normalized pixel across DINOv2-Reg4 / EVA-02 / SigLIP / ConvNeXt-L -> OOF 0.9308!).
  2. Re-calibrates Elite-9 (`M01,M06,M07,M05,M03,M08,M14,M04,M09`) on shared `canonical_cids.npz`
     via Left-Tail Robust-Z and combines with orthogonal Track A channels (`M16_svd`, `M18_opt`, `SP4_top2`).
  3. Validates strictly under BOTH:
     - Standard Nested 5-Fold Unseen CV (`runs/folds_unseen.json`, 10,821 Train images)
     - `Sim-LB-800` 200-World Fragmented Variable-Prevalence Stress Gate (Train-only!)
  4. Exports verified black-box submission CSV (`runs/submissions/sub_e20_track_ab_final.csv`)
     and machine-readable summary (`runs/track_b_v2/track_b_v2_summary.json`).

ZERO Test images are ever rendered, saved, or visually inspected.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score

from cluster_v2_and_recal import left_tail_robust_z
from run_15model_ensemble import export_verified_submission
from run_track_b_ensemble_lab import calibrate, fit_nn_bce


def build_product_sim_worlds(
    cids: np.ndarray,
    y: np.ndarray,
    n_worlds: int = 200,
    world_size: int = 800,
    seed: int = 42,
) -> list[np.ndarray]:
    """Build 200 deterministic 800-image fragmented variable-prevalence worlds strictly from visual product cids."""
    rng = np.random.default_rng(seed)
    pools: list[tuple[np.ndarray, np.ndarray]] = []
    for c in np.unique(cids):
        if c < 0:
            continue
        idx = np.where(cids == c)[0]
        pos = idx[y[idx] == 1]
        neg = idx[y[idx] == 0]
        if len(pos) >= 5 and len(neg) >= 20:
            pools.append((neg, pos))

    worlds: list[np.ndarray] = []
    n_pools = len(pools)
    for _ in range(n_worlds):
        # Sample 7~11 products with heavy-tailed sizes (small clusters n_c=6..25 + large) and pi_c in [0.08, 0.50]
        k_active = int(rng.integers(min(7, n_pools), n_pools + 1))
        chosen = rng.choice(n_pools, size=k_active, replace=False)
        raw_sizes = rng.lognormal(mean=3.5, sigma=0.90, size=k_active)
        sizes = np.maximum(6, np.round(raw_sizes / raw_sizes.sum() * world_size).astype(int))
        diff = world_size - int(sizes.sum())
        if diff > 0:
            sizes[np.argmax(sizes)] += diff
        elif diff < 0:
            for oi in np.argsort(sizes)[::-1]:
                take = min(-diff, max(0, sizes[oi] - 6))
                sizes[oi] -= take
                diff += take
                if diff == 0:
                    break
        w_idx = []
        for pi_idx, sz in zip(chosen, sizes):
            neg_p, pos_p = pools[pi_idx]
            pi_c = float(rng.uniform(0.08, 0.50))
            n_pos = max(1, min(int(round(sz * pi_c)), sz - 1))
            n_neg = sz - n_pos
            w_idx.append(rng.choice(neg_p, size=n_neg, replace=(n_neg > len(neg_p))))
            w_idx.append(rng.choice(pos_p, size=n_pos, replace=(n_pos > len(pos_p))))
        worlds.append(np.concatenate(w_idx, axis=0))
    return worlds


def lt_per_fold(raw: np.ndarray, cids: np.ndarray, fold_arr: np.ndarray) -> np.ndarray:
    out = np.zeros_like(raw, dtype=np.float64)
    for f in range(5):
        m = fold_arr == f
        out[m] = left_tail_robust_z(raw[m], cids[m])
    return out


def normalize_maps_per_cluster(maps: np.ndarray, cids: np.ndarray, fold_arr: np.ndarray | None = None) -> np.ndarray:
    """Per-cluster Left-Tail pixel normalization (removes per-product & per-backbone offset/scale)."""
    out = np.zeros_like(maps, dtype=np.float32)
    if fold_arr is None:
        splits = [np.ones(len(maps), dtype=bool)]
    else:
        splits = [fold_arr == f for f in range(5)]

    for sm in splits:
        for c in np.unique(cids[sm]):
            mc = sm & (cids == c)
            sub = maps[mc]
            p35 = float(np.percentile(sub, 35))
            sig = max(float((np.percentile(sub, 50) - np.percentile(sub, 15)) / 0.967), 1e-4)
            out[mc] = (sub - p35) / sig
    return out


def pool_spatial_map(mp: np.ndarray) -> np.ndarray:
    """Multi-scale spatial pooling: 0.50*top0.5%(5px) + 0.30*3x3_avgpool_max + 0.20*top5%(51px)."""
    t = torch.from_numpy(mp.astype(np.float32)).unsqueeze(1)
    flat = t.reshape(len(mp), -1)
    s_mic = torch.topk(flat, k=5, dim=1).values.mean(dim=1).numpy()
    s_mes = torch.topk(flat, k=51, dim=1).values.mean(dim=1).numpy()
    s_sp = F.avg_pool2d(t, kernel_size=3, stride=1, padding=1).amax(dim=(1, 2, 3)).numpy()
    return (0.50 * s_mic + 0.30 * s_sp + 0.20 * s_mes).astype(np.float64)


def load_aligned_pair(npz_path: Path, key_tr: str, key_te: str, canon_tr: list[str], canon_te: list[str]) -> tuple[np.ndarray, np.ndarray]:
    d = np.load(npz_path, allow_pickle=True)
    m_tr = dict(zip([str(x) for x in d["oof_ids"]], d[key_tr].astype(np.float64)))
    m_te = dict(zip([str(x) for x in d["test_ids"]], d[key_te].astype(np.float64)))
    return (
        np.array([m_tr[i] for i in canon_tr], dtype=np.float64),
        np.array([m_te[i] for i in canon_te], dtype=np.float64),
    )


def eval_nested_and_sim(
    name: str,
    tr_list: list[np.ndarray],
    te_list: list[np.ndarray],
    y_tr: np.ndarray,
    fold_arr: np.ndarray,
    worlds: list[np.ndarray],
    cal_mode: str,
    l2_reg: float = 1.0,
    top2_alpha: float = 0.0,
    feat_names: list[str] | None = None,
) -> tuple[dict, np.ndarray, np.ndarray]:
    Z_tr = calibrate(np.stack(tr_list, axis=0), cal_mode)
    Z_te = calibrate(np.stack(te_list, axis=0), cal_mode)

    oof = np.zeros(len(y_tr), dtype=np.float64)
    for f in range(5):
        tr_m, va_m = fold_arr != f, fold_arr == f
        wf = fit_nn_bce(Z_tr[:, tr_m], y_tr[tr_m], l2_reg=l2_reg)
        oof[va_m] = wf @ Z_tr[:, va_m]
    w_full = fit_nn_bce(Z_tr, y_tr, l2_reg=l2_reg)
    te_pred = w_full @ Z_te

    if top2_alpha > 0.0:
        oof = oof + top2_alpha * np.sort(Z_tr, axis=0)[-2:].mean(axis=0)
        te_pred = te_pred + top2_alpha * np.sort(Z_te, axis=0)[-2:].mean(axis=0)

    ov_auc = float(roc_auc_score(y_tr, oof))
    per_fold = {f"fold_{f}": round(float(roc_auc_score(y_tr[fold_arr == f], oof[fold_arr == f])), 4) for f in range(5)}
    mean_fold = float(np.mean(list(per_fold.values())))

    # Sim-LB-800 200-World Fragmented Variable-Prevalence Evaluation (strictly on nested OOF!)
    w_aucs = [float(roc_auc_score(y_tr[wi], oof[wi])) for wi in worlds]
    sim_mean = float(np.mean(w_aucs))
    sim_p10 = float(np.percentile(w_aucs, 10))
    sim_std = float(np.std(w_aucs))

    weights_dict = (
        {fn: round(float(w), 4) for fn, w in zip(feat_names, w_full)}
        if feat_names is not None
        else {f"f{i}": round(float(w), 4) for i, w in enumerate(w_full)}
    )
    info = {
        "name": name,
        "cal_mode": cal_mode,
        "l2_reg": l2_reg,
        "top2_alpha": top2_alpha,
        "nested_5fold_oof": round(ov_auc, 4),
        "mean_fold_auroc": round(mean_fold, 4),
        "per_fold": per_fold,
        "sim_lb_800": {
            "mean_auroc": round(sim_mean, 4),
            "p10_worst_auroc": round(sim_p10, 4),
            "std_auroc": round(sim_std, 4),
        },
        "weights": weights_dict,
    }
    print(
        f"{name:44s} | OOF={ov_auc:.4f} (Mean={mean_fold:.4f}, F0={per_fold['fold_0']:.4f}) "
        f"| SimLB={sim_mean:.4f} (P10={sim_p10:.4f})",
        flush=True,
    )
    return info, oof, te_pred


def main() -> None:
    parser = argparse.ArgumentParser(description="Track B v2 Final Integration (M01~M19 + 32x32 Spatial Fusion)")
    parser.add_argument("--folds-json", type=str, default="runs/folds_unseen.json")
    parser.add_argument("--canonical-cids", type=str, default="runs/day0_recal/canonical_cids.npz")
    parser.add_argument("--models-dir", type=str, default="runs/queue15/models")
    parser.add_argument("--maps-dir", type=str, default="runs/musc_v2")
    parser.add_argument("--sample-sub", type=str, default="data/raw/open/sample_submission.csv")
    parser.add_argument("--out-dir", type=str, default="runs/track_b_v2")
    parser.add_argument("--sub-out", type=str, default="runs/submissions/sub_e20_track_ab_final.csv")
    parser.add_argument("--n-worlds", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    models_dir = Path(args.models_dir)
    maps_dir = Path(args.maps_dir)

    # 1. Load canonical Train & Test IDs + Cluster-v2 Zero-Singleton IDs
    folds_payload = json.loads(Path(args.folds_json).read_text(encoding="utf-8"))
    labels_map = folds_payload["labels"]
    fold_map = {img: int(f) for f, d in folds_payload["folds"].items() for img in d["val_ids"]}
    canon_tr = sorted(labels_map.keys())
    y_tr = np.array([labels_map[i] for i in canon_tr], dtype=np.int64)
    fold_arr = np.array([fold_map[i] for i in canon_tr], dtype=np.int64)

    with Path(args.sample_sub).open("r", encoding="utf-8-sig", newline="") as f:
        canon_te = [r["image_path"] for r in csv.DictReader(f)]

    cd = np.load(args.canonical_cids, allow_pickle=True)
    tr_cid_map = dict(zip([str(x) for x in cd["train_ids"]], cd["train_cids"].astype(np.int64)))
    te_cid_map = dict(zip([str(x) for x in cd["test_ids"]], cd["test_cids"].astype(np.int64)))
    tr_cids = np.array([tr_cid_map[i] for i in canon_tr], dtype=np.int64)
    te_cids = np.array([te_cid_map[i] for i in canon_te], dtype=np.int64)

    # 2. Compute 4-Backbone Pixel-Aligned 32x32 Spatial Anomaly Fusion (Train per-fold + Test)
    track_a_models = [
        "M16_musc_dinov2_reg4_l",
        "M17_musc_eva02_l448",
        "M18_musc_siglip_so400m_384",
        "M19_musc_convnext_l384",
    ]
    tr_maps_norm, te_maps_norm = {}, {}
    for mn in track_a_models:
        d = np.load(maps_dir / mn / f"{mn}_maps32.npz", allow_pickle=True)
        tr_idx = {str(x): k for k, x in enumerate(d["oof_ids"])}
        te_idx = {str(x): k for k, x in enumerate(d["test_ids"])}
        m_tr = d["oof_maps_32"][[tr_idx[i] for i in canon_tr]].astype(np.float32)
        m_te = d["test_maps_32"][[te_idx[i] for i in canon_te]].astype(np.float32)
        tr_maps_norm[mn[:3]] = normalize_maps_per_cluster(m_tr, tr_cids, fold_arr)
        te_maps_norm[mn[:3]] = normalize_maps_per_cluster(m_te, te_cids, None)

    tr_stack4 = np.stack([tr_maps_norm[k] for k in ("M16", "M17", "M18", "M19")], axis=0)
    te_stack4 = np.stack([te_maps_norm[k] for k in ("M16", "M17", "M18", "M19")], axis=0)
    sp4_top2_tr = lt_per_fold(pool_spatial_map(np.sort(tr_stack4, axis=0)[2]), tr_cids, fold_arr)
    sp4_top2_te = left_tail_robust_z(pool_spatial_map(np.sort(te_stack4, axis=0)[2]), te_cids)

    sp3_add_tr = lt_per_fold(
        pool_spatial_map(0.25 * tr_maps_norm["M16"] + 0.25 * tr_maps_norm["M17"] + 0.50 * tr_maps_norm["M18"]),
        tr_cids,
        fold_arr,
    )
    sp3_add_te = left_tail_robust_z(
        pool_spatial_map(0.25 * te_maps_norm["M16"] + 0.25 * te_maps_norm["M17"] + 0.50 * te_maps_norm["M18"]),
        te_cids,
    )
    print(
        f"[SPATIAL 32x32] SP4_top2 OOF={roc_auc_score(y_tr, sp4_top2_tr):.4f} | "
        f"SP3_add OOF={roc_auc_score(y_tr, sp3_add_tr):.4f}",
        flush=True,
    )

    # 3. Load Left-Tail Recalibrated Elite-9 + Track A Orthogonal Channels
    elite_keys = [
        "M01_musc_dinov2_l518",
        "M06_musc_dinov2_b518",
        "M07_vand_clip_l14_laion2b",
        "M05_siglip_so400m_384",
        "M03_winclip_l14_laion2b",
        "M08_patchmil_dinov2_l518",
        "M14_patchmil_convnext_l384",
        "M04_patchmil_convnext_b384",
        "M09_patchmil_swin_b384",
    ]
    e9_tr, e9_te = {}, {}
    for mk in elite_keys:
        r_tr, r_te = load_aligned_pair(models_dir / f"{mk}.npz", "oof_raw", "test_raw", canon_tr, canon_te)
        e9_tr[mk[:3]] = lt_per_fold(r_tr, tr_cids, fold_arr)
        e9_te[mk[:3]] = left_tail_robust_z(r_te, te_cids)

    # Track A optimal orthogonal channels
    m16_s_tr, m16_s_te = load_aligned_pair(models_dir / "M16_musc_dinov2_reg4_l.npz", "oof_svd", "test_svd", canon_tr, canon_te)
    m16_svd_tr, m16_svd_te = lt_per_fold(m16_s_tr, tr_cids, fold_arr), left_tail_robust_z(m16_s_te, te_cids)

    m17_m_tr, m17_m_te = load_aligned_pair(models_dir / "M17_musc_eva02_l448.npz", "oof_musc", "test_musc", canon_tr, canon_te)
    m17_musc_tr, m17_musc_te = lt_per_fold(m17_m_tr, tr_cids, fold_arr), left_tail_robust_z(m17_m_te, te_cids)

    m18_m_tr, m18_m_te = load_aligned_pair(models_dir / "M18_musc_siglip_so400m_384.npz", "oof_musc", "test_musc", canon_tr, canon_te)
    m18_s_tr, m18_s_te = load_aligned_pair(models_dir / "M18_musc_siglip_so400m_384.npz", "oof_svd", "test_svd", canon_tr, canon_te)
    m18_l_tr, m18_l_te = load_aligned_pair(models_dir / "M18_musc_siglip_so400m_384.npz", "oof_logical", "test_logical", canon_tr, canon_te)
    m18_opt_tr = 0.75 * lt_per_fold(m18_m_tr, tr_cids, fold_arr) + 0.15 * lt_per_fold(m18_s_tr, tr_cids, fold_arr) + 0.10 * lt_per_fold(m18_l_tr, tr_cids, fold_arr)
    m18_opt_te = 0.75 * left_tail_robust_z(m18_m_te, te_cids) + 0.15 * left_tail_robust_z(m18_s_te, te_cids) + 0.10 * left_tail_robust_z(m18_l_te, te_cids)

    m19_s_tr, m19_s_te = load_aligned_pair(models_dir / "M19_musc_convnext_l384.npz", "oof_svd", "test_svd", canon_tr, canon_te)
    m19_svd_tr, m19_svd_te = lt_per_fold(m19_s_tr, tr_cids, fold_arr), left_tail_robust_z(m19_s_te, te_cids)

    # 4. Build deterministic Sim-LB-800 200-World Stress Gate strictly from visual product cids (ZERO score leakage!)
    worlds = build_product_sim_worlds(tr_cids, y_tr, n_worlds=args.n_worlds, seed=42)

    print("\n=== [Track B v2 Final Evaluation: Nested 5-Fold Unseen CV + Sim-LB-800 (200 Worlds)] ===", flush=True)
    base_names = [k[:3] for k in elite_keys]
    base_tr = [e9_tr[k] for k in base_names]
    base_te = [e9_te[k] for k in base_names]

    results = []
    preds = {}

    # Baseline: E15-Day0 (no Track A)
    r_e15, oof_e15, te_e15 = eval_nested_and_sim(
        "E15_Baseline (LT-Elite9 + 0.20*Top2)",
        base_tr, base_te, y_tr, fold_arr, worlds, "robust_z", 1.0, 0.20, base_names,
    )
    results.append(r_e15)
    preds[r_e15["name"]] = (oof_e15, te_e15)

    # Candidate A: LT-Elite9 + M16_svd + M18_opt (z_plus_rg, BCE)
    f_a = base_names + ["M16_svd", "M18_opt"]
    r_a, oof_a, te_a = eval_nested_and_sim(
        "E20a_Pruned (LT-E9+M16svd+M18opt, z+rg)",
        base_tr + [m16_svd_tr, m18_opt_tr],
        base_te + [m16_svd_te, m18_opt_te],
        y_tr, fold_arr, worlds, "z_plus_rg", 1.0, 0.0, f_a,
    )
    results.append(r_a)
    preds[r_a["name"]] = (oof_a, te_a)

    # Candidate B: LT-Elite9 + M16_svd + M18_opt + SP4_top2 (ALL 4 backbones via 32x32 pixel consensus, z_plus_rg)
    f_b = base_names + ["M16_svd", "M18_opt", "SP4_top2"]
    r_b, oof_b, te_b = eval_nested_and_sim(
        "E20b_Spatial (LT-E9+M16s+M18o+SP4, z+rg)",
        base_tr + [m16_svd_tr, m18_opt_tr, sp4_top2_tr],
        base_te + [m16_svd_te, m18_opt_te, sp4_top2_te],
        y_tr, fold_arr, worlds, "z_plus_rg", 1.0, 0.0, f_b,
    )
    results.append(r_b)
    preds[r_b["name"]] = (oof_b, te_b)

    # Candidate C: LT-Elite9 + M16_svd + M18_opt + SP4_top2 (winsor_z + 0.15*Top2Mean tail consensus)
    r_c, oof_c, te_c = eval_nested_and_sim(
        "E20c_SpatialTop2 (winsor_z + 0.15*Top2)",
        base_tr + [m16_svd_tr, m18_opt_tr, sp4_top2_tr],
        base_te + [m16_svd_te, m18_opt_te, sp4_top2_te],
        y_tr, fold_arr, worlds, "winsor_z", 1.0, 0.15, f_b,
    )
    results.append(r_c)
    preds[r_c["name"]] = (oof_c, te_c)

    # Candidate D: 6-Tower Paradigm-Hierarchical (ALL M01..M19 + SP4 + SP3, zero feature selection bias)
    tower_defs = {
        "T1_MuSc_DINO": ([e9_tr["M01"], e9_tr["M06"], m16_svd_tr], [e9_te["M01"], e9_te["M06"], m16_svd_te]),
        "T2_MuSc_VLM_MIM": ([m18_opt_tr, m17_musc_tr], [m18_opt_te, m17_musc_te]),
        "T3_Spatial_32x32": ([sp4_top2_tr, sp3_add_tr], [sp4_top2_te, sp3_add_te]),
        "T4_VAND_ZeroShot": ([e9_tr["M07"], e9_tr["M05"], e9_tr["M03"]], [e9_te["M07"], e9_te["M05"], e9_te["M03"]]),
        "T5_PatchMIL_ViT": ([e9_tr["M08"]], [e9_te["M08"]]),
        "T6_CNN_Hier": ([e9_tr["M14"], e9_tr["M04"], e9_tr["M09"], m19_svd_tr], [e9_te["M14"], e9_te["M04"], e9_te["M09"], m19_svd_te]),
    }
    t_names = list(tower_defs.keys())
    t_tr = [calibrate(np.stack(v[0], axis=0), "z_plus_rg").mean(axis=0) for v in tower_defs.values()]
    t_te = [calibrate(np.stack(v[1], axis=0), "z_plus_rg").mean(axis=0) for v in tower_defs.values()]
    r_d, oof_d, te_d = eval_nested_and_sim(
        "E20d_6Tower_All19 (z_plus_rg, BCE 0.2)",
        t_tr, t_te, y_tr, fold_arr, worlds, "robust_z", 0.2, 0.0, t_names,
    )
    results.append(r_d)
    preds[r_d["name"]] = (oof_d, te_d)

    # Select winner strictly by balanced Nested 5-Fold OOF + Mean Fold + Sim-LB-800 Mean (among Track A+B candidates)
    candidates = results[1:]
    winner = max(
        candidates,
        key=lambda r: 0.50 * r["nested_5fold_oof"] + 0.25 * r["mean_fold_auroc"] + 0.25 * r["sim_lb_800"]["mean_auroc"],
    )
    win_oof, win_te = preds[winner["name"]]

    # Export verified black-box submission CSV
    sub_path = Path(args.sub_out)
    checks = export_verified_submission(sub_path, Path(args.sample_sub), canon_te, win_te, is_rank=False)

    # Also export 6-Tower zero-selection-bias backup CSV
    sub_6t = sub_path.with_name("sub_e20_6tower_all19.csv")
    checks_6t = export_verified_submission(sub_6t, Path(args.sample_sub), canon_te, te_d, is_rank=False)

    # Save Train-only OOF predictions for error analysis + summary JSON
    oof_records = [
        {"image_path": img, "label": int(lbl), "fold": int(fl), "score": round(float(sc), 6)}
        for img, lbl, fl, sc in zip(canon_tr, y_tr, fold_arr, win_oof)
    ]
    (out_dir / "val_oof_e20.json").write_text(
        json.dumps({"winner": winner["name"], "oof_auroc": winner["nested_5fold_oof"], "records": oof_records}),
        encoding="utf-8",
    )
    summary = {
        "schema_version": 1,
        "spatial_32x32_single_auroc": {
            "SP4_top2": round(float(roc_auc_score(y_tr, sp4_top2_tr)), 4),
            "SP3_add": round(float(roc_auc_score(y_tr, sp3_add_tr)), 4),
        },
        "winner": winner,
        "all_candidates": results,
        "submission_path": str(sub_path),
        "submission_checks": checks,
        "backup_6tower_path": str(sub_6t),
        "backup_6tower_checks": checks_6t,
    }
    (out_dir / "track_b_v2_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(
        f"\n[WINNER] {winner['name']} -> Nested 5-Fold OOF={winner['nested_5fold_oof']:.4f} "
        f"(Mean={winner['mean_fold_auroc']:.4f}, F0={winner['per_fold']['fold_0']:.4f}) | "
        f"Sim-LB-800={winner['sim_lb_800']['mean_auroc']:.4f} (P10={winner['sim_lb_800']['p10_worst_auroc']:.4f})\n"
        f"  Verified CSV : {sub_path} {checks}\n"
        f"  6-Tower CSV  : {sub_6t} {checks_6t}",
        flush=True,
    )


if __name__ == "__main__":
    main()
