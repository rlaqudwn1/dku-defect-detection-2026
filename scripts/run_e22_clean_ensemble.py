"""E22 Clean Ensemble Builder: 3-Gate Pre-Submission Audit + Leave-One-Out MVP Ranking.

Fixes all 3 root causes of the E21 (-0.0119) drop:
  1. Strips noisy `test_rev` completely: uses clean forward `v2` (`M16~M21`) + `M22_musc_only` (SigLIP2-512 forward!)
  2. Applies `z_plus_rg` (Winsorized [-4, +6] + Rank-Gauss) to neutralize +13~+22 outlier spikes
  3. Preserves proven `E20 (0.926148 LB)` weapon (`SP4_top2` 32x32 2D spatial consensus) via calibrated blend

Also prints an exact Leave-One-Out (LOO) Marginal Contribution + Diversity MVP table across all clean models.
ZERO Test images are ever rendered, saved, or visually inspected.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from cluster_v2_and_recal import left_tail_robust_z
from run_15model_ensemble import export_verified_submission
from run_track_b_ensemble_lab import calibrate, fit_nn_bce
from run_track_b_v2_final import build_product_sim_worlds


def main() -> None:
    folds_payload = json.loads(Path("runs/folds_unseen.json").read_text(encoding="utf-8"))
    labels_map = folds_payload["labels"]
    fold_map = {img_id: int(f_str) for f_str, f_data in folds_payload["folds"].items() for img_id in f_data["val_ids"]}
    canon_tr = sorted(labels_map.keys())
    y = np.array([labels_map[i] for i in canon_tr], dtype=np.int64)
    folds = np.array([fold_map[i] for i in canon_tr], dtype=np.int64)

    sample_sub = Path("data/raw/open/sample_submission.csv")
    with sample_sub.open("r", encoding="utf-8-sig", newline="") as f:
        canon_te = [r["image_path"] for r in csv.DictReader(f)]

    cd = np.load("runs/day0_recal/canonical_cids.npz", allow_pickle=True)
    cid_tr_map = dict(zip([str(x) for x in cd["train_ids"]], cd["train_cids"].astype(np.int64)))
    cids_tr = np.array([cid_tr_map[i] for i in canon_tr], dtype=np.int64)
    cids_te = cd["test_cids"].astype(np.int64)
    worlds = build_product_sim_worlds(cids_tr, y, n_worlds=200, world_size=800, seed=42)

    # Load E20 (LB 0.926148) Test predictions for Gate-2 sanity check & Champion SP4 preservation
    e20_csv = Path("runs/e20_wandb_dl/sub_e20_track_ab_final.csv")
    with e20_csv.open("r", encoding="utf-8-sig", newline="") as f:
        e20_m = {r["image_path"]: float(r["anomaly_score"]) for r in csv.DictReader(f)}
    s_e20_te = np.array([e20_m[i] for i in canon_te], dtype=np.float64)
    e20_logit_te = np.log(np.clip(s_e20_te, 1e-6, 1 - 1e-6) / (1.0 - np.clip(s_e20_te, 1e-6, 1 - 1e-6)))

    mdir = Path("runs/queue15/models")
    tr_map, te_map = {}, {}
    for p in sorted(mdir.glob("M*.npz")):
        d = np.load(p, allow_pickle=True)
        m_tr = dict(zip([str(x) for x in d["oof_ids"]], d["oof_z"].astype(np.float64)))
        m_te = dict(zip([str(x) for x in d["test_ids"]], d["test_z"].astype(np.float64)))
        if all(i in m_tr for i in canon_tr) and all(i in m_te for i in canon_te):
            tr_map[p.stem] = np.array([m_tr[i] for i in canon_tr], dtype=np.float64)
            te_map[p.stem] = np.array([m_te[i] for i in canon_te], dtype=np.float64)
        # Extract clean forward-only `musc` channel for M22 SigLIP2-512 (zero test_rev noise!)
        if p.stem == "M22_v3_siglip2_so400m_512" and "oof_musc" in d.files:
            mm_tr = dict(zip([str(x) for x in d["oof_ids"]], d["oof_musc"].astype(np.float64)))
            mm_te = dict(zip([str(x) for x in d["test_ids"]], d["test_musc"].astype(np.float64)))
            raw_tr = np.array([mm_tr[i] for i in canon_tr], dtype=np.float64)
            raw_te = np.array([mm_te[i] for i in canon_te], dtype=np.float64)
            z_tr = np.zeros_like(raw_tr)
            for f in range(5):
                mf = folds == f
                z_tr[mf] = left_tail_robust_z(raw_tr[mf], cids_tr[mf])
            z_te = left_tail_robust_z(raw_te, cids_te)
            tr_map["M22_siglip2_512_musc_only"] = z_tr
            te_map["M22_siglip2_512_musc_only"] = z_te

    clean_unsup = [
        "M16_musc_dinov2_reg4_l",
        "M17_musc_eva02_l448",
        "M18_musc_siglip_so400m_384",
        "M19_musc_convnext_l384",
        "M20_musc_dinov2_l518_v2",
        "M21_musc_dinov2_b518_v2",
        "M22_siglip2_512_musc_only",
    ]
    clean_zs = ["M05_siglip_so400m_384", "M03_winclip_l14_laion2b"]
    clean_sup = [
        "M07_vand_clip_l14_laion2b",
        "M08_patchmil_dinov2_l518",
        "M14_patchmil_convnext_l384",
        "M04_patchmil_convnext_b384",
        "M09_patchmil_swin_b384",
    ]
    pool_keys = clean_unsup + clean_zs + clean_sup

    # === GATE 1 & 2: Pre-Submission Audit Table ===
    print("=== [GATE 1 & 2: Pre-Submission Model Audit (Zero test_rev, z_plus_rg clipping)] ===")
    print(f"{'Model':30s} | {'Single OOF':10s} | {'te_max':6s} | {'rho_E20(0.9261)':15s} | {'Status'}")
    print("-" * 82)
    for k in pool_keys:
        s_oof = roc_auc_score(y, tr_map[k])
        rho20 = spearmanr(te_map[k], s_e20_te).statistic
        tmax = float(np.max(te_map[k]))
        flag = "CLIPPED(z+rg)" if tmax > 8.0 else "OK"
        print(f"{k:30s} | {s_oof:10.4f} | {tmax:6.2f} | {rho20:15.4f} | {flag}")

    # Evaluate Clean Full-14 & Clean Pure-Unsup-9 with `z_plus_rg`
    Z_tr_full = calibrate(np.stack([tr_map[k] for k in pool_keys], axis=0), "z_plus_rg")
    Z_te_full = calibrate(np.stack([te_map[k] for k in pool_keys], axis=0), "z_plus_rg")

    oof_full = np.zeros(len(y), dtype=np.float64)
    for f in range(5):
        tr_m, va_m = folds != f, folds == f
        wf = fit_nn_bce(Z_tr_full[:, tr_m], y[tr_m], l2_reg=1.0)
        oof_full[va_m] = wf @ Z_tr_full[:, va_m]
    w_full = fit_nn_bce(Z_tr_full, y, l2_reg=1.0)
    te_full = w_full @ Z_te_full

    auc_full = float(roc_auc_score(y, oof_full))
    pfs_full = {f"fold_{f}": round(float(roc_auc_score(y[folds == f], oof_full[folds == f])), 4) for f in range(5)}
    sim_full = np.array([roc_auc_score(y[wi], oof_full[wi]) for wi in worlds])

    # Pure Unsup + ZS (9 models, ZERO Supervised)
    pure_keys = clean_unsup + clean_zs
    Z_tr_pure = calibrate(np.stack([tr_map[k] for k in pure_keys], axis=0), "z_plus_rg")
    Z_te_pure = calibrate(np.stack([te_map[k] for k in pure_keys], axis=0), "z_plus_rg")
    oof_pure = np.zeros(len(y), dtype=np.float64)
    for f in range(5):
        tr_m, va_m = folds != f, folds == f
        wf = fit_nn_bce(Z_tr_pure[:, tr_m], y[tr_m], l2_reg=1.0)
        oof_pure[va_m] = wf @ Z_tr_pure[:, va_m]
    w_pure = fit_nn_bce(Z_tr_pure, y, l2_reg=1.0)
    te_pure = w_pure @ Z_te_pure
    auc_pure = float(roc_auc_score(y, oof_pure))
    sim_pure = np.array([roc_auc_score(y[wi], oof_pure[wi]) for wi in worlds])

    # === Leave-One-Out Marginal Contribution MVP Ranking ===
    print(f"\n=== [Ensemble MVP Ranking (LOO Marginal Gain vs Full-14 OOF={auc_full:.4f})] ===")
    mvp_rows = []
    for drop_i, dk in enumerate(pool_keys):
        keep = [j for j in range(len(pool_keys)) if j != drop_i]
        oof_d = np.zeros(len(y), dtype=np.float64)
        for f in range(5):
            tr_m, va_m = folds != f, folds == f
            wf = fit_nn_bce(Z_tr_full[keep][:, tr_m], y[tr_m], l2_reg=1.0)
            oof_d[va_m] = wf @ Z_tr_full[keep][:, va_m]
        auc_d = float(roc_auc_score(y, oof_d))
        contrib = auc_full - auc_d  # positive = model HELPS ensemble!
        avg_corr = float(np.mean([spearmanr(tr_map[dk], tr_map[ok]).statistic for ok in pool_keys if ok != dk]))
        mvp_rows.append((contrib, dk, float(roc_auc_score(y, tr_map[dk])), float(w_full[drop_i]), avg_corr))

    mvp_rows.sort(reverse=True)
    print(f"{'Rank':4s} | {'Model':30s} | {'LOO Contrib':11s} | {'Weight':6s} | {'Single OOF':10s} | {'Avg Corr':8s}")
    print("-" * 82)
    for rank, (c, dk, so, wt, ac) in enumerate(mvp_rows, 1):
        print(f"{rank:4d} | {dk:30s} | {c:+11.4f} | {wt:6.3f} | {so:10.4f} | {ac:8.3f}")

    # === GATE 3: Champion Weapon Preservation Blend (50% E20 SP4 + 50% Clean E22 Full-14) ===
    te_full_std = (te_full - np.median(te_full)) / (np.percentile(te_full, 75) - np.percentile(te_full, 25) + 1e-6)
    e20_std = (e20_logit_te - np.median(e20_logit_te)) / (np.percentile(e20_logit_te, 75) - np.percentile(e20_logit_te, 25) + 1e-6)
    te_blend_e20 = 0.50 * e20_std + 0.50 * te_full_std

    print("\n=== [GATE 3: Final Candidate Summary & Test Alignment vs E20(0.9261 LB)] ===")
    print(f"  E22a Clean-14 (z_plus_rg, NO rev) : OOF={auc_full:.4f} | Mean={np.mean(list(pfs_full.values())):.4f} | SimLB={sim_full.mean():.4f} (P10={np.percentile(sim_full, 10):.4f}) | rho_E20={spearmanr(te_full, s_e20_te).statistic:.4f}")
    print(f"  E22b Blend(50% E20_SP4 + 50% E22a): preserves SP4 32x32 maps + adds M20/M21/M22_musc | rho_E20={spearmanr(te_blend_e20, s_e20_te).statistic:.4f}")
    print(f"  E22c Pure-Unsup+ZS-9 (ZERO Sup)   : OOF={auc_pure:.4f} | SimLB={sim_pure.mean():.4f} (P10={np.percentile(sim_pure, 10):.4f}) | rho_E20={spearmanr(te_pure, s_e20_te).statistic:.4f}")

    sub_blend = Path("runs/submissions/sub_e22_clean_blend_e20_m22.csv")
    sub_clean = Path("runs/submissions/sub_e22_clean14_zrg_no_rev.csv")
    sub_pure = Path("runs/submissions/sub_e22_clean_pure_unsup9.csv")
    chk_b = export_verified_submission(sub_blend, sample_sub, canon_te, te_blend_e20, is_rank=False)
    chk_c = export_verified_submission(sub_clean, sample_sub, canon_te, te_full, is_rank=False)
    chk_p = export_verified_submission(sub_pure, sample_sub, canon_te, te_pure, is_rank=False)
    print(f"\n[VERIFIED] {sub_blend} | {chk_b}")
    print(f"[VERIFIED] {sub_clean} | {chk_c}")
    print(f"[VERIFIED] {sub_pure} | {chk_p}")


if __name__ == "__main__":
    main()
