"""E21-v3 Ensemble Builder: M16~M22 v3/v2 Integration (Zero Legacy M01/M06 19% Singleton).

Evaluates strictly under BOTH:
  - Nested 5-Fold Unseen-Product CV (`runs/folds_unseen.json`, 10,821 Train images)
  - `Sim-LB-800` (200 fragmented variable-prevalence Train mini-worlds)

Exports verified black-box Test submission CSVs:
  1. `runs/submissions/sub_e21_v3_full14.csv`       (Primary Champion: v3+v2+ZS+Sup, ZERO legacy M01/M06)
  2. `runs/submissions/sub_e21_v3_pure_unsup9.csv`  (Private Shake-up Barbell: ZERO Supervised models!)

ZERO Test images are ever rendered, saved, or visually inspected.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

from run_15model_ensemble import export_verified_submission
from run_track_b_ensemble_lab import calibrate, fit_nn_bce
from run_track_b_v2_final import build_product_sim_worlds


def load_models(models_dir: Path, canon_tr: list[str], canon_te: list[str]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    tr_map, te_map = {}, {}
    for p in sorted(models_dir.glob("M*.npz")):
        d = np.load(p, allow_pickle=True)
        m_tr = dict(zip([str(x) for x in d["oof_ids"]], d["oof_z"].astype(np.float64)))
        m_te = dict(zip([str(x) for x in d["test_ids"]], d["test_z"].astype(np.float64)))
        if all(i in m_tr for i in canon_tr) and all(i in m_te for i in canon_te):
            tr_map[p.stem] = np.array([m_tr[i] for i in canon_tr], dtype=np.float64)
            te_map[p.stem] = np.array([m_te[i] for i in canon_te], dtype=np.float64)
    return tr_map, te_map


def eval_candidate(
    name: str,
    keys: list[str],
    tr_map: dict[str, np.ndarray],
    te_map: dict[str, np.ndarray],
    y: np.ndarray,
    folds: np.ndarray,
    m_cid2: np.ndarray,
    worlds: list[np.ndarray],
    cal: str = "robust_z",
    l2_reg: float = 1.0,
    top2_alpha: float = 0.0,
) -> dict:
    Z_tr = calibrate(np.stack([tr_map[k] for k in keys], axis=0), cal)
    Z_te = calibrate(np.stack([te_map[k] for k in keys], axis=0), cal)

    oof_lin = np.zeros(len(y), dtype=np.float64)
    for f in range(5):
        tr_m, va_m = folds != f, folds == f
        wf = fit_nn_bce(Z_tr[:, tr_m], y[tr_m], l2_reg=l2_reg)
        oof_lin[va_m] = wf @ Z_tr[:, va_m]
    w_full = fit_nn_bce(Z_tr, y, l2_reg=l2_reg)
    te_lin = w_full @ Z_te

    if top2_alpha > 0.0:
        oof = oof_lin + top2_alpha * np.sort(Z_tr, axis=0)[-2:].mean(axis=0)
        te = te_lin + top2_alpha * np.sort(Z_te, axis=0)[-2:].mean(axis=0)
    else:
        oof, te = oof_lin, te_lin

    auc = float(roc_auc_score(y, oof))
    pfs = {f"fold_{f}": round(float(roc_auc_score(y[folds == f], oof[folds == f])), 4) for f in range(5)}
    mean_f = float(np.mean(list(pfs.values())))
    c2 = float(roc_auc_score(y[m_cid2], oof[m_cid2]))
    sim_aucs = np.array([roc_auc_score(y[w_idx], oof[w_idx]) for w_idx in worlds], dtype=np.float64)

    print(
        f"{name:34s} | OOF={auc:.4f} | Mean={mean_f:.4f} | cid2={c2:.4f} | "
        f"SimLB={sim_aucs.mean():.4f} (P10={np.percentile(sim_aucs, 10):.4f}) | "
        + " ".join(f"F{f}:{pfs[f'fold_{f}']:.4f}" for f in range(5))
    )
    return {
        "name": name,
        "keys": keys,
        "cal": cal,
        "top2_alpha": top2_alpha,
        "nested_oof": round(auc, 4),
        "fold_mean": round(mean_f, 4),
        "cid2_auroc": round(c2, 4),
        "per_fold": pfs,
        "sim_lb_mean": round(float(sim_aucs.mean()), 4),
        "sim_lb_p10": round(float(np.percentile(sim_aucs, 10)), 4),
        "weights": {k: round(float(w), 4) for k, w in zip(keys, w_full)},
        "oof": oof,
        "test": te,
    }


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
    cid_map = dict(zip([str(x) for x in cd["train_ids"]], cd["train_cids"].astype(np.int64)))
    cids = np.array([cid_map[i] for i in canon_tr], dtype=np.int64)
    m_cid2 = cids == 2
    worlds = build_product_sim_worlds(cids, y, n_worlds=200, world_size=800, seed=42)

    tr_map, te_map = load_models(Path("runs/queue15/models"), canon_tr, canon_te)

    unsup_v3_v2 = [
        "M16_v3_dinov2_reg4_l",
        "M17_v3_eva02_l448",
        "M18_v3_siglip_so400m_384",
        "M19_v3_convnext_l384",
        "M22_v3_siglip2_so400m_512",
        "M20_musc_dinov2_l518_v2",
        "M21_musc_dinov2_b518_v2",
    ]
    zs_vlm = ["M05_siglip_so400m_384", "M03_winclip_l14_laion2b"]
    sup_elite = [
        "M07_vand_clip_l14_laion2b",
        "M08_patchmil_dinov2_l518",
        "M14_patchmil_convnext_l384",
        "M04_patchmil_convnext_b384",
        "M09_patchmil_swin_b384",
    ]

    pure9_keys = unsup_v3_v2 + zs_vlm
    full14_keys = unsup_v3_v2 + zs_vlm + sup_elite

    print("=== [E21-v3 Strict Nested 5-Fold Unseen CV & Sim-LB-800 Validation] ===")
    c_pure9 = eval_candidate("E21b_PureUnsupZS9 (Linear)", pure9_keys, tr_map, te_map, y, folds, m_cid2, worlds, "robust_z", 1.0, 0.0)
    c_pure9_t2 = eval_candidate("E21b_PureUnsupZS9 (+0.15*Top2)", pure9_keys, tr_map, te_map, y, folds, m_cid2, worlds, "robust_z", 1.0, 0.15)
    c_full14 = eval_candidate("E21a_Full14_v3 (Linear)", full14_keys, tr_map, te_map, y, folds, m_cid2, worlds, "robust_z", 1.0, 0.0)
    c_full14_t2 = eval_candidate("E21a_Full14_v3 (+0.15*Top2)", full14_keys, tr_map, te_map, y, folds, m_cid2, worlds, "robust_z", 1.0, 0.15)

    best_full = c_full14_t2 if c_full14_t2["nested_oof"] > c_full14["nested_oof"] else c_full14
    best_pure = c_pure9_t2 if c_pure9_t2["nested_oof"] > c_pure9["nested_oof"] else c_pure9

    sub_full = Path("runs/submissions/sub_e21_v3_full14.csv")
    sub_pure = Path("runs/submissions/sub_e21_v3_pure_unsup9.csv")
    chk_full = export_verified_submission(sub_full, sample_sub, canon_te, best_full["test"], is_rank=False)
    chk_pure = export_verified_submission(sub_pure, sample_sub, canon_te, best_pure["test"], is_rank=False)

    out_dir = Path("runs/track_b_v3")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "primary_full14": {k: v for k, v in best_full.items() if k not in ("oof", "test")},
        "barbell_pure_unsup9": {k: v for k, v in best_pure.items() if k not in ("oof", "test")},
        "submission_full14": str(sub_full),
        "checks_full14": chk_full,
        "submission_pure9": str(sub_pure),
        "checks_pure9": chk_pure,
    }
    (out_dir / "e21_v3_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    # Export OOF for Train error analysis
    oof_recs = [
        {"image_path": img_id, "label": int(lbl), "fold": int(fld), "score": round(float(sc), 6)}
        for img_id, lbl, fld, sc in zip(canon_tr, y, folds, best_full["oof"])
    ]
    (out_dir / "val_oof_e21.json").write_text(
        json.dumps({"method": best_full["name"], "oof_auroc": best_full["nested_oof"], "records": oof_recs}),
        encoding="utf-8",
    )

    print(f"\n[VERIFIED PRIMARY] {sub_full} | OOF={best_full['nested_oof']} | SimLB={best_full['sim_lb_mean']} | checks={chk_full}")
    print(f"[VERIFIED BARBELL] {sub_pure} | OOF={best_pure['nested_oof']} | SimLB={best_pure['sim_lb_mean']} | checks={chk_pure}")
    print("\nPrimary Weights:", best_full["weights"])


if __name__ == "__main__":
    main()
