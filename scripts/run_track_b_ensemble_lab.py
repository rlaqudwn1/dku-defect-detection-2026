"""Track B Ensemble Lab: Systematic Calibration, Meta-Learner & Subset Optimization.

Evaluates strictly under Nested 5-Fold Unseen-Product CV (`runs/folds_unseen.json`)
across all cached models in `runs/queue15/models/M*.npz` (ZERO Test image access).

Benchmarks:
  1. Score Calibrations: Robust-Z, Rank-Gauss (Inverse-Normal CDF), Soft-Clipped Z
  2. Meta-Learners:
     - Equal-Z & 5-Tower Paradigm-Hierarchical
     - Non-Negative BCE Logistic Stack (L2-to-0)
     - Direct Pairwise AUROC Surrogate Optimizer (smooth WMW ranking loss!)
     - Tail-Consensus / Max-Disagreement Hybrid (weighted_mean + alpha * std/top2)
  3. Subset Selection: Full-15 vs 12m (excl collapsed) vs Elite-9 vs Diversity-Weighted
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm, rankdata
from sklearn.metrics import roc_auc_score


def load_models(queue_dir: Path, folds_json: Path, sample_sub: Path):
    folds_payload = json.loads(folds_json.read_text(encoding="utf-8"))
    labels_map = folds_payload["labels"]
    fold_map = {
        img_id: int(f_str)
        for f_str, f_data in folds_payload["folds"].items()
        for img_id in f_data["val_ids"]
    }
    canon_tr = sorted(labels_map.keys())
    y_tr = np.array([labels_map[i] for i in canon_tr], dtype=np.int64)
    fold_arr = np.array([fold_map[i] for i in canon_tr], dtype=np.int64)

    with sample_sub.open("r", encoding="utf-8-sig", newline="") as f:
        canon_te = [r["image_path"] for r in csv.DictReader(f)]

    model_ids, Z_tr_raw, Z_te_raw = [], [], []
    for p in sorted((queue_dir / "models").glob("M*.npz")):
        d = np.load(p, allow_pickle=True)
        oof_m = dict(zip([str(x) for x in d["oof_ids"]], d["oof_z"].astype(np.float64)))
        te_m = dict(zip([str(x) for x in d["test_ids"]], d["test_z"].astype(np.float64)))
        if not all(i in oof_m for i in canon_tr):
            continue
        model_ids.append(p.stem)
        Z_tr_raw.append(np.array([oof_m[i] for i in canon_tr], dtype=np.float64))
        Z_te_raw.append(np.array([te_m[i] for i in canon_te], dtype=np.float64))

    return model_ids, np.stack(Z_tr_raw, 0), np.stack(Z_te_raw, 0), y_tr, fold_arr, canon_tr, canon_te


def calibrate(Z: np.ndarray, mode: str) -> np.ndarray:
    out = np.zeros_like(Z, dtype=np.float64)
    for i in range(Z.shape[0]):
        v = Z[i]
        med = np.median(v)
        iqr = np.percentile(v, 75) - np.percentile(v, 25) + 1e-6
        z = (v - med) / iqr
        if mode == "robust_z":
            out[i] = z
        elif mode == "winsor_z":
            out[i] = np.clip(z, -3.5, 6.0)
        elif mode == "rank_gauss":
            u = (rankdata(v, method="average") - 0.5) / len(v)
            out[i] = norm.ppf(np.clip(u, 1e-5, 1.0 - 1e-5))
        elif mode == "z_plus_rg":
            u = (rankdata(v, method="average") - 0.5) / len(v)
            rg = norm.ppf(np.clip(u, 1e-5, 1.0 - 1e-5))
            out[i] = 0.5 * np.clip(z, -4.0, 6.0) + 0.5 * rg
        else:
            raise ValueError(mode)
    return out


def fit_nn_bce(Z_in: np.ndarray, y_in: np.ndarray, l2_reg: float = 1.0) -> np.ndarray:
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


def fit_direct_auc(Z_in: np.ndarray, y_in: np.ndarray, l2_reg: float = 0.05, tau: float = 0.5) -> np.ndarray:
    """Directly optimize smooth pairwise AUROC surrogate (sampled pos-neg pairs, deterministic seed)."""
    m = Z_in.shape[0]
    pos_idx = np.where(y_in == 1)[0]
    neg_idx = np.where(y_in == 0)[0]
    rng = np.random.default_rng(42)
    n_pairs = 25000
    pi = rng.choice(pos_idx, size=n_pairs, replace=True)
    ni = rng.choice(neg_idx, size=n_pairs, replace=True)
    D = (Z_in[:, pi] - Z_in[:, ni]).T  # (n_pairs, M)

    def _loss_and_grad(w):
        margins = (D @ w) / tau
        # log(1 + exp(-margin))
        loss_vec = np.logaddexp(0.0, -margins)
        sig = 1.0 / (1.0 + np.exp(np.clip(margins, -30.0, 30.0)))  # sigmoid(-margin)
        grad = -(D.T @ sig) / (n_pairs * tau) + 2.0 * l2_reg * w
        return float(np.mean(loss_vec) + l2_reg * np.sum(w ** 2)), grad

    init = np.full(m, 1.0 / m)
    bounds = [(0.0, 5.0)] * m
    res = minimize(_loss_and_grad, init, jac=True, method="L-BFGS-B", bounds=bounds, options={"maxiter": 200})
    w = np.maximum(0.0, res.x)
    return w / (w.sum() + 1e-12)


def eval_nested(name: str, oof: np.ndarray, y_tr: np.ndarray, fold_arr: np.ndarray) -> float:
    auc = float(roc_auc_score(y_tr, oof))
    pfs = [float(roc_auc_score(y_tr[fold_arr == f], oof[fold_arr == f])) for f in range(5)]
    print(f"{name:48s} | OOF={auc:.4f} | Mean={np.mean(pfs):.4f} | " + " ".join(f"F{f}:{v:.4f}" for f, v in enumerate(pfs)))
    return auc


def run_benchmark():
    mids, Z_tr_raw, Z_te_raw, y_tr, fold_arr, _, _ = load_models(
        Path("runs/queue15"), Path("runs/folds_unseen.json"), Path("data/raw/open/sample_submission.csv")
    )
    idx_of = {m[:3]: i for i, m in enumerate(mids)}
    elite9 = [idx_of[k] for k in ("M01", "M06", "M07", "M05", "M03", "M08", "M14", "M04", "M09")]
    clean12 = [idx_of[k] for k in ("M01", "M02", "M03", "M04", "M05", "M06", "M07", "M08", "M09", "M11", "M13", "M14")]

    print("=== [Track B-1] Calibration Comparison (Nested 5-Fold, NN-BCE l2=1.0) ===")
    for cal_mode in ["robust_z", "winsor_z", "rank_gauss", "z_plus_rg"]:
        Zc = calibrate(Z_tr_raw, cal_mode)
        for sname, sidx in [("Clean-12", clean12), ("Elite-9", elite9)]:
            oof = np.zeros(len(y_tr))
            for f in range(5):
                tr, va = fold_arr != f, fold_arr == f
                w = fit_nn_bce(Zc[sidx][:, tr], y_tr[tr], l2_reg=1.0)
                oof[va] = w @ Zc[sidx][:, va]
            eval_nested(f"{sname:8s} [{cal_mode:10s}] NN-BCE(1.0)", oof, y_tr, fold_arr)

    print("\n=== [Track B-2] Direct Pairwise AUROC Surrogate Meta-Learner (Nested 5-Fold) ===")
    for cal_mode in ["robust_z", "rank_gauss", "z_plus_rg"]:
        Zc = calibrate(Z_tr_raw, cal_mode)
        for sname, sidx in [("All-15", list(range(len(mids)))), ("Clean-12", clean12), ("Elite-9", elite9)]:
            for l2 in [0.02, 0.1]:
                oof = np.zeros(len(y_tr))
                for f in range(5):
                    tr, va = fold_arr != f, fold_arr == f
                    w = fit_direct_auc(Zc[sidx][:, tr], y_tr[tr], l2_reg=l2, tau=0.5)
                    oof[va] = w @ Zc[sidx][:, va]
                eval_nested(f"{sname:8s} [{cal_mode:10s}] DirectAUC(l2={l2})", oof, y_tr, fold_arr)

    print("\n=== [Track B-3] Tail-Consensus / Top-K Disagreement Hybrid (Elite-9, robust_z & z_plus_rg) ===")
    for cal_mode in ["robust_z", "z_plus_rg"]:
        Zc = calibrate(Z_tr_raw, cal_mode)[elite9]
        # Base nested linear score
        oof_lin = np.zeros(len(y_tr))
        for f in range(5):
            tr, va = fold_arr != f, fold_arr == f
            w = fit_nn_bce(Zc[:, tr], y_tr[tr], l2_reg=1.0)
            oof_lin[va] = w @ Zc[:, va]
        top2_mean = np.sort(Zc, axis=0)[-2:].mean(axis=0)
        std_acc = Zc.std(axis=0)
        for alpha in [0.05, 0.15, 0.25]:
            eval_nested(f"Elite-9 [{cal_mode:10s}] + {alpha:.2f}*Top2Mean", oof_lin + alpha * top2_mean, y_tr, fold_arr)
            eval_nested(f"Elite-9 [{cal_mode:10s}] + {alpha:.2f}*Std", oof_lin + alpha * std_acc, y_tr, fold_arr)

    print("\n=== [Track B-4] 5-Tower Paradigm-Hierarchical DirectAUC vs NN-BCE (Clean-12, zero selection bias) ===")
    towers = {
        "A_MuSc": [idx_of["M01"], idx_of["M06"]],
        "B_VAND": [idx_of["M02"], idx_of["M07"], idx_of["M11"]],
        "C_ZeroShot": [idx_of["M03"], idx_of["M05"]],
        "D_PatchMIL_ViT": [idx_of["M08"], idx_of["M13"]],
        "E_CNN_Hier": [idx_of["M04"], idx_of["M14"], idx_of["M09"]],
    }
    for cal_mode in ["robust_z", "rank_gauss", "z_plus_rg"]:
        Zc = calibrate(Z_tr_raw, cal_mode)
        T = np.stack([Zc[idxs].mean(axis=0) for idxs in towers.values()], axis=0)
        T = calibrate(T, "robust_z")
        for mname, fn in [("NN-BCE(0.2)", lambda Z, y: fit_nn_bce(Z, y, 0.2)), ("DirectAUC(0.05)", lambda Z, y: fit_direct_auc(Z, y, 0.05))]:
            oof = np.zeros(len(y_tr))
            for f in range(5):
                tr, va = fold_arr != f, fold_arr == f
                w = fn(T[:, tr], y_tr[tr])
                oof[va] = w @ T[:, va]
            eval_nested(f"5-Tower-12m [{cal_mode:10s}] {mname}", oof, y_tr, fold_arr)
            w_full = fn(T, y_tr)
            print("   weights:", ", ".join(f"{k}:{v:.2f}" for k, v in zip(towers.keys(), w_full)))


if __name__ == "__main__":
    run_benchmark()
