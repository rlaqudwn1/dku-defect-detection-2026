"""Log publication-ready Two-Track Competition Report charts & tables to W&B.

Logs strictly Train Unseen 5-Fold OOF metrics, ensemble ablations, Spearman diversity
heatmap, and Public LB progression to W&B (`entity/project` configurable via CLI/env).
ZERO Test images are ever accessed or logged.

Usage:
    .venv/bin/python scripts/log_wandb_report.py --run-name e14_15models_trackB_report
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import wandb


LB_HISTORY = [
    ("E04", "ResNet18 Patch kNN", None, 0.5593),
    ("E06", "DINOv2-B + WinCLIP raw", 0.7893, 0.6941),
    ("E06-clean", "Remove fake MuSc", None, 0.6968),
    ("E07", "Category Logit-Z", None, 0.7167),
    ("E12-T2", "Wave-1 4-Model (No MuSc)", 0.9054, 0.8231),
    ("E11", "E09 MuSc + E10 VAND", None, 0.8752),
    ("E14-Elite9", "15-Pool Pruned Elite-9", 0.9520, 0.8846),
]


def make_report_figures(summary: dict, fig_dir: Path, track_b_best: tuple[str, float] | None = None) -> dict[str, Path]:
    fig_dir.mkdir(parents=True, exist_ok=True)
    figs: dict[str, Path] = {}

    # 1. Single-Model vs Ensemble Progression Bar Chart
    singles = sorted(summary["single_models"], key=lambda x: x["oof_auroc"])
    labels = [r["model_id"].replace("_", " ") for r in singles]
    scores = [r["oof_auroc"] for r in singles]
    colors = ["#4C78A8" if "musc" not in r["model_id"] else "#F58518" for r in singles]

    # Append key ensemble milestones
    em = summary["ensemble_methods"]
    for key, disp, col in [
        ("4_nested_nn_logistic_stack", "Ens: Naive-15 Stack", "#72B7B2"),
        ("6_paradigm_5tower_nested_stack", "Ens: 5-Tower (12m)", "#54A24B"),
        ("7_pruned_elite9_nested_stack", "Ens: Pruned Elite-9", "#E45756"),
    ]:
        if key in em:
            labels.append(disp)
            scores.append(em[key]["nested_5fold_oof"])
            colors.append(col)
    if track_b_best is not None:
        labels.append(track_b_best[0])
        scores.append(track_b_best[1])
        colors.append("#B279A2")

    fig, ax = plt.subplots(figsize=(10, 7), dpi=160)
    y_pos = np.arange(len(labels))
    bars = ax.barh(y_pos, scores, color=colors, edgecolor="none", height=0.72)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0.62, 1.00)
    ax.set_xlabel("Train 5-Fold Unseen-Product OOF AUROC", fontsize=10)
    ax.set_title(
        f"Single-Model ({summary['completed_models']} Models) vs Nested 5-Fold Ensembles",
        fontsize=11.5,
        fontweight="bold",
    )
    ax.grid(axis="x", linestyle="--", alpha=0.4)
    for bar, sc in zip(bars, scores):
        ax.text(sc + 0.004, bar.get_y() + bar.get_height() / 2, f"{sc:.4f}", va="center", fontsize=8)
    plt.tight_layout()
    p1 = fig_dir / "single_vs_ensemble_oof.png"
    fig.savefig(p1)
    plt.close(fig)
    figs["single_vs_ensemble_oof"] = p1

    # 2. Pairwise Spearman Rank Correlation Heatmap (Diversity Evidence)
    corr_dict = summary["spearman_corr_matrix"]
    mids = list(corr_dict.keys())
    short_names = [m[:14] for m in mids]
    mat = np.array([[corr_dict[i][j] for j in mids] for i in mids], dtype=np.float64)

    fig, ax = plt.subplots(figsize=(9.5, 8.2), dpi=160)
    im = ax.imshow(mat, vmin=0.0, vmax=1.0, cmap="YlGnBu")
    ax.set_xticks(np.arange(len(mids)))
    ax.set_yticks(np.arange(len(mids)))
    ax.set_xticklabels(short_names, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(short_names, fontsize=8)
    for i in range(len(mids)):
        for j in range(len(mids)):
            val = mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=6.5, color="white" if val > 0.65 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Spearman Rank Correlation")
    ax.set_title("Pairwise Spearman Correlation Across Diverse Models (Train OOF)", fontsize=11, fontweight="bold")
    plt.tight_layout()
    p2 = fig_dir / "spearman_diversity_heatmap.png"
    fig.savefig(p2)
    plt.close(fig)
    figs["spearman_diversity_heatmap"] = p2

    # 3. Public LB Progression Curve
    exps = [x[0] for x in LB_HISTORY]
    lbs = [x[3] for x in LB_HISTORY]
    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=160)
    ax.plot(range(len(exps)), lbs, marker="o", linewidth=2.4, markersize=8, color="#E45756")
    ax.set_xticks(range(len(exps)))
    ax.set_xticklabels(exps, fontsize=9.5)
    ax.set_ylim(0.50, 0.96)
    ax.set_ylabel("Public LB AUROC", fontsize=10)
    ax.set_title("Competition Public LB Progression (E04 -> E14)", fontsize=11.5, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.45)
    for idx, (exp, desc, _, lb) in enumerate(LB_HISTORY):
        ax.annotate(f"{lb:.4f}", (idx, lb), textcoords="offset points", xytext=(0, 9), ha="center", fontsize=8.5, fontweight="bold")
    plt.tight_layout()
    p3 = fig_dir / "public_lb_progression.png"
    fig.savefig(p3)
    plt.close(fig)
    figs["public_lb_progression"] = p3

    return figs


def main() -> None:
    parser = argparse.ArgumentParser(description="Log Two-Track competition report artifacts to W&B")
    parser.add_argument("--summary-json", type=str, default="runs/queue15/ensemble_summary.json")
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY", "rlaqudwn123456-dankook-university"))
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "CV"))
    parser.add_argument("--run-name", type=str, default="e14_15models_two_track_report")
    parser.add_argument("--group", type=str, default="two-track-report")
    parser.add_argument("--track-b-best-name", type=str, default="TrackB: Elite9+0.25*Top2Mean")
    parser.add_argument("--track-b-best-oof", type=float, default=0.9533)
    parser.add_argument("--public-lb", type=float, default=0.884610)
    args = parser.parse_args()

    summary_path = Path(args.summary_json)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    fig_dir = summary_path.parent / "report_figs"
    figs = make_report_figures(summary, fig_dir, track_b_best=(args.track_b_best_name, args.track_b_best_oof))

    run = wandb.init(
        entity=args.entity,
        project=args.project,
        name=args.run_name,
        group=args.group,
        tags=["two-track", f"{summary['completed_models']}-models", "unseen-5fold", "report"],
        config={
            "completed_models": summary["completed_models"],
            "best_single_model": summary["best_single_model"]["model_id"],
            "best_ensemble_method": summary["best_ensemble_method"],
            "track_b_best_method": args.track_b_best_name,
        },
    )

    # 1. Single-Model Leaderboard Table
    single_cols = ["model_id", "paradigm", "backbone", "oof_auroc", "mean_fold_auroc", "fold_0", "fold_1", "fold_2", "fold_3", "fold_4", "license"]
    single_rows = []
    for r in sorted(summary["single_models"], key=lambda x: -x["oof_auroc"]):
        pf = r["per_fold"]
        single_rows.append([
            r["model_id"], r["paradigm"], r["backbone"], r["oof_auroc"], r["mean_fold_auroc"],
            pf["fold_0"], pf["fold_1"], pf["fold_2"], pf["fold_3"], pf["fold_4"], r["license"],
        ])
    single_tbl = wandb.Table(columns=single_cols, data=single_rows)

    # 2. Ensemble Methods Ablation Table
    ens_cols = ["method", "nested_5fold_oof", "rationale", "active_weights"]
    ens_rows = []
    for mname, mdata in summary["ensemble_methods"].items():
        active_w = ", ".join(f"{k}:{v:.2f}" for k, v in mdata["weights"].items() if v >= 0.01)
        ens_rows.append([mname, mdata["nested_5fold_oof"], mdata["why"], active_w])
    ens_rows.append([
        args.track_b_best_name,
        args.track_b_best_oof,
        "Track B Tail-Consensus Hybrid: linear NN-BCE + 0.25*Top2Mean preserves sharp specialist anomaly alarms.",
        "Elite-9 + 0.25*Top2Mean",
    ])
    ens_tbl = wandb.Table(columns=ens_cols, data=ens_rows)

    # 3. Public LB Progression Table
    lb_tbl = wandb.Table(
        columns=["exp", "description", "train_5fold_oof", "public_lb"],
        data=[[e, d, o if o is not None else float("nan"), lb] for e, d, o, lb in LB_HISTORY],
    )

    log_payload = {
        "tables/single_models_leaderboard": single_tbl,
        "tables/ensemble_methods_ablation": ens_tbl,
        "tables/public_lb_progression": lb_tbl,
        "charts/single_models_bar": wandb.plot.bar(single_tbl, "model_id", "oof_auroc", title="Single-Model 5-Fold Unseen OOF AUROC"),
        "charts/ensemble_ablation_bar": wandb.plot.bar(ens_tbl, "method", "nested_5fold_oof", title="Nested 5-Fold Ensemble Ablation"),
        "summary/best_single_oof": summary["best_single_model"]["oof_auroc"],
        "summary/best_ensemble_oof": summary["best_ensemble_nested_5fold_oof"],
        "summary/track_b_best_oof": args.track_b_best_oof,
        "summary/public_lb_champion": args.public_lb,
        "summary/completed_models": summary["completed_models"],
    }
    for k, p in figs.items():
        log_payload[f"figures/{k}"] = wandb.Image(str(p))

    run.log(log_payload)
    # Also log LB progression as a W&B line-series step metric for native W&B Report line plots
    for step_idx, (_, _, _, lb) in enumerate(LB_HISTORY):
        run.log({"progression/public_lb": lb}, step=step_idx)

    url = run.url
    run.finish()
    print(f"\n[W&B REPORT LOGGED] URL: {url}")


if __name__ == "__main__":
    main()
