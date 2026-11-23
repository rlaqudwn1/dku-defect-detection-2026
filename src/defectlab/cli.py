"""Lean ML competition CLI; heavy dependencies are imported only when needed."""
import argparse
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(prog="defectlab")
    commands = root.add_subparsers(dest="command", required=True)

    p = commands.add_parser("verify", help="Verify dataset files, labels and optional manifest")
    p.add_argument("data_root")
    p.add_argument("--manifest")

    p = commands.add_parser("profile", help="Decode images and build dataset profile + contact sheet")
    p.add_argument("data_root")
    p.add_argument("output_dir")

    p = commands.add_parser("audit-duplicates", help="Audit file, pixel and dHash duplicates")
    p.add_argument("data_root")
    p.add_argument("output_dir")
    p.add_argument("--split", help="Optional split JSON with train_ids and val_ids")
    p.add_argument("--max-hamming", type=int, default=6, help="Maximum dHash distance (default: 6)")
    p.add_argument("--top-k", type=int, default=100, help="Pairs per candidate list (default: 100)")

    p = commands.add_parser("compare-transforms", help="Compare resize/pad with original image detail")
    p.add_argument("data_root")
    p.add_argument("output_dir")
    p.add_argument("--ids", nargs="+", required=True, help="Image IDs, a JSON array file, or a text list file")
    p.add_argument("--sizes", nargs="+", type=int, default=[256, 384, 512], help="Sizes (default: 256 384 512)")

    p = commands.add_parser("split", help="Create a leakage-safe stratified group holdout split")
    p.add_argument("profile")
    p.add_argument("output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--groups")

    p = commands.add_parser("folds", help="Create a leakage-safe Stratified Group K-Fold split")
    p.add_argument("profile")
    p.add_argument("output")
    p.add_argument("--n-splits", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--groups")

    p = commands.add_parser("train", help="Train a model on a holdout split or a specific CV fold")
    p.add_argument("data_root")
    p.add_argument("split")
    p.add_argument("config")
    p.add_argument("output_dir")
    p.add_argument("--resume")
    p.add_argument("--fold", type=int, default=None, help="Fold index when using folds.json")
    p.add_argument("--fast", action="store_true", help="Skip full-dataset SHA-256 re-hashing")

    p = commands.add_parser("evaluate", help="Evaluate a checkpoint on validation/fold split (GPU + optional TTA)")
    p.add_argument("data_root")
    p.add_argument("split")
    p.add_argument("checkpoint")
    p.add_argument("output")
    p.add_argument("--fold", type=int, default=None, help="Fold index when using folds.json")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--tta", action="store_true", help="Enable 4x flip Test-Time Augmentation")
    p.add_argument("--fast", action="store_true", help="Skip full-dataset SHA-256 re-hashing")

    p = commands.add_parser("predict", help="Generate submission CSV on test set (GPU + optional TTA)")
    p.add_argument("data_root")
    p.add_argument("template")
    p.add_argument("checkpoint")
    p.add_argument("output")
    p.add_argument("--device", choices=["cpu", "cuda"], default=None)
    p.add_argument("--tta", action="store_true", help="Enable 4x flip Test-Time Augmentation")
    p.add_argument("--fast", action="store_true", help="Skip full-dataset SHA-256 re-hashing")

    p = commands.add_parser("errors", help="Error-driven EDA: render FP Top-K and FN Top-K contact sheets")
    p.add_argument("data_root")
    p.add_argument("validation")
    p.add_argument("output_dir")
    p.add_argument("--top-k", type=int, default=20, help="Number of top FP/FN images (default: 20)")

    p = commands.add_parser("tune", help="Run bounded sequential candidate configs")
    p.add_argument("data_root")
    p.add_argument("split")
    p.add_argument("plan")
    p.add_argument("output_dir")

    p = commands.add_parser("ensemble", help="Blend validation or test predictions (prob, rank, or logit)")
    p.add_argument("output")
    p.add_argument("members", nargs="+")
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--method", choices=["prob", "rank", "logit"], default="prob")
    p.add_argument("--recipe")
    p.add_argument("--frozen-recipe")

    p = commands.add_parser("reproduce", help="Verify numerical equivalence between two prediction files")
    p.add_argument("reference")
    p.add_argument("current")
    p.add_argument("--probability-atol", type=float, required=True)
    p.add_argument("--auroc-atol", type=float, required=True)

    p = commands.add_parser("brief", help="Print compact AI session boot context (git, STATE, runs, errors)")
    p.add_argument("--repo-root", default=".")
    p.add_argument("--runs-dir", default="runs")
    p.add_argument("--top-n", type=int, default=5)

    return root


def dispatch(a):
    if a.command == "brief":
        from .brief import build_brief
        return build_brief(a.repo_root, a.runs_dir, a.top_n)
    if a.command in {"verify", "profile", "split", "folds"}:
        from . import data
        if a.command == "verify":
            return data.verify(a.data_root, a.manifest)
        if a.command == "profile":
            return data.profile(a.data_root, a.output_dir)
        if a.command == "split":
            return data.make_split(a.profile, a.output, a.seed, a.val_fraction, a.groups)
        return data.make_folds(a.profile, a.output, a.n_splits, a.seed, a.groups)
    if a.command == "audit-duplicates":
        from .eda import audit_duplicates
        report = audit_duplicates(a.data_root, a.output_dir, a.split, a.max_hamming, a.top_k)
        return {
            "schema_version": report["schema_version"],
            "ok": report["ok"],
            "report_path": str(Path(a.output_dir).resolve() / "duplicate_audit.json"),
            "dataset_hash": report["dataset_hash"],
            "split_sha256": report["split_sha256"],
            "split_file_sha256": report["split_file_sha256"],
            "max_hamming": report["max_hamming"],
            "top_k_per_candidate_list": report["top_k_per_candidate_list"],
            "image_count": len(report["images"]),
            "decode_failure_count": len(report["decode_failures"]),
            "exact_file_duplicate_group_count": len(report["exact_file_duplicate_groups"]),
            "exact_pixel_duplicate_group_count": len(report["exact_pixel_duplicate_groups"]),
            "near_duplicate_candidate_count": report["near_duplicate_candidate_count"],
            "cross_split_candidate_count": report["cross_split_candidate_count"],
            "train_test_candidate_count": report["train_test_candidate_count"],
        }
    if a.command == "compare-transforms":
        from .eda import compare_transforms
        return compare_transforms(a.data_root, a.output_dir, a.ids, a.sizes)
    if a.command == "errors":
        from .errors import analyze_errors
        return analyze_errors(a.data_root, a.validation, a.output_dir, a.top_k)
    from . import engine
    if a.command == "train":
        return engine.train(a.data_root, a.split, a.config, a.output_dir, a.resume,
                            fold=a.fold, fast_validate=True if a.fast else None)
    if a.command == "evaluate":
        return engine.evaluate(a.data_root, a.split, a.checkpoint, a.output,
                               device=a.device, tta=a.tta, fold=a.fold, fast_validate=a.fast)
    if a.command == "predict":
        return engine.predict(a.data_root, a.template, a.checkpoint, a.output,
                              device=a.device, tta=a.tta, fast_validate=a.fast)
    if a.command == "tune":
        return engine.tune(a.data_root, a.split, a.plan, a.output_dir)
    if a.command == "ensemble":
        return engine.ensemble(a.members, a.weights, a.output, a.recipe, a.frozen_recipe, method=a.method)
    return engine.reproduce(a.reference, a.current, a.probability_atol, a.auroc_atol)


def main(argv=None):
    try:
        result = dispatch(parser().parse_args(argv))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if isinstance(result, dict) and (result.get("passed") is False or result.get("ok") is False) else 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"defectlab: {exc}", file=sys.stderr)
        return 2
