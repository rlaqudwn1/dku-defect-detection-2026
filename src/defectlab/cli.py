"""Explicit local commands; heavy dependencies are imported only when needed."""
import argparse
import json
from pathlib import Path
import sys


def parser():
    root = argparse.ArgumentParser(prog="defectlab")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("verify")
    p.add_argument("data_root")
    p.add_argument("--manifest")
    p = commands.add_parser("profile")
    p.add_argument("data_root")
    p.add_argument("output_dir")
    p = commands.add_parser("audit-duplicates", help="Audit file, pixel and dHash duplicates")
    p.add_argument("data_root")
    p.add_argument("output_dir")
    p.add_argument("--split", help="Optional split JSON with train_ids and val_ids")
    p.add_argument("--max-hamming", type=int, default=6, help="Maximum dHash distance (default: 6)")
    p.add_argument("--top-k", type=int, default=100, help="Pairs per candidate list (default: 100)")
    p = commands.add_parser("sample-eda", help="Draw stratified samples and build blind review sheets")
    p.add_argument("profile")
    p.add_argument("data_root")
    p.add_argument("output_dir")
    p.add_argument("--config", help="Sampling JSON (defaults to the eda-first-look-v1 settings)")
    p = commands.add_parser("compare-transforms", help="Compare baseline resize/pad with original image detail")
    p.add_argument("data_root")
    p.add_argument("output_dir")
    p.add_argument("--ids", nargs="+", required=True, help="Image IDs, a JSON array file, or a text list file")
    p.add_argument("--sizes", nargs="+", type=int, default=[256, 384, 512], help="Sizes (default: 256 384 512)")
    p = commands.add_parser("split")
    p.add_argument("profile")
    p.add_argument("output")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.2)
    p.add_argument("--groups")
    for name in ["train", "evaluate", "predict", "tune"]:
        p = commands.add_parser(name)
        p.add_argument("data_root")
        if name == "train":
            p.add_argument("split")
            p.add_argument("config")
            p.add_argument("output_dir")
            p.add_argument("--resume")
        elif name == "evaluate":
            p.add_argument("split")
            p.add_argument("checkpoint")
            p.add_argument("output")
        elif name == "predict":
            p.add_argument("template")
            p.add_argument("checkpoint")
            p.add_argument("output")
        else:
            p.add_argument("split")
            p.add_argument("plan")
            p.add_argument("output_dir")
    p = commands.add_parser("ensemble")
    p.add_argument("output")
    p.add_argument("members", nargs="+")
    p.add_argument("--weights", type=float, nargs="+", required=True)
    p.add_argument("--recipe")
    p.add_argument("--frozen-recipe")
    p = commands.add_parser("reproduce")
    p.add_argument("reference")
    p.add_argument("current")
    p.add_argument("--probability-atol", type=float, required=True)
    p.add_argument("--auroc-atol", type=float, required=True)
    p = commands.add_parser("record")
    p.add_argument("ledger")
    p.add_argument("record")
    p = commands.add_parser("status")
    p.add_argument("ledger")
    p = commands.add_parser("workflow")
    p.add_argument("plan")
    p.add_argument("ledger")
    p = commands.add_parser("results")
    p.add_argument("output_dir")
    p.add_argument("runs", nargs="+")
    p = commands.add_parser("context")
    p.add_argument("ledger")
    p.add_argument("brief")
    p.add_argument("output")
    for name in ["freshness", "freeze"]:
        p = commands.add_parser(name)
        p.add_argument("context")
        if name == "freeze":
            p.add_argument("output")
    p = commands.add_parser("select")
    p.add_argument("validation")
    p.add_argument("test")
    p.add_argument("reproduction")
    p.add_argument("output")
    p.add_argument("--reason", required=True)
    return root


def dispatch(a):
    if a.command == "workflow":
        from .workflow import resume
        return resume(a.plan, a.ledger)
    if a.command == "select":
        from .selection import select
        return select(a.validation, a.test, a.reproduction, a.output, a.reason)
    if a.command in {"verify", "profile", "split"}:
        from . import data
        if a.command == "verify":
            return data.verify(a.data_root, a.manifest)
        if a.command == "profile":
            return data.profile(a.data_root, a.output_dir)
        return data.make_split(a.profile, a.output, a.seed, a.val_fraction, a.groups)
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
    if a.command == "sample-eda":
        from .eda import sample_eda
        manifest = sample_eda(a.profile, a.data_root, a.output_dir, a.config)
        output_dir = Path(a.output_dir).resolve()
        return {
            "schema_version": manifest["schema_version"],
            "protocol_id": manifest["protocol_id"],
            "manifest_path": str(output_dir / "sample_manifest.json"),
            "blank_observations_path": str(output_dir / "blank_observations.json"),
            "blind_key_path": str(output_dir / "blind_key.json"),
            "config_sha256": manifest["config_sha256"],
            "profile_sha256": manifest["profile_sha256"],
            "dataset_hash": manifest["dataset_hash"],
            "seed": manifest["seed"],
            "stratum_count": len(manifest["strata"]),
            "sample_count": len(manifest["samples"]),
            "phase_counts": {
                phase: sum(phase in row["selections"] for row in manifest["samples"])
                for phase in ("calibration", "survey", "discovery")
            },
            "blind_sheet_pages": {name: len(ids) for name, ids in manifest["blind_sheet_pages"].items()},
        }
    if a.command == "compare-transforms":
        from .eda import compare_transforms
        return compare_transforms(a.data_root, a.output_dir, a.ids, a.sizes)
    if a.command in {"train", "evaluate", "predict", "tune", "ensemble", "reproduce"}:
        from . import engine
        if a.command == "train":
            return engine.train(a.data_root, a.split, a.config, a.output_dir, a.resume)
        if a.command == "evaluate":
            return engine.evaluate(a.data_root, a.split, a.checkpoint, a.output)
        if a.command == "predict":
            return engine.predict(a.data_root, a.template, a.checkpoint, a.output)
        if a.command == "tune":
            return engine.tune(a.data_root, a.split, a.plan, a.output_dir)
        if a.command == "ensemble":
            return engine.ensemble(a.members, a.weights, a.output, a.recipe, a.frozen_recipe)
        return engine.reproduce(a.reference, a.current, a.probability_atol, a.auroc_atol)
    from . import governance
    if a.command == "record":
        return governance.record(a.ledger, a.record)
    if a.command == "status":
        return governance.status(a.ledger)
    if a.command == "results":
        return governance.results(a.runs, a.output_dir)
    if a.command == "context":
        return governance.context(a.ledger, a.brief, a.output)
    if a.command == "freshness":
        return governance.freshness(a.context)
    return governance.freeze(a.context, a.output)


def main(argv=None):
    try:
        result = dispatch(parser().parse_args(argv))
        print(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False))
        return 1 if isinstance(result, dict) and (result.get("passed") is False or result.get("ok") is False) else 0
    except (ValueError, OSError, KeyError, ImportError) as exc:
        print(f"defectlab: {exc}", file=sys.stderr)
        return 2
