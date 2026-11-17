"""Explicit local commands; heavy dependencies are imported only when needed."""
import argparse
import json
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
