"""Zero-dependency AI context snapshot generator for fast, drift-free session handoffs."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path


def _git_info(root: Path) -> dict:
    commit, dirty = None, None
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
        dirty = bool(
            subprocess.check_output(
                ["git", "status", "--porcelain", "--untracked-files=no"], cwd=root, stderr=subprocess.DEVNULL
            )
        )
    except (OSError, subprocess.CalledProcessError):
        pass
    return {"commit": commit, "dirty": dirty}


def _scan_runs(root: Path, runs_dir: Path, top_n: int = 5) -> list[dict]:
    if not runs_dir.is_dir():
        return []
    found = []
    for path in sorted(runs_dir.rglob("run.json")):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if data.get("status") != "completed":
            continue
        auroc = data.get("best_auroc")
        if auroc is None:
            auroc = (data.get("metrics") or {}).get("auroc")
        if auroc is None:
            continue
        try:
            rel = path.parent.relative_to(root).as_posix()
        except ValueError:
            rel = path.parent.name
        found.append({
            "run_dir": rel,
            "run_id": data.get("run_id"),
            "fold": data.get("fold"),
            "best_auroc": round(float(auroc), 6),
            "epochs": data.get("completed_epochs"),
        })
    found.sort(key=lambda r: (-r["best_auroc"], r["run_dir"]))
    return found[:top_n]


def _scan_errors(root: Path, runs_dir: Path, limit: int = 3) -> list[dict]:
    if not runs_dir.is_dir():
        return []
    found = []
    for path in sorted(runs_dir.rglob("errors.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        try:
            rel = path.parent.relative_to(root).as_posix()
        except ValueError:
            rel = path.parent.name
        found.append({
            "error_dir": rel,
            "auroc": round(float(data["auroc"]), 6) if data.get("auroc") is not None else None,
            "top_k": data.get("top_k"),
            "total_samples": data.get("total_samples"),
        })
        if len(found) >= limit:
            break
    return found


def _experiment_rows(exp_path: Path, tail: int = 10) -> dict:
    if not exp_path.is_file():
        return {"exists": False, "total_rows": 0, "recent_rows": []}
    lines = [
        line.strip()
        for line in exp_path.read_text(encoding="utf-8").splitlines()
        if line.strip().startswith("|")
    ]
    # Skip header + separator
    data_rows = [row for row in lines[2:] if not set(row.replace("|", "").replace("-", "").strip()) == set()]
    return {"exists": True, "total_rows": len(data_rows), "recent_rows": data_rows[-tail:]}


def build_brief(repo_root=".", runs_dir="runs", top_n=5) -> dict:
    """Assemble a compact, token-efficient context snapshot for an AI session."""
    root = Path(repo_root).resolve()
    runs = Path(runs_dir)
    if not runs.is_absolute():
        runs = (root / runs).resolve()

    state_path = root / "STATE.md"
    state_lines = state_path.read_text(encoding="utf-8").splitlines() if state_path.is_file() else []

    return {
        "schema_version": 1,
        "git": _git_info(root),
        "boot_files": [name for name in ("AGENTS.md", "STATE.md", "EXPERIMENTS.md", "LOOP_PLAN.md")
                       if (root / name).is_file()],
        "state_line_count": len(state_lines),
        "state_bloat_warning": len(state_lines) > 60,
        "experiments": _experiment_rows(root / "EXPERIMENTS.md"),
        "top_completed_runs": _scan_runs(root, runs, top_n=top_n),
        "recent_error_reports": _scan_errors(root, runs),
    }
