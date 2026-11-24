"""One-Command Auto Sync, Ensemble Update & W&B Report Hub.

Called automatically at the end of any Track A model (`run_musc_v2_blackwell.py --auto-sync`)
or manually after Track B updates:
  1. Pushes new model caches (`runs/queue15/models/M*.npz`) to Google Drive via `rclone` (if configured)
  2. Re-runs `scripts/run_15model_ensemble.py` to update `ensemble_summary.json` & `queue_report_tables.md`
  3. Uploads updated report figures & tables to W&B (graceful offline fallback if not logged in)
  4. Prints a 1-line `EXPERIMENTS.md` snippet ready to append.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path


def run_step(cmd: list[str], label: str, required: bool = False) -> bool:
    print(f"\n[AUTO-HUB] >>> {label}: {' '.join(cmd)}", flush=True)
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        print(f"[AUTO-HUB] WARN: {label} exited with code {res.returncode}", flush=True)
        if required:
            sys.exit(res.returncode)
        return False
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto sync model caches, update ensemble, and push W&B report")
    parser.add_argument("--run-name", type=str, default="auto_two_track_update")
    parser.add_argument("--public-lb", type=float, default=0.884610)
    parser.add_argument("--skip-sync", action="store_true")
    parser.add_argument("--skip-wandb", action="store_true")
    args = parser.parse_args()

    py = sys.executable

    # 1. Google Drive Sync (graceful skip if rclone/gdrive not configured on this host)
    if not args.skip_sync and shutil.which("rclone"):
        remotes = subprocess.run(["rclone", "listremotes"], capture_output=True, text=True, check=False).stdout
        if "gdrive:" in remotes or os.environ.get("RCLONE_REMOTE"):
            run_step(["bash", "scripts/sync_artifacts.sh", "push"], "Google Drive rclone push", required=False)
        else:
            print("[AUTO-HUB] rclone installed but no gdrive: remote found; skipping cloud push.", flush=True)
    elif not args.skip_sync:
        print("[AUTO-HUB] rclone not found; skipping cloud push.", flush=True)

    # 2. Recompute Nested 5-Fold Ensemble & Markdown Report Tables
    run_step([py, "scripts/run_15model_ensemble.py"], "Nested 5-Fold Ensemble & Report Tables", required=True)

    # 3. Upload W&B Report Dashboard (auto offline fallback if no credentials)
    if not args.skip_wandb:
        env = os.environ.copy()
        if not env.get("WANDB_API_KEY") and not (Path.home() / ".netrc").exists():
            env["WANDB_MODE"] = "offline"
            print("[AUTO-HUB] No W&B credentials detected -> using WANDB_MODE=offline (sync later via `wandb sync`).", flush=True)
        cmd = [py, "scripts/log_wandb_report.py", "--run-name", args.run_name, "--public-lb", str(args.public_lb)]
        print(f"\n[AUTO-HUB] >>> W&B Report Upload: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, env=env, check=False)

    # 4. Print ready-to-use EXPERIMENTS.md summary line
    summary_p = Path("runs/queue15/ensemble_summary.json")
    if summary_p.exists():
        s = json.loads(summary_p.read_text(encoding="utf-8"))
        best_m = s["best_single_model"]
        print("\n=== [AUTO-HUB SUMMARY] ===")
        print(f"  Completed Models : {s['completed_models']}")
        print(f"  Best Single      : {best_m['model_id']} (OOF={best_m['oof_auroc']:.4f})")
        print(f"  Best Ensemble    : {s['best_ensemble_method']} (Nested 5-Fold OOF={s['best_ensemble_nested_5fold_oof']:.4f})")
        print(f"  Verified CSV     : {s['submission_path']}")


if __name__ == "__main__":
    main()
