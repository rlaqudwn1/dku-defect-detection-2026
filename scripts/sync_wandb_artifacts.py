"""Zero-OAuth Model Cache Sync via W&B Artifacts (Local <-> Headless Blackwell Server).

Replaces rclone/Google Drive OAuth on headless servers: uses the exact same W&B login
(`WANDB_API_KEY` or `~/.netrc`) already used for experiment tracking and reporting!

Syncs (~2 MB total):
  - `runs/queue15/models/M*.npz`
  - `runs/day0_recal/canonical_cids.npz`
  - `runs/day0_recal/recal_summary.json`
  - `runs/folds_unseen.json`

Usage:
  python scripts/sync_wandb_artifacts.py push   # Upload local caches to W&B Artifact
  python scripts/sync_wandb_artifacts.py pull   # Download latest caches from W&B Artifact
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import wandb


ARTIFACT_NAME = "dku-model-cache"


def main() -> None:
    parser = argparse.ArgumentParser(description="Push/Pull model caches via W&B Artifacts (Zero OAuth)")
    parser.add_argument("action", choices=["push", "pull"])
    parser.add_argument("--entity", type=str, default=os.environ.get("WANDB_ENTITY", "rlaqudwn123456-dankook-university"))
    parser.add_argument("--project", type=str, default=os.environ.get("WANDB_PROJECT", "CV"))
    args = parser.parse_args()

    if args.action == "push":
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            job_type="artifact-sync-push",
            name="sync_push_model_cache",
            group="artifact-sync",
        )
        art = wandb.Artifact(ARTIFACT_NAME, type="model-cache", description="Synced M01..MXX .npz caches + canonical_cids + folds")
        count = 0
        models_dir = Path("runs/queue15/models")
        if models_dir.exists():
            for p in sorted(models_dir.glob("M*.npz")):
                art.add_file(str(p), name=f"queue15/models/{p.name}")
                count += 1
        # Also sync lightweight 32x32 2D spatial anomaly maps (~19MB each) for cross-backbone SP_top2 fusion!
        musc_dir = Path("runs/musc_v2")
        if musc_dir.exists():
            for mp in sorted(musc_dir.glob("*/*_maps32.npz")):
                art.add_file(str(mp), name=f"musc_v2/{mp.parent.name}/{mp.name}")
                count += 1
        for rel in [
            "folds_unseen.json",
            "day0_recal/canonical_cids.npz",
            "day0_recal/recal_summary.json",
            "queue15/ensemble_summary.json",
            "queue15/queue_report_tables.md",
        ]:
            fp = Path("runs") / rel
            if fp.exists():
                art.add_file(str(fp), name=rel)
                count += 1
        run.log_artifact(art, aliases=["latest"])
        run.finish()
        print(f"[W&B SYNC PUSH] Uploaded {count} files to {args.entity}/{args.project}/{ARTIFACT_NAME}:latest")

    elif args.action == "pull":
        run = wandb.init(
            entity=args.entity,
            project=args.project,
            job_type="artifact-sync-pull",
            name="sync_pull_model_cache",
            group="artifact-sync",
        )
        art = run.use_artifact(f"{args.entity}/{args.project}/{ARTIFACT_NAME}:latest", type="model-cache")
        dl_dir = Path(art.download())
        copied = 0
        for src in dl_dir.rglob("*"):
            if src.is_file():
                rel = src.relative_to(dl_dir)
                dst = Path("runs") / rel
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src, dst)
                copied += 1
        run.finish()
        print(f"[W&B SYNC PULL] Synced {copied} files into runs/ from {ARTIFACT_NAME}:latest")


if __name__ == "__main__":
    main()
