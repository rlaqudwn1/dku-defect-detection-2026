#!/usr/bin/env bash
# Automated artifact sync between Local, Blackwell Server, and Colab/Kaggle via rclone.
# Usage:
#   bash scripts/sync_artifacts.sh push    # Upload model caches (.npz), reports, folds
#   bash scripts/sync_artifacts.sh pull    # Download latest model caches (.npz), reports, folds
#   bash scripts/sync_artifacts.sh status  # List synced model caches on remote
#
# Override remote with env var if needed (never hardcode private Drive IDs in Git):
#   RCLONE_REMOTE="gdrive:dku-defect-2026" bash scripts/sync_artifacts.sh push

set -euo pipefail

REMOTE="${RCLONE_REMOTE:-gdrive:dku-defect-2026}"
ACTION="${1:-status}"

if ! command -v rclone >/dev/null 2>&1; then
  echo "[ERROR] rclone is not installed. Install with: curl -s https://rclone.org/install.sh | bash (or put binary in ~/.local/bin)"
  exit 1
fi

case "${ACTION}" in
  push)
    echo "[SYNC PUSH] Local -> ${REMOTE}"
    mkdir -p runs/queue15/models runs/submissions
    rclone copy runs/queue15/models "${REMOTE}/queue15/models" --include "M*.npz" --transfers 8 --progress
    for f in runs/folds_unseen.json runs/day0_recal/canonical_cids.npz runs/day0_recal/recal_summary.json runs/queue15/ensemble_summary.json runs/queue15/queue_report_tables.md; do
      if [[ -f "${f}" ]]; then
        rclone copyto "${f}" "${REMOTE}/${f#runs/}"
      fi
    done
    echo "[SYNC PUSH] Done. Remote models:"
    rclone ls "${REMOTE}/queue15/models" | sort -k2
    ;;
  pull)
    echo "[SYNC PULL] ${REMOTE} -> Local"
    mkdir -p runs/queue15/models runs/submissions runs/day0_recal
    rclone copy "${REMOTE}/queue15/models" runs/queue15/models --include "M*.npz" --transfers 8 --progress
    for f in folds_unseen.json day0_recal/canonical_cids.npz day0_recal/recal_summary.json queue15/ensemble_summary.json queue15/queue_report_tables.md; do
      rclone copyto "${REMOTE}/${f}" "runs/${f}" 2>/dev/null || true
    done
    echo "[SYNC PULL] Done. Local models:"
    ls -lh runs/queue15/models/M*.npz 2>/dev/null || echo "(none)"
    ;;
  status)
    echo "[SYNC STATUS] Remote: ${REMOTE}"
    rclone ls "${REMOTE}/queue15/models" 2>/dev/null | sort -k2 || echo "(remote folder not initialized yet)"
    ;;
  *)
    echo "Usage: $0 {push|pull|status}"
    exit 1
    ;;
esac
