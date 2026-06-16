#!/usr/bin/env bash
set -euo pipefail
# Supports --config --run_id --device --dry_run --resume
PYTHONPATH="${PYTHONPATH:-src}" python scripts/smoke.py --stage data "$@"
