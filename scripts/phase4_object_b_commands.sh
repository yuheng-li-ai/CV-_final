#!/usr/bin/env bash
set -euo pipefail

device="auto"
which="all"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --device)
      device="$2"
      shift 2
      ;;
    --which)
      which="$2"
      shift 2
      ;;
    -h|--help)
      cat <<'USAGE'
Usage: bash scripts/phase4_object_b_commands.sh [--device auto|cuda:N] [--which all|simple|detailed|style]

Print formal Phase4 Object B threestudio/SDS training commands.
The generated nohup commands use --device auto by default so RunManager selects the freest GPU at launch time.
USAGE
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

emit_one() {
  local name="$1"
  local config="$2"
  local run_id="$3"
  local output_dir="$4"

  cat <<EOF

### object_b_sds_${name}
mkdir -p outputs/${run_id}
nohup bash scripts/run_object_b_sds.sh --config ${config} --run_id ${run_id} --device ${device} > outputs/${run_id}/nohup.log 2>&1 &

# progress bar
python scripts/watch_threestudio_progress.py --config ${config} --run_id ${run_id} --interval 5

# raw logs
tail -f outputs/${run_id}/log.txt
tail -f outputs/${run_id}/nohup.log

# resume if interrupted
nohup bash scripts/run_object_b_sds.sh --config ${config} --run_id ${run_id} --device ${device} --resume > outputs/${run_id}/nohup.log 2>&1 &

# expected outputs
find ${output_dir}/main -maxdepth 3 -type f | sort | tail -80
EOF
}

case "$which" in
  all)
    emit_one simple configs/scene/text3d_simple.yaml object_b_sds_simple runs/scene/text3d_simple
    emit_one detailed configs/scene/text3d_detailed.yaml object_b_sds_detailed runs/scene/text3d_detailed
    emit_one style configs/scene/text3d_style_constrained.yaml object_b_sds_style_constrained runs/scene/text3d_style_constrained
    ;;
  simple)
    emit_one simple configs/scene/text3d_simple.yaml object_b_sds_simple runs/scene/text3d_simple
    ;;
  detailed)
    emit_one detailed configs/scene/text3d_detailed.yaml object_b_sds_detailed runs/scene/text3d_detailed
    ;;
  style|style_constrained)
    emit_one style configs/scene/text3d_style_constrained.yaml object_b_sds_style_constrained runs/scene/text3d_style_constrained
    ;;
  *)
    echo "Unknown --which value: $which" >&2
    exit 2
    ;;
esac
