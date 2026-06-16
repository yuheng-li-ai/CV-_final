#!/usr/bin/env bash
set -euo pipefail

wait_checkpoint=""
wait_pattern=""
config=""
run_id=""
device="auto"
interval=60
min_free_mb=0
post_launch_min_step=20
post_launch_timeout=3600
post_launch_interval=30
resume=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --wait_checkpoint)
      wait_checkpoint="$2"
      shift 2
      ;;
    --wait_pattern)
      wait_pattern="$2"
      shift 2
      ;;
    --config)
      config="$2"
      shift 2
      ;;
    --run_id)
      run_id="$2"
      shift 2
      ;;
    --device)
      device="$2"
      shift 2
      ;;
    --interval)
      interval="$2"
      shift 2
      ;;
    --min_free_mb)
      min_free_mb="$2"
      shift 2
      ;;
    --post_launch_min_step)
      post_launch_min_step="$2"
      shift 2
      ;;
    --post_launch_timeout)
      post_launch_timeout="$2"
      shift 2
      ;;
    --post_launch_interval)
      post_launch_interval="$2"
      shift 2
      ;;
    --resume)
      resume="--resume"
      shift
      ;;
    -h|--help)
      cat <<'EOF'
Wait for one Magic123 run to finish, then launch another one with nohup.

Required:
  --wait_checkpoint PATH   Checkpoint that must exist before launching.
  --wait_pattern TEXT      Process-table pattern that should disappear.
  --config PATH            Next Magic123 config.
  --run_id ID              Next run_id.
  --device DEVICE          Next device, e.g. cuda:1.

Optional:
  --interval SECONDS       Poll interval. Default: 60.
  --min_free_mb MB         Also wait until target GPU has at least this free memory.
  --post_launch_min_step N Wait until the launched run reaches at least this step. Default: 20.
  --post_launch_timeout S  Maximum seconds to wait for post-launch stability. Default: 3600.
  --post_launch_interval S Post-launch poll interval. Default: 30.
  --resume                 Pass --resume to the next run.
EOF
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

if [[ -z "${wait_checkpoint}" || -z "${wait_pattern}" || -z "${config}" || -z "${run_id}" ]]; then
  echo "Missing required arguments. Use --help." >&2
  exit 2
fi

if [[ ! "${interval}" =~ ^[0-9]+$ ]] || [[ "${interval}" -lt 5 ]]; then
  echo "--interval must be an integer >= 5 seconds." >&2
  exit 2
fi

if [[ ! "${min_free_mb}" =~ ^[0-9]+$ ]]; then
  echo "--min_free_mb must be a non-negative integer." >&2
  exit 2
fi

if [[ ! "${post_launch_min_step}" =~ ^[0-9]+$ ]]; then
  echo "--post_launch_min_step must be a non-negative integer." >&2
  exit 2
fi

if [[ ! "${post_launch_timeout}" =~ ^[0-9]+$ ]] || [[ "${post_launch_timeout}" -lt 60 ]]; then
  echo "--post_launch_timeout must be an integer >= 60 seconds." >&2
  exit 2
fi

if [[ ! "${post_launch_interval}" =~ ^[0-9]+$ ]] || [[ "${post_launch_interval}" -lt 5 ]]; then
  echo "--post_launch_interval must be an integer >= 5 seconds." >&2
  exit 2
fi

target_gpu=""
if [[ "${device}" =~ ^cuda:([0-9]+)$ ]]; then
  target_gpu="${BASH_REMATCH[1]}"
fi

echo "WAIT_CHECKPOINT: ${wait_checkpoint}"
echo "WAIT_PATTERN: ${wait_pattern}"
echo "NEXT_CONFIG: ${config}"
echo "NEXT_RUN_ID: ${run_id}"
echo "NEXT_DEVICE: ${device}"
echo "INTERVAL: ${interval}s"
echo "MIN_FREE_MB: ${min_free_mb}"
echo "POST_LAUNCH_MIN_STEP: ${post_launch_min_step}"
echo "POST_LAUNCH_TIMEOUT: ${post_launch_timeout}s"
echo "POST_LAUNCH_INTERVAL: ${post_launch_interval}s"

while true; do
  timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"
  checkpoint_ready=0
  process_running=0
  gpu_ready=1
  gpu_free_mb="unknown"

  if [[ -f "${wait_checkpoint}" ]]; then
    checkpoint_ready=1
  fi

  while read -r pid ppid args; do
    if [[ "${args}" == *"wait_then_run_magic123.sh"* ]]; then
      continue
    fi
    if [[ "${pid}" != "$$" && "${ppid}" != "$$" && "${args}" == *"${wait_pattern}"* ]]; then
      process_running=1
      break
    fi
  done < <(ps -eo pid=,ppid=,args=)

  if [[ "${min_free_mb}" -gt 0 && -n "${target_gpu}" ]]; then
    gpu_free_mb="$(nvidia-smi --id="${target_gpu}" --query-gpu=memory.free --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
    if [[ ! "${gpu_free_mb}" =~ ^[0-9]+$ || "${gpu_free_mb}" -lt "${min_free_mb}" ]]; then
      gpu_ready=0
    fi
  fi

  echo "[${timestamp}] checkpoint=${checkpoint_ready} process_running=${process_running} gpu_ready=${gpu_ready} gpu_free_mb=${gpu_free_mb}"

  if [[ "${checkpoint_ready}" -eq 1 && "${process_running}" -eq 0 && "${gpu_ready}" -eq 1 ]]; then
    break
  fi

  sleep "${interval}"
done

mkdir -p "outputs/${run_id}"
echo "LAUNCH_TIME: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo "LAUNCH_COMMAND: nohup bash scripts/run_object_c_magic123.sh --config ${config} --run_id ${run_id} --device ${device} ${resume} > outputs/${run_id}/nohup.log 2>&1 &"
nohup bash scripts/run_object_c_magic123.sh \
  --config "${config}" \
  --run_id "${run_id}" \
  --device "${device}" \
  ${resume} \
  > "outputs/${run_id}/nohup.log" 2>&1 &
launched_pid="$!"
echo "LAUNCHED_PID: ${launched_pid}"

if [[ "${post_launch_min_step}" -eq 0 ]]; then
  exit 0
fi

run_log="outputs/${run_id}/log.txt"
nohup_log="outputs/${run_id}/nohup.log"
deadline=$((SECONDS + post_launch_timeout))
while [[ "${SECONDS}" -lt "${deadline}" ]]; do
  timestamp="$(date '+%Y-%m-%d %H:%M:%S %Z')"

  combined_tail=""
  if [[ -f "${run_log}" ]]; then
    combined_tail="${combined_tail}"$'\n'"$(tail -n 200 "${run_log}")"
  fi
  if [[ -f "${nohup_log}" ]]; then
    combined_tail="${combined_tail}"$'\n'"$(tail -n 120 "${nohup_log}")"
  fi

  if echo "${combined_tail}" | grep -E "Traceback|RuntimeError|CUDA out of memory|FileExistsError|No such file" >/dev/null 2>&1; then
    echo "[${timestamp}] POST_LAUNCH_FAILED: error detected in ${run_log} or ${nohup_log}"
    exit 1
  fi

  step_line="$(echo "${combined_tail}" | grep -Eo "Train \[Step\][[:space:]]+[0-9]+/[0-9]+" | tail -n 1 || true)"
  current_step=0
  if [[ -n "${step_line}" ]]; then
    current_step="${step_line##*] }"
    current_step="${current_step%%/*}"
  fi

  process_alive=0
  if kill -0 "${launched_pid}" >/dev/null 2>&1; then
    process_alive=1
  fi

  echo "[${timestamp}] post_launch_step=${current_step} target_step=${post_launch_min_step} process_alive=${process_alive}"

  if [[ "${current_step}" =~ ^[0-9]+$ && "${current_step}" -ge "${post_launch_min_step}" ]]; then
    echo "POST_LAUNCH_STABLE: reached step ${current_step} for ${run_id}"
    exit 0
  fi

  if [[ "${process_alive}" -eq 0 && "${current_step}" -lt "${post_launch_min_step}" ]]; then
    echo "[${timestamp}] POST_LAUNCH_FAILED: launched process exited before reaching step ${post_launch_min_step}"
    exit 1
  fi

  sleep "${post_launch_interval}"
done

echo "POST_LAUNCH_TIMEOUT: did not reach step ${post_launch_min_step} within ${post_launch_timeout}s"
exit 1
