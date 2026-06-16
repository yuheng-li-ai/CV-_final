#!/usr/bin/env bash
set -euo pipefail
# Supports --config --run_id --device --dry_run --resume
export HF_HOME="${HF_HOME:-/home/yuhengli/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TCNN_CUDA_ARCHITECTURES="${TCNN_CUDA_ARCHITECTURES:-86}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
PYTHONPATH="${PYTHONPATH:-src}" python scripts/run_task1.py --cli_command generate-text3d "$@"
