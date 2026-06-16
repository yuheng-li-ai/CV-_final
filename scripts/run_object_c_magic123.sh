#!/usr/bin/env bash
set -euo pipefail
# Supports --config --run_id --device --dry_run --resume
export HF_HOME="${HF_HOME:-/home/yuhengli/.cache/huggingface}"
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TRANSFORMERS_OFFLINE="${TRANSFORMERS_OFFLINE:-1}"
export DIFFUSERS_OFFLINE="${DIFFUSERS_OFFLINE:-1}"
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export CC="/usr/bin/gcc-11"
export CXX="/usr/bin/g++-11"
export CUDAHOSTCXX="/usr/bin/g++-11"
export CMAKE_CUDA_HOST_COMPILER="/usr/bin/g++-11"
export MAX_JOBS="${MAX_JOBS:-2}"
export TORCH_EXTENSIONS_DIR="${TORCH_EXTENSIONS_DIR:-/home/yuhengli/study/CV/assignment3/.cache/torch_extensions_magic123}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/home/yuhengli/study/CV/assignment3/.cache/matplotlib_magic123}"
mkdir -p "${TORCH_EXTENSIONS_DIR}" "${MPLCONFIGDIR}"
PYTHONPATH="${PYTHONPATH:-src}" python scripts/patch_magic123_compat.py external/Magic123
PYTHONPATH="${PYTHONPATH:-src}" python scripts/run_task1.py --cli_command generate-image3d "$@"
