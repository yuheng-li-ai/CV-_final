#!/usr/bin/env bash
set -euo pipefail

repo="${1:-external/Magic123}"

if [ ! -f "${repo}/main.py" ]; then
  echo "Missing Magic123 repo: ${repo}" >&2
  exit 2
fi

export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-12.4}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.6}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/tmp/pip-cache-magic123}"
export MAX_JOBS="${MAX_JOBS:-2}"

if [ -x /usr/bin/gcc-11 ] && [ -x /usr/bin/g++-11 ]; then
  export CC="${CC:-/usr/bin/gcc-11}"
  export CXX="${CXX:-/usr/bin/g++-11}"
  export CUDAHOSTCXX="${CUDAHOSTCXX:-/usr/bin/g++-11}"
fi

# Minimal install for the current cvhw3-scene environment. Do not reinstall
# torch/torchvision/CUDA or downgrade shared packages used by Phase4.
python -m pip install --no-deps \
  easydict \
  torch-ema \
  tensorboardX \
  dearpygui \
  pymeshlab==2022.2.post4

if ! python -c "import cubvh" >/dev/null 2>&1; then
  NVCC_PREPEND_FLAGS="${NVCC_PREPEND_FLAGS:--allow-unsupported-compiler}" \
    python -m pip install --no-deps --no-build-isolation git+https://github.com/ashawkey/cubvh \
    || echo "WARNING: cubvh install failed; continuing because Magic123 only needs cubvh for base_mesh initialization."
fi

(
  cd "${repo}"
  for extension in raymarching shencoder freqencoder gridencoder; do
    if ! python -c "import ${extension}" >/dev/null 2>&1; then
      python -m pip install --no-deps --no-build-isolation "./${extension}"
    fi
  done
)

echo "Magic123 Python dependencies installed without reinstalling torch/torchvision or CUDA."
