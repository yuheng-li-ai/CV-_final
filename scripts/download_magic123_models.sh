#!/usr/bin/env bash
set -euo pipefail

repo="${1:-external/Magic123}"
repo="$(cd "$(dirname "$repo")" && pwd)/$(basename "$repo")"

mkdir -p "${repo}/pretrained/zero123" "${repo}/pretrained/midas"

file_size() {
  if [ -f "$1" ]; then
    stat -c%s "$1"
  else
    echo 0
  fi
}

download_if_incomplete() {
  local url="$1"
  local dest="$2"
  local min_bytes="$3"
  local current_size
  current_size="$(file_size "${dest}")"
  if [ "${current_size}" -lt "${min_bytes}" ]; then
    echo "Downloading $(basename "${dest}") (${current_size}/${min_bytes}+ bytes present)"
    wget -c --progress=bar:force:noscroll "${url}" -O "${dest}"
  fi
  current_size="$(file_size "${dest}")"
  if [ "${current_size}" -lt "${min_bytes}" ]; then
    echo "ERROR: incomplete download for ${dest}: ${current_size} bytes" >&2
    exit 1
  fi
}

download_if_incomplete \
  https://huggingface.co/cvlab/zero123-weights/resolve/main/105000.ckpt \
  "${repo}/pretrained/zero123/105000.ckpt" \
  15000000000

download_if_incomplete \
  https://github.com/isl-org/MiDaS/releases/download/v3_1/dpt_beit_large_512.pt \
  "${repo}/pretrained/midas/dpt_beit_large_512.pt" \
  1000000000

echo "Magic123 model files are ready under ${repo}/pretrained"
