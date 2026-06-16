#!/usr/bin/env bash
set -euo pipefail

repo="${1:-external/Magic123}"

if [ -d "${repo}/.git" ]; then
  echo "Magic123 repo already exists: ${repo}"
  exit 0
fi

if [ -e "${repo}" ]; then
  echo "Path exists but is not a git repo: ${repo}" >&2
  exit 2
fi

mkdir -p "$(dirname "${repo}")"
git clone https://github.com/guochengqian/Magic123.git "${repo}"
echo "Magic123 repo cloned to ${repo}"
echo "Next: inspect ${repo}/install.sh before running it in the intended conda environment."
