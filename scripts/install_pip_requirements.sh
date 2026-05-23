#!/usr/bin/env bash
set -u

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/install_pip_requirements.sh <requirements.txt> [index-url]" >&2
  exit 2
fi

REQ_FILE="$1"
INDEX_URL="${2:-https://pypi.tuna.tsinghua.edu.cn/simple}"
MAX_ATTEMPTS="${PIP_MAX_ATTEMPTS:-5}"

if [ ! -f "$REQ_FILE" ]; then
  echo "Requirements file not found: $REQ_FILE" >&2
  exit 2
fi

python -m pip install --upgrade pip

FAILED=0
while IFS= read -r raw_line || [ -n "$raw_line" ]; do
  package="$(printf '%s' "$raw_line" | sed 's/[[:space:]]*#.*$//' | xargs)"

  if [ -z "$package" ]; then
    continue
  fi

  attempt=1
  installed=0
  while [ "$attempt" -le "$MAX_ATTEMPTS" ]; do
    echo "Installing ${package} (attempt ${attempt}/${MAX_ATTEMPTS})"
    if python -m pip install \
      -i "$INDEX_URL" \
      --timeout 180 \
      --retries 10 \
      --no-cache-dir \
      "$package"; then
      installed=1
      break
    fi
    attempt=$((attempt + 1))
    sleep 5
  done

  if [ "$installed" -ne 1 ]; then
    echo "FAILED: ${package}" >&2
    FAILED=1
  fi
done < "$REQ_FILE"

exit "$FAILED"
