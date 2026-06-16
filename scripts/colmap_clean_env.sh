#!/usr/bin/env bash
set -euo pipefail

# System COLMAP conflicts with this conda environment's shared libraries.
# Drop LD_LIBRARY_PATH so /usr/bin/colmap resolves system Ceres/glog.
env -u LD_LIBRARY_PATH /usr/bin/colmap "$@"
