#!/usr/bin/env bash
set -euo pipefail

COLMAP_CONFIG="configs/scene/object_a_mouse_colmap_final.yaml"
COLMAP_RUN_ID="object_a_mouse_colmap_final"
TDGS_CONFIG="configs/scene/object_a_mouse_2dgs_final.yaml"
TDGS_RUN_ID="object_a_mouse_2dgs_final"
WORKSPACE="runs/scene/object_a_mouse_colmap_final/colmap"
IMAGE_DIR="data/scene/object_a_mouse/frames_dense"
DENSE_DIR="${WORKSPACE}/dense"
SPARSE_DIR="${WORKSPACE}/sparse/0"
DENSE_SPARSE_FLAT="${DENSE_DIR}/sparse"
DENSE_SPARSE_NESTED="${DENSE_DIR}/sparse/0"

mkdir -p "outputs/${COLMAP_RUN_ID}" "outputs/${TDGS_RUN_ID}"

echo "A_CHAIN_STAGE: colmap"
bash scripts/run_object_a_colmap.sh \
  --config "${COLMAP_CONFIG}" \
  --run_id "${COLMAP_RUN_ID}" \
  --device cpu \
  > "outputs/${COLMAP_RUN_ID}/nohup.log" 2>&1

if [[ ! -f "${DENSE_SPARSE_NESTED}/cameras.bin" && ! -f "${DENSE_SPARSE_FLAT}/cameras.bin" ]]; then
  if [[ ! -d "${SPARSE_DIR}" ]]; then
    echo "A_CHAIN_FAILED: missing COLMAP sparse model at ${SPARSE_DIR}" >&2
    exit 1
  fi
  echo "A_CHAIN_STAGE: image_undistorter"
  scripts/colmap_clean_env.sh image_undistorter \
    --image_path "${IMAGE_DIR}" \
    --input_path "${SPARSE_DIR}" \
    --output_path "${DENSE_DIR}" \
    --output_type COLMAP \
    >> "outputs/${COLMAP_RUN_ID}/nohup.log" 2>&1
fi

if [[ ! -f "${DENSE_SPARSE_NESTED}/cameras.bin" && -f "${DENSE_SPARSE_FLAT}/cameras.bin" ]]; then
  mkdir -p "${DENSE_SPARSE_NESTED}"
  cp "${DENSE_SPARSE_FLAT}/cameras.bin" "${DENSE_SPARSE_FLAT}/images.bin" "${DENSE_SPARSE_FLAT}/points3D.bin" "${DENSE_SPARSE_NESTED}/"
fi

if [[ ! -f "${DENSE_SPARSE_NESTED}/cameras.bin" || ! -d "${DENSE_DIR}/images" ]]; then
  echo "A_CHAIN_FAILED: dense COLMAP scene is incomplete under ${DENSE_DIR}" >&2
  exit 1
fi

echo "A_CHAIN_STAGE: 2dgs"
bash scripts/run_object_a_2dgs.sh \
  --config "${TDGS_CONFIG}" \
  --run_id "${TDGS_RUN_ID}" \
  --device cuda:1 \
  > "outputs/${TDGS_RUN_ID}/nohup.log" 2>&1

echo "A_CHAIN_DONE"
