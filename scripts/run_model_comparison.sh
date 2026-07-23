#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-/home/zhong/anaconda3/envs/torch/bin/python}"
readonly MAESTRO_ROOT="${MAESTRO_ROOT:-${PROJECT_DIR}/data/maestro-v3.0.0}"
readonly CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/cache/maestro-v3.0.0}"
readonly RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/model_comparison}"
readonly CURRENT_DIR="${RUN_ROOT}/current_fixed"
readonly V2_DIR="${RUN_ROOT}/ddsp_v2"
readonly EXPORT_DIR="${PROJECT_DIR}/exports"
readonly MIDI_DIR="${PROJECT_DIR}/midi"
readonly MIDI_TEST_DIR="${EXPORT_DIR}/midi_tests"
readonly EPOCHS="${EPOCHS:-20}"
readonly STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-2000}"
readonly VALIDATION_BATCHES="${VALIDATION_BATCHES:-16}"

cd "${PROJECT_DIR}"
mkdir -p "${RUN_ROOT}" "${EXPORT_DIR}"

"${PYTHON_BIN}" train.py \
  --maestro-root "${MAESTRO_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --prepare-only

current_start=()
if [[ -f "${CURRENT_DIR}/checkpoints/last.pt" ]]; then
  current_start=(--resume "${CURRENT_DIR}/checkpoints/last.pt")
fi
"${PYTHON_BIN}" train.py \
  --maestro-root "${MAESTRO_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --experiment-dir "${CURRENT_DIR}" \
  --model-variant current \
  --phase 1 \
  --epochs "${EPOCHS}" \
  --steps-per-epoch "${STEPS_PER_EPOCH}" \
  --validation-batches "${VALIDATION_BATCHES}" \
  --device "${DEVICE:-cuda}" \
  --amp \
  "${current_start[@]}"

v2_start=()
if [[ -f "${V2_DIR}/checkpoints/last.pt" ]]; then
  v2_start=(--resume "${V2_DIR}/checkpoints/last.pt")
fi
"${PYTHON_BIN}" train.py \
  --maestro-root "${MAESTRO_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --experiment-dir "${V2_DIR}" \
  --model-variant v2 \
  --phase 1 \
  --epochs "${EPOCHS}" \
  --steps-per-epoch "${STEPS_PER_EPOCH}" \
  --validation-batches "${VALIDATION_BATCHES}" \
  --device "${DEVICE:-cuda}" \
  --amp \
  "${v2_start[@]}"

"${PYTHON_BIN}" scripts/export_onnx.py \
  --checkpoint "${CURRENT_DIR}/checkpoints/best.pt" \
  --output "${EXPORT_DIR}/piano_current_fixed.onnx"

"${PYTHON_BIN}" scripts/export_onnx.py \
  --checkpoint "${V2_DIR}/checkpoints/best.pt" \
  --model-variant v2 \
  --output "${EXPORT_DIR}/piano_ddsp_v2.onnx"

"${PYTHON_BIN}" scripts/render_onnx.py \
  --model "${EXPORT_DIR}/piano_current_fixed.onnx" \
  --midi-dir "${MIDI_DIR}" \
  --output-dir "${MIDI_TEST_DIR}/current_fixed"

"${PYTHON_BIN}" scripts/render_onnx.py \
  --model "${EXPORT_DIR}/piano_ddsp_v2.onnx" \
  --midi-dir "${MIDI_DIR}" \
  --output-dir "${MIDI_TEST_DIR}/v2"

printf 'Comparison exports written to:\n  %s\n  %s\n' \
  "${EXPORT_DIR}/piano_current_fixed.onnx" \
  "${EXPORT_DIR}/piano_ddsp_v2.onnx"
