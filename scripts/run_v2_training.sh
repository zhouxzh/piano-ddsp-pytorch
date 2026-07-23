#!/usr/bin/env bash
set -euo pipefail

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-/home/zhong/anaconda3/envs/torch/bin/python}"
readonly MAESTRO_ROOT="${MAESTRO_ROOT:-${PROJECT_DIR}/data/maestro-v3.0.0}"
readonly CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/cache/maestro-v3.0.0}"
readonly RUN_DIR="${RUN_DIR:-${PROJECT_DIR}/runs/model_comparison/ddsp_v2}"
readonly EXPORT_DIR="${PROJECT_DIR}/exports"
readonly MIDI_DIR="${PROJECT_DIR}/midi"
readonly MIDI_TEST_DIR="${EXPORT_DIR}/midi_tests"
readonly STATE_DIR="${PROJECT_DIR}/.training-state"
readonly LOG_FILE="${STATE_DIR}/ddsp-v2.log"
readonly LOCK_FILE="${STATE_DIR}/ddsp-v2.lock"
readonly EPOCHS="${EPOCHS:-20}"
readonly STEPS_PER_EPOCH="${STEPS_PER_EPOCH:-2000}"
readonly VALIDATION_BATCHES="${VALIDATION_BATCHES:-16}"

mkdir -p "${RUN_DIR}" "${EXPORT_DIR}" "${STATE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] another DDSP-Piano v2 pipeline is already running\n' \
    "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
  exit 0
fi

exec >>"${LOG_FILE}" 2>&1
cd "${PROJECT_DIR}"
printf '[%s] starting DDSP-Piano v2 full training\n' "$(date --iso-8601=seconds)"

resume_args=()
if [[ -f "${RUN_DIR}/checkpoints/last.pt" ]]; then
  resume_args=(--resume "${RUN_DIR}/checkpoints/last.pt")
fi

"${PYTHON_BIN}" train.py \
  --maestro-root "${MAESTRO_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --experiment-dir "${RUN_DIR}" \
  --model-variant v2 \
  --phase 1 \
  --batch-size 1 \
  --epochs "${EPOCHS}" \
  --steps-per-epoch "${STEPS_PER_EPOCH}" \
  --validation-batches "${VALIDATION_BATCHES}" \
  --num-workers 4 \
  --device cuda \
  --amp \
  "${resume_args[@]}"

printf '[%s] exporting and validating v2 ONNX\n' "$(date --iso-8601=seconds)"
"${PYTHON_BIN}" scripts/export_onnx.py \
  --checkpoint "${RUN_DIR}/checkpoints/best.pt" \
  --model-variant v2 \
  --output "${EXPORT_DIR}/piano_ddsp_v2.onnx" \
  --verify-steps 4

printf '[%s] rendering current-fixed MIDI reference WAVs\n' "$(date --iso-8601=seconds)"
"${PYTHON_BIN}" scripts/render_onnx.py \
  --model "${EXPORT_DIR}/piano_current_fixed.onnx" \
  --midi-dir "${MIDI_DIR}" \
  --output-dir "${MIDI_TEST_DIR}/current_fixed"

printf '[%s] rendering v2 MIDI comparison WAVs\n' "$(date --iso-8601=seconds)"
"${PYTHON_BIN}" scripts/render_onnx.py \
  --model "${EXPORT_DIR}/piano_ddsp_v2.onnx" \
  --midi-dir "${MIDI_DIR}" \
  --output-dir "${MIDI_TEST_DIR}/v2"

printf '[%s] DDSP-Piano v2 pipeline completed successfully\n' "$(date --iso-8601=seconds)"
