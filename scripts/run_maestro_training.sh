#!/usr/bin/env bash
set -u

readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly PYTHON_BIN="${PYTHON_BIN:-/home/zhong/anaconda3/envs/torch/bin/python}"
readonly MAESTRO_ROOT="${MAESTRO_ROOT:-${PROJECT_DIR}/data/maestro-v3.0.0}"
readonly CACHE_DIR="${CACHE_DIR:-${PROJECT_DIR}/cache/maestro-v3.0.0}"
readonly RUN_ROOT="${RUN_ROOT:-${PROJECT_DIR}/runs/maestro_vst}"
readonly PHASE1_DIR="${RUN_ROOT}/phase1"
readonly PHASE2_DIR="${RUN_ROOT}/phase2"
readonly EXPORT_PATH="${PROJECT_DIR}/exports/piano_maestro_realtime_controls.onnx"
readonly PHASE2_EXPORT_PATH="${PROJECT_DIR}/exports/piano_maestro_phase2_realtime_controls.onnx"
readonly STATE_DIR="${PROJECT_DIR}/.training-state"
readonly LOG_FILE="${STATE_DIR}/maestro-vst.log"
readonly LOCK_FILE="${STATE_DIR}/maestro-vst.lock"
readonly CHECK_INTERVAL_SECONDS=600
readonly RETRY_DELAY_SECONDS=60

mkdir -p "${STATE_DIR}" "${RUN_ROOT}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] another MAESTRO training pipeline is already running\n' \
    "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
  exit 0
fi

if [[ ! -x "${PYTHON_BIN}" ]]; then
  printf '[%s] error: torch environment Python is not executable: %s\n' \
    "$(date --iso-8601=seconds)" "${PYTHON_BIN}" >>"${LOG_FILE}"
  exit 127
fi

log_status() {
  local cache_size metric gpu
  cache_size="$(du -sh "${CACHE_DIR}" 2>/dev/null | cut -f1)"
  metric="$(tail -n 1 "${PHASE2_DIR}/metrics.jsonl" 2>/dev/null || tail -n 1 "${PHASE1_DIR}/metrics.jsonl" 2>/dev/null || true)"
  gpu="$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | head -n 1)"
  printf '[%s] status: cache=%s gpu=%s metric=%s\n' \
    "$(date --iso-8601=seconds)" "${cache_size:-0}" "${gpu:-unavailable}" "${metric:-none}" >>"${LOG_FILE}"
}

run_monitored() {
  "$@" >>"${LOG_FILE}" 2>&1 &
  local child_pid=$!
  while kill -0 "${child_pid}" 2>/dev/null; do
    if timeout "${CHECK_INTERVAL_SECONDS}" tail --pid="${child_pid}" -f /dev/null; then
      break
    fi
    log_status
  done
  wait "${child_pid}"
}

retry_command() {
  local label=$1
  shift
  while true; do
    printf '[%s] starting %s\n' "$(date --iso-8601=seconds)" "${label}" >>"${LOG_FILE}"
    run_monitored "$@"
    local exit_code=$?
    if (( exit_code == 0 )); then
      printf '[%s] completed %s\n' "$(date --iso-8601=seconds)" "${label}" >>"${LOG_FILE}"
      return 0
    fi
    printf '[%s] %s exited with code %d; retrying in %d seconds\n' \
      "$(date --iso-8601=seconds)" "${label}" "${exit_code}" "${RETRY_DELAY_SECONDS}" >>"${LOG_FILE}"
    sleep "${RETRY_DELAY_SECONDS}"
  done
}

cd "${PROJECT_DIR}"
log_status

retry_command "MAESTRO preprocessing" \
  "${PYTHON_BIN}" train.py \
  --maestro-root "${MAESTRO_ROOT}" \
  --cache-dir "${CACHE_DIR}" \
  --prepare-only \
  --prepare-workers 4

while true; do
  phase1_resume=()
  if [[ -f "${PHASE1_DIR}/checkpoints/last.pt" ]]; then
    phase1_resume=(--resume "${PHASE1_DIR}/checkpoints/last.pt")
  fi
  if run_monitored \
    "${PYTHON_BIN}" train.py \
    --maestro-root "${MAESTRO_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --experiment-dir "${PHASE1_DIR}" \
    --phase 1 \
    --batch-size 1 \
    --epochs 20 \
    --steps-per-epoch 2000 \
    --validation-batches 16 \
    --num-workers 4 \
    --device cuda \
    --amp \
    "${phase1_resume[@]}"; then
    break
  fi
  printf '[%s] phase 1 failed; retrying from last checkpoint in %d seconds\n' \
    "$(date --iso-8601=seconds)" "${RETRY_DELAY_SECONDS}" >>"${LOG_FILE}"
  sleep "${RETRY_DELAY_SECONDS}"
done

while true; do
  if [[ -f "${PHASE2_DIR}/checkpoints/last.pt" ]]; then
    phase2_start=(--resume "${PHASE2_DIR}/checkpoints/last.pt")
  else
    phase2_start=(--weights "${PHASE1_DIR}/checkpoints/best.pt")
  fi
  if run_monitored \
    "${PYTHON_BIN}" train.py \
    --maestro-root "${MAESTRO_ROOT}" \
    --cache-dir "${CACHE_DIR}" \
    --experiment-dir "${PHASE2_DIR}" \
    --phase 2 \
    --batch-size 1 \
    --epochs 5 \
    --steps-per-epoch 1000 \
    --validation-batches 16 \
    --num-workers 4 \
    --device cuda \
    --amp \
    "${phase2_start[@]}"; then
    break
  fi
  printf '[%s] phase 2 failed; retrying from last checkpoint in %d seconds\n' \
    "$(date --iso-8601=seconds)" "${RETRY_DELAY_SECONDS}" >>"${LOG_FILE}"
  sleep "${RETRY_DELAY_SECONDS}"
done

retry_command "phase-1 ONNX export and CPU verification" \
  "${PYTHON_BIN}" scripts/export_onnx.py \
  --checkpoint "${PHASE1_DIR}/checkpoints/best.pt" \
  --output "${EXPORT_PATH}"

retry_command "phase-2 comparison ONNX export and CPU verification" \
  "${PYTHON_BIN}" scripts/export_onnx.py \
  --checkpoint "${PHASE2_DIR}/checkpoints/best.pt" \
  --output "${PHASE2_EXPORT_PATH}"

log_status
printf '[%s] full MAESTRO training pipeline completed successfully\n' \
  "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
