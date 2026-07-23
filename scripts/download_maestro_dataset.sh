#!/usr/bin/env bash
set -u

readonly REPO_ID="ddPn08/maestro-v3.0.0"
readonly PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
readonly TARGET_DIR="${PROJECT_DIR}/data/maestro-v3.0.0"
readonly STATE_DIR="${PROJECT_DIR}/.download-state"
readonly LOG_FILE="${STATE_DIR}/maestro-v3.0.0.log"
readonly LOCK_FILE="${STATE_DIR}/maestro-v3.0.0.lock"
readonly CHECK_INTERVAL_SECONDS=600
readonly RETRY_DELAY_SECONDS=30

mkdir -p "${TARGET_DIR}" "${STATE_DIR}"
exec 9>"${LOCK_FILE}"
if ! flock -n 9; then
  printf '[%s] Another MAESTRO download monitor is already running.\n' "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
  exit 0
fi

export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-120}"

if [[ -z "${HF_BIN:-}" ]]; then
  HF_BIN="/home/zhong/anaconda3/envs/torch/bin/hf"
fi
if [[ ! -x "${HF_BIN}" ]]; then
  HF_BIN="$(command -v hf 2>/dev/null || true)"
fi
if [[ -z "${HF_BIN}" || ! -x "${HF_BIN}" ]]; then
  printf '[%s] error: hf executable not found (HF_BIN=%s)\n' \
    "$(date --iso-8601=seconds)" "${HF_BIN:-unset}" >>"${LOG_FILE}"
  exit 127
fi
readonly HF_BIN

log_status() {
  local bytes file_count size
  bytes="$(du -sb "${TARGET_DIR}" 2>/dev/null | cut -f1)"
  file_count="$(find "${TARGET_DIR}" -type f 2>/dev/null | wc -l)"
  size="$(du -sh "${TARGET_DIR}" 2>/dev/null | cut -f1)"
  printf '[%s] status: downloaded=%s bytes=%s files=%s endpoint=%s\n' \
    "$(date --iso-8601=seconds)" "${size:-0}" "${bytes:-0}" \
    "${file_count:-0}" "${HF_ENDPOINT}" >>"${LOG_FILE}"
}

while true; do
  log_status
  printf '[%s] starting/resuming hf download\n' "$(date --iso-8601=seconds)" >>"${LOG_FILE}"

  "${HF_BIN}" download "${REPO_ID}" \
    --type dataset \
    --local-dir "${TARGET_DIR}" >>"${LOG_FILE}" 2>&1 &
  download_pid=$!

  while kill -0 "${download_pid}" 2>/dev/null; do
    if timeout "${CHECK_INTERVAL_SECONDS}" tail --pid="${download_pid}" -f /dev/null; then
      break
    fi
    log_status
  done

  wait "${download_pid}"
  exit_code=$?
  if (( exit_code == 0 )); then
    log_status
    printf '[%s] download completed successfully\n' "$(date --iso-8601=seconds)" >>"${LOG_FILE}"
    exit 0
  fi

  printf '[%s] hf download exited with code %d; retrying in %d seconds\n' \
    "$(date --iso-8601=seconds)" "${exit_code}" "${RETRY_DELAY_SECONDS}" >>"${LOG_FILE}"
  sleep "${RETRY_DELAY_SECONDS}"
done
