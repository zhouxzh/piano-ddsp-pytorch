#!/usr/bin/env bash
set -euo pipefail

# Dataset traffic must stay on the Hugging Face mirror. In particular, do not
# inherit Clash settings from an interactive shell when downloading large data.
unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY ALL_PROXY all_proxy
export HF_ENDPOINT="https://hf-mirror.com"
export HF_HUB_ETAG_TIMEOUT="${HF_HUB_ETAG_TIMEOUT:-60}"
export HF_HUB_DOWNLOAD_TIMEOUT="${HF_HUB_DOWNLOAD_TIMEOUT:-600}"

dataset_root="${DATASET_ROOT:-/data/dataset}"
hf_bin="${HF_BIN:-/home/zhong/anaconda3/envs/torch/bin/hf}"
workers="${HF_MAX_WORKERS:-8}"
small_file_workers="${HF_SMALL_FILE_WORKERS:-2}"
target="${1:-all}"

export HF_HOME="${HF_HOME:-${dataset_root}/.hf-cache}"
mkdir -p "${dataset_root}" "${HF_HOME}"

if [[ ! -x "${hf_bin}" ]]; then
  echo "Hugging Face CLI not found or not executable: ${hf_bin}" >&2
  exit 1
fi

require_count() {
  local directory="$1"
  local pattern="$2"
  local expected="$3"
  local count
  count=$(find "${directory}" -type f -name "${pattern}" | wc -l)
  if [[ "${count}" -ne "${expected}" ]]; then
    echo "Incomplete download: ${directory} has ${count} ${pattern} files; expected ${expected}." >&2
    exit 1
  fi
}

require_no_wav() {
  local directory="$1"
  local count
  count=$(find "${directory}" -type f -iname "*.wav" | wc -l)
  if [[ "${count}" -ne 0 ]]; then
    echo "MIDI-only download unexpectedly contains ${count} WAV files: ${directory}" >&2
    exit 1
  fi
}

download_nsynth() {
  "${hf_bin}" download jg583/NSynth \
    --repo-type dataset \
    --revision 5ba57c8114d9d4d7cf076c4cb3c8f508f105fca8 \
    --local-dir "${dataset_root}/nsynth-hf-jg583" \
    --max-workers "${workers}"
  require_count "${dataset_root}/nsynth-hf-jg583" "*.parquet" 37
}

download_groove_midi() {
  "${hf_bin}" download schism-audio/groove-midi-dataset \
    --repo-type dataset \
    --revision 5ab68d3ff4d44d93b9bf8a107a242fee725b0f83 \
    --include "*.mid" \
    --include "*.csv" \
    --include "README*" \
    --include "LICENSE*" \
    --local-dir "${dataset_root}/groove-midi-hf-schism-audio" \
    --max-workers "${small_file_workers}"
  require_count "${dataset_root}/groove-midi-hf-schism-audio" "*.mid" 1150
  require_no_wav "${dataset_root}/groove-midi-hf-schism-audio"
}

download_e_gmd_midi() {
  "${hf_bin}" download schism-audio/e-gmd \
    --repo-type dataset \
    --revision 4ab131ef425da2486cf2838febb555d6cfb639c8 \
    --include "*.midi" \
    --include "*.csv" \
    --include "README*" \
    --include "LICENSE*" \
    --local-dir "${dataset_root}/e-gmd-midi-hf-schism-audio" \
    --max-workers "${small_file_workers}"
  require_count "${dataset_root}/e-gmd-midi-hf-schism-audio" "*.midi" 45537
  require_no_wav "${dataset_root}/e-gmd-midi-hf-schism-audio"
}

case "${target}" in
  all)
    download_nsynth
    download_groove_midi
    download_e_gmd_midi
    ;;
  nsynth)
    download_nsynth
    ;;
  groove-midi)
    download_groove_midi
    ;;
  e-gmd-midi)
    download_e_gmd_midi
    ;;
  *)
    echo "Usage: $0 [all|nsynth|groove-midi|e-gmd-midi]" >&2
    exit 2
    ;;
esac
