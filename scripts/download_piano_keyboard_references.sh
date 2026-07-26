#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
COLLECTION_DIR=${PIANO_KEYBOARD_REFERENCE_DIR:-"$ROOT_DIR/references/piano-keyboard-topic"}
CLONE_WORKERS=${CLONE_WORKERS:-6}
PROXY_PREFIX=${GITHUB_CLONE_PROXY_PREFIX:-}
ARCHIVE_PROXY_PREFIX=${GITHUB_ARCHIVE_PROXY_PREFIX:-}
FETCH_MODE=${REFERENCE_FETCH_MODE:-clone}
REFRESH_INVENTORY=${REFERENCE_REFRESH_INVENTORY:-1}
PRUNE_UNSELECTED=${PIANO_KEYBOARD_PRUNE_UNSELECTED:-0}
API_URL="https://api.github.com/search/repositories?q=topic%3Apiano-keyboard&sort=stars&order=desc&per_page=100"
CURATED_REPOSITORIES=(
  "Calbabreaker/piano"
  "dy/piano-keyboard"
  "scottroot/Musical-Dynamics-Training-Software"
  "sightread/sightread"
)

if ! [[ "$CLONE_WORKERS" =~ ^[1-9][0-9]*$ ]]; then
  echo "CLONE_WORKERS must be a positive integer" >&2
  exit 2
fi
if [[ "$FETCH_MODE" != "clone" && "$FETCH_MODE" != "proxy" && "$FETCH_MODE" != "archive" ]]; then
  echo "REFERENCE_FETCH_MODE must be clone, proxy, or archive" >&2
  exit 2
fi
if [[ "$FETCH_MODE" == "proxy" && -z "$PROXY_PREFIX" ]]; then
  echo "GITHUB_CLONE_PROXY_PREFIX is required in proxy mode" >&2
  exit 2
fi
if [[ "$REFRESH_INVENTORY" != "0" && "$REFRESH_INVENTORY" != "1" ]]; then
  echo "REFERENCE_REFRESH_INVENTORY must be 0 or 1" >&2
  exit 2
fi
if [[ "$PRUNE_UNSELECTED" != "0" && "$PRUNE_UNSELECTED" != "1" ]]; then
  echo "PIANO_KEYBOARD_PRUNE_UNSELECTED must be 0 or 1" >&2
  exit 2
fi
for command in curl git jq tar timeout; do
  if ! command -v "$command" >/dev/null 2>&1; then
    echo "Required command not found: $command" >&2
    exit 2
  fi
done

mkdir -p "$COLLECTION_DIR"
find "$COLLECTION_DIR" -mindepth 1 -maxdepth 1 -type d -name '.fetch.*' \
  -exec rm -rf -- {} +
TEMP_DIR=$(mktemp -d "$COLLECTION_DIR/.inventory.XXXXXX")
trap 'rm -rf "$TEMP_DIR"' EXIT

if [[ "$REFRESH_INVENTORY" == "1" ]]; then
  inventory_complete=1
  for page in 1 2 3; do
    if ! curl --fail --silent --show-error --location --retry 5 --retry-delay 3 \
        -H "Accept: application/vnd.github+json" \
        -H "X-GitHub-Api-Version: 2022-11-28" \
        "$API_URL&page=$page" >"$TEMP_DIR/page-$page.json"; then
      inventory_complete=0
      break
    fi
  done
  if [[ "$inventory_complete" == "1" ]]; then
    jq -s '
      {
        generated_at: (now | todateiso8601),
        topic: "piano-keyboard",
        api_total_count: .[0].total_count,
        repositories: (
          [.[].items[]]
          | unique_by(.full_name)
          | sort_by([-.stargazers_count, .full_name])
          | map({
              full_name,
              clone_url,
              html_url,
              default_branch,
              description,
              language,
              stargazers_count,
              archived,
              fork,
              size_kb: .size,
              pushed_at
            })
        )
      }
    ' "$TEMP_DIR"/page-*.json >"$TEMP_DIR/repositories.json"
    mv "$TEMP_DIR/repositories.json" "$COLLECTION_DIR/repositories.json"
  elif [[ -s "$COLLECTION_DIR/repositories.json" ]]; then
    echo "GitHub API refresh failed; reusing cached repositories.json" >&2
  else
    echo "GitHub API refresh failed and no cached inventory exists" >&2
    exit 1
  fi
elif [[ ! -s "$COLLECTION_DIR/repositories.json" ]]; then
  echo "Cached repositories.json not found; inventory refresh is required" >&2
  exit 1
fi

curated_repositories_json=$(
  printf '%s\n' "${CURATED_REPOSITORIES[@]}" \
    | jq -Rsc 'split("\n") | map(select(length > 0))'
)
jq --argjson curated "$curated_repositories_json" '
  {
    generated_at,
    topic,
    api_total_count,
    selection: {
      policy: "realtime-midi-piano-player",
      repositories: $curated
    },
    repositories: [
      .repositories[]
      | select(.full_name as $name | ($curated | index($name)) != null)
    ]
  }
' "$COLLECTION_DIR/repositories.json" \
  >"$COLLECTION_DIR/selected-repositories.json"

selected_count=$(jq '.repositories | length' "$COLLECTION_DIR/selected-repositories.json")
if [[ "$selected_count" -ne "${#CURATED_REPOSITORIES[@]}" ]]; then
  echo "Curated repository list is not fully represented in the GitHub inventory" >&2
  exit 1
fi

if [[ "$PRUNE_UNSELECTED" == "1" ]]; then
  pruned_count=0
  while IFS= read -r full_name; do
    destination="$COLLECTION_DIR/${full_name//\//__}"
    [[ -e "$destination" ]] || continue
    if [[ "$destination" != "$COLLECTION_DIR/"* ]]; then
      echo "Refusing to prune path outside collection: $destination" >&2
      exit 1
    fi
    if [[ -d "$destination/.git" ]] \
        && git -C "$destination" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
      if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
        echo "Refusing to prune modified checkout: $full_name" >&2
        continue
      fi
    elif [[ ! -f "$destination/.topic-source.json" ]]; then
      echo "Refusing to prune unrecognized directory: $full_name" >&2
      continue
    fi
    rm -rf -- "$destination"
    pruned_count=$((pruned_count + 1))
  done < <(
    jq -r --argjson curated "$curated_repositories_json" \
      '.repositories[]
       | select(.full_name as $name | ($curated | index($name)) == null)
       | .full_name' \
      "$COLLECTION_DIR/repositories.json"
  )
  printf 'Pruned unselected sources: %d\n' "$pruned_count"
fi

: >"$COLLECTION_DIR/download-failures.tsv"

fetch_one() {
  local full_name=$1
  local clone_url=$2
  local default_branch=$3
  local pushed_at=$4
  local size_kb=$5
  local destination="$COLLECTION_DIR/${full_name//\//__}"
  if [[ -d "$destination/.git" ]] \
      && git -C "$destination" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    echo "present $full_name"
    return 0
  fi
  if [[ -f "$destination/.topic-source.json" ]]; then
    echo "present $full_name"
    return 0
  fi
  if [[ -e "$destination" ]]; then
    printf '%s\t%s\n' "$full_name" "destination exists but is not a recognized snapshot" \
      >>"$COLLECTION_DIR/download-failures.tsv"
    echo "failed  $full_name (invalid existing destination)" >&2
    return 0
  fi

  local temporary
  if [[ "$FETCH_MODE" == "clone" ]]; then
    temporary=$(mktemp -d "$COLLECTION_DIR/.fetch.XXXXXX")
    rmdir "$temporary"
    if timeout 90 env GIT_LFS_SKIP_SMUDGE=1 git clone \
        --depth 1 --single-branch --no-tags "$clone_url" "$temporary" \
        >/dev/null 2>&1; then
      mv "$temporary" "$destination"
      echo "cloned   $full_name"
      return 0
    fi
    rm -rf "$temporary"

  fi

  if [[ "$FETCH_MODE" != "archive" && -n "$PROXY_PREFIX" ]]; then
    temporary=$(mktemp -d "$COLLECTION_DIR/.fetch.XXXXXX")
    rmdir "$temporary"
    if timeout 180 env GIT_LFS_SKIP_SMUDGE=1 git clone \
        --depth 1 --single-branch --no-tags \
        "${PROXY_PREFIX}${clone_url}" "$temporary" >/dev/null 2>&1; then
      mv "$temporary" "$destination"
      echo "proxied  $full_name"
      return 0
    fi
    rm -rf "$temporary"
  fi

  temporary=$(mktemp -d "$COLLECTION_DIR/.fetch.XXXXXX")
  mkdir "$temporary/content"
  local archive_source="github-codeload"
  local archive_downloaded=0
  if [[ -n "$ARCHIVE_PROXY_PREFIX" ]] \
      && curl --fail --location --retry 3 --retry-delay 3 \
        --connect-timeout 20 --speed-limit 1024 --speed-time 90 --max-time 1800 \
        "${ARCHIVE_PROXY_PREFIX}https://github.com/${full_name}/archive/refs/heads/${default_branch}.tar.gz" \
        >"$temporary/source.tar.gz" 2>/dev/null; then
    archive_source="github-archive-proxy"
    archive_downloaded=1
  fi
  if [[ "$archive_downloaded" == "0" ]] \
      && curl --fail --location --retry 5 --retry-delay 3 \
        --connect-timeout 20 --speed-limit 1024 --speed-time 90 --max-time 1800 \
        "https://codeload.github.com/${full_name}/tar.gz/refs/heads/${default_branch}" \
        >"$temporary/source.tar.gz" 2>/dev/null; then
    archive_source="github-codeload"
    archive_downloaded=1
  fi
  if [[ "$archive_downloaded" == "1" ]] \
      && tar -xzf "$temporary/source.tar.gz" -C "$temporary/content" \
        --strip-components=1; then
    jq -n \
      --arg full_name "$full_name" \
      --arg branch "$default_branch" \
      --arg pushed_at "$pushed_at" \
      --arg source_type "$archive_source" \
      '{source_type:$source_type, full_name:$full_name, ref:$branch,
        pushed_at:$pushed_at, fetched_at:(now | todateiso8601)}' \
      >"$temporary/content/.topic-source.json"
    rm "$temporary/source.tar.gz"
    mv "$temporary/content" "$destination"
    rmdir "$temporary"
    echo "archived $full_name"
    return 0
  fi
  rm -rf "$temporary"

  printf '%s\t%s\t%s\t%s\n' "$full_name" "$clone_url" "$default_branch" "$size_kb" \
    >>"$COLLECTION_DIR/download-failures.tsv"
  echo "failed   $full_name" >&2
}
export COLLECTION_DIR PROXY_PREFIX ARCHIVE_PROXY_PREFIX FETCH_MODE
export -f fetch_one

jq -r '.repositories[] | [.full_name, .clone_url, .default_branch, .pushed_at, .size_kb] | @tsv' \
  "$COLLECTION_DIR/selected-repositories.json" \
  | xargs -P "$CLONE_WORKERS" -d '\n' -n 1 bash -c \
      'IFS=$'"'"'\t'"'"' read -r full_name clone_url default_branch pushed_at size_kb <<<"$1"; fetch_one "$full_name" "$clone_url" "$default_branch" "$pushed_at" "$size_kb"' _

sightread_soundfonts="$COLLECTION_DIR/sightread__sightread/public/soundfonts"
if [[ -d "$sightread_soundfonts" ]]; then
  rm -rf -- "$sightread_soundfonts"
  echo "pruned   sightread/sightread public soundfont assets"
fi

{
  printf 'full_name\tsource_type\trevision\n'
  jq -r '.repositories[].full_name' "$COLLECTION_DIR/selected-repositories.json" \
    | while IFS= read -r full_name; do
        destination="$COLLECTION_DIR/${full_name//\//__}"
        if [[ -d "$destination/.git" ]] \
            && git -C "$destination" rev-parse HEAD >/dev/null 2>&1; then
          printf '%s\tshallow-git\t%s\n' \
            "$full_name" "$(git -C "$destination" rev-parse HEAD)"
        elif [[ -f "$destination/.topic-source.json" ]]; then
          printf '%s\t%s\t%s\n' \
            "$full_name" \
            "$(jq -r '.source_type' "$destination/.topic-source.json")" \
            "$(jq -r '.ref + "@" + .pushed_at' "$destination/.topic-source.json")"
        fi
      done
} >"$COLLECTION_DIR/sources.tsv"

topic_count=$(jq '.repositories | length' "$COLLECTION_DIR/repositories.json")
repository_count=$(jq '.repositories | length' "$COLLECTION_DIR/selected-repositories.json")
download_count=$(($(wc -l <"$COLLECTION_DIR/sources.tsv") - 1))
failure_count=$(wc -l <"$COLLECTION_DIR/download-failures.tsv")
printf 'Topic repositories: %d\nCurated realtime MIDI piano repositories: %d\nLocal sources: %d\nFailures: %d\n' \
  "$topic_count" "$repository_count" "$download_count" "$failure_count"
