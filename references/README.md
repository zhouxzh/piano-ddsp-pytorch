# Local Reference Inventory

`references/` is the single local home for third-party source code, papers,
and reference models used by this repository. The inventory is committed, but
all downloaded content below this file is ignored by the parent repository.
Each source checkout retains its own `.git` directory and upstream history.

## Source Repositories

| Directory | Upstream | Pinned revision | Primary use |
|---|---|---|---|
| `ddsp-piano` | `lrenault/ddsp-piano` | `e868b7ccd3fe` | Piano architecture and v2 training design |
| `ddsp-piano-pytorch` | `ytsrt66589/ddsp-piano-pytorch` | `2c9e17aa0c17` | Direct historical baseline for this repository |
| `acids-ddsp_pytorch` | `acids-ircam/ddsp_pytorch` | `9db246f48dba` | Stateful PyTorch DDSP and realtime export |
| `google-ddsp` | `magenta/ddsp` | `cf5e62dfe5d5` | Canonical DDSP implementation and VST model |
| `midi-ddsp` | `magenta/midi-ddsp` | `d7af42704a63` | MIDI expression and synthesis generators |
| `ddsp-vst` | `magenta/ddsp-vst` | `f2996e97f946` | Realtime host, buffering, and DSP implementation |
| `ddsp-realtime` | `woosukji/ddsp-realtime` | `6cdfb583e5e9` | Framework-independent realtime C++ core |
| `realtime-ddsp` | `hyakuchiki/realtimeDDSP` | `3f2f79039413` | Streaming oscillator, noise, and reverb state |
| `sweetcocoa-ddsp-pytorch` | `sweetcocoa/ddsp-pytorch` | `ea5f25318dd4` | Early general PyTorch DDSP implementation |
| `ascend-cann-samples` | `huqi/ascend-cann-samples` | `6511a5f4a45a` | Ascend audio API reference sample |
| `magenta` | `magenta/magenta` | `c15687ebd1c1` | Archived Magenta models and data pipelines |
| `note-seq` | `magenta/note-seq` | `358255088853` | MIDI parsing, quantization, and alignment |
| `mt3` | `magenta/mt3` | `fa53e12321ac` | Multi-task transcription data contracts |
| `music-spectrogram-diffusion` | `magenta/music-spectrogram-diffusion` | `24cd4b0df1b1` | High-quality MIDI-conditioned synthesis reference |
| `magenta-js` | `magenta/magenta-js` | `96bad8837da4` | Browser MIDI and audio interaction |
| `magenta-realtime` | `magenta/magenta-realtime` | `2a854047691f` | Streaming state and realtime inference |
| `magenta-studio` | `magenta/magenta-studio` | `30bca2674f34` | Packaged MIDI generation tools |
| `chamber-ensemble-generator` | `lukewys/chamber-ensemble-generator` | `4647edbce671` | CocoChorales generation and synthesis controls |
| `ascend310-case3` | `zhouxzh/Ascend310` | `fb17de9dd50b` | Latest case3 intelligent-piano MIDI and keyboard interaction reference |

`ascend-cann-samples` intentionally retains the local CANN 8.3 audio-device
and frame-size compatibility patch described in
`doc/upstream-reference-review.md`. Some imported repositories may also appear
dirty because their Windows CRLF line endings were preserved; do not normalize
or update a reference checkout as part of product-code changes.

`ascend310-case3` is a shallow sparse checkout of the upstream repository. Its
complete `samples/case3/` directory is available locally, including `midi.py`,
the scenario README, and the CAD model. The realtime browser service in this
repository reuses its F3-to-E5 key range, computer-key mapping, velocity, and
Note On/Off interaction semantics; case3 itself does not contain a network
audio server.

`piano-keyboard-topic/` is a task-focused subset of the public repositories
returned by the GitHub Search API for the `piano-keyboard` topic. Selection is
based on direct relevance to the remote realtime MIDI-to-ONNX piano player, not
only star count. The full API inventory remains local for traceability. The
retained sources are:

| Repository | Reference use |
|---|---|
| `Calbabreaker/piano` | Web MIDI, keyboard/touch input, velocity, sustain, MIDI-file playback, and WebSocket Note On/Off |
| `sightread/sightread` | MIDI parsing, playback clock, input lifecycle, transport controls, and tests |
| `scottroot/Musical-Dynamics-Training-Software` | Hardware MIDI velocity ranges, 88-key mapping, and velocity-zero Note Off handling |
| `dy/piano-keyboard` | Minimal DOM keyboard, active-note deduplication, and reliable Note Off cleanup |

The large `sightread` soundfont bank is intentionally omitted because this
project streams ONNX-generated PCM from the server and does not use browser
soundfont synthesis. Repositories for fingering, MIDI-to-video, Android or IoT
keyboards, Roblox autoplay, 3D rendering, generic local synthesizers, and
duplicate piano UIs are not retained.

Direct Git downloads use one-level checkouts with Git LFS payloads disabled. When Git and
an optional proxy are unavailable, the downloader falls back to an official
GitHub codeload snapshot and writes `.topic-source.json` in that source tree.
The ignored files `repositories.json`, `selected-repositories.json`,
`sources.tsv`, and `download-failures.tsv` record the full API snapshot,
selection, local source type/revision, and retry list. Refresh or resume with
`scripts/download_piano_keyboard_references.sh`; set
`GITHUB_CLONE_PROXY_PREFIX` for a Git proxy prefix, or set
`REFERENCE_FETCH_MODE=proxy` to skip direct Git. Set it to `archive` to skip
Git entirely and avoid a known-bad Git endpoint. During a rate limit, set
`REFERENCE_REFRESH_INVENTORY=0` to resume from the last API snapshot. An
optional `GITHUB_ARCHIVE_PROXY_PREFIX` accelerates only the codeload fallback
and is recorded as `github-archive-proxy` in the source manifest.
Set `PIANO_KEYBOARD_PRUNE_UNSELECTED=1` to remove clean, recognized checkouts
outside the task-focused allowlist.

## Papers And Models

- `papers/`: the single flat directory for all locally verified papers. Its
  generated `SHA256SUMS` file covers every PDF.
- `papers/ddsp-2001.04643.pdf`: the DDSP paper, SHA-256
  `41aa87d9710abd0c66837502684d5a5f58e0cb1e34a62eb2d14e25944dbb0856`.
- `models/ddsp-vst-flute/Flute.onnx`: the fixed-shape DDSP-VST reference model,
  SHA-256 `908d9233f93dca3dcf6f9eb9961f28bdc3cbc2093a42831ae97b20fbde44edf2`.
- `models/ddsp-vst-flute/Flute.json`: the model's numerical comparison record,
  SHA-256 `dac6ac0b05cef468589af65be3ffe5b8d5e042033f5cb2c3d359cfe4b9673133`.

Generated smoke-test MIDI and WAV files are not references. Legacy files from
the former `_upstream/` directory were moved to
`evaluations/legacy_reference_smoke/`. Temporary Windows TFLite inspection
packages were moved to `.download-state/tflite-python-tools-windows/`.

See `UPSTREAM.md` for provenance and `doc/upstream-reference-review.md` for the
engineering assessment of these sources. Dataset provenance, mirror revisions,
and download policy are documented in `doc/magenta-reference-materials.md`.
