# PyTorch DDSP-Piano Training

This is a standalone NVIDIA training package for a polyphonic MIDI-to-audio
DDSP piano model. It uses PyTorch and the server's existing CUDA-matched
`torchaudio`; it does not require TensorFlow, Magenta, or `note_seq`.

The package contains no MAESTRO MIDI, WAV, preprocessing cache, checkpoints,
or audio examples. Download datasets into their own `data/<dataset-name>/`
directory; their contents are ignored by Git.

## Server Setup

Install the pinned dependencies that were validated on ace2. The PyTorch and
torchaudio wheels target CUDA 12.8, so use a matching NVIDIA environment.

```bash
cd piano-ddsp-pytorch
conda activate torch
conda install -c conda-forge ffmpeg
python -m pip install -r requirements.txt
python scripts/check_environment.py
```

TorchCodec uses the system FFmpeg libraries for WAV decoding and encoding.
`scripts/check_environment.py` exits nonzero when a required Python package,
CUDA, or FFmpeg is unavailable.

`cuda_available=True` is required for practical training. This model is large;
start with batch size 1 and increase only after observing GPU memory use.

## Dataset Requirement

Training requires a complete extracted MAESTRO v3.0.0 root containing both the
aligned WAV recordings and MIDI files referenced in `maestro-v3.0.0.csv`. A
MIDI-only download cannot learn a piano timbre.

After installing `requirements.txt`, download the complete dataset directly
from the Hugging Face mirror into `data/maestro-v3.0.0/`:

```bash
export HF_ENDPOINT=https://hf-mirror.com

hf download ddPn08/maestro-v3.0.0 \
  --type dataset \
  --local-dir data/maestro-v3.0.0 \
  --max-workers 8
```

For unreliable connections, `scripts/download_maestro_dataset.sh` uses the
mirror, resumes partial files, retries after 30 seconds, and logs progress to
`.download-state/maestro-v3.0.0.log` every 10 minutes.

This repository stores the extracted dataset as separate files organized by
year, so no `unzip` step is required. The download contains 1,276 aligned WAV
and MIDI pairs plus `maestro-v3.0.0.csv`. A MIDI-only download does not contain
the WAV recordings required for training.

Validate the server-side dataset before preprocessing:

```bash
python scripts/validate_maestro.py --maestro-root data/maestro-v3.0.0
```

The command exits nonzero when audio or MIDI files named by the metadata are
missing.

## Preprocess

Cache audio, MIDI conditioning, sustain-pedal controls, and stable polyphonic
slots on the server. The cache can be regenerated, so it is ignored by Git and
excluded from the ZIP package.

```bash
python train.py \
  --maestro-root data/maestro-v3.0.0 \
  --cache-dir cache/maestro-v3.0.0 \
  --prepare-only
```

`--prepare-workers 4` is the default. Every track cache is written atomically,
so an interrupted preprocessing run can be started again without rebuilding
complete tracks.

The default configuration is 16 kHz audio, 250 MIDI-control frames per second,
3-second segments, 50% overlap, and 16-note polyphony. Keep these settings the
same for preprocessing and training.

## Train

Phase 1 trains the main synthesis and reverb components. It is the appropriate
starting point; phase 2 only fine-tunes detuning and inharmonicity components.

```bash
python train.py \
  --maestro-root data/maestro-v3.0.0 \
  --cache-dir cache/maestro-v3.0.0 \
  --experiment-dir runs/maestro_phase1 \
  --batch-size 1 \
  --epochs 20 \
  --device cuda \
  --amp
```

Training writes `metrics.jsonl` and checkpoints under the experiment directory.
Resume an interrupted run with `--resume runs/maestro_phase1/checkpoints/last.pt`.

The checked-in unattended pipeline performs resumable preprocessing, phase 1,
phase 2, and verified ONNX export in the `torch` Conda environment:

```bash
systemd-run --user --unit=maestro-ddsp-training \
  --property=Restart=on-failure --property=RestartSec=60s \
  --working-directory="$PWD" "$PWD/scripts/run_maestro_training.sh"

tail -f .training-state/maestro-vst.log
systemctl --user status maestro-ddsp-training.service --no-pager
```

The default run executes 40,000 phase-1 steps followed by 5,000 phase-2 steps.
It resumes `last.pt` after a process failure and records cache, GPU, and metric
status every 10 minutes. Current-model training uses a 0.7 dry / 0.3 wet
spectral objective, a fixed 0.25 wet gain, and an IR-tail constraint; phase 2
remains an A/B experiment.

To train the repaired current model and the independent DDSP-Piano v2-style
model with identical data settings, use:

```bash
scripts/run_model_comparison.sh
```

The script writes `piano_current_fixed.onnx` and `piano_ddsp_v2.onnx` after
running CPU ONNX checks. It then renders every `.mid`/`.midi` file in `midi/`
to `exports/midi_tests/current_fixed/` and `exports/midi_tests/v2/` for a
same-score listening comparison. Override `EPOCHS`, `STEPS_PER_EPOCH`,
`DEVICE`, `MAESTRO_ROOT`, and `RUN_ROOT` for a smoke run or a full experiment.

## Pipeline Smoke Test

No real data is included, but the optional `scripts/make_smoke_maestro.py`
command creates a tiny synthetic MAESTRO-shaped directory for checking CSV
parsing, MIDI conditioning, and cache creation. It does not validate sound
quality.

```bash
python scripts/make_smoke_maestro.py
python train.py \
  --maestro-root data/smoke_maestro/maestro-v3.0.0 \
  --cache-dir cache/smoke_maestro \
  --prepare-only
```

## Stateful ONNX Export for Ascend 310B

Like DDSP-VST, deployment is block based and carries recurrent state between
calls. The complete training model contains complex FFT operations that are not
placed in the Ascend-facing graph. The exported fixed-shape neural model emits
DDSP controls; the host runs harmonic phase accumulation, filtered-noise
synthesis, and convolution reverb.

```bash
python scripts/export_onnx.py \
  --checkpoint runs/maestro_vst/phase1/checkpoints/best.pt \
  --output exports/piano_maestro_realtime_controls.onnx
```

Phase 1 is the production default. Official DDSP-Piano reports that its second
frequency-only training phase does not improve listening quality, so this
pipeline keeps phase 2 only as an explicitly named A/B export instead of
silently replacing the phase-1 model.

The default contract is opset 13 and FP32, with batch size 1, one 250 Hz control
frame per call, 16 polyphony slots, and 64 audio samples per 16 kHz block.

Inputs:

- `conditioning [1,1,16,2]`: active MIDI pitch and onset velocity, FP32.
- `pedal [1,1,4]`: MIDI CC 64 through 67 normalized to `[0,1]`, FP32.
- `piano_model [1]`: MAESTRO year embedding index, INT32.
- `extended_pitch [1,1,16,1]`: host-maintained one-second release pitch, FP32.
- `context_state [1,1,64]`: context GRU state, FP32.
- `monophonic_state [1,16,192]`: per-slot synthesis GRU state, FP32.

Outputs for the repaired current model are `amplitudes [1,1,16,1]`,
`harmonic_distribution [1,1,16,96]`, `inharmonicity [1,1,16,1]`,
`f0_hz [1,1,16,1]` for the phase-1 production model (or
`[1,1,16,2]` for the phase-2 detuned-string comparison),
`noise_magnitudes [1,1,16,64]`,
`reverb_ir [1,24000]`, and the two next-state tensors matching their inputs.
`ddsp_piano.deployment.extend_pitch_for_release` is the CPU reference for the
host release-state rule. The amplitude, harmonic, and noise tensors are raw
neural controls and must use the same DDSP scaling, band-limiting, and
normalization as the training synthesizer;
`ddsp_piano.deployment.scale_controls_for_synthesis` is the NumPy CPU reference
for those operations. Cache `reverb_ir` until `piano_model` changes instead of
transferring it for every audio block.

Export runs `onnx.checker` and compares every output against PyTorch CPU using
ONNX Runtime. The adjacent JSON records shapes, dtypes, operator counts, byte
sizes, and per-output errors. Ascend deployment supports FP16 or FP32 only;
BF16 is intentionally absent. CANN/ATC conversion and on-device memory and
performance tests are outside the current server validation scope; this project
currently gates the model on PyTorch CPU, ONNX graph validation, and ONNX
Runtime numerical equivalence only.

The v2 export keeps the first five control outputs and replaces `reverb_ir`
with `reverb_controls [1,9]`. It emits 128 harmonic bins and 96 noise bands.
The JSON marks FDN as host-side postprocessing and records its delay lines, so
the same WAV renderer can compare both models without changing the neural
input/state contract.

## Listen to the ONNX Model

Render every test score in `midi/` through ONNX Runtime and the
training-matched CPU DDSP synthesis boundary:

```bash
conda run -n torch python scripts/render_onnx.py \
  --model exports/piano_current_fixed.onnx \
  --midi-dir midi \
  --output-dir exports/midi_tests/current_fixed
```

The command loads the ONNX model and its adjacent deployment JSON, but no
PyTorch checkpoint. It carries both GRU states and the one-second MIDI release
state between 250 Hz calls and across bounded-memory render chunks, uses the
upstream DDSP-Piano 0.5-second silent warm-up, appends 2.5 seconds for release
and reverb decay, and gives every voice an independent seeded white-noise
excitation as in DDSP-Piano and MIDI-DDSP. It then renders each score to a
normalized mono 16 kHz WAV and writes `manifest.json` beside the files. The
manifest records source/render duration and any frames exceeding the fixed
16-voice deployment limit. For a v2 JSON, the host applies the recorded FDN
controls instead of a learned IR.
`--piano-model` selects one of the MAESTRO-year embeddings listed in the JSON
and defaults to index 9 (2018).

Use `--midi path/to/file.mid --output path/to/file.wav` for a single score.
With neither MIDI option, the renderer retains the built-in 7.5-second passage
as a smoke test.

To render the current `midi/` directory through every compatible ONNX file in
`exports/`, run:

```bash
python scripts/render_all_onnx_models.py \
  --model-dir exports \
  --midi-dir midi \
  --output-root exports/midi_tests/all_models
```

The command validates every ONNX graph and fixed deployment contract first.
It writes a content-addressed test-set directory, one subdirectory per model,
and an `index.json` containing model hashes, exclusions, piano embedding IDs,
and completion status. Non-stateful or incomplete smoke graphs are listed as
excluded instead of being treated as comparable model versions.

Add `--decompose` to also write separately normalized harmonic, filtered-noise,
and unreverberated stems next to the main WAV. These stems are intended for
diagnosing model quality and should not be mixed after their independent
normalization.

For a deterministic listening comparison, render the same MIDI directory
through both exports:

```bash
python scripts/render_onnx.py --model exports/piano_current_fixed.onnx \
  --midi-dir midi --output-dir exports/midi_tests/current_fixed
python scripts/render_onnx.py --model exports/piano_ddsp_v2.onnx \
  --midi-dir midi --output-dir exports/midi_tests/v2
```

The two directories use identical WAV basenames, MIDI conditioning, piano
embedding, warm-up, release tail, and noise seed.

## Local References

The requested references are downloaded under the ignored `references/`
directory: ACIDS IRCAM's PyTorch DDSP implementation, Google's DDSP repository,
and the DDSP paper PDF (`arXiv:2001.04643`). See `UPSTREAM.md` for revisions.

## Engineering Review

The current architecture, realtime readiness, `_upstream/` comparison, and
Ascend 310B implementation gates are documented in [`doc/README.md`](doc/README.md).

## Provenance

See `UPSTREAM.md` for the upstream PyTorch DDSP-Piano reference revision and
the scope of the changes made in this standalone package.
