# PyTorch DDSP-Piano Training

This is a standalone NVIDIA training package for a polyphonic MIDI-to-audio
DDSP piano model. It uses PyTorch and the server's existing CUDA-matched
`torchaudio`; it does not require TensorFlow, Magenta, or `note_seq`.

The package contains no MAESTRO MIDI, WAV, preprocessing cache, checkpoints,
or audio examples. The original dataset stays outside this directory.

## Server Setup

The server already has PyTorch and torchaudio. Do not reinstall them from this
package, because the installed builds must remain matched to the NVIDIA CUDA
driver. Install only the small Python helpers:

```bash
cd piano-ddsp-pytorch
python -m pip install -r requirements-extra.txt
python scripts/check_environment.py
```

`cuda_available=True` is required for practical training. This model is large;
start with batch size 1 and increase only after observing GPU memory use.

## Dataset Requirement

Training requires a complete extracted MAESTRO v3.0.0 root containing both the
aligned WAV recordings and MIDI files referenced in `maestro-v3.0.0.csv`. A
MIDI-only download cannot learn a piano timbre.

Validate the server-side dataset before preprocessing:

```bash
python scripts/validate_maestro.py --maestro-root /data/maestro-v3.0.0
```

The command exits nonzero when audio or MIDI files named by the metadata are
missing.

## Preprocess

Cache audio, MIDI conditioning, sustain-pedal controls, and stable polyphonic
slots on the server. The cache can be regenerated, so it is ignored by Git and
excluded from the ZIP package.

```bash
python scripts/prepare_maestro.py \
  --maestro-root /data/maestro-v3.0.0 \
  --cache-dir /data/ddsp_piano_cache
```

The default configuration is 16 kHz audio, 250 MIDI-control frames per second,
3-second segments, 50% overlap, and 16-note polyphony. Keep these settings the
same for preprocessing and training.

## Train

Phase 1 trains the main synthesis and reverb components. It is the appropriate
starting point; phase 2 only fine-tunes detuning and inharmonicity components.

```bash
python train.py \
  --maestro-root /data/maestro-v3.0.0 \
  --cache-dir /data/ddsp_piano_cache \
  --experiment-dir runs/maestro_phase1 \
  --batch-size 1 \
  --epochs 20 \
  --device cuda \
  --amp
```

Training writes `metrics.jsonl` and checkpoints under the experiment directory.
Resume an interrupted run with `--resume runs/maestro_phase1/checkpoints/last.pt`.

## Pipeline Smoke Test

No real data is included, but the optional `scripts/make_smoke_maestro.py`
command creates a tiny synthetic MAESTRO-shaped directory for checking CSV
parsing, MIDI conditioning, and cache creation. It does not validate sound
quality.

## Provenance

See `UPSTREAM.md` for the upstream PyTorch DDSP-Piano reference revision and
the scope of the changes made in this standalone package.
