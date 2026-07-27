# DDSP Piano PyTorch

[中文说明](README.zh-CN.md)

This repository trains, exports, and evaluates causal MIDI-conditioned piano
control models. Its first stable publication is the four-model
`model-suite-v1.0.0`:

| Model ID | Architecture | Harmonics / noise | Host reverb |
| --- | --- | ---: | --- |
| `paper_ir` | DAFx22 paper-style control networks | 96 / 64 | learned IR, wet 0.25 |
| `film_fdn` | later MAESTRO v2 FiLM/deep network | 128 / 96 | FDN controls |
| `calibrated_ir` | legacy controls with perceptual calibration | 96 / 64 | learned IR, wet 1.0 |
| `calibrated_film_ir` | FiLM/deep/joint controls with perceptual calibration | 96 / 64 | learned IR, wet 1.0 |

The names identify structures, not a quality ranking. All four models are
formal downstream conversion candidates. Existing listening results do not
justify selecting a single winner.

## Quick Start

Use Python 3.11. CPU inference and validation:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r requirements.txt
python -m unittest discover -v
```

Download the `model-suite-v1.0.0` GitHub Release assets into
`artifacts/model-suite-v1.0.0/`, then verify them:

```bash
cd artifacts/model-suite-v1.0.0
sha256sum -c SHA256SUMS
```

Render a local MIDI file or start the browser listening service:

```bash
python scripts/render_onnx.py --model-id paper_ir --midi path/to/input.mid --output output.wav
python scripts/realtime_midi_server.py --host 0.0.0.0 --port 8765
```

The browser UI lists every installed release model and resets neural and DSP
state whenever the model is changed.

## Deployment Boundary

The exported graph is FP32 ONNX opset 13 with fixed batch 1, one 250 Hz frame,
16 voices, 16 kHz audio, and explicit recurrent state. Harmonic phase,
filtered-noise synthesis, reverb, and the one-second MIDI release state remain
in the host.

This repository validates PyTorch CPU and ONNX Runtime behavior. It does not
convert or validate OM models. Ascend 310B/CANN conversion, memory, numerical,
and real-time tests belong in the downstream deployment repository.

## Documentation

- [Models and release assets](docs/models.md)
- [Training and ONNX export](docs/training-and-export.md)
- [Standard evaluation and local MIDI corpus](docs/evaluation.md)
- [Real-time browser player](docs/realtime.md)
- [Ascend 310B handoff contract](docs/ascend-310b.md)
- [Upstream provenance and licenses](docs/provenance.md)

Project additions are MIT-licensed. Adapted DDSP-Piano portions retain their
Apache-2.0 obligations; see [LICENSE](LICENSE) and
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).
