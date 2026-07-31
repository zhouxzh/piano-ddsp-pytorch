---
license: cc-by-nc-sa-4.0
library_name: onnxruntime
tags:
  - audio
  - midi
  - piano
  - ddsp
  - onnx
  - real-time
  - ascend-310
  - ascend-310b
datasets:
  - ddPn08/maestro-v3.0.0
---

# Piano DDSP for Ascend 310

This repository publishes four causal MIDI-conditioned DDSP piano control
models from `zhouxzh/piano-ddsp-pytorch`. The model IDs describe architecture,
not a quality ranking:

| Model ID | Architecture | Reverb output |
| --- | --- | --- |
| `gru_ir_96_64` | recurrent context and monophonic controls | learned IR |
| `film_fdn_128_96` | MAESTRO v2 FiLM/deep control networks | FDN controls |
| `gru_ir_fullwet_96_64` | recurrent controls with perceptual calibration | learned IR |
| `film_ir_fullwet_96_64` | FiLM/deep controls with perceptual calibration | learned IR |

## Intended Use

This repository is reserved for real-time MIDI piano synthesis models targeting
the Ascend 310 product family. The current `model-suite-v1.0.1` contract targets
Ascend 310B specifically. It does not claim compatibility with later 310
devices until a separate hardware and CANN validation is published. The ONNX
files are controlled conversion inputs for the downstream CANN/OM workflow.
PyTorch CPU and ONNX Runtime exist only for numerical comparison and
model-quality testing; other deployment platforms are outside the supported
scope.

## License

The published checkpoints, ONNX graphs, model parameters, and any later OM
derivatives are licensed under CC BY-NC-SA 4.0 for non-commercial use. MAESTRO
v3.0.0, used for training, is provided by Google LLC under the same license.
Apache-2.0 notices for the adapted DDSP-Piano implementation are retained in
`THIRD_PARTY_NOTICES.md` and `LICENSES/Apache-2.0.txt`. No MAESTRO audio, MIDI,
or local listening MIDI is distributed in this model repository.

The fixed graph contract is FP32 ONNX opset 13, batch 1, one 250 Hz control
frame, 16 voices, 16 kHz audio, and explicit recurrent state. Harmonic phase,
filtered-noise synthesis, reverb, and MIDI release handling remain in the host.

All four ONNX files passed PyTorch CPU reference comparison, ONNX validation,
and 100 stateful ONNX Runtime steps. This does not constitute OM/CANN or Ascend
310B device validation. See `VALIDATION.md`, each adjacent JSON contract, and
`model-suite.json` before conversion.

Download an immutable release and verify it:

```bash
HF_ENDPOINT=https://huggingface.co hf download zhouxzh/piano-ddsp-ascend310 \
  --revision model-suite-v1.0.1 \
  --local-dir artifacts/model-suite-v1.0.1
cd artifacts/model-suite-v1.0.1
sha256sum -c SHA256SUMS
```

Source, training instructions, ONNX export code, evaluation tools, and the
real-time browser player are maintained at
https://github.com/zhouxzh/piano-ddsp-pytorch.

The current release contains ONNX artifacts only. Ascend 310B OM artifacts may
be added by the downstream deployment project in a later immutable release,
after the exact CANN toolchain and target device have been validated.
