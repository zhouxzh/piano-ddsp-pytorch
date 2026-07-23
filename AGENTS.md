# AI Engineering Instructions

## Deployment Target

The model trained in this repository is intended to be deployed on an
Ascend 310B device after training. Treat Ascend 310B compatibility as a
first-class requirement for every model, preprocessing, export, and
configuration change.

The NVIDIA GPU is used for development and training only. CUDA-specific code
is acceptable in the training path when it improves training performance, but
the exported inference graph must not depend on CUDA, PyTorch autograd, or
training-only functionality.

## Required Constraints

- Keep the inference model exportable to ONNX with a fixed, documented input
  contract and output contract.
- Prefer ONNX/Ascend-supported operators and conservative tensor operations.
  Check every new operation against the target Ascend 310B/CANN toolchain.
- Avoid dynamic Python control flow, data-dependent module creation, custom
  autograd functions, unsupported FFT/STFT behavior, and runtime filesystem or
  network access in the deployable forward path.
- Keep preprocessing and postprocessing explicit and reproducible. Document
  sample rate, frame rate, segment length, MIDI conditioning layout, tensor
  shapes, dtypes, and normalization in the repository.
- Preserve a CPU reference path for numerical comparison with the exported
  ONNX model before Ascend deployment.
- Do not assume CUDA AMP, CUDA-only kernels, or NVIDIA-specific libraries are
  available on the Ascend device. AMP changes are training-only unless their
  exported numerical behavior is verified.
- Ascend 310B deployment in this project targets FP16 or FP32 only. Do not add
  BF16 model, export, calibration, or deployment paths because the target
  device does not support BF16.
- Keep model dimensions, polyphony limits, harmonic counts, FFT sizes, and
  sequence lengths configurable only when the ONNX export and deployment
  memory budget are checked for each variant.
- When changing model code or defaults, inspect ONNX exportability and Ascend
  memory/performance impact before considering the change complete.

## Required Validation Before Handoff

For changes affecting the model or data path, run or update checks covering:

1. PyTorch CPU reference inference.
2. ONNX export and ONNX graph validation.
3. Numerical comparison between PyTorch and ONNX outputs.
4. Ascend 310B/CANN compatibility, including unsupported operators and memory
   use, when the target toolchain is available.
5. Documentation of the exact export command, input shapes, output shapes,
   opset, dtype, and deployment assumptions.

If the Ascend toolchain is unavailable, state that explicitly and do not claim
that the model is deployment-ready based only on successful CUDA training.

## Scope Discipline

Keep training-only code separate from deployment-facing code. Do not remove
or weaken the Ascend constraints merely to make a CUDA smoke test pass.
Update `README.md`, export scripts, configuration defaults, and this file when
the deployment contract changes.

## Current Deployment Contract

- Export `scripts/export_onnx.py` as FP32 ONNX opset 13.
- Use fixed batch 1, one 250 Hz control frame, 16 polyphony slots, and a
  16 kHz sample rate (64 audio samples per call).
- Carry the context and monophonic GRU states explicitly between calls.
- Maintain the one-second MIDI release state in host preprocessing using
  `ddsp_piano.deployment.extend_pitch_for_release` as the CPU reference.
- Keep harmonic phase accumulation, filtered-noise FFT synthesis, and reverb
  convolution outside the Ascend ONNX graph until the exact CANN version has
  verified those operators and the resulting memory use.
- Treat the JSON emitted beside the ONNX file as the machine-readable input,
  output, dtype, operator, size, and numerical-comparison contract.
