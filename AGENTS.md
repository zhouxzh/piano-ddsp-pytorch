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

## Non-Negotiable Product Goal

Every ONNX model produced by this repository is an intermediate deployment
artifact, not the final product. The intended final model artifact is an OM
model converted from ONNX and executed on an Ascend 310B device as part of a
real-time, causal MIDI-to-piano synthesis system.

This repository's responsibility ends at training, exporting, and rigorously
validating the ONNX model and its CPU reference path. It is the ONNX model
development and test repository; it is not the final OM conversion or Ascend
310B device-validation repository. Actual ATC conversion, OM artifact
validation, CANN integration, and Ascend 310B on-device numerical, memory,
latency, and realtime tests belong in the user's Ascend 310B deployment example
repository. Changes here must deliver the ONNX model, adjacent machine-readable
JSON contract, CPU reference behavior, and reproducible handoff information
needed by that downstream repository.

Treat the complete target pipeline as:

```text
MIDI events
  -> host preprocessing, pedal handling, and polyphonic voice allocation
  -> Ascend 310B OM neural control inference with explicit recurrent state
  -> stateful host harmonic, filtered-noise, and reverb DSP
  -> continuous audio output
```

- Use the upstream Ascend CANN samples and the project's documented successful
  DDSP-VST-to-OM case as implementation references. Reuse applicable CANN,
  device-memory, model-loading, execution, and audio integration patterns, but
  do not assume that a reference model's successful conversion proves that a
  new piano graph is compatible.
- A model architecture is acceptable only if it preserves a credible,
  documented ONNX-to-OM path for Ascend 310B. Successful PyTorch, CUDA, or ONNX
  Runtime execution is necessary development evidence, but is not the final
  deployment result.
- The deployed system must be causal and continuously stateful. It must not
  require future MIDI events, full-song preprocessing, dynamic graph changes,
  or non-deterministic training code in the inference path.
- Real-time readiness includes the complete MIDI-to-audio path, not only neural
  inference latency. Validate voice allocation, pedal and note-release state,
  cross-block phase/noise/reverb continuity, audio-thread scheduling, buffer
  underruns, and end-to-end P95/P99/P99.9 latency before calling the system
  real-time ready.
- Validation in this repository stops at the ONNX boundary. Preserve
  conversion-ready artifacts and reproducible contracts, mark OM/device
  validation as a downstream responsibility, and never reinterpret this
  repository boundary as changing the final Ascend 310B product goal.

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
4. Static review of the ONNX graph against the known Ascend 310B/CANN
   constraints and the pinned deployment examples, including an operator and
   size inventory for downstream conversion.
5. Documentation of the exact export command, input shapes, output shapes,
   opset, dtype, and deployment assumptions.

Do not use this repository as proof of OM or Ascend 310B device validation, even
if an Ascend toolchain happens to be installed on the development server. Report
a passing artifact as "ONNX validated and ready for downstream conversion",
not "Ascend deployment-ready". Successful CUDA training or ONNX Runtime
inference does not waive the downstream OM conversion and Ascend 310B real-time
system validation requirements.

## Scope Discipline

Keep training-only code separate from deployment-facing code. Do not remove
or weaken the Ascend constraints merely to make a CUDA smoke test pass.
Update `README.md`, export scripts, configuration defaults, and this file when
the deployment contract changes.

## Current Deployment Contract

- Treat ONNX as the portable conversion input and OM as the final neural-model
  artifact executed by the Ascend 310B deployment example.
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
