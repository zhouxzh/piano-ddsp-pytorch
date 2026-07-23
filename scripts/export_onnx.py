#!/usr/bin/env python3
"""Export and numerically verify the stateful DDSP piano control model."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.default_model import get_model, get_v2_model
from ddsp_piano.deployment import PianoRealtimeControlModel


INPUT_NAMES = [
    "conditioning",
    "pedal",
    "piano_model",
    "extended_pitch",
    "context_state",
    "monophonic_state",
]
CURRENT_OUTPUT_NAMES = [
    "amplitudes",
    "harmonic_distribution",
    "inharmonicity",
    "f0_hz",
    "noise_magnitudes",
    "reverb_ir",
    "next_context_state",
    "next_monophonic_state",
]
V2_OUTPUT_NAMES = [
    "amplitudes",
    "harmonic_distribution",
    "inharmonicity",
    "f0_hz",
    "noise_magnitudes",
    "reverb_controls",
    "next_context_state",
    "next_monophonic_state",
]


def _checkpoint_value(checkpoint: dict, name: str, fallback: int) -> int:
    value = checkpoint.get("args", {}).get(name, fallback)
    return int(value)


def _shape(value: torch.Tensor | np.ndarray) -> list[int]:
    return [int(size) for size in value.shape]


def _comparison(
    reference: tuple[torch.Tensor, ...],
    actual: list[np.ndarray],
    output_names: list[str],
    atol: float,
    rtol: float,
) -> dict[str, dict[str, float | bool]]:
    report: dict[str, dict[str, float | bool]] = {}
    for name, torch_value, ort_value in zip(output_names, reference, actual):
        expected = torch_value.detach().cpu().numpy()
        absolute = np.abs(expected - ort_value)
        relative = absolute / np.maximum(np.abs(expected), 1e-6)
        report[name] = {
            "allclose": bool(np.allclose(expected, ort_value, atol=atol, rtol=rtol)),
            "max_abs": float(absolute.max(initial=0.0)),
            "max_rel": float(relative.max(initial=0.0)),
        }
    return report


def _merge_comparison(
    aggregate: dict[str, dict[str, float | bool]],
    current: dict[str, dict[str, float | bool]],
) -> dict[str, dict[str, float | bool]]:
    for name, values in current.items():
        if name not in aggregate:
            aggregate[name] = dict(values)
            continue
        aggregate[name]["allclose"] = bool(aggregate[name]["allclose"] and values["allclose"])
        aggregate[name]["max_abs"] = max(float(aggregate[name]["max_abs"]), float(values["max_abs"]))
        aggregate[name]["max_rel"] = max(float(aggregate[name]["max_rel"]), float(values["max_rel"]))
    return aggregate


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("exports/piano_controls.onnx"))
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--frame-rate", type=int)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--max-polyphony", type=int)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--verify-steps", type=int, default=4)
    parser.add_argument("--model-variant", choices=("auto", "current", "v2"), default="auto")
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.verify_steps <= 0:
        raise ValueError("--verify-steps must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    piano_models = checkpoint.get("piano_models")
    if not piano_models:
        raise ValueError("Checkpoint does not contain piano_models")

    sample_rate = args.sample_rate or _checkpoint_value(checkpoint, "sample_rate", 16_000)
    frame_rate = args.frame_rate or _checkpoint_value(checkpoint, "frame_rate", 250)
    max_polyphony = args.max_polyphony or _checkpoint_value(checkpoint, "max_polyphony", 16)
    if sample_rate % frame_rate:
        raise ValueError("sample rate must be divisible by frame rate")

    checkpoint_variant = checkpoint.get("args", {}).get("model_variant", "current")
    model_variant = checkpoint_variant if args.model_variant == "auto" else args.model_variant
    model_builder = get_v2_model if model_variant == "v2" else get_model
    output_names = V2_OUTPUT_NAMES if model_variant == "v2" else CURRENT_OUTPUT_NAMES
    model = model_builder(
        inference=True,
        n_synths=max_polyphony,
        n_piano_models=len(piano_models),
        sample_rate=sample_rate,
        duration=args.frames / frame_rate,
        frame_rate=frame_rate,
    )
    model.load_state_dict(checkpoint["model"])
    phase = int(checkpoint.get("args", {}).get("phase", 1))
    model.alternate_training(first_phase=phase == 1)
    model.eval()
    export_model = PianoRealtimeControlModel(model).eval()

    conditioning = torch.zeros(1, args.frames, max_polyphony, 2, dtype=torch.float32)
    conditioning[:, :, 0, 0] = 60.0
    conditioning[:, 0, 0, 1] = 0.8
    pedal = torch.zeros(1, args.frames, 4, dtype=torch.float32)
    piano_model = torch.zeros(1, dtype=torch.int32)
    extended_pitch = conditioning[..., :1].clone()
    context_state = torch.zeros(1, 1, model.context_network.gru.hidden_size, dtype=torch.float32)
    monophonic_state = torch.zeros(
        1,
        max_polyphony,
        model.monophonic_network.gru.hidden_size,
        dtype=torch.float32,
    )
    inputs = (
        conditioning,
        pedal,
        piano_model,
        extended_pitch,
        context_state,
        monophonic_state,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        export_model,
        inputs,
        args.output,
        input_names=INPUT_NAMES,
        output_names=output_names,
        opset_version=args.opset,
        do_constant_folding=True,
        dynamo=False,
    )
    onnx_model = onnx.load(str(args.output))
    onnx.checker.check_model(onnx_model)

    session = ort.InferenceSession(str(args.output), providers=["CPUExecutionProvider"])
    torch_inputs = list(inputs)
    ort_inputs = {name: value.detach().cpu().numpy() for name, value in zip(INPUT_NAMES, inputs)}
    comparison: dict[str, dict[str, float | bool]] = {}
    with torch.inference_mode():
        for _ in range(args.verify_steps):
            torch_outputs = export_model(*torch_inputs)
            ort_outputs = session.run(output_names, ort_inputs)
            comparison = _merge_comparison(
                comparison,
                _comparison(torch_outputs, ort_outputs, output_names, args.atol, args.rtol),
            )
            torch_inputs[-2] = torch_outputs[-2]
            torch_inputs[-1] = torch_outputs[-1]
            ort_inputs["context_state"] = ort_outputs[-2]
            ort_inputs["monophonic_state"] = ort_outputs[-1]
    if not all(result["allclose"] for result in comparison.values()):
        raise RuntimeError(f"PyTorch/ONNX numerical comparison failed: {comparison}")

    input_contract = {name: _shape(value) for name, value in zip(INPUT_NAMES, inputs)}
    output_contract = {name: _shape(value) for name, value in zip(output_names, ort_outputs)}
    operator_counts = dict(sorted(Counter(node.op_type for node in onnx_model.graph.node).items()))
    parameter_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    metadata = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.output),
        "opset": args.opset,
        "dtype": "FP32",
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "frames_per_call": args.frames,
        "audio_samples_per_call": args.frames * (sample_rate // frame_rate),
        "release_frames": frame_rate,
        "training_phase": phase,
        "model_variant": model_variant,
        "n_harmonics": output_contract["harmonic_distribution"][-1],
        "n_noise_bands": output_contract["noise_magnitudes"][-1],
        "n_substrings": output_contract["f0_hz"][-1],
        "piano_model_index_to_maestro_year": piano_models,
        "inputs": input_contract,
        "outputs": output_contract,
        "onnx_runtime_comparison": comparison,
        "onnx_runtime_stateful_steps": args.verify_steps,
        "reverb_output": output_names[5],
        "reverb_wet_gain": float(getattr(model.reverb_module, "wet_gain", 1.0)),
        "reverb_ir_postprocess": (
            {
                "embedded_in_onnx": True,
                "type": "exponential_decay",
                "decay_start_samples": 16_000,
                "decay_exponent": 4.0,
                "reference": "ddsp_piano.modules.sub_modules.MultiInstrumentReverb",
            }
            if model_variant != "v2"
            else {
                "embedded_in_onnx": False,
                "type": "fdn",
                "control_shape": output_contract["reverb_controls"],
                "delays_samples": [149, 211, 263, 293],
                "reference": "ddsp_piano.ddsp_pytorch.fdn.FDNReverb",
            }
        ),
        "operator_counts": operator_counts,
        "parameter_bytes": parameter_bytes,
        "onnx_file_bytes": args.output.stat().st_size,
        "dsp_boundary": (
            "The ONNX graph predicts synthesis controls. Harmonic phase accumulation, "
            "filtered-noise synthesis, reverb processing, and the 1-second MIDI release "
            "state run in the deployment host."
        ),
        "control_postprocess_reference": "ddsp_piano.deployment.scale_controls_for_synthesis",
        "validation_scope": (
            "PyTorch CPU reference, onnx.checker, and ONNX Runtime stateful numerical "
            "equivalence are required for this server. OM/CANN validation is out of scope."
        ),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"onnx": str(args.output), "metadata": str(metadata_path), "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
