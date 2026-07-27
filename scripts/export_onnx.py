#!/usr/bin/env python3
"""Export and numerically verify the stateful DDSP piano control model."""

from __future__ import annotations

import argparse
import hashlib
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

from ddsp_piano.default_model import build_configurable_model, build_paper_model
from ddsp_piano.deployment import PianoRealtimeControlModel
from ddsp_piano.model_registry import load_model_registry


INPUT_NAMES = [
    "conditioning",
    "pedal",
    "piano_model",
    "extended_pitch",
    "context_state",
    "monophonic_state",
]
IR_OUTPUT_NAMES = [
    "amplitudes",
    "harmonic_distribution",
    "inharmonicity",
    "f0_hz",
    "noise_magnitudes",
    "reverb_ir",
    "next_context_state",
    "next_monophonic_state",
]
FDN_OUTPUT_NAMES = [
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


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


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
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-rate", type=int)
    parser.add_argument("--frame-rate", type=int)
    parser.add_argument("--frames", type=int, default=1)
    parser.add_argument("--max-polyphony", type=int)
    parser.add_argument("--opset", type=int, default=13)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--verify-steps", type=int, default=100)
    parser.add_argument("--model-id", required=True)
    args = parser.parse_args()

    if args.frames <= 0:
        raise ValueError("--frames must be positive")
    if args.verify_steps <= 0:
        raise ValueError("--verify-steps must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    registry = load_model_registry()
    model_spec = registry.require(args.model_id)
    piano_models = checkpoint.get("piano_models")
    if not piano_models:
        raise ValueError("Checkpoint does not contain piano_models")

    sample_rate = args.sample_rate or _checkpoint_value(checkpoint, "sample_rate", 16_000)
    frame_rate = args.frame_rate or _checkpoint_value(checkpoint, "frame_rate", 250)
    max_polyphony = args.max_polyphony or _checkpoint_value(checkpoint, "max_polyphony", 16)
    if sample_rate % frame_rate:
        raise ValueError("sample rate must be divisible by frame rate")

    checkpoint_args = checkpoint.get("args", {})
    model_config = model_spec.model
    n_harmonics = int(model_config["n_harmonics"])
    n_noise_bands = int(model_config["n_noise_bands"])
    reverb_type = str(model_config["reverb_type"])
    context_type = str(model_config["context_type"])
    monophonic_type = str(model_config["monophonic_type"])
    inharmonicity_type = str(model_config["inharmonicity_type"])
    reverb_wet_gain_arg = float(model_config["reverb_wet_gain"])
    model_builder = (
        build_configurable_model
        if model_spec.architecture == "configurable"
        else build_paper_model
    )
    output_names = FDN_OUTPUT_NAMES if reverb_type == "fdn" else IR_OUTPUT_NAMES
    model_kwargs = dict(
        inference=True,
        n_synths=max_polyphony,
        n_piano_models=len(piano_models),
        sample_rate=sample_rate,
        duration=args.frames / frame_rate,
        frame_rate=frame_rate,
        reverb_wet_gain=reverb_wet_gain_arg,
    )
    if model_spec.architecture == "configurable":
        model_kwargs.update(
            n_harmonics=n_harmonics,
            n_noise_filter_banks=n_noise_bands,
            reverb_type=reverb_type,
            context_type=context_type,
            monophonic_type=monophonic_type,
            inharmonicity_type=inharmonicity_type,
        )
    model = model_builder(**model_kwargs)
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
    reverb_wet_gain = float(getattr(model.reverb_module, "wet_gain", 1.0))
    loss_calibration = checkpoint.get("loss_calibration")
    if isinstance(loss_calibration, dict):
        loss_calibration = dict(loss_calibration)
        loss_calibration.pop("quality_manifest", None)
    initialization = checkpoint.get("initialization")
    if isinstance(initialization, dict):
        initialization = {
            key: initialization[key]
            for key in ("checkpoint_sha256", "loaded_tensors", "target_tensors")
            if key in initialization
        }
    metadata = {
        "schema": "ddsp-piano-model/v1",
        "model_suite_release": registry.release,
        "model_id": model_spec.model_id,
        "display_name": model_spec.display_name,
        "description": model_spec.description,
        "architecture": model_spec.architecture,
        "lineage": model_spec.lineage,
        "checkpoint": f"{model_spec.asset_basename}.pt",
        "checkpoint_sha256": _sha256_file(args.checkpoint),
        "onnx": args.output.name,
        "artifact_name": model_spec.asset_basename,
        "opset": args.opset,
        "dtype": "FP32",
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "frames_per_call": args.frames,
        "audio_samples_per_call": args.frames * (sample_rate // frame_rate),
        "release_frames": frame_rate,
        "training_phase": phase,
        "onnx_status": "verified",
        "om_status": "pending",
        "quality_status": model_spec.quality_status,
        "host_dsp_profile": (
            f"learned-ir-wet-{reverb_wet_gain:g}"
            if reverb_type == "ir"
            else "fdn-controls"
        ),
        "model_config": {
            "n_harmonics": n_harmonics,
            "n_noise_bands": n_noise_bands,
            "reverb_type": reverb_type,
            "context_type": context_type,
            "monophonic_type": monophonic_type,
            "inharmonicity_type": inharmonicity_type,
            "loss_version": str(checkpoint_args.get("loss_version", "legacy")),
            "energy_loss_weight": float(checkpoint_args.get("energy_loss_weight", 0.0)),
            "onset_loss_weight": float(checkpoint_args.get("onset_loss_weight", 0.0)),
            "velocity_loss_weight": float(
                checkpoint_args.get("velocity_loss_weight", 0.0)
            ),
            "loss_calibration": loss_calibration,
            "initialization": initialization,
            "validation_corpus_sha256": checkpoint.get("validation_corpus_sha256"),
        },
        "n_harmonics": output_contract["harmonic_distribution"][-1],
        "n_noise_bands": output_contract["noise_magnitudes"][-1],
        "n_substrings": output_contract["f0_hz"][-1],
        "piano_model_index_to_maestro_year": piano_models,
        "inputs": input_contract,
        "outputs": output_contract,
        "onnx_runtime_comparison": comparison,
        "onnx_runtime_stateful_steps": args.verify_steps,
        "reverb_output": output_names[5],
        "reverb_wet_gain": reverb_wet_gain,
        "reverb_ir_postprocess": (
            {
                "embedded_in_onnx": True,
                "type": "exponential_decay",
                "decay_start_samples": 16_000,
                "decay_exponent": 4.0,
            "reference": "ddsp_piano.modules.sub_modules.MultiInstrumentReverb",
            }
            if reverb_type == "ir"
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
            "equivalence are required in this repository. OM/CANN validation is out of scope."
        ),
    }
    metadata_path = args.output.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({"onnx": str(args.output), "metadata": str(metadata_path), "comparison": comparison}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
