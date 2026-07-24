#!/usr/bin/env python3
"""Render identical cached MAESTRO segments through two ONNX exports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.maestro import MaestroSegmentDataset, PreprocessConfig
from scripts.render_onnx import _run_onnx, _shape, _synthesize, _write_normalized_wav


def _rms(value: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))


def _spectral_l1(prediction: np.ndarray, target: np.ndarray) -> float:
    size = min(prediction.size, target.size)
    pred = np.abs(np.fft.rfft(prediction[:size]))
    truth = np.abs(np.fft.rfft(target[:size]))
    return float(np.mean(np.abs(pred - truth)) / max(float(np.mean(truth)), 1e-7))


def _render_segment(model_path: Path, metadata: dict, segment, warm_up_seconds: float, seed: int):
    audio, conditioning, pedal, piano_model = segment
    conditioning = conditioning.numpy().astype(np.float32, copy=False)
    pedal = pedal.numpy().astype(np.float32, copy=False)
    piano_id = int(piano_model.item())
    sample_rate = int(metadata["sample_rate"])
    frame_rate = int(metadata["frame_rate"])
    max_polyphony = int(_shape(metadata, "inputs", "conditioning")[2])
    samples_per_frame = sample_rate // frame_rate
    warm_up_frames = int(round(warm_up_seconds * frame_rate))
    if warm_up_frames:
        conditioning = np.pad(conditioning, ((warm_up_frames, 0), (0, 0), (0, 0)))
        pedal = np.pad(pedal, ((warm_up_frames, 0), (0, 0)))
    reverb_name = str(metadata.get("reverb_output", "reverb_ir"))
    reverb_type = str(metadata.get("reverb_ir_postprocess", {}).get("type", "ir"))
    reverb_wet_gain = float(metadata.get("reverb_wet_gain", 1.0))
    controls, reverb_condition = _run_onnx(
        model_path,
        metadata,
        conditioning,
        pedal,
        piano_id,
        reverb_name,
    )
    signals = _synthesize(
        controls,
        reverb_condition,
        sample_rate,
        samples_per_frame,
        seed,
        reverb_type,
        reverb_wet_gain,
    )
    if warm_up_frames:
        warm_up_samples = warm_up_frames * samples_per_frame
        signals = {name: value[warm_up_samples:] for name, value in signals.items()}
    target = audio.numpy().astype(np.float32, copy=False)
    return signals, target, piano_id


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--v1",
        "--current",
        dest="v1",
        type=Path,
        required=True,
        help="v1 ONNX export (--current remains a compatibility alias)",
    )
    parser.add_argument("--v2", type=Path, required=True)
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("exports/model_comparison"))
    parser.add_argument("--indices", default="0,-1", help="Comma-separated cached segment indices")
    parser.add_argument("--warm-up-seconds", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    v1_metadata = json.loads(args.v1.with_suffix(".json").read_text(encoding="utf-8"))
    v2_metadata = json.loads(args.v2.with_suffix(".json").read_text(encoding="utf-8"))
    config = PreprocessConfig(
        sample_rate=int(v1_metadata["sample_rate"]),
        frame_rate=int(v1_metadata["frame_rate"]),
        segment_seconds=3.0,
        max_polyphony=int(_shape(v1_metadata, "inputs", "conditioning")[2]),
    )
    dataset = MaestroSegmentDataset(
        args.maestro_root,
        args.cache_dir,
        "validation",
        config,
        require_cache=True,
    )
    indices = []
    for value in args.indices.split(","):
        index = int(value.strip())
        index = index if index >= 0 else len(dataset) + index
        if not 0 <= index < len(dataset):
            raise ValueError(f"segment index out of range: {value}")
        if index not in indices:
            indices.append(index)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report = {"segments": [], "models": {"v1": str(args.v1), "v2": str(args.v2)}}
    for segment_index in indices:
        segment = dataset[segment_index]
        entry = {"index": segment_index}
        for label, model_path, metadata in (
            ("v1", args.v1, v1_metadata),
            ("v2", args.v2, v2_metadata),
        ):
            signals, target, piano_id = _render_segment(
                model_path, metadata, segment, args.warm_up_seconds, args.seed
            )
            output = args.output_dir / f"segment_{segment_index:05d}_{label}.wav"
            wav_report = _write_normalized_wav(output, signals["wet"], int(metadata["sample_rate"]))
            entry[label] = {
                "output": str(output),
                "piano_model": piano_id,
                "wet_rms": _rms(signals["wet"]),
                "dry_rms": _rms(signals["dry"]),
                "harmonic_rms": _rms(signals["harmonic"]),
                "noise_rms": _rms(signals["noise"]),
                "wet_dry_rms_ratio": _rms(signals["wet"]) / max(_rms(signals["dry"]), 1e-7),
                "target_spectral_l1": _spectral_l1(signals["wet"], target),
                "wav": wav_report,
            }
        report["segments"].append(entry)

    report_path = args.output_dir / "comparison.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
