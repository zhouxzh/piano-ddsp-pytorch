#!/usr/bin/env python3
"""Verify that a v3 candidate starts numerically equivalent to its parent."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.default_model import get_v3_model
from ddsp_piano.deployment import PianoRealtimeControlModel
from ddsp_piano.evaluation import sha256_file
from train import build_control_anchor, load_partial_initialization


OUTPUT_NAMES = (
    "amplitudes",
    "harmonic_distribution",
    "inharmonicity",
    "f0_hz",
    "noise_magnitudes",
    "reverb_ir",
    "next_context_state",
    "next_monophonic_state",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--conditioning-gate",
        choices=("none", "velocity_onset"),
        required=True,
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--atol", type=float, default=1e-5)
    args = parser.parse_args()
    if args.steps <= 0:
        raise ValueError("--steps must be positive")

    checkpoint = torch.load(
        args.parent_checkpoint, map_location="cpu", weights_only=False
    )
    piano_models = checkpoint.get("piano_models")
    if not piano_models:
        raise ValueError("Parent checkpoint does not contain piano_models")
    parent = build_control_anchor(
        args.parent_checkpoint, torch.device("cpu"), 1 / 250, "serial"
    )
    candidate = get_v3_model(
        duration=1 / 250,
        n_synths=16,
        n_piano_models=len(piano_models),
        frame_rate=250,
        sample_rate=16_000,
        conditioning_gate=args.conditioning_gate,
    )
    initialization = load_partial_initialization(
        candidate, args.parent_checkpoint, torch.device("cpu")
    )
    parent.alternate_training(first_phase=True)
    candidate.alternate_training(first_phase=True)
    parent.eval()
    candidate.eval()
    parent_wrapper = PianoRealtimeControlModel(parent).eval()
    candidate_wrapper = PianoRealtimeControlModel(candidate).eval()

    generator = torch.Generator().manual_seed(20260726)
    parent_context = torch.zeros(1, 1, 64)
    candidate_context = torch.zeros_like(parent_context)
    parent_mono = torch.zeros(1, 16, 192)
    candidate_mono = torch.zeros_like(parent_mono)
    maxima = {name: 0.0 for name in OUTPUT_NAMES}
    with torch.inference_mode():
        for step in range(args.steps):
            conditioning = torch.zeros(1, 1, 16, 2)
            active = 1 + step % 4
            conditioning[0, 0, :active, 0] = torch.randint(
                36, 85, (active,), generator=generator
            ).float()
            conditioning[0, 0, :active, 1] = torch.rand(
                active, generator=generator
            )
            extended_pitch = conditioning[..., :1].clone()
            pedal = torch.rand(1, 1, 4, generator=generator) * 0.2
            piano_model = torch.tensor([step % len(piano_models)], dtype=torch.int32)
            parent_outputs = parent_wrapper(
                conditioning,
                pedal,
                piano_model,
                extended_pitch,
                parent_context,
                parent_mono,
            )
            candidate_outputs = candidate_wrapper(
                conditioning,
                pedal,
                piano_model,
                extended_pitch,
                candidate_context,
                candidate_mono,
            )
            for name, expected, actual in zip(
                OUTPUT_NAMES, parent_outputs, candidate_outputs
            ):
                maxima[name] = max(
                    maxima[name], float((expected - actual).abs().max())
                )
            parent_context, parent_mono = parent_outputs[-2:]
            candidate_context, candidate_mono = candidate_outputs[-2:]

    passed = all(value <= args.atol for value in maxima.values())
    report = {
        "schema": "ddsp-piano-v3-initialization-check/v1",
        "parent_checkpoint": str(args.parent_checkpoint.resolve()),
        "parent_checkpoint_sha256": sha256_file(args.parent_checkpoint),
        "conditioning_gate": args.conditioning_gate,
        "steps": args.steps,
        "atol": args.atol,
        "max_abs": maxima,
        "passed": passed,
        "initialization": initialization,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not passed:
        raise RuntimeError(f"v3 initialization equivalence failed: {maxima}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
