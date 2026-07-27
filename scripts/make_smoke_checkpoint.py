#!/usr/bin/env python3
"""Create a deterministic random checkpoint for ONNX export smoke tests."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.default_model import build_configurable_model, build_paper_model
from ddsp_piano.model_registry import load_model_registry


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(20260727)
    spec = load_model_registry().require(args.model_id)
    kwargs = {
        "n_synths": 16,
        "n_piano_models": 10,
        "sample_rate": 16_000,
        "frame_rate": 250,
        "duration": 3.0,
        "reverb_wet_gain": float(spec.model["reverb_wet_gain"]),
    }
    if spec.architecture == "configurable":
        kwargs.update(
            n_harmonics=int(spec.model["n_harmonics"]),
            n_noise_filter_banks=int(spec.model["n_noise_bands"]),
            reverb_type=str(spec.model["reverb_type"]),
            context_type=str(spec.model["context_type"]),
            monophonic_type=str(spec.model["monophonic_type"]),
            inharmonicity_type=str(spec.model["inharmonicity_type"]),
        )
        model = build_configurable_model(**kwargs)
    else:
        model = build_paper_model(**kwargs)
    model.alternate_training(first_phase=True)
    checkpoint_args = {
        "model_id": spec.model_id,
        "architecture": spec.architecture,
        "sample_rate": 16_000,
        "frame_rate": 250,
        "max_polyphony": 16,
        "phase": 1,
        **spec.model,
        **spec.training,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "args": checkpoint_args,
            "piano_models": [2004, 2006, 2008, 2009, 2011, 2013, 2014, 2015, 2017, 2018],
        },
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
