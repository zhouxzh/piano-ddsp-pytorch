#!/usr/bin/env python3
"""Build the train-only curriculum and MIDI-velocity calibration manifest."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.maestro import MaestroSegmentDataset, PreprocessConfig
from ddsp_piano.training_quality import build_quality_manifest, write_quality_manifest


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--maestro-root", type=Path, required=True)
    result.add_argument("--cache-dir", type=Path, required=True)
    result.add_argument("--output", type=Path, required=True)
    result.add_argument("--seed", type=int, default=20260725)
    result.add_argument("--sample-rate", type=int, default=16_000)
    result.add_argument("--frame-rate", type=int, default=250)
    result.add_argument("--segment-seconds", type=float, default=3.0)
    result.add_argument("--overlap", type=float, default=0.5)
    result.add_argument("--max-polyphony", type=int, default=16)
    return result


def main() -> int:
    args = parser().parse_args()
    config = PreprocessConfig(
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        segment_seconds=args.segment_seconds,
        overlap=args.overlap,
        max_polyphony=args.max_polyphony,
    )
    dataset = MaestroSegmentDataset(
        args.maestro_root, args.cache_dir, "train", config, require_cache=True
    )
    manifest = build_quality_manifest(dataset, args.sample_rate, args.frame_rate, args.seed)
    write_quality_manifest(args.output, manifest)
    print(
        json.dumps(
            {
                "output": str(args.output.resolve()),
                "entries": manifest["entry_count"],
                "dataset_index_sha256": manifest["dataset_index_sha256"],
                "global_velocity_slope": manifest["velocity_response"]["global_slope"],
            }
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
