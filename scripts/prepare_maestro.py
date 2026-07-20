#!/usr/bin/env python3
"""Precompute disk-backed MAESTRO caches for PyTorch DDSP-Piano training."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.maestro import PreprocessConfig, prepare_split


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache")
    parser.add_argument("--splits", default="train,validation,test")
    parser.add_argument("--limit-tracks", type=int)
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--frame-rate", type=int, default=250)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-polyphony", type=int, default=16)
    args = parser.parse_args()
    config = PreprocessConfig(
        args.sample_rate,
        args.frame_rate,
        args.segment_seconds,
        args.overlap,
        args.max_polyphony,
    )
    config.validate()
    for split in (value.strip() for value in args.splits.split(",")):
        if split:
            print(split, prepare_split(args.maestro_root, args.cache_dir, split, config, args.limit_tracks))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
