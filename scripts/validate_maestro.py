#!/usr/bin/env python3
"""Check that an extracted MAESTRO directory contains aligned MIDI and audio files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.maestro import validate_maestro


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--splits", default="train,validation,test")
    args = parser.parse_args()
    splits = [value.strip() for value in args.splits.split(",") if value.strip()]
    report = validate_maestro(args.maestro_root, splits)
    print(json.dumps(report, indent=2))
    return int(any(value for key, value in report.items() if key.endswith("missing_audio") or key.endswith("missing_midi")))


if __name__ == "__main__":
    raise SystemExit(main())
