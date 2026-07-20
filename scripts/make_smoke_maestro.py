#!/usr/bin/env python3
"""Create a tiny synthetic MAESTRO-shaped dataset for pipeline smoke tests."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import mido
import torch
import torchaudio


def write_track(root: Path, name: str, phase: float) -> tuple[str, str]:
    relative_dir = Path("2004")
    output_dir = root / relative_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    audio_path = output_dir / f"{name}.wav"
    midi_path = output_dir / f"{name}.midi"
    sample_rate = 16_000
    duration = 4.0
    time = torch.arange(int(sample_rate * duration), dtype=torch.float32) / sample_rate
    audio = 0.15 * torch.sin(2.0 * math.pi * 261.63 * time + phase)
    audio += 0.10 * torch.sin(2.0 * math.pi * 329.63 * time + phase)
    torchaudio.save(str(audio_path), audio.unsqueeze(0), sample_rate)

    midi = mido.MidiFile(ticks_per_beat=480)
    track = mido.MidiTrack()
    midi.tracks.append(track)
    track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
    track.append(mido.Message("note_on", note=60, velocity=96, time=0))
    track.append(mido.Message("note_on", note=64, velocity=80, time=0))
    track.append(mido.Message("control_change", control=64, value=127, time=480))
    track.append(mido.Message("note_off", note=60, velocity=0, time=480))
    track.append(mido.Message("note_off", note=64, velocity=0, time=0))
    track.append(mido.Message("control_change", control=64, value=0, time=480))
    track.append(mido.MetaMessage("end_of_track", time=0))
    midi.save(str(midi_path))
    return str(relative_dir / audio_path.name), str(relative_dir / midi_path.name)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/smoke_maestro/maestro-v3.0.0"))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    train_audio, train_midi = write_track(args.output, "smoke_train", 0.0)
    validation_audio, validation_midi = write_track(args.output, "smoke_validation", 0.2)
    with (args.output / "maestro-v3.0.0.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "year", "midi_filename", "audio_filename"])
        writer.writeheader()
        writer.writerow({"split": "train", "year": 2004, "midi_filename": train_midi, "audio_filename": train_audio})
        writer.writerow({"split": "validation", "year": 2004, "midi_filename": validation_midi, "audio_filename": validation_audio})
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
