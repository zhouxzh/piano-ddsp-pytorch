from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import mido

from ddsp_piano.maestro import load_midi_conditioning


class MidiConditioningTest(unittest.TestCase):
    def _write_midi(self, path: Path) -> None:
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        track.append(mido.Message("control_change", control=1, value=80, time=0))
        track.append(mido.Message("note_on", note=60, velocity=100, time=0))
        track.append(mido.Message("control_change", control=64, value=127, time=480))
        track.append(mido.Message("note_off", note=60, velocity=0, time=480))
        track.append(mido.Message("control_change", control=64, value=0, time=480))
        track.append(mido.MetaMessage("end_of_track", time=0))
        midi.save(path)

    def test_file_conditioning_ignores_unmapped_controls_and_applies_sustain(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            midi_path = Path(temporary) / "pedal.mid"
            self._write_midi(midi_path)
            result = load_midi_conditioning(
                midi_path,
                frame_rate=10,
                max_polyphony=2,
                tail_seconds=0.5,
            )

        self.assertEqual(result.conditioning.shape, (21, 2, 2))
        self.assertEqual(result.pedal.shape, (21, 4))
        self.assertAlmostEqual(result.conditioning[0, 0, 1], 100 / 127)
        self.assertEqual(result.conditioning[14, 0, 0], 60.0)
        self.assertEqual(result.conditioning[15, 0, 0], 0.0)
        self.assertEqual(result.pedal[5, 0], 1.0)
        self.assertEqual(result.pedal[15, 0], 0.0)
        self.assertEqual(result.max_observed_polyphony, 1)
        self.assertEqual(result.overflow_frames, 0)


if __name__ == "__main__":
    unittest.main()
