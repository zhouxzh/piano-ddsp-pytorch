import io
import tempfile
import unittest
import wave
from pathlib import Path

import mido
import numpy as np

from ddsp_piano.realtime import (
    LiveMidiState,
    MidiTimeline,
    PartitionedConvolver,
    ScheduledMidiEvent,
    VoiceReleaseEnvelope,
    apply_scheduled_midi_event,
    encode_wav_chunk,
    load_midi_timeline,
    restore_midi_timeline_state,
)


class RealtimeTest(unittest.TestCase):
    def test_midi_timeline_applies_tempo_changes_and_normalizes_note_off(self):
        midi = mido.MidiFile(ticks_per_beat=480)
        track = mido.MidiTrack()
        midi.tracks.append(track)
        track.append(mido.MetaMessage("set_tempo", tempo=500_000, time=0))
        track.append(mido.Message("note_on", note=60, velocity=96, time=0))
        track.append(mido.Message("control_change", control=64, value=127, time=480))
        track.append(mido.MetaMessage("set_tempo", tempo=1_000_000, time=0))
        track.append(mido.Message("note_on", note=60, velocity=0, time=480))
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "tempo.mid"
            midi.save(path)
            timeline = load_midi_timeline(path)

        self.assertAlmostEqual(timeline.duration_seconds, 1.5)
        self.assertEqual(
            [(event.kind, event.data1, event.data2) for event in timeline.events],
            [
                ("note_on", 60, 96),
                ("control_change", 64, 127),
                ("note_off", 60, 0),
            ],
        )
        np.testing.assert_allclose(
            [event.time_seconds for event in timeline.events], [0.0, 0.5, 1.5]
        )

    def test_live_midi_state_preserves_slots_and_consumes_onset_velocity(self):
        state = LiveMidiState(max_polyphony=2)
        self.assertTrue(state.note_on(60, 96))
        self.assertTrue(state.note_on(64, 64))

        conditioning, pedal = state.render_frames(3)

        np.testing.assert_array_equal(
            conditioning[:, :, 0], [[60, 64], [60, 64], [60, 64]]
        )
        np.testing.assert_allclose(conditioning[0, :, 1], [96 / 127, 64 / 127])
        np.testing.assert_array_equal(conditioning[1:, :, 1], 0.0)
        np.testing.assert_array_equal(pedal, 0.0)

        next_conditioning, _ = state.render_frames(1)
        np.testing.assert_array_equal(next_conditioning[0, :, 1], 0.0)

    def test_live_midi_state_applies_sustain_and_voice_stealing(self):
        state = LiveMidiState(max_polyphony=2)
        state.note_on(60, 100)
        state.note_on(64, 100)
        self.assertTrue(state.control_change(64, 127))
        self.assertTrue(state.note_off(60))
        self.assertEqual(state.snapshot().active_notes, (60, 64))

        state.note_on(67, 100)
        self.assertEqual(state.snapshot().active_notes, (64, 67))
        self.assertEqual(state.snapshot().voice_steals, 1)

        self.assertTrue(state.control_change(64, 0))
        self.assertEqual(state.snapshot().active_notes, (64, 67))
        self.assertFalse(state.snapshot().sustain)

        state.note_off(64)
        self.assertEqual(state.snapshot().active_notes, (67,))
        state.panic()
        self.assertEqual(state.snapshot().active_notes, ())

    def test_seek_restores_held_notes_and_pedal_state(self):
        timeline = MidiTimeline(
            duration_seconds=2.0,
            events=(
                ScheduledMidiEvent(0.0, "note_on", 60, 96),
                ScheduledMidiEvent(0.5, "control_change", 64, 127),
                ScheduledMidiEvent(1.0, "note_off", 60),
                ScheduledMidiEvent(1.25, "note_on", 64, 80),
                ScheduledMidiEvent(1.5, "control_change", 64, 0),
            ),
        )
        state = LiveMidiState(max_polyphony=4)

        next_event = restore_midi_timeline_state(state, timeline, 1.1)

        self.assertEqual(next_event, 3)
        self.assertEqual(state.snapshot().active_notes, (60,))
        self.assertTrue(state.snapshot().sustain)

        next_event = restore_midi_timeline_state(state, timeline, 1.75)

        self.assertEqual(next_event, len(timeline.events))
        self.assertEqual(state.snapshot().active_notes, (64,))
        self.assertFalse(state.snapshot().sustain)

    def test_scheduled_midi_event_rejects_unknown_kind(self):
        state = LiveMidiState(max_polyphony=2)
        with self.assertRaisesRegex(ValueError, "Unsupported scheduled MIDI event"):
            apply_scheduled_midi_event(
                state, ScheduledMidiEvent(0.0, "aftertouch", 60, 64)
            )

    def test_keyoff_gate_waits_for_sustain_release(self):
        state = LiveMidiState(max_polyphony=2)
        state.note_on(60, 100)
        _, _, gate = state.render_block(1)
        np.testing.assert_array_equal(gate, [True, False])

        state.control_change(64, 127)
        state.note_off(60)
        _, _, gate = state.render_block(1)
        np.testing.assert_array_equal(gate, [True, False])

        state.control_change(64, 0)
        conditioning, _, gate = state.render_block(1)
        np.testing.assert_array_equal(conditioning[0, :, 0], 0.0)
        np.testing.assert_array_equal(gate, [False, False])

    def test_voice_release_envelope_damps_only_released_voice(self):
        envelope = VoiceReleaseEnvelope(
            max_polyphony=2, sample_rate=1_000, release_ms=50.0
        )
        active = envelope.render(np.asarray([True, True]), n_samples=10)
        np.testing.assert_array_equal(active, 1.0)

        releasing = envelope.render(np.asarray([False, True]), n_samples=25)
        self.assertAlmostEqual(float(releasing[0, 0]), 0.98, places=6)
        self.assertAlmostEqual(float(releasing[0, -1]), 0.5, places=6)
        np.testing.assert_array_equal(releasing[1], 1.0)

        finished = envelope.render(np.asarray([False, True]), n_samples=25)
        self.assertAlmostEqual(float(finished[0, -1]), 0.0, places=6)
        np.testing.assert_array_equal(finished[1], 1.0)

    def test_partitioned_convolver_matches_linear_convolution_across_blocks(self):
        rng = np.random.default_rng(7)
        audio = rng.normal(size=32).astype(np.float32)
        impulse = rng.normal(size=19).astype(np.float32)
        convolver = PartitionedConvolver(impulse, block_size=8)

        rendered = np.concatenate(
            [convolver.process(audio[index : index + 8]) for index in range(0, 32, 8)]
        )
        expected = np.convolve(audio, impulse)[: audio.size]

        np.testing.assert_allclose(rendered, expected, rtol=1e-5, atol=1e-5)

    def test_encode_wav_chunk_writes_self_contained_pcm16_wav(self):
        encoded, clipped = encode_wav_chunk(
            np.asarray([-1.5, -0.5, 0.0, 0.5, 1.5], dtype=np.float32),
            sample_rate=16_000,
        )

        self.assertEqual(encoded[:4], b"RIFF")
        self.assertEqual(encoded[8:12], b"WAVE")
        self.assertEqual(clipped, 2)
        with wave.open(io.BytesIO(encoded), "rb") as wav:
            self.assertEqual(wav.getnchannels(), 1)
            self.assertEqual(wav.getsampwidth(), 2)
            self.assertEqual(wav.getframerate(), 16_000)
            self.assertEqual(wav.getnframes(), 5)

    def test_panic_clears_all_live_midi_state(self):
        state = LiveMidiState(max_polyphony=2)
        state.note_on(60, 100)
        state.control_change(64, 127)
        state.note_off(60)

        state.panic()
        conditioning, pedal = state.render_frames(1)

        np.testing.assert_array_equal(conditioning, 0.0)
        np.testing.assert_array_equal(pedal, 0.0)


if __name__ == "__main__":
    unittest.main()
