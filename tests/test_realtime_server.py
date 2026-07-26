import json
import unittest
from pathlib import Path

from ddsp_piano.realtime import LiveMidiState, MidiTimeline, ScheduledMidiEvent
from scripts.realtime_midi_server import ClientSession, MidiCatalogEntry


class FakeWebSocket:
    def __init__(self) -> None:
        self.closed = False
        self.messages: list[dict] = []

    async def send_str(self, message: str) -> None:
        self.messages.append(json.loads(message))


class FakeSynth:
    def __init__(self) -> None:
        self.midi = LiveMidiState(max_polyphony=4)
        self.reset_count = 0

    def hard_reset(self) -> None:
        self.reset_count += 1
        self.midi.panic()


class RealtimeServerTest(unittest.IsolatedAsyncioTestCase):
    def make_session(self) -> tuple[ClientSession, FakeWebSocket, FakeSynth]:
        timeline = MidiTimeline(
            duration_seconds=10.0,
            events=(
                ScheduledMidiEvent(0.0, "note_on", 60, 96),
                ScheduledMidiEvent(5.0, "control_change", 64, 127),
                ScheduledMidiEvent(8.0, "note_off", 60),
                ScheduledMidiEvent(9.0, "control_change", 64, 0),
            ),
        )
        entry = MidiCatalogEntry(
            id="test.mid",
            name="test",
            path=Path("test.mid"),
            timeline=timeline,
        )
        websocket = FakeWebSocket()
        session = ClientSession(websocket, object(), {entry.id: entry})
        synth = FakeSynth()
        session.synth = synth
        return session, websocket, synth

    async def test_transport_restores_state_across_pause_seek_and_resume(self):
        session, websocket, synth = self.make_session()

        await session.handle_event(
            {
                "type": "play_midi",
                "midi_id": "test.mid",
                "position_seconds": 2.0,
                "tempo_scale": 1.5,
                "loop": True,
            }
        )
        self.assertEqual(session._midi_state, "playing")
        self.assertEqual(synth.midi.snapshot().active_notes, (60,))
        self.assertEqual(session._midi_playback_snapshot(0.0)["tempo_scale"], 1.5)
        self.assertTrue(session._midi_playback_snapshot(0.0)["loop"])

        await session.handle_event({"type": "pause_midi"})
        self.assertEqual(session._midi_state, "paused")
        self.assertEqual(synth.midi.snapshot().active_notes, ())

        await session.handle_event(
            {"type": "seek_midi", "position_seconds": 7.0}
        )
        self.assertEqual(session._midi_state, "paused")
        await session.handle_event({"type": "resume_midi"})
        self.assertEqual(session._midi_state, "playing")
        self.assertEqual(synth.midi.snapshot().active_notes, (60,))
        self.assertTrue(synth.midi.snapshot().sustain)

        await session.handle_event(
            {"type": "set_midi_transport", "tempo_scale": 0.75, "loop": False}
        )
        snapshot = session._midi_playback_snapshot(
            session._midi_started_at or 0.0
        )
        self.assertEqual(snapshot["tempo_scale"], 0.75)
        self.assertFalse(snapshot["loop"])
        self.assertGreaterEqual(synth.reset_count, 4)
        self.assertEqual(websocket.messages[-1]["reason"], "configured")

        await session.handle_event({"type": "stop_midi"})
        self.assertEqual(session._midi_state, "stopped")
        self.assertIsNone(session.midi_task)

    async def test_transport_rejects_non_boolean_loop(self):
        session, _, _ = self.make_session()
        with self.assertRaisesRegex(ValueError, "loop must be a boolean"):
            await session.handle_event(
                {
                    "type": "play_midi",
                    "midi_id": "test.mid",
                    "loop": "yes",
                }
            )
