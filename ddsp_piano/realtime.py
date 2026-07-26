"""Stateful host-side MIDI and DSP runtime for realtime ONNX evaluation."""

from __future__ import annotations

import io
import json
import math
import threading
import time
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from mido import MidiFile, merge_tracks, tick2second

from ddsp_piano.ddsp_pytorch.fdn import fdn_impulse_response
from ddsp_piano.versioning import model_output_label
from scripts.render_onnx import (
    _StreamingDrySynthesizer,
    _StreamingOnnxRunner,
    _validate_contract,
)


PIANO_MIDI_MIN = 21
PIANO_MIDI_MAX = 108
PEDAL_CONTROLLERS = (64, 65, 66, 67)


@dataclass(frozen=True)
class MidiSnapshot:
    active_notes: tuple[int, ...]
    sustain: bool
    voice_steals: int


@dataclass(frozen=True)
class ScheduledMidiEvent:
    time_seconds: float
    kind: str
    data1: int
    data2: int = 0


@dataclass(frozen=True)
class MidiTimeline:
    duration_seconds: float
    events: tuple[ScheduledMidiEvent, ...]


def load_midi_timeline(path: Path) -> MidiTimeline:
    """Convert a standard MIDI file into realtime note and pedal events."""

    midi = MidiFile(Path(path))
    tempo = 500_000
    elapsed = 0.0
    events: list[ScheduledMidiEvent] = []
    for message in merge_tracks(midi.tracks):
        elapsed += tick2second(message.time, midi.ticks_per_beat, tempo)
        if message.type == "set_tempo":
            tempo = message.tempo
        elif message.type == "note_on" and PIANO_MIDI_MIN <= message.note <= PIANO_MIDI_MAX:
            if message.velocity > 0:
                events.append(
                    ScheduledMidiEvent(
                        elapsed, "note_on", message.note, message.velocity
                    )
                )
            else:
                events.append(ScheduledMidiEvent(elapsed, "note_off", message.note))
        elif message.type == "note_off" and PIANO_MIDI_MIN <= message.note <= PIANO_MIDI_MAX:
            events.append(ScheduledMidiEvent(elapsed, "note_off", message.note))
        elif message.type == "control_change" and message.control in PEDAL_CONTROLLERS:
            events.append(
                ScheduledMidiEvent(
                    elapsed, "control_change", message.control, message.value
                )
            )
    return MidiTimeline(duration_seconds=elapsed, events=tuple(events))


class LiveMidiState:
    """Map asynchronous MIDI events onto stable model polyphony slots."""

    def __init__(self, max_polyphony: int) -> None:
        if max_polyphony <= 0:
            raise ValueError("max_polyphony must be positive")
        self.max_polyphony = max_polyphony
        self._pitch = np.zeros(max_polyphony, dtype=np.int16)
        self._key_down = np.zeros(max_polyphony, dtype=np.bool_)
        self._pending_velocity = np.zeros(max_polyphony, dtype=np.float32)
        self._gate_target = np.zeros(max_polyphony, dtype=np.bool_)
        self._started = np.zeros(max_polyphony, dtype=np.int64)
        self._pedal = np.zeros(4, dtype=np.float32)
        self._counter = 0
        self._voice_steals = 0
        self._lock = threading.Lock()

    def note_on(self, pitch: int, velocity: int) -> bool:
        if velocity <= 0:
            return self.note_off(pitch)
        if not PIANO_MIDI_MIN <= pitch <= PIANO_MIDI_MAX:
            return False
        velocity = min(127, int(velocity))
        with self._lock:
            matching = np.flatnonzero(self._pitch == pitch)
            if matching.size:
                slot = int(matching[0])
            else:
                free = np.flatnonzero(self._pitch == 0)
                if free.size:
                    slot = int(free[0])
                else:
                    released = np.flatnonzero(~self._key_down)
                    candidates = released if released.size else np.arange(self.max_polyphony)
                    slot = int(candidates[np.argmin(self._started[candidates])])
                    self._voice_steals += 1
            self._counter += 1
            self._pitch[slot] = pitch
            self._key_down[slot] = True
            self._pending_velocity[slot] = np.float32(velocity / 127.0)
            self._gate_target[slot] = True
            self._started[slot] = self._counter
        return True

    def note_off(self, pitch: int) -> bool:
        with self._lock:
            matching = np.flatnonzero(self._pitch == pitch)
            if not matching.size:
                return False
            for slot_value in matching:
                slot = int(slot_value)
                self._key_down[slot] = False
                if self._pedal[0] < 0.5:
                    self._release_slot(slot)
        return True

    def control_change(self, controller: int, value: int) -> bool:
        if controller not in PEDAL_CONTROLLERS:
            return False
        index = PEDAL_CONTROLLERS.index(controller)
        normalized = np.float32(min(127, max(0, int(value))) / 127.0)
        with self._lock:
            was_sustained = self._pedal[0] >= 0.5
            self._pedal[index] = normalized
            is_sustained = self._pedal[0] >= 0.5
            if index == 0 and was_sustained and not is_sustained:
                for slot_value in np.flatnonzero(~self._key_down & (self._pitch != 0)):
                    self._release_slot(int(slot_value))
        return True

    def panic(self) -> None:
        with self._lock:
            self._pitch.fill(0)
            self._key_down.fill(False)
            self._pending_velocity.fill(0.0)
            self._gate_target.fill(False)
            self._pedal.fill(0.0)

    def render_frames(self, n_frames: int) -> tuple[np.ndarray, np.ndarray]:
        conditioning, pedal, _ = self.render_block(n_frames)
        return conditioning, pedal

    def render_block(
        self, n_frames: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if n_frames <= 0:
            raise ValueError("n_frames must be positive")
        with self._lock:
            conditioning = np.zeros(
                (n_frames, self.max_polyphony, 2), dtype=np.float32
            )
            conditioning[:, :, 0] = self._pitch[np.newaxis, :]
            conditioning[0, :, 1] = self._pending_velocity
            self._pending_velocity.fill(0.0)
            pedal = np.repeat(self._pedal[np.newaxis, :], n_frames, axis=0)
            gate_target = self._gate_target.copy()
        return conditioning, pedal, gate_target

    def snapshot(self) -> MidiSnapshot:
        with self._lock:
            notes = tuple(sorted(int(pitch) for pitch in self._pitch if pitch))
            return MidiSnapshot(
                active_notes=notes,
                sustain=bool(self._pedal[0] >= 0.5),
                voice_steals=self._voice_steals,
            )

    def _release_slot(self, slot: int) -> None:
        self._pitch[slot] = 0
        self._key_down[slot] = False
        self._pending_velocity[slot] = 0.0
        self._gate_target[slot] = False


def apply_scheduled_midi_event(
    state: LiveMidiState,
    event: ScheduledMidiEvent,
) -> None:
    """Apply one normalized timeline event to the realtime MIDI state."""

    if event.kind == "note_on":
        state.note_on(event.data1, event.data2)
    elif event.kind == "note_off":
        state.note_off(event.data1)
    elif event.kind == "control_change":
        state.control_change(event.data1, event.data2)
    else:
        raise ValueError(f"Unsupported scheduled MIDI event: {event.kind!r}")


def restore_midi_timeline_state(
    state: LiveMidiState,
    timeline: MidiTimeline,
    position_seconds: float,
) -> int:
    """Rebuild notes and pedals at a seek position and return the next event."""

    position_seconds = float(position_seconds)
    if not math.isfinite(position_seconds) or not 0.0 <= position_seconds <= timeline.duration_seconds:
        raise ValueError("position_seconds must be inside the MIDI timeline")
    state.panic()
    next_event = 0
    for next_event, event in enumerate(timeline.events):
        if event.time_seconds > position_seconds:
            return next_event
        apply_scheduled_midi_event(state, event)
    return len(timeline.events)


class VoiceReleaseEnvelope:
    """Apply click-free per-voice damping after KeyOff."""

    def __init__(
        self,
        max_polyphony: int,
        sample_rate: int,
        release_ms: float = 60.0,
    ) -> None:
        if max_polyphony <= 0 or sample_rate <= 0 or release_ms <= 0:
            raise ValueError("polyphony, sample rate, and release must be positive")
        self.max_polyphony = max_polyphony
        self.sample_rate = sample_rate
        self.release_ms = release_ms
        self.release_samples = max(1, round(sample_rate * release_ms / 1000.0))
        self._gain = np.zeros(max_polyphony, dtype=np.float32)

    def render(self, targets: np.ndarray, n_samples: int) -> np.ndarray:
        targets = np.asarray(targets, dtype=np.bool_).reshape(-1)
        if targets.shape != (self.max_polyphony,):
            raise ValueError(
                f"targets must have shape ({self.max_polyphony},), "
                f"received {targets.shape}"
            )
        if n_samples <= 0:
            raise ValueError("n_samples must be positive")
        envelopes = np.empty((self.max_polyphony, n_samples), dtype=np.float32)
        release_step = np.float32(1.0 / self.release_samples)
        sample_index = np.arange(1, n_samples + 1, dtype=np.float32)
        for voice in range(self.max_polyphony):
            if targets[voice]:
                envelopes[voice].fill(1.0)
                self._gain[voice] = 1.0
            else:
                envelopes[voice] = np.maximum(
                    self._gain[voice] - release_step * sample_index, 0.0
                )
                self._gain[voice] = envelopes[voice, -1]
        return envelopes

    def reset(self) -> None:
        self._gain.fill(0.0)


class PartitionedConvolver:
    """Uniform overlap-add convolution with state carried between blocks."""

    def __init__(self, impulse_response: np.ndarray, block_size: int) -> None:
        impulse_response = np.asarray(impulse_response, dtype=np.float32).reshape(-1)
        if impulse_response.size == 0:
            raise ValueError("impulse_response must not be empty")
        if not np.isfinite(impulse_response).all():
            raise ValueError("impulse_response contains NaN or infinity")
        if block_size <= 0:
            raise ValueError("block_size must be positive")

        self.block_size = block_size
        self.fft_size = block_size * 2
        n_partitions = math.ceil(impulse_response.size / block_size)
        padded_ir = np.zeros(n_partitions * block_size, dtype=np.float32)
        padded_ir[: impulse_response.size] = impulse_response
        partition_time = np.zeros((n_partitions, self.fft_size), dtype=np.float32)
        partition_time[:, :block_size] = padded_ir.reshape(n_partitions, block_size)
        self._responses = np.fft.rfft(partition_time, axis=-1)
        self._history = np.zeros_like(self._responses)
        self._overlap = np.zeros(block_size, dtype=np.float32)

    def process(self, audio: np.ndarray) -> np.ndarray:
        audio = np.asarray(audio, dtype=np.float32).reshape(-1)
        if audio.size != self.block_size:
            raise ValueError(
                f"Expected {self.block_size} audio samples, received {audio.size}"
            )
        block = np.zeros(self.fft_size, dtype=np.float32)
        block[: self.block_size] = audio
        if self._history.shape[0] > 1:
            self._history[1:] = self._history[:-1].copy()
        self._history[0] = np.fft.rfft(block)
        spectrum = np.sum(self._responses * self._history, axis=0)
        convolved = np.fft.irfft(spectrum, n=self.fft_size).astype(np.float32)
        output = convolved[: self.block_size] + self._overlap
        self._overlap = convolved[self.block_size :].copy()
        return output


class StreamingReverb:
    """Apply model-provided reverb while retaining convolution history."""

    def __init__(
        self,
        metadata: dict,
        reverb_condition: np.ndarray,
        block_size: int,
    ) -> None:
        reverb_type = metadata.get("reverb_ir_postprocess", {}).get("type", "ir")
        condition = np.asarray(reverb_condition, dtype=np.float32)
        if reverb_type == "fdn":
            with torch.inference_mode():
                impulse, wet_mix = fdn_impulse_response(
                    torch.from_numpy(condition), int(metadata["sample_rate"]), 24_000
                )
            impulse_response = impulse[0].numpy().copy()
            self.wet_gain = float(wet_mix.reshape(-1)[0])
        elif reverb_type in {"exponential_decay", "ir"}:
            impulse_response = condition.reshape(-1).copy()
            self.wet_gain = float(metadata.get("reverb_wet_gain", 1.0))
        else:
            raise ValueError(f"Unsupported reverb type: {reverb_type}")
        impulse_response[0] = 0.0
        self._convolver = PartitionedConvolver(impulse_response, block_size)

    def process(self, dry: np.ndarray) -> np.ndarray:
        return np.asarray(dry, dtype=np.float32) + self.wet_gain * self._convolver.process(dry)


@dataclass(frozen=True)
class RealtimeChunk:
    audio: np.ndarray
    render_seconds: float
    snapshot: MidiSnapshot


class RealtimeOnnxSynthesizer:
    """Run the one-frame ONNX contract as a continuous realtime instrument."""

    def __init__(
        self,
        model_path: Path,
        metadata_path: Path,
        piano_model: int = 9,
        chunk_frames: int = 8,
        seed: int = 0,
        keyoff_fade_ms: float = 60.0,
        all_notes_off_fade_ms: float = 120.0,
        onnx_intra_op_threads: int = 1,
        onnx_inter_op_threads: int = 1,
    ) -> None:
        self.model_path = Path(model_path)
        self.metadata_path = Path(metadata_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(f"ONNX model not found: {self.model_path}")
        if not self.metadata_path.is_file():
            raise FileNotFoundError(f"ONNX metadata not found: {self.metadata_path}")
        self.metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        self.sample_rate, self.frame_rate, self.max_polyphony, self.samples_per_frame = (
            _validate_contract(self.metadata)
        )
        if chunk_frames <= 0:
            raise ValueError("chunk_frames must be positive")
        years = self.metadata.get("piano_model_index_to_maestro_year", [])
        if years and not 0 <= piano_model < len(years):
            raise ValueError(f"piano_model must be between 0 and {len(years) - 1}")

        self.piano_model = piano_model
        self.chunk_frames = chunk_frames
        self.chunk_samples = chunk_frames * self.samples_per_frame
        self.chunk_seconds = self.chunk_samples / self.sample_rate
        self.midi = LiveMidiState(self.max_polyphony)
        reverb_output = str(self.metadata.get("reverb_output", "reverb_ir"))
        self._runner = _StreamingOnnxRunner(
            self.model_path,
            self.metadata,
            piano_model,
            reverb_output,
            intra_op_threads=onnx_intra_op_threads,
            inter_op_threads=onnx_inter_op_threads,
        )
        self._dry_synth = _StreamingDrySynthesizer(
            self.metadata, self.sample_rate, self.samples_per_frame, seed
        )
        self._voice_envelope = VoiceReleaseEnvelope(
            self.max_polyphony, self.sample_rate, keyoff_fade_ms
        )
        self._output_envelope = VoiceReleaseEnvelope(
            1, self.sample_rate, all_notes_off_fade_ms
        )
        self._reverb: StreamingReverb | None = None
        self._render_lock = threading.Lock()
        self._seed = seed
        self._hard_silence = True

    def render_chunk(self) -> RealtimeChunk:
        started = time.perf_counter()
        with self._render_lock:
            conditioning, pedal, gate_target = self.midi.render_block(self.chunk_frames)
            block_active = bool(np.any(gate_target))
            controls = self._runner.run(conditioning, pedal)
            voice_envelopes = self._voice_envelope.render(
                gate_target, self.chunk_samples
            )
            output_envelope = self._output_envelope.render(
                np.asarray([block_active]), self.chunk_samples
            )[0]
            harmonic, noise = self._dry_synth.render(controls, voice_envelopes)
            dry = harmonic + noise
            if self._reverb is None:
                if self._runner.reverb_condition is None:
                    raise RuntimeError("ONNX inference produced no reverb condition")
                self._reverb = StreamingReverb(
                    self.metadata, self._runner.reverb_condition, self.chunk_samples
                )
            elif block_active and self._hard_silence:
                self._reverb = StreamingReverb(
                    self.metadata, self._runner.reverb_condition, self.chunk_samples
                )
            was_hard_silence = self._hard_silence
            audio = self._reverb.process(dry) * output_envelope
            self._hard_silence = bool(
                not block_active and output_envelope[-1] <= 0.0
            )
            if was_hard_silence and not block_active:
                audio = np.zeros_like(audio)
        if not np.isfinite(audio).all():
            raise RuntimeError("Realtime synthesis produced NaN or infinity")
        return RealtimeChunk(
            audio=audio,
            render_seconds=time.perf_counter() - started,
            snapshot=self.midi.snapshot(),
        )

    def warm_up(self, seconds: float) -> int:
        chunks = max(0, math.ceil(seconds / self.chunk_seconds))
        for _ in range(chunks):
            self.render_chunk()
        return chunks

    def hard_reset(self) -> None:
        """Clear every recurrent/DSP state for panic or device reset."""

        with self._render_lock:
            self.midi.panic()
            self._runner.context_state.fill(0.0)
            self._runner.monophonic_state.fill(0.0)
            self._runner.held_pitch.fill(0.0)
            self._runner.released_frames.fill(0)
            self._runner.reverb_condition = None
            self._dry_synth = _StreamingDrySynthesizer(
                self.metadata, self.sample_rate, self.samples_per_frame, self._seed
            )
            self._voice_envelope.reset()
            self._output_envelope.reset()
            self._reverb = None
            self._hard_silence = True

    def describe(self) -> dict:
        years = self.metadata.get("piano_model_index_to_maestro_year", [])
        return {
            "model": self.model_path.name,
            "release_version": model_output_label(self.model_path, self.metadata),
            "host_dsp_profile": self.metadata.get("host_dsp_profile", "legacy"),
            "sample_rate": self.sample_rate,
            "frame_rate": self.frame_rate,
            "chunk_frames": self.chunk_frames,
            "chunk_samples": self.chunk_samples,
            "chunk_ms": self.chunk_seconds * 1000.0,
            "keyoff_fade_ms": self._voice_envelope.release_ms,
            "all_notes_off_fade_ms": self._output_envelope.release_ms,
            "max_polyphony": self.max_polyphony,
            "piano_model": self.piano_model,
            "piano_year": years[self.piano_model] if years else None,
            "piano_model_years": years,
            "reverb_type": self.metadata.get("reverb_ir_postprocess", {}).get("type"),
            "onnx_intra_op_threads": self._runner.intra_op_threads,
            "onnx_inter_op_threads": self._runner.inter_op_threads,
            "torch_threads": torch.get_num_threads(),
            "torch_interop_threads": torch.get_num_interop_threads(),
        }


def encode_wav_chunk(
    audio: np.ndarray,
    sample_rate: int,
    gain: float = 1.0,
) -> tuple[bytes, int]:
    """Encode one self-contained mono PCM16 WAV block and report clipped samples."""

    audio = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio.size == 0:
        raise ValueError("audio must not be empty")
    if sample_rate <= 0 or gain <= 0:
        raise ValueError("sample_rate and gain must be positive")
    if not np.isfinite(audio).all():
        raise ValueError("audio contains NaN or infinity")
    scaled = audio * np.float32(gain)
    clipped = int(np.count_nonzero(np.abs(scaled) > 1.0))
    pcm = np.round(np.clip(scaled, -1.0, 1.0) * 32767.0).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue(), clipped
