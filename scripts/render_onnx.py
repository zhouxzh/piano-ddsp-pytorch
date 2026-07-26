#!/usr/bin/env python3
"""Render demo or MIDI-driven audio from the stateful ONNX control model."""

from __future__ import annotations

import argparse
import json
import math
import sys
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.ddsp_pytorch.core import (
    frequency_impulse_response,
    get_fft_size,
    scale_function,
)
from ddsp_piano.ddsp_pytorch.fdn import FDNReverb
from ddsp_piano.ddsp_pytorch.noise import Noise
from ddsp_piano.ddsp_pytorch.reverb import Reverb
from ddsp_piano.deployment import extend_pitch_for_release
from ddsp_piano.maestro import MidiConditioning, load_midi_conditioning
from ddsp_piano.modules.inharm_synth import MultiInharmonic
from ddsp_piano.versioning import model_output_label


INPUT_NAMES = [
    "conditioning",
    "pedal",
    "piano_model",
    "extended_pitch",
    "context_state",
    "monophonic_state",
]
CONTROL_OUTPUT_NAMES = [
    "amplitudes",
    "harmonic_distribution",
    "inharmonicity",
    "f0_hz",
    "noise_magnitudes",
]


@dataclass(frozen=True)
class Note:
    start: float
    duration: float
    pitch: int
    velocity: float


DEMO_NOTES = (
    Note(0.20, 0.65, 60, 0.72),
    Note(0.65, 0.65, 64, 0.76),
    Note(1.10, 0.65, 67, 0.80),
    Note(1.55, 0.85, 72, 0.84),
    Note(2.55, 0.70, 48, 0.78),
    Note(2.55, 0.70, 60, 0.72),
    Note(2.55, 0.70, 64, 0.76),
    Note(2.55, 0.70, 67, 0.80),
    Note(3.40, 0.70, 53, 0.76),
    Note(3.40, 0.70, 60, 0.72),
    Note(3.40, 0.70, 65, 0.78),
    Note(3.40, 0.70, 69, 0.80),
    Note(4.25, 0.70, 55, 0.78),
    Note(4.25, 0.70, 59, 0.74),
    Note(4.25, 0.70, 62, 0.76),
    Note(4.25, 0.70, 67, 0.82),
    Note(5.10, 1.00, 48, 0.82),
    Note(5.10, 1.00, 60, 0.76),
    Note(5.10, 1.00, 64, 0.80),
    Note(5.10, 1.00, 67, 0.84),
    Note(5.10, 1.00, 72, 0.86),
)
DEMO_SECONDS = 7.5


def _shape(metadata: dict, group: str, name: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in metadata[group][name])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"Metadata is missing a valid {group}.{name} shape") from error


def _validate_contract(metadata: dict) -> tuple[int, int, int, int]:
    sample_rate = int(metadata["sample_rate"])
    frame_rate = int(metadata["frame_rate"])
    frames_per_call = int(metadata["frames_per_call"])
    conditioning_shape = _shape(metadata, "inputs", "conditioning")
    if frames_per_call != 1 or conditioning_shape[:2] != (1, 1):
        raise ValueError("The renderer requires the fixed batch-1, one-frame ONNX contract")
    if len(conditioning_shape) != 4 or conditioning_shape[-1] != 2:
        raise ValueError("conditioning must have shape [1, 1, polyphony, 2]")
    if sample_rate <= 0 or frame_rate <= 0 or sample_rate % frame_rate:
        raise ValueError("sample_rate must be positive and divisible by frame_rate")
    return sample_rate, frame_rate, conditioning_shape[2], sample_rate // frame_rate


def _build_demo_conditioning(
    frame_rate: int,
    max_polyphony: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_frames = int(round(DEMO_SECONDS * frame_rate))
    roll = np.zeros((n_frames, 88, 2), dtype=np.float32)
    for note in DEMO_NOTES:
        if not 21 <= note.pitch <= 108:
            raise ValueError(f"MIDI pitch is outside the piano range: {note.pitch}")
        start = max(0, int(round(note.start * frame_rate)))
        end = min(n_frames, int(round((note.start + note.duration) * frame_rate)))
        if end <= start:
            continue
        index = note.pitch - 21
        roll[start:end, index, 0] = 1.0
        roll[start, index, 1] = np.float32(note.velocity)

    conditioning = np.zeros((n_frames, max_polyphony, 2), dtype=np.float32)
    slots = np.zeros(max_polyphony, dtype=np.int16)
    for frame in range(n_frames):
        active_indices = np.flatnonzero(roll[frame, :, 0])
        active_pitches = {int(index + 21) for index in active_indices}
        for slot, pitch in enumerate(slots):
            if pitch and pitch not in active_pitches:
                slots[slot] = 0

        assigned = {int(pitch) for pitch in slots if pitch}
        for pitch in sorted(active_pitches - assigned, reverse=True):
            free_slots = np.flatnonzero(slots == 0)
            if free_slots.size == 0:
                raise ValueError(f"Demo exceeds the ONNX polyphony limit of {max_polyphony}")
            slots[int(free_slots[0])] = pitch

        for slot, pitch in enumerate(slots):
            if pitch:
                conditioning[frame, slot, 0] = pitch
                conditioning[frame, slot, 1] = roll[frame, pitch - 21, 1]

    pedal = np.zeros((n_frames, 4), dtype=np.float32)
    return conditioning, pedal


def _run_onnx(
    model_path: Path,
    metadata: dict,
    conditioning: np.ndarray,
    pedal: np.ndarray,
    piano_model: int,
    reverb_output_name: str,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    output_names = CONTROL_OUTPUT_NAMES + [reverb_output_name, "next_context_state", "next_monophonic_state"]
    if [value.name for value in session.get_inputs()] != INPUT_NAMES:
        raise ValueError("ONNX input names do not match the deployment contract")
    if [value.name for value in session.get_outputs()] != output_names:
        raise ValueError("ONNX output names do not match the deployment contract")

    n_frames, max_polyphony, _ = conditioning.shape
    context_state = np.zeros(_shape(metadata, "inputs", "context_state"), dtype=np.float32)
    monophonic_state = np.zeros(
        _shape(metadata, "inputs", "monophonic_state"), dtype=np.float32
    )
    held_pitch = np.zeros((1, max_polyphony), dtype=np.float32)
    released_frames = np.zeros((1, max_polyphony), dtype=np.int32)
    release_frames = int(metadata["release_frames"])
    instrument = np.asarray([piano_model], dtype=np.int32)

    control_names = CONTROL_OUTPUT_NAMES
    controls = {
        name: np.empty((n_frames,) + _shape(metadata, "outputs", name)[2:], dtype=np.float32)
        for name in control_names
    }
    reverb_ir: np.ndarray | None = None

    for frame in range(n_frames):
        conditioning_block = conditioning[np.newaxis, frame : frame + 1]
        pedal_block = pedal[np.newaxis, frame : frame + 1]
        extended_pitch, held_pitch, released_frames = extend_pitch_for_release(
            conditioning_block,
            held_pitch,
            released_frames,
            release_frames,
        )
        outputs = session.run(
            output_names,
            {
                "conditioning": conditioning_block,
                "pedal": pedal_block,
                "piano_model": instrument,
                "extended_pitch": extended_pitch,
                "context_state": context_state,
                "monophonic_state": monophonic_state,
            },
        )
        for name, value in zip(control_names, outputs[:5]):
            controls[name][frame] = value[0, 0]
        if reverb_ir is None:
            reverb_ir = outputs[5].astype(np.float32, copy=True)
        context_state = outputs[6]
        monophonic_state = outputs[7]

        completed = frame + 1
        if completed % max(1, n_frames // 10) == 0 or completed == n_frames:
            print(f"ONNX controls: {completed}/{n_frames} frames", flush=True)

    if reverb_ir is None:
        raise RuntimeError("ONNX inference produced no reverb impulse response")
    return controls, reverb_ir


class _StreamingOnnxRunner:
    """Carry all explicit ONNX and host release states across render chunks."""

    def __init__(
        self,
        model_path: Path,
        metadata: dict,
        piano_model: int,
        reverb_output_name: str,
        intra_op_threads: int | None = None,
        inter_op_threads: int | None = None,
    ) -> None:
        for name, value in (
            ("intra_op_threads", intra_op_threads),
            ("inter_op_threads", inter_op_threads),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when supplied")
        session_options = None
        if intra_op_threads is not None or inter_op_threads is not None:
            session_options = ort.SessionOptions()
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
            if intra_op_threads is not None:
                session_options.intra_op_num_threads = intra_op_threads
            if inter_op_threads is not None:
                session_options.inter_op_num_threads = inter_op_threads
        self.intra_op_threads = intra_op_threads
        self.inter_op_threads = inter_op_threads
        self.session = ort.InferenceSession(
            str(model_path),
            sess_options=session_options,
            providers=["CPUExecutionProvider"],
        )
        self.output_names = CONTROL_OUTPUT_NAMES + [
            reverb_output_name,
            "next_context_state",
            "next_monophonic_state",
        ]
        if [value.name for value in self.session.get_inputs()] != INPUT_NAMES:
            raise ValueError("ONNX input names do not match the deployment contract")
        if [value.name for value in self.session.get_outputs()] != self.output_names:
            raise ValueError("ONNX output names do not match the deployment contract")

        max_polyphony = _shape(metadata, "inputs", "conditioning")[2]
        self.metadata = metadata
        self.context_state = np.zeros(
            _shape(metadata, "inputs", "context_state"), dtype=np.float32
        )
        self.monophonic_state = np.zeros(
            _shape(metadata, "inputs", "monophonic_state"), dtype=np.float32
        )
        self.held_pitch = np.zeros((1, max_polyphony), dtype=np.float32)
        self.released_frames = np.zeros((1, max_polyphony), dtype=np.int32)
        self.instrument = np.asarray([piano_model], dtype=np.int32)
        self.release_frames = int(metadata["release_frames"])
        self.reverb_condition: np.ndarray | None = None

    def run(self, conditioning: np.ndarray, pedal: np.ndarray) -> dict[str, np.ndarray]:
        n_frames = conditioning.shape[0]
        controls = {
            name: np.empty(
                (n_frames,) + _shape(self.metadata, "outputs", name)[2:],
                dtype=np.float32,
            )
            for name in CONTROL_OUTPUT_NAMES
        }
        for frame in range(n_frames):
            conditioning_block = conditioning[np.newaxis, frame : frame + 1]
            pedal_block = pedal[np.newaxis, frame : frame + 1]
            extended_pitch, self.held_pitch, self.released_frames = extend_pitch_for_release(
                conditioning_block,
                self.held_pitch,
                self.released_frames,
                self.release_frames,
            )
            outputs = self.session.run(
                self.output_names,
                {
                    "conditioning": conditioning_block,
                    "pedal": pedal_block,
                    "piano_model": self.instrument,
                    "extended_pitch": extended_pitch,
                    "context_state": self.context_state,
                    "monophonic_state": self.monophonic_state,
                },
            )
            for name, value in zip(CONTROL_OUTPUT_NAMES, outputs[:5]):
                controls[name][frame] = value[0, 0]
            if self.reverb_condition is None:
                self.reverb_condition = outputs[5].astype(np.float32, copy=True)
            self.context_state = outputs[6]
            self.monophonic_state = outputs[7]
        return controls


class _StreamingDrySynthesizer:
    """Bound memory while preserving oscillator phase and filtered-noise overlap."""

    def __init__(
        self,
        metadata: dict,
        sample_rate: int,
        samples_per_frame: int,
        seed: int,
    ) -> None:
        amplitude_shape = _shape(metadata, "outputs", "amplitudes")
        harmonic_shape = _shape(metadata, "outputs", "harmonic_distribution")
        f0_shape = _shape(metadata, "outputs", "f0_hz")
        noise_shape = _shape(metadata, "outputs", "noise_magnitudes")
        self.max_polyphony = amplitude_shape[2]
        self.n_harmonics = harmonic_shape[3]
        self.n_substrings = f0_shape[3]
        self.n_noise_bands = noise_shape[3]
        self.sample_rate = sample_rate
        self.samples_per_frame = samples_per_frame
        self.phase = torch.zeros(
            self.max_polyphony,
            self.n_substrings,
            self.n_harmonics,
            dtype=torch.float32,
        )
        self.noise_tail = [torch.zeros(0, dtype=torch.float32) for _ in range(self.max_polyphony)]
        self.generators = []
        for voice in range(self.max_polyphony):
            generator = torch.Generator(device="cpu")
            generator.manual_seed(seed + voice)
            self.generators.append(generator)
        self.ratios = torch.arange(1, self.n_harmonics + 1, dtype=torch.float32)
        self.noise_ir_size = 2 * (self.n_noise_bands - 1)
        self.noise_fft_size = get_fft_size(self.samples_per_frame, self.noise_ir_size)
        self.overlap_filter = torch.eye(self.noise_fft_size, dtype=torch.float32).unsqueeze(1)

    def _render_harmonic_voice(
        self,
        voice: int,
        amplitudes: torch.Tensor,
        harmonic_distribution: torch.Tensor,
        inharmonicity: torch.Tensor,
        f0_hz: torch.Tensor,
    ) -> torch.Tensor:
        n_frames = amplitudes.shape[0]
        n_samples = n_frames * self.samples_per_frame
        scaled_amplitudes = scale_function(amplitudes)
        distribution = scale_function(harmonic_distribution)
        inharmonic_factor = torch.sqrt(
            1.0 + inharmonicity * torch.square(self.ratios).unsqueeze(0)
        )
        reference_frequencies = f0_hz[:, :1] * self.ratios.unsqueeze(0) * inharmonic_factor
        distribution = distribution * (
            (reference_frequencies < self.sample_rate / 2).to(distribution.dtype) + 1e-4
        )
        scaled_amplitudes = scaled_amplitudes * (
            (f0_hz[:, :1] > 20.0).to(scaled_amplitudes.dtype) + 1e-4
        )
        distribution = distribution / distribution.sum(dim=-1, keepdim=True)
        partial_amplitudes = scaled_amplitudes * distribution / self.n_substrings

        harmonic = torch.zeros(n_samples, dtype=torch.float32)
        for substring in range(self.n_substrings):
            frequencies = (
                f0_hz[:, substring : substring + 1]
                * self.ratios.unsqueeze(0)
                * inharmonic_factor
            )
            if torch.any(frequencies[:, 0] > 20.0):
                sample_frequencies = frequencies.repeat_interleave(
                    self.samples_per_frame, dim=0
                )
                sample_amplitudes = partial_amplitudes.repeat_interleave(
                    self.samples_per_frame, dim=0
                )
                sample_amplitudes = sample_amplitudes * (
                    (sample_frequencies < self.sample_rate / 2).to(sample_amplitudes.dtype)
                    + 1e-4
                )
                phase = self.phase[voice, substring] + torch.cumsum(
                    sample_frequencies * (2.0 * math.pi / self.sample_rate), dim=0
                )
                harmonic.add_(torch.sum(sample_amplitudes * torch.sin(phase), dim=-1))
                self.phase[voice, substring] = torch.remainder(phase[-1], 2.0 * math.pi)
            else:
                phase_delta = (
                    frequencies.sum(dim=0)
                    * self.samples_per_frame
                    * (2.0 * math.pi / self.sample_rate)
                )
                self.phase[voice, substring] = torch.remainder(
                    self.phase[voice, substring] + phase_delta,
                    2.0 * math.pi,
                )
        return harmonic

    def _render_noise_voice(self, voice: int, magnitudes: torch.Tensor) -> torch.Tensor:
        n_frames = magnitudes.shape[0]
        n_samples = n_frames * self.samples_per_frame
        noise = (
            torch.rand(
                n_samples,
                generator=self.generators[voice],
                dtype=torch.float32,
            )
            * 2.0
            - 1.0
        )
        noise_frames = noise.reshape(n_frames, self.samples_per_frame)
        impulse_response = frequency_impulse_response(
            scale_function(magnitudes).unsqueeze(0)
        ).squeeze(0)
        with torch.autocast(device_type="cpu", enabled=False):
            filtered_frames = torch.fft.irfft(
                torch.fft.rfft(noise_frames, self.noise_fft_size)
                * torch.fft.rfft(impulse_response, self.noise_fft_size),
                n=self.noise_fft_size,
            )
        overlap = F.conv_transpose1d(
            filtered_frames.transpose(0, 1).unsqueeze(0),
            self.overlap_filter,
            stride=self.samples_per_frame,
        ).squeeze(0).squeeze(0)
        previous_tail = self.noise_tail[voice]
        if previous_tail.numel():
            overlap[: previous_tail.numel()].add_(previous_tail)
        self.noise_tail[voice] = overlap[n_samples:].clone()
        return overlap[:n_samples]

    def render(
        self,
        controls: dict[str, np.ndarray],
        voice_envelopes: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n_frames = controls["amplitudes"].shape[0]
        n_samples = n_frames * self.samples_per_frame
        harmonic_mix = torch.zeros(n_samples, dtype=torch.float32)
        noise_mix = torch.zeros(n_samples, dtype=torch.float32)
        tensors = {name: torch.from_numpy(value) for name, value in controls.items()}
        envelope_tensor = None
        if voice_envelopes is not None:
            voice_envelopes = np.asarray(voice_envelopes, dtype=np.float32)
            expected_shape = (self.max_polyphony, n_samples)
            if voice_envelopes.shape != expected_shape:
                raise ValueError(
                    f"voice_envelopes must have shape {expected_shape}, "
                    f"received {voice_envelopes.shape}"
                )
            envelope_tensor = torch.from_numpy(voice_envelopes)
        with torch.inference_mode():
            for voice in range(self.max_polyphony):
                harmonic_voice = self._render_harmonic_voice(
                    voice,
                    tensors["amplitudes"][:, voice],
                    tensors["harmonic_distribution"][:, voice],
                    tensors["inharmonicity"][:, voice],
                    tensors["f0_hz"][:, voice],
                )
                noise_voice = self._render_noise_voice(
                    voice, tensors["noise_magnitudes"][:, voice]
                )
                if envelope_tensor is not None:
                    harmonic_voice.mul_(envelope_tensor[voice])
                    noise_voice.mul_(envelope_tensor[voice])
                harmonic_mix.add_(harmonic_voice)
                noise_mix.add_(noise_voice)
        return harmonic_mix.numpy(), noise_mix.numpy()


def _apply_reverb(
    dry: np.ndarray,
    reverb_condition: np.ndarray,
    sample_rate: int,
    reverb_type: str,
    reverb_wet_gain: float,
) -> np.ndarray:
    dry_tensor = torch.from_numpy(dry).unsqueeze(0)
    condition_tensor = torch.from_numpy(reverb_condition)
    with torch.inference_mode():
        if reverb_type == "fdn":
            wet = FDNReverb(sample_rate=sample_rate, length=24_000).eval()(
                dry_tensor, condition_tensor
            )
        elif reverb_type in {"exponential_decay", "ir"}:
            wet = Reverb(wet_gain=reverb_wet_gain).eval()(dry_tensor, condition_tensor)
        else:
            raise ValueError(f"Unsupported reverb type in deployment JSON: {reverb_type}")
    return wet.squeeze(0).numpy()


def _synthesize(
    controls: dict[str, np.ndarray],
    reverb_condition: np.ndarray,
    sample_rate: int,
    samples_per_frame: int,
    seed: int,
    reverb_type: str,
    reverb_wet_gain: float = 1.0,
) -> dict[str, np.ndarray]:
    n_frames, max_polyphony, _ = controls["amplitudes"].shape
    n_samples = n_frames * samples_per_frame
    harmonic_synth = MultiInharmonic(n_samples=n_samples, sample_rate=sample_rate).eval()
    noise_synth = Noise().eval()
    harmonic_mix = torch.zeros(1, n_samples, dtype=torch.float32)
    noise_mix = torch.zeros(1, n_samples, dtype=torch.float32)
    torch.manual_seed(seed)

    with torch.inference_mode():
        for voice in range(max_polyphony):
            amplitudes = torch.from_numpy(controls["amplitudes"][:, voice]).unsqueeze(0)
            harmonic_distribution = torch.from_numpy(
                controls["harmonic_distribution"][:, voice]
            ).unsqueeze(0)
            inharmonicity = torch.from_numpy(
                controls["inharmonicity"][:, voice]
            ).unsqueeze(0)
            f0_hz = torch.from_numpy(controls["f0_hz"][:, voice]).unsqueeze(0)
            noise_magnitudes = torch.from_numpy(
                controls["noise_magnitudes"][:, voice]
            ).unsqueeze(0)

            scaled = harmonic_synth.get_controls(
                amplitudes,
                harmonic_distribution,
                inharmonicity,
                f0_hz,
            )
            harmonic = harmonic_synth(
                scaled["amplitudes"],
                scaled["harmonic_distribution"],
                scaled["harmonic_shifts"],
                scaled["f0_hz"],
            )
            harmonic_mix.add_(harmonic)
            noise_mix.add_(noise_synth(harmonic, noise_magnitudes))
            print(f"CPU DDSP synthesis: {voice + 1}/{max_polyphony} voices", flush=True)

        dry = harmonic_mix + noise_mix
        if reverb_type == "fdn":
            wet = FDNReverb(sample_rate=sample_rate, length=24_000).eval()(
                dry, torch.from_numpy(reverb_condition)
            )
        elif reverb_type in {"exponential_decay", "ir"}:
            wet = Reverb(wet_gain=reverb_wet_gain).eval()(
                dry, torch.from_numpy(reverb_condition)
            )
        else:
            raise ValueError(f"Unsupported reverb type in deployment JSON: {reverb_type}")

    return {
        "harmonic": harmonic_mix.squeeze(0).numpy(),
        "noise": noise_mix.squeeze(0).numpy(),
        "dry": dry.squeeze(0).numpy(),
        "wet": wet.squeeze(0).numpy(),
    }


def _write_normalized_wav(output: Path, audio: np.ndarray, sample_rate: int) -> dict[str, float]:
    if audio.ndim != 1 or audio.size == 0:
        raise ValueError("Rendered audio must be a non-empty mono signal")
    if not np.isfinite(audio).all():
        raise ValueError("Rendered audio contains NaN or infinity")

    source_peak = float(np.max(np.abs(audio)))
    source_rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
    if source_peak <= 1e-8:
        raise ValueError("Rendered audio is silent")
    target_peak = math.pow(10.0, -1.0 / 20.0)
    normalized = np.clip(audio * (target_peak / source_peak), -1.0, 1.0)
    pcm = np.round(normalized * 32767.0).astype("<i2")
    output.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(output), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(pcm.tobytes())
    return {
        "duration_seconds": audio.size / sample_rate,
        "source_peak": source_peak,
        "source_rms": source_rms,
        "written_peak_dbfs": 20.0 * math.log10(float(np.max(np.abs(normalized)))),
    }


def _render_streaming_conditioning(
    model_path: Path,
    metadata: dict,
    conditioning: np.ndarray,
    pedal: np.ndarray,
    piano_model: int,
    warm_up_frames: int,
    chunk_frames: int,
    seed: int,
    reverb_output_name: str,
    reverb_type: str,
    reverb_wet_gain: float,
) -> dict[str, np.ndarray]:
    sample_rate, _, _, samples_per_frame = _validate_contract(metadata)
    if warm_up_frames:
        conditioning = np.pad(conditioning, ((warm_up_frames, 0), (0, 0), (0, 0)))
        pedal = np.pad(pedal, ((warm_up_frames, 0), (0, 0)))

    runner = _StreamingOnnxRunner(
        model_path,
        metadata,
        piano_model,
        reverb_output_name,
    )
    synthesizer = _StreamingDrySynthesizer(
        metadata,
        sample_rate,
        samples_per_frame,
        seed,
    )
    total_frames = conditioning.shape[0]
    total_samples = total_frames * samples_per_frame
    harmonic = np.empty(total_samples, dtype=np.float32)
    noise = np.empty(total_samples, dtype=np.float32)
    progress_interval = max(chunk_frames, total_frames // 10)
    next_progress = progress_interval

    for start in range(0, total_frames, chunk_frames):
        end = min(start + chunk_frames, total_frames)
        controls = runner.run(conditioning[start:end], pedal[start:end])
        harmonic_chunk, noise_chunk = synthesizer.render(controls)
        sample_start = start * samples_per_frame
        sample_end = end * samples_per_frame
        harmonic[sample_start:sample_end] = harmonic_chunk
        noise[sample_start:sample_end] = noise_chunk
        if end >= next_progress or end == total_frames:
            print(f"ONNX + CPU DDSP: {end}/{total_frames} frames", flush=True)
            next_progress = end + progress_interval

    if runner.reverb_condition is None:
        raise RuntimeError("ONNX inference produced no reverb condition")
    dry = harmonic + noise
    wet = _apply_reverb(
        dry,
        runner.reverb_condition,
        sample_rate,
        reverb_type,
        reverb_wet_gain,
    )
    warm_up_samples = warm_up_frames * samples_per_frame
    if warm_up_samples:
        harmonic = harmonic[warm_up_samples:]
        noise = noise[warm_up_samples:]
        dry = dry[warm_up_samples:]
        wet = wet[warm_up_samples:]
    return {"harmonic": harmonic, "noise": noise, "dry": dry, "wet": wet}


def _write_rendered_signals(
    output_path: Path,
    signals: dict[str, np.ndarray],
    sample_rate: int,
    decompose: bool,
) -> tuple[dict[str, float], dict[str, str]]:
    report = _write_normalized_wav(output_path, signals["wet"], sample_rate)
    stem_paths: dict[str, str] = {}
    if decompose:
        for name in ("harmonic", "noise", "dry"):
            stem_path = output_path.with_name(f"{output_path.stem}_{name}{output_path.suffix}")
            _write_normalized_wav(stem_path, signals[name], sample_rate)
            stem_paths[name] = str(stem_path)
    return report, stem_paths


def _render_midi_file(
    midi_path: Path,
    output_path: Path,
    model_path: Path,
    metadata: dict,
    piano_model: int,
    warm_up_seconds: float,
    tail_seconds: float,
    chunk_seconds: float,
    seed: int,
    decompose: bool,
) -> dict[str, object]:
    sample_rate, frame_rate, max_polyphony, _ = _validate_contract(metadata)
    midi_data: MidiConditioning = load_midi_conditioning(
        midi_path,
        frame_rate=frame_rate,
        max_polyphony=max_polyphony,
        tail_seconds=tail_seconds,
    )
    warm_up_frames = int(round(warm_up_seconds * frame_rate))
    chunk_frames = max(1, int(round(chunk_seconds * frame_rate)))
    reverb_output_name = str(metadata.get("reverb_output", "reverb_ir"))
    reverb_type = str(metadata.get("reverb_ir_postprocess", {}).get("type", "ir"))
    reverb_wet_gain = float(metadata.get("reverb_wet_gain", 1.0))
    print(f"Rendering MIDI: {midi_path}", flush=True)
    signals = _render_streaming_conditioning(
        model_path,
        metadata,
        midi_data.conditioning,
        midi_data.pedal,
        piano_model,
        warm_up_frames,
        chunk_frames,
        seed,
        reverb_output_name,
        reverb_type,
        reverb_wet_gain,
    )
    wav_report, stem_paths = _write_rendered_signals(
        output_path,
        signals,
        sample_rate,
        decompose,
    )
    return {
        "midi": str(midi_path.resolve()),
        "output": str(output_path.resolve()),
        "midi_duration_seconds": midi_data.duration_seconds,
        "tail_seconds": tail_seconds,
        "sample_rate": sample_rate,
        "frame_rate": frame_rate,
        "max_polyphony_contract": max_polyphony,
        "max_observed_polyphony": midi_data.max_observed_polyphony,
        "polyphony_overflow_frames": midi_data.overflow_frames,
        "piano_model": piano_model,
        "warm_up_seconds": warm_up_frames / frame_rate,
        "noise_seed": seed,
        "harmonic_rms": float(
            np.sqrt(np.mean(np.square(signals["harmonic"], dtype=np.float64)))
        ),
        "noise_rms": float(np.sqrt(np.mean(np.square(signals["noise"], dtype=np.float64)))),
        "dry_rms": float(np.sqrt(np.mean(np.square(signals["dry"], dtype=np.float64)))),
        "stems": stem_paths,
        "wav": wav_report,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("exports/piano_v1.onnx"),
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        help="Deployment JSON; defaults to the ONNX path with a .json suffix",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Output WAV for demo or --midi mode",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory for --midi-dir mode",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--midi", type=Path, help="Render one MIDI file")
    source.add_argument("--midi-dir", type=Path, help="Render every .mid/.midi file in a directory")
    parser.add_argument(
        "--piano-model",
        type=int,
        default=9,
        help="Piano embedding index listed in the deployment JSON (default: 9)",
    )
    parser.add_argument(
        "--warm-up-seconds",
        type=float,
        default=0.5,
        help="Silent recurrent-state warm-up cropped from the WAV (default: 0.5)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=20260722,
        help="Seed for independent filtered-noise excitation (default: 20260722)",
    )
    parser.add_argument(
        "--tail-seconds",
        type=float,
        default=2.5,
        help="Silence appended after MIDI end for note release and reverb (default: 2.5)",
    )
    parser.add_argument(
        "--chunk-seconds",
        type=float,
        default=4.0,
        help="Bounded-memory synthesis chunk duration (default: 4.0)",
    )
    parser.add_argument(
        "--reverb-wet-gain",
        type=float,
        help="Override the metadata IR wet gain for a host-DSP listening ablation",
    )
    parser.add_argument(
        "--decompose",
        action="store_true",
        help="Also write normalized harmonic, noise, and unreverberated stems",
    )
    args = parser.parse_args()

    model_path = args.model.resolve()
    metadata_path = (args.metadata or args.model.with_suffix(".json")).resolve()
    if not model_path.is_file():
        raise FileNotFoundError(f"ONNX model not found: {model_path}")
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Deployment metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if args.reverb_wet_gain is not None:
        if args.reverb_wet_gain < 0:
            raise ValueError("--reverb-wet-gain must be non-negative")
        metadata = dict(metadata)
        metadata["reverb_wet_gain"] = float(args.reverb_wet_gain)
    sample_rate, frame_rate, max_polyphony, samples_per_frame = _validate_contract(metadata)
    output_label = model_output_label(model_path, metadata)
    reverb_output_name = str(metadata.get("reverb_output", "reverb_ir"))
    reverb_type = str(metadata.get("reverb_ir_postprocess", {}).get("type", "ir"))
    reverb_wet_gain = float(metadata.get("reverb_wet_gain", 1.0))

    piano_models = metadata.get("piano_model_index_to_maestro_year", [])
    if not 0 <= args.piano_model < len(piano_models):
        raise ValueError(
            f"--piano-model must be between 0 and {len(piano_models) - 1} for this export"
        )
    if args.warm_up_seconds < 0:
        raise ValueError("--warm-up-seconds must be non-negative")
    if args.tail_seconds < 0:
        raise ValueError("--tail-seconds must be non-negative")
    if args.chunk_seconds <= 0:
        raise ValueError("--chunk-seconds must be positive")

    if args.midi_dir is not None:
        midi_dir = args.midi_dir.resolve()
        if not midi_dir.is_dir():
            raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")
        midi_paths = sorted(
            path for path in midi_dir.iterdir()
            if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
        )
        if not midi_paths:
            raise FileNotFoundError(f"No .mid or .midi files found in: {midi_dir}")
        output_dir = (
            args.output_dir or Path("exports/midi_tests") / output_label
        ).resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = output_dir / "manifest.json"
        manifest: dict[str, object] = {
            "model": str(model_path),
            "metadata": str(metadata_path),
            "release_version": metadata.get("release_version", output_label),
            "midi_directory": str(midi_dir),
            "piano_model": args.piano_model,
            "maestro_year": piano_models[args.piano_model],
            "effective_reverb_wet_gain": reverb_wet_gain,
            "files": [],
        }
        for midi_path in midi_paths:
            entry = _render_midi_file(
                midi_path,
                output_dir / f"{midi_path.stem}.wav",
                model_path,
                metadata,
                args.piano_model,
                args.warm_up_seconds,
                args.tail_seconds,
                args.chunk_seconds,
                args.seed,
                args.decompose,
            )
            manifest["files"].append(entry)
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(json.dumps(manifest, indent=2))
        return 0

    if args.midi is not None:
        midi_path = args.midi.resolve()
        output_path = (
            args.output
            or Path("exports/midi_tests") / output_label / f"{midi_path.stem}.wav"
        ).resolve()
        report = _render_midi_file(
            midi_path,
            output_path,
            model_path,
            metadata,
            args.piano_model,
            args.warm_up_seconds,
            args.tail_seconds,
            args.chunk_seconds,
            args.seed,
            args.decompose,
        )
        print(json.dumps(report, indent=2))
        return 0

    conditioning, pedal = _build_demo_conditioning(frame_rate, max_polyphony)
    warm_up_frames = int(round(args.warm_up_seconds * frame_rate))
    if warm_up_frames:
        conditioning = np.pad(conditioning, ((warm_up_frames, 0), (0, 0), (0, 0)))
        pedal = np.pad(pedal, ((warm_up_frames, 0), (0, 0)))
    controls, reverb_ir = _run_onnx(
        model_path,
        metadata,
        conditioning,
        pedal,
        args.piano_model,
        reverb_output_name,
    )
    signals = _synthesize(
        controls,
        reverb_ir,
        sample_rate,
        samples_per_frame,
        args.seed,
        reverb_type,
        reverb_wet_gain,
    )
    warm_up_samples = warm_up_frames * samples_per_frame
    if warm_up_samples:
        signals = {name: value[warm_up_samples:] for name, value in signals.items()}
    output_path = (args.output or Path("exports/piano_maestro_onnx_demo.wav")).resolve()
    report, stem_paths = _write_rendered_signals(
        output_path,
        signals,
        sample_rate,
        args.decompose,
    )
    report.update(
        {
            "output": str(output_path),
            "sample_rate": sample_rate,
            "piano_model": args.piano_model,
            "maestro_year": piano_models[args.piano_model],
            "warm_up_seconds": warm_up_frames / frame_rate,
            "noise_seed": args.seed,
            "harmonic_rms": float(
                np.sqrt(np.mean(np.square(signals["harmonic"], dtype=np.float64)))
            ),
            "noise_rms": float(
                np.sqrt(np.mean(np.square(signals["noise"], dtype=np.float64)))
            ),
            "dry_rms": float(
                np.sqrt(np.mean(np.square(signals["dry"], dtype=np.float64)))
            ),
            "stems": stem_paths,
        }
    )
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
