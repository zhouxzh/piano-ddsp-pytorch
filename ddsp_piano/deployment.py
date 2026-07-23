"""Deployment-facing wrappers that exclude FFT-based audio synthesis."""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn


class PianoControlModel(nn.Module):
    """Export the fixed-shape neural control network for Ascend deployment."""

    def __init__(self, piano_model: nn.Module) -> None:
        super().__init__()
        self.piano_model = piano_model

    def forward(
        self,
        conditioning: torch.Tensor,
        pedal: torch.Tensor,
        piano_model: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        return self.piano_model.predict_controls(conditioning, pedal, piano_model)


class PianoRealtimeControlModel(nn.Module):
    """Fixed-block control model with explicit recurrent state for ONNX."""

    def __init__(self, piano_model: nn.Module) -> None:
        super().__init__()
        self.piano_model = piano_model

    @staticmethod
    def _batch_first(value: torch.Tensor) -> torch.Tensor:
        return value.permute(1, 2, 0, 3)

    def forward(
        self,
        conditioning: torch.Tensor,
        pedal: torch.Tensor,
        piano_model: torch.Tensor,
        extended_pitch: torch.Tensor,
        context_state: torch.Tensor,
        monophonic_state: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        outputs = self.piano_model.predict_controls_stateful(
            conditioning,
            pedal,
            piano_model.to(torch.int64),
            extended_pitch,
            context_state,
            monophonic_state,
        )
        controls = tuple(self._batch_first(value) for value in outputs[:5])
        return controls + outputs[5:]


def extend_pitch_for_release(
    conditioning: np.ndarray,
    held_pitch: np.ndarray,
    released_frames: np.ndarray,
    release_frames: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the training release rule to a streaming MIDI conditioning block.

    Shapes are ``conditioning=[B,T,P,2]`` and state ``[B,P]``. The function
    returns ``extended_pitch=[B,T,P,1]`` plus the next state arrays.
    """
    if conditioning.ndim != 4 or conditioning.shape[-1] != 2:
        raise ValueError("conditioning must have shape [batch, frames, polyphony, 2]")
    expected_state_shape = (conditioning.shape[0], conditioning.shape[2])
    if held_pitch.shape != expected_state_shape or released_frames.shape != expected_state_shape:
        raise ValueError(f"release states must have shape {expected_state_shape}")
    if release_frames < 0:
        raise ValueError("release_frames must be non-negative")

    held_pitch = held_pitch.astype(np.float32, copy=True)
    released_frames = released_frames.astype(np.int32, copy=True)
    extended = np.empty(conditioning.shape[:-1] + (1,), dtype=np.float32)
    for frame in range(conditioning.shape[1]):
        current_pitch = conditioning[:, frame, :, 0]
        active = current_pitch > 0
        released_frames = np.where(active, 0, released_frames + 1)
        held_pitch = np.where(active, current_pitch, held_pitch)
        held_pitch = np.where(released_frames <= release_frames, held_pitch, 0.0)
        extended[:, frame, :, 0] = held_pitch
    return extended, held_pitch, released_frames


def scale_controls_for_synthesis(
    amplitudes: np.ndarray,
    harmonic_distribution: np.ndarray,
    inharmonicity: np.ndarray,
    f0_hz: np.ndarray,
    noise_magnitudes: np.ndarray,
    sample_rate: int,
) -> dict[str, np.ndarray]:
    """CPU reference for turning raw ONNX outputs into DDSP controls."""
    prefix = amplitudes.shape[:-1]
    if amplitudes.shape[-1] != 1:
        raise ValueError("amplitudes must end in one channel")
    if harmonic_distribution.shape[:-1] != prefix:
        raise ValueError("harmonic_distribution prefix must match amplitudes")
    if inharmonicity.shape != amplitudes.shape:
        raise ValueError("inharmonicity must match amplitudes")
    if f0_hz.shape[:-1] != prefix:
        raise ValueError("f0_hz prefix must match amplitudes")
    if noise_magnitudes.shape[:-1] != prefix:
        raise ValueError("noise_magnitudes prefix must match amplitudes")
    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive")

    def scale(value: np.ndarray) -> np.ndarray:
        value = value.astype(np.float32, copy=False)
        sigmoid = 1.0 / (1.0 + np.exp(-np.clip(value, -60.0, 60.0)))
        return (2.0 * np.power(sigmoid, math.log(10.0)) + 1e-7).astype(np.float32)

    harmonic_count = harmonic_distribution.shape[-1]
    partial = np.arange(1, harmonic_count + 1, dtype=np.float32)
    reshape = (1,) * len(prefix) + (harmonic_count,)
    partial = partial.reshape(reshape)
    inharmonic_factor = np.sqrt(1.0 + inharmonicity * np.square(partial))
    harmonic_shifts = inharmonic_factor - 1.0
    first_string_frequencies = f0_hz[..., :1] * partial * inharmonic_factor

    scaled_harmonics = scale(harmonic_distribution)
    nyquist_mask = (first_string_frequencies < sample_rate / 2).astype(np.float32) + 1e-4
    scaled_harmonics *= nyquist_mask
    scaled_harmonics /= np.sum(scaled_harmonics, axis=-1, keepdims=True)

    audible = (f0_hz[..., :1] > 20.0).astype(np.float32) + 1e-4
    scaled_amplitudes = scale(amplitudes) * audible / f0_hz.shape[-1]
    all_partial_frequencies = f0_hz[..., np.newaxis] * partial * inharmonic_factor[..., np.newaxis, :]
    return {
        "amplitudes": scaled_amplitudes.astype(np.float32),
        "harmonic_distribution": scaled_harmonics.astype(np.float32),
        "harmonic_shifts": harmonic_shifts.astype(np.float32),
        "partial_frequencies_hz": all_partial_frequencies.astype(np.float32),
        "noise_magnitudes": scale(noise_magnitudes),
    }
