"""Small deterministic FDN-style reverb used outside the ONNX graph."""

from __future__ import annotations

import torch
from torch import nn

from ddsp_piano.ddsp_pytorch.core import fft_convolve


FDN_DELAYS = (149, 211, 263, 293)


def fdn_impulse_response(
    controls: torch.Tensor,
    sample_rate: int,
    length: int = 24_000,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert nine FDN controls into a bounded impulse response and wet mix.

    The response is a fixed-delay feedback network approximation. It uses only
    tensor arithmetic and is intentionally kept in the host/training DSP path;
    the stateful ONNX graph exports ``controls`` instead.
    """
    if controls.ndim == 1:
        controls = controls.unsqueeze(0)
    if controls.ndim != 2 or controls.shape[-1] != 9:
        raise ValueError("FDN controls must have shape [batch, 9]")
    if sample_rate <= 0 or length <= 0:
        raise ValueError("sample_rate and length must be positive")

    controls = controls.to(torch.float32)
    gains = 0.35 * torch.sigmoid(controls[:, :4])
    feedback = 0.92 * torch.sigmoid(controls[:, 4:8])
    wet_mix = 0.8 * torch.sigmoid(controls[:, 8:9])
    time = torch.arange(length, device=controls.device, dtype=controls.dtype)
    ir = torch.zeros(controls.shape[0], length, device=controls.device, dtype=controls.dtype)
    damping = torch.exp(-time / controls.new_tensor(float(sample_rate) * 2.4))
    for index, delay in enumerate(FDN_DELAYS):
        pulse = (torch.remainder(time, delay) == 0).to(controls.dtype)
        repeat = torch.floor(time / delay)
        comb = pulse.unsqueeze(0) * feedback[:, index:index + 1].pow(repeat.unsqueeze(0))
        ir = ir + gains[:, index:index + 1] * comb * damping.unsqueeze(0)
    ir = ir / ir.abs().amax(dim=-1, keepdim=True).clamp_min(1e-4)
    return ir, wet_mix


class FDNReverb(nn.Module):
    """Apply the host-side FDN response represented by compact controls."""

    def __init__(self, sample_rate: int = 16_000, length: int = 24_000) -> None:
        super().__init__()
        self.sample_rate = sample_rate
        self.length = length

    def forward(self, audio: torch.Tensor, controls: torch.Tensor) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError("audio must have shape [batch, samples]")
        ir, wet_mix = fdn_impulse_response(controls, self.sample_rate, self.length)
        ir = torch.cat([torch.zeros_like(ir[:, :1]), ir[:, 1:]], dim=-1)
        convolved = fft_convolve(audio, ir, padding="same", delay_compensation=0)
        return audio + wet_mix * convolved
