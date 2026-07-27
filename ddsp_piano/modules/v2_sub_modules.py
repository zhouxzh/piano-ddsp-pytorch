"""PyTorch equivalents of the main DDSP-Piano v2 control modules."""

from __future__ import annotations

import torch
from torch import nn

from ddsp_piano.modules import sub_modules


class ResidualFiLMContextNetwork(nn.Module):
    """Identity-initialized FiLM adapter around the v1 context network."""

    def __init__(self, z_dim: int = 16, context_dim: int = 32) -> None:
        super().__init__()
        self.base = sub_modules.ContextNetwork()
        self.film = nn.Linear(z_dim, context_dim * 2)
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)

    @property
    def gru(self) -> nn.GRU:
        return self.base.gru

    def _adapt(self, context: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        z = z.expand(-1, context.shape[1], -1)
        scale, bias = self.film(z).chunk(2, dim=-1)
        return context * (1.0 + scale) + bias

    def forward_stateful(self, conditioning, pedal, z, hidden_state=None):
        context, next_state = self.base.forward_stateful(
            conditioning, pedal, z, hidden_state
        )
        return self._adapt(context, z), next_state

    def forward(self, conditioning, pedal, z):
        context, _ = self.forward_stateful(conditioning, pedal, z)
        return context


class ResidualDeepMonophonicNetwork(nn.Module):
    """Zero-output deep adapter around the fixed 96/64 v1 decoder."""

    def __init__(self, context_dim: int = 32, hidden_dim: int = 64) -> None:
        super().__init__()
        self.base = sub_modules.MonophonicNetwork()
        self.midi_norm = 128.0
        self.residual = nn.Sequential(
            nn.Linear(context_dim + 3, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, 161),
        )
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)

    @property
    def gru(self) -> nn.GRU:
        return self.base.gru

    def _residual(self, conditioning, extended_pitch, context):
        normalized = torch.cat(
            [
                extended_pitch / self.midi_norm,
                conditioning / conditioning.new_tensor([self.midi_norm, 1.0]),
                context,
            ],
            dim=-1,
        )
        return torch.split(self.residual(normalized), [1, 96, 64], dim=-1)

    def forward_stateful(self, conditioning, extended_pitch, context, hidden_state=None):
        amplitude, harmonics, noise, next_state = self.base.forward_stateful(
            conditioning, extended_pitch, context, hidden_state
        )
        delta_amplitude, delta_harmonics, delta_noise = self._residual(
            conditioning, extended_pitch, context
        )
        return (
            amplitude + delta_amplitude,
            harmonics + delta_harmonics,
            noise + delta_noise,
            next_state,
        )

    def forward(self, conditioning, extended_pitch, context):
        return self.forward_stateful(conditioning, extended_pitch, context)[:-1]


class ResidualJointInharmonicity(nn.Module):
    """Identity-initialized pitch-curve correction around the v1 prior."""

    def __init__(self) -> None:
        super().__init__()
        self.base = sub_modules.InharmonicityNetwork()
        self.slopes_modifier = nn.Parameter(torch.zeros(2))
        self.offsets_modifier = nn.Parameter(torch.zeros(2))

    def forward(self, extended_pitch, global_inharm=None):
        base = self.base(extended_pitch, global_inharm)
        pitch = extended_pitch / 128.0
        correction = (
            self.slopes_modifier * pitch + self.offsets_modifier
        ).sum(dim=-1, keepdim=True)
        return base * torch.exp(correction.clamp(-2.0, 2.0))


class FiLMContextNetwork(nn.Module):
    """Context encoder with piano-conditioned feature-wise modulation."""

    def __init__(
        self,
        z_dim: int = 16,
        hidden_dim: int = 64,
        context_dim: int = 32,
        n_synths: int = 16,
    ) -> None:
        super().__init__()
        self.n_synths = n_synths
        self.conditioning_head = nn.Sequential(
            nn.Linear(2 * n_synths, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
        )
        self.pedal_head = nn.Sequential(
            nn.Linear(4, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
        )
        self.main = nn.Linear(hidden_dim * 2, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, 1, batch_first=True)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.film = nn.Linear(z_dim, hidden_dim * 2)
        self.output = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, context_dim),
        )

    @staticmethod
    def collapse_conditioning(conditioning: torch.Tensor) -> torch.Tensor:
        return conditioning.reshape(conditioning.shape[0], conditioning.shape[1], -1)

    def forward_stateful(self, conditioning, pedal, z, hidden_state=None):
        conditioning = self.collapse_conditioning(conditioning)
        conditioning = conditioning / conditioning.new_tensor([128.0, 1.0]).repeat(
            self.n_synths
        )
        x = torch.cat([self.conditioning_head(conditioning), self.pedal_head(pedal)], dim=-1)
        x = torch.nn.functional.leaky_relu(self.main(x), 0.1)
        x, next_state = self.gru(x, hidden_state)
        x = torch.nn.functional.leaky_relu(self.layer_norm(x), 0.1)
        # ``expand`` accepts both the one-frame export shape and the full
        # training segment shape without data-dependent Python control flow.
        z = z.expand(-1, x.shape[1], -1)
        film = self.film(z)
        coefficient, bias = film.chunk(2, dim=-1)
        return self.output(x * coefficient + bias), next_state

    def forward(self, conditioning, pedal, z):
        context, _ = self.forward_stateful(conditioning, pedal, z)
        return context


class MonophonicDeepNetwork(nn.Module):
    """Deep input/output stacks around a causal GRU decoder."""

    def __init__(self, context_dim: int = 32, hidden_dim: int = 64, rnn_dim: int = 192,
                 n_harmonics: int = 128, n_noise_bands: int = 96) -> None:
        super().__init__()
        self.midi_norm = 128.0

        def stack(input_dim: int) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
                nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
            )

        self.pitch_stack = stack(1)
        self.conditioning_stack = stack(2)
        self.context_stack = stack(context_dim)
        self.gru = nn.GRU(hidden_dim * 3, rnn_dim, 1, batch_first=True)
        self.out_stack = nn.Sequential(
            nn.Linear(hidden_dim * 3 + rnn_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
            nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(0.1),
        )
        self.dense_out = nn.Linear(hidden_dim, 1 + n_harmonics + n_noise_bands)
        self.n_harmonics = n_harmonics
        self.n_noise_bands = n_noise_bands

    def forward_stateful(self, conditioning, extended_pitch, context, hidden_state=None):
        pitch = self.pitch_stack(extended_pitch / self.midi_norm)
        normalized_conditioning = conditioning / conditioning.new_tensor([self.midi_norm, 1.0])
        condition = self.conditioning_stack(normalized_conditioning)
        context_features = self.context_stack(context)
        latent = torch.cat([pitch, condition, context_features], dim=-1)
        recurrent, next_state = self.gru(latent, hidden_state)
        output = self.dense_out(self.out_stack(torch.cat([latent, recurrent], dim=-1)))
        return (*torch.split(output, [1, self.n_harmonics, self.n_noise_bands], dim=-1), next_state)

    def forward(self, conditioning, extended_pitch, context):
        return self.forward_stateful(conditioning, extended_pitch, context)[:-1]


class JointParametricInharmTuning(nn.Module):
    """Joint bass/treble inharmonicity curve used by the v2 model."""

    def __init__(self) -> None:
        super().__init__()
        self.slopes_modifier = nn.Parameter(torch.zeros(2))
        self.offsets_modifier = nn.Parameter(torch.zeros(2))
        # DAFx/DDSP piano priors expressed in the normalized MIDI domain.
        # The slope is scaled by 128 so the resulting coefficient stays in the
        # physical piano range (roughly 1e-5..1e-2), rather than saturating the
        # oscillator's inharmonicity control.
        self.register_buffer("base_slopes", torch.tensor([-10.84, 11.85]))
        self.register_buffer("base_offsets", torch.tensor([0.536, -1.15]))

    def forward(self, extended_pitch, global_inharm=None):
        pitch = extended_pitch / 128.0
        slopes = self.base_slopes + self.slopes_modifier
        offsets = self.base_offsets + self.offsets_modifier
        curve = slopes * (pitch + offsets)
        coefficient = torch.exp(curve).sum(dim=-1, keepdim=True)
        if global_inharm is not None:
            coefficient = coefficient * torch.exp(0.1 * global_inharm)
        return coefficient.clamp(1e-5, 0.2)


class V2FDNReverbControls(nn.Module):
    """Per-instrument FDN parameters exported as a compact host control vector."""

    def __init__(self, n_instruments: int, control_dim: int = 9) -> None:
        super().__init__()
        self.embedding = nn.Embedding(n_instruments, control_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=0.05)
        self.control_dim = control_dim

    def forward(self, piano_model: torch.Tensor) -> torch.Tensor:
        if piano_model.ndim > 1:
            piano_model = piano_model[..., 0]
        return self.embedding(piano_model.to(torch.int64))
