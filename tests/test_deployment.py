from __future__ import annotations

import math
import unittest

import numpy as np
import torch
import tempfile
from pathlib import Path

from ddsp_piano.default_model import get_model, get_v2_model, get_v3_model
from train import load_partial_initialization
from ddsp_piano.ddsp_pytorch.fdn import fdn_impulse_response
from ddsp_piano.ddsp_pytorch.reverb import Reverb
from ddsp_piano.ddsp_pytorch.core import remove_frequencies_above_nyquist
from ddsp_piano.ddsp_pytorch.harmonic_oscillator import HarmonicOscillator
from ddsp_piano.ddsp_pytorch.noise import Noise
from ddsp_piano.deployment import (
    PianoRealtimeControlModel,
    extend_pitch_for_release,
    scale_controls_for_synthesis,
)
from ddsp_piano.modules.inharm_synth import InHarmonic


class DeploymentTest(unittest.TestCase):
    def test_bandlimit_uses_actual_partial_frequencies(self) -> None:
        amplitudes = torch.ones(1, 1, 2)
        frequencies = torch.tensor([[[1_000.0, 9_000.0]]])
        result = remove_frequencies_above_nyquist(amplitudes, frequencies, 16_000)
        torch.testing.assert_close(result, torch.tensor([[[1.0001, 0.0001]]]))

    def test_release_state_is_carried_between_blocks(self) -> None:
        conditioning = np.zeros((1, 3, 2, 2), dtype=np.float32)
        conditioning[0, 0, 0, 0] = 60.0
        held = np.zeros((1, 2), dtype=np.float32)
        released = np.zeros((1, 2), dtype=np.int32)

        extended, held, released = extend_pitch_for_release(
            conditioning, held, released, release_frames=2
        )
        self.assertEqual(extended[0, :, 0, 0].tolist(), [60.0, 60.0, 60.0])

        empty_block = np.zeros((1, 1, 2, 2), dtype=np.float32)
        extended, held, released = extend_pitch_for_release(
            empty_block, held, released, release_frames=2
        )
        self.assertEqual(extended[0, 0, 0, 0], 0.0)
        self.assertEqual(held[0, 0], 0.0)
        self.assertEqual(released[0, 0], 3)

    def test_control_scaling_contract(self) -> None:
        shape = (1, 1, 2)
        controls = scale_controls_for_synthesis(
            amplitudes=np.zeros(shape + (1,), dtype=np.float32),
            harmonic_distribution=np.zeros(shape + (3,), dtype=np.float32),
            inharmonicity=np.zeros(shape + (1,), dtype=np.float32),
            f0_hz=np.full(shape + (2,), 440.0, dtype=np.float32),
            noise_magnitudes=np.zeros(shape + (4,), dtype=np.float32),
            sample_rate=16_000,
        )
        np.testing.assert_allclose(
            controls["harmonic_distribution"].sum(axis=-1),
            np.ones(shape, dtype=np.float32),
            atol=1e-6,
        )
        self.assertEqual(controls["partial_frequencies_hz"].shape, shape + (2, 3))
        self.assertEqual(controls["noise_magnitudes"].shape, shape + (4,))

        torch_synth = InHarmonic(n_samples=64, sample_rate=16_000)
        torch_controls = torch_synth.get_controls(
            torch.zeros(2, 1, 1),
            torch.zeros(2, 1, 3),
            torch.zeros(2, 1, 1),
            torch.full((2, 1, 1), 440.0),
        )
        np.testing.assert_allclose(
            controls["amplitudes"].reshape(2, 1, 1),
            torch_controls["amplitudes"].numpy() / 2,
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            controls["harmonic_distribution"].reshape(2, 1, 3),
            torch_controls["harmonic_distribution"].numpy(),
            rtol=1e-6,
        )
        np.testing.assert_allclose(
            controls["harmonic_shifts"].reshape(2, 1, 3),
            torch_controls["harmonic_shifts"].numpy(),
            rtol=1e-6,
        )

    def test_oscillator_uses_configured_sample_count(self) -> None:
        oscillator = HarmonicOscillator(fs=24_000, n_samples=96)
        audio = oscillator(
            f0=torch.full((1, 1, 1), 440.0),
            amplitudes=torch.ones(1, 1, 1),
            harmonic_distribution=torch.ones(1, 1, 2) / 2,
        )
        self.assertEqual(tuple(audio.shape), (1, 96))

    def test_noise_uses_independent_reproducible_realizations(self) -> None:
        harmonic = torch.zeros(1, 64)
        magnitudes = torch.zeros(1, 1, 4)
        noise = Noise().eval()

        torch.manual_seed(1234)
        first = noise(harmonic, magnitudes)
        second = noise(harmonic, magnitudes)
        self.assertFalse(torch.equal(first, second))

        torch.manual_seed(1234)
        repeated = noise(harmonic, magnitudes)
        torch.testing.assert_close(first, repeated)

    def test_inference_reverb_applies_midi_ddsp_decay(self) -> None:
        model = get_model(
            inference=True,
            n_synths=1,
            n_piano_models=1,
            duration=1 / 250,
            frame_rate=250,
            sample_rate=16_000,
        )
        model.reverb_model.reverb_dict.weight.data.fill_(1.0)
        impulse = model.reverb_model(torch.zeros(1, 1, dtype=torch.int64))
        self.assertEqual(impulse[0, 15_999], 1.0)
        torch.testing.assert_close(impulse[0, -1], torch.tensor(math.exp(-4.0)))

    def test_realtime_control_contract(self) -> None:
        polyphony = 16
        model = get_model(
            n_synths=polyphony,
            n_piano_models=1,
            duration=1 / 250,
            frame_rate=250,
            sample_rate=16_000,
        ).eval()
        wrapper = PianoRealtimeControlModel(model).eval()
        inputs = (
            torch.zeros(1, 1, polyphony, 2),
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, 1, polyphony, 1),
            torch.zeros(1, 1, 64),
            torch.zeros(1, polyphony, 192),
        )
        with torch.inference_mode():
            outputs = wrapper(*inputs)

        self.assertEqual(
            [tuple(output.shape) for output in outputs],
            [
                (1, 1, 16, 1),
                (1, 1, 16, 96),
                (1, 1, 16, 1),
                (1, 1, 16, 2),
                (1, 1, 16, 64),
                (1, 24_000),
                (1, 1, 64),
                (1, 16, 192),
            ],
        )

    def test_v2_control_contract_uses_deep_outputs_and_fdn_controls(self) -> None:
        model = get_v2_model(
            n_synths=16,
            n_piano_models=1,
            duration=1 / 250,
            frame_rate=250,
            sample_rate=16_000,
        ).eval()
        wrapper = PianoRealtimeControlModel(model).eval()
        inputs = (
            torch.zeros(1, 1, 16, 2),
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, 1, 16, 1),
            torch.zeros(1, 1, 64),
            torch.zeros(1, 16, 192),
        )
        with torch.inference_mode():
            outputs = wrapper(*inputs)
        self.assertEqual(tuple(outputs[1].shape), (1, 1, 16, 128))
        self.assertEqual(tuple(outputs[4].shape), (1, 1, 16, 96))
        self.assertEqual(tuple(outputs[5].shape), (1, 9))
        self.assertTrue(torch.isfinite(outputs[5]).all())

    def test_v2_ir_ablation_keeps_configured_control_dimensions(self) -> None:
        model = get_v2_model(
            n_synths=2,
            n_piano_models=1,
            duration=1 / 250,
            frame_rate=250,
            sample_rate=16_000,
            n_harmonics=96,
            n_noise_filter_banks=64,
            reverb_type="ir",
        ).eval()
        wrapper = PianoRealtimeControlModel(model).eval()
        inputs = (
            torch.zeros(1, 1, 2, 2),
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, 1, 2, 1),
            torch.zeros(1, 1, 64),
            torch.zeros(1, 2, 192),
        )
        with torch.inference_mode():
            outputs = wrapper(*inputs)
        self.assertEqual(tuple(outputs[1].shape), (1, 1, 2, 96))
        self.assertEqual(tuple(outputs[4].shape), (1, 1, 2, 64))
        self.assertEqual(tuple(outputs[5].shape), (1, 24_000))

    def test_v2a_residual_film_is_identity_after_v1_initialization(self) -> None:
        kwargs = {
            "n_synths": 16,
            "n_piano_models": 1,
            "duration": 1 / 250,
            "frame_rate": 250,
            "sample_rate": 16_000,
            "reverb_wet_gain": 1.0,
        }
        torch.manual_seed(7)
        v1 = get_model(**kwargs).eval()
        v2a = get_v2_model(
            **kwargs,
            n_harmonics=96,
            n_noise_filter_banks=64,
            reverb_type="ir",
            context_type="residual_film",
            monophonic_type="legacy",
            inharmonicity_type="legacy",
        ).eval()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "v1.pt"
            torch.save({"model": v1.state_dict()}, checkpoint)
            report = load_partial_initialization(v2a, checkpoint, torch.device("cpu"))
        self.assertGreater(report["loaded_tensors"], 0)
        inputs = (
            torch.cat(
                [
                    torch.tensor([[[[60.0, 0.8]]]]),
                    torch.zeros(1, 1, 15, 2),
                ],
                dim=2,
            ),
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.int32),
            torch.cat(
                [torch.tensor([[[[60.0]]]]), torch.zeros(1, 1, 15, 1)],
                dim=2,
            ),
            torch.zeros(1, 1, 64),
            torch.zeros(1, 16, 192),
        )
        with torch.inference_mode():
            v1_outputs = PianoRealtimeControlModel(v1)(*inputs)
            v2a_outputs = PianoRealtimeControlModel(v2a)(*inputs)
        for expected, actual in zip(v1_outputs, v2a_outputs):
            torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)

    def test_v3_factorized_heads_preserve_parent_outputs_at_initialization(self) -> None:
        kwargs = {
            "n_synths": 16,
            "n_piano_models": 1,
            "duration": 1 / 250,
            "frame_rate": 250,
            "sample_rate": 16_000,
            "reverb_wet_gain": 1.0,
        }
        torch.manual_seed(11)
        parent = get_model(**kwargs).eval()
        inputs = (
            torch.zeros(1, 1, 16, 2),
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, 1, 16, 1),
            torch.zeros(1, 1, 64),
            torch.zeros(1, 16, 192),
        )
        inputs[0][0, 0, 0] = torch.tensor([60.0, 0.8])
        inputs[3][0, 0, 0, 0] = 60.0
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "parent.pt"
            torch.save({"model": parent.state_dict()}, checkpoint)
            for gate in ("none", "velocity_onset"):
                candidate = get_v3_model(**kwargs, conditioning_gate=gate).eval()
                report = load_partial_initialization(
                    candidate, checkpoint, torch.device("cpu")
                )
                self.assertIn(
                    "monophonic_network.amplitude_head.weight", report["mapping"]
                )
                with torch.inference_mode():
                    expected = PianoRealtimeControlModel(parent)(*inputs)
                    actual = PianoRealtimeControlModel(candidate)(*inputs)
                for parent_value, candidate_value in zip(expected, actual):
                    torch.testing.assert_close(
                        candidate_value, parent_value, rtol=1e-6, atol=1e-6
                    )

    def test_v3_realtime_contract_keeps_fixed_state_and_control_shapes(self) -> None:
        model = get_v3_model(
            n_synths=16,
            n_piano_models=1,
            duration=1 / 250,
            frame_rate=250,
            sample_rate=16_000,
            conditioning_gate="velocity_onset",
        ).eval()
        wrapper = PianoRealtimeControlModel(model).eval()
        inputs = (
            torch.zeros(1, 1, 16, 2),
            torch.zeros(1, 1, 4),
            torch.zeros(1, dtype=torch.int32),
            torch.zeros(1, 1, 16, 1),
            torch.zeros(1, 1, 64),
            torch.zeros(1, 16, 192),
        )
        with torch.inference_mode():
            outputs = wrapper(*inputs)
        self.assertEqual(tuple(outputs[0].shape), (1, 1, 16, 1))
        self.assertEqual(tuple(outputs[1].shape), (1, 1, 16, 96))
        self.assertEqual(tuple(outputs[4].shape), (1, 1, 16, 64))
        self.assertEqual(tuple(outputs[-2].shape), (1, 1, 64))
        self.assertEqual(tuple(outputs[-1].shape), (1, 16, 192))

    def test_fdn_controls_are_bounded_and_renderable(self) -> None:
        controls = torch.zeros(1, 9)
        impulse, wet_mix = fdn_impulse_response(controls, sample_rate=16_000, length=512)
        self.assertEqual(tuple(impulse.shape), (1, 512))
        self.assertEqual(tuple(wet_mix.shape), (1, 1))
        self.assertTrue(torch.isfinite(impulse).all())
        self.assertLessEqual(float(impulse.abs().max()), 1.0 + 1e-6)
        self.assertGreater(float(wet_mix), 0.0)
        self.assertLess(float(wet_mix), 1.0)

    def test_current_reverb_wet_gain_is_explicit(self) -> None:
        audio = torch.zeros(1, 32)
        audio[:, 0] = 1.0
        impulse = torch.zeros(1, 32)
        impulse[:, 1] = 1.0
        full = Reverb(wet_gain=1.0)(audio, impulse)
        limited = Reverb(wet_gain=0.25)(audio, impulse)
        self.assertGreater(float((full - audio).abs().sum()), 0.0)
        torch.testing.assert_close(limited - audio, 0.25 * (full - audio))


if __name__ == "__main__":
    unittest.main()
