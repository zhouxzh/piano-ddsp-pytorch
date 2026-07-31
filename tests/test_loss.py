from __future__ import annotations

import unittest

import torch

from ddsp_piano.modules.loss import HybridLoss, ReverbRegularizer


class LossTest(unittest.TestCase):
    def test_mean_reverb_regularizer_is_independent_of_ir_length(self) -> None:
        regularizer = ReverbRegularizer(weight=0.1, reduction="mean")
        short = regularizer(torch.ones(2, 8))
        long = regularizer(torch.ones(2, 24_000))
        self.assertAlmostEqual(float(short), 0.1, places=6)
        self.assertAlmostEqual(float(long), 0.1, places=6)

    def test_legacy_reverb_regularizer_preserves_sum_semantics(self) -> None:
        regularizer = ReverbRegularizer(weight=0.1, reduction="sum_per_sample")
        self.assertAlmostEqual(float(regularizer(torch.ones(2, 8))), 0.8, places=6)

    def test_energy_and_onset_components_are_trainable(self) -> None:
        prediction = torch.zeros(1, 2048, requires_grad=True)
        target = torch.linspace(-0.2, 0.2, 2048).reshape(1, -1)
        loss = HybridLoss(
            [64],
            torch.nn.Identity(),
            phase=True,
            reverb_mode="fdn",
            energy_weight=0.1,
            onset_weight=0.05,
            sample_rate=16_000,
        )
        components = loss.components(
            prediction,
            target,
            torch.zeros(1, 9),
            dry_pred=prediction,
        )
        self.assertEqual(len(components), 8)
        self.assertGreater(float(components[4].detach()), 0.0)
        self.assertGreater(float(components[5].detach()), 0.0)
        components[0].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())

    def test_perceptual_v2_does_not_compare_dry_audio_to_wet_target(self) -> None:
        wet = torch.linspace(-0.1, 0.1, 2048).reshape(1, -1).requires_grad_()
        target = wet.detach().clone()
        dry = torch.zeros_like(wet)
        loss = HybridLoss(
            [64],
            torch.nn.Identity(),
            phase=True,
            dry_weight=100.0,
            wet_weight=0.75,
            loss_version="perceptual_v2",
        )
        components = loss.components(wet, target, dry_pred=dry)
        self.assertEqual(float(components[3]), 0.0)
        self.assertAlmostEqual(float(components[0].detach()), 0.0, places=5)

    def test_velocity_monotonic_penalty_detects_inverted_response(self) -> None:
        mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        good = HybridLoss.velocity_monotonic_loss(
            torch.tensor([[[[0.0]]]]), torch.tensor([[[[1.0]]]]), mask
        )
        bad = HybridLoss.velocity_monotonic_loss(
            torch.tensor([[[[1.0]]]]), torch.tensor([[[[0.0]]]]), mask
        )
        self.assertLess(float(good), float(bad))

    def test_velocity_response_penalty_matches_calibrated_ratio(self) -> None:
        mask = torch.ones(1, 1, 1, 1, dtype=torch.bool)
        low = torch.tensor([[[[0.0]]]])
        expected = torch.tensor([[[[0.4]]]])
        low_scaled = torch.nn.functional.softplus(low)
        matching_high = torch.log(torch.expm1(low_scaled * torch.exp(expected)))
        matching = HybridLoss.velocity_response_loss(low, matching_high, mask, expected)
        flat = HybridLoss.velocity_response_loss(low, low, mask, expected)
        self.assertLess(float(matching), float(flat))

    def test_centroid_and_release_tail_components_are_trainable(self) -> None:
        torch.manual_seed(4)
        target = torch.randn(1, 4096) * 0.01
        prediction = (target * 0.8).detach().requires_grad_()
        conditioning = torch.zeros(1, 64, 2, 2)
        conditioning[:, :24, 0, 0] = 60.0
        loss = HybridLoss(
            [64],
            torch.nn.Identity(),
            phase=True,
            wet_weight=0.0,
            centroid_weight=0.1,
            tail_weight_audio=0.1,
            sample_rate=16_000,
            frame_rate=250,
            loss_version="perceptual_v2",
        )
        components = loss.components(
            prediction,
            target,
            conditioning=conditioning,
        )
        self.assertGreaterEqual(float(components[6].detach()), 0.0)
        self.assertGreater(float(components[7].detach()), 0.0)
        components[0].backward()
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
