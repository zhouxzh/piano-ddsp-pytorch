from __future__ import annotations

import unittest

import torch

from ddsp_piano.modules.loss import HybridLoss


class LossTest(unittest.TestCase):
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
        self.assertEqual(len(components), 6)
        self.assertGreater(float(components[4].detach()), 0.0)
        self.assertGreater(float(components[5].detach()), 0.0)
        components[0].backward()
        self.assertIsNotNone(prediction.grad)
        self.assertTrue(torch.isfinite(prediction.grad).all())


if __name__ == "__main__":
    unittest.main()
