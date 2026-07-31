from __future__ import annotations

import unittest
from pathlib import Path

from scripts.train_quality_recovery import (
    promoted_models,
    replace_finetune_with_resume,
    training_command,
)


class QualityRecoveryTest(unittest.TestCase):
    def test_training_command_anchors_to_stable_checkpoint_and_freezes_reverb(self) -> None:
        command = training_command(
            python="python",
            registry=Path("registry.json"),
            model_id="gru_ir_96_64",
            source_checkpoint=Path("stable.pt"),
            experiment_dir=Path("run/pilot"),
            maestro_root=Path("maestro"),
            cache_dir=Path("cache"),
            quality_manifest=Path("quality.json"),
            phase_settings={"epochs": 1, "steps_per_epoch": 10, "validation_batches": 2},
            model_settings={"batch_size": 8, "learning_rate": 1e-5},
            common={
                "stage": "controls",
                "sampling_mode": "coverage",
                "curriculum_tail_fraction": 0.2,
                "train_workers": 8,
                "validation_workers": 2,
                "log_every": 100,
                "reverb_regularizer_reduction": "mean",
                "freeze_reverb": True,
            },
        )
        self.assertIn("--finetune-from", command)
        self.assertEqual(command[command.index("--finetune-from") + 1], "stable.pt")
        self.assertIn("--freeze-reverb", command)
        self.assertEqual(
            command[command.index("--reverb-regularizer-reduction") + 1], "mean"
        )
        resumed = replace_finetune_with_resume(command, Path("last.pt"))
        self.assertNotIn("--finetune-from", resumed)
        self.assertEqual(resumed[resumed.index("--resume") + 1], "last.pt")
        self.assertIn("--freeze-reverb", resumed)

    def test_only_promoted_models_continue_to_full_training(self) -> None:
        summary = {
            "promotions": {
                "gru_ir_96_64": {"decision": "baseline"},
                "film_fdn_128_96": {"decision": "candidate"},
            }
        }
        self.assertEqual(promoted_models(summary), ["film_fdn_128_96"])


if __name__ == "__main__":
    unittest.main()
