from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from scripts.sweep_stage_checkpoints import (
    candidate_stages,
    promotion_decisions,
    report_row,
)


class CheckpointSweepTest(unittest.TestCase):
    def test_candidate_stages_discovers_and_filters_best_checkpoints(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            run_root = Path(temporary)
            paper = run_root / "training" / "gru_ir_96_64" / "controls" / "checkpoints" / "best.pt"
            film = run_root / "training" / "film_fdn_128_96" / "refine" / "checkpoints" / "best.pt"
            paper.parent.mkdir(parents=True)
            film.parent.mkdir(parents=True)
            paper.touch()
            film.touch()
            self.assertEqual(
                [(model, stage) for model, stage, _ in candidate_stages(run_root, {"gru_ir_96_64"})],
                [("gru_ir_96_64", "controls")],
            )

    def test_report_row_extracts_cross_stage_gate_inputs(self) -> None:
        report = {
            "comparison": {
                "composite_ratio": {"median": 0.97, "p95": 1.1},
                "groups": {"a": {"median": 1.01}, "b": {"median": 1.08}},
                "metric_ratios": {"mrstft": {"median": 0.95}},
            },
            "config": {"gates": {"group_median": 1.05}},
            "summary": {
                "baseline": {"wet_dry_rms_ratio": {"median": 2.0}},
                "candidate": {"wet_dry_rms_ratio": {"median": 1.8}},
            },
            "models": {
                "baseline": {"latency_ms": {"p95": 1.0}},
                "candidate": {
                    "latency_ms": {"p95": 1.1},
                    "unexpected_operators": [],
                    "numerical_allclose": True,
                },
            },
            "verdict": {
                "objective_eligible": False,
                "objective_failures": ["group_regression"],
                "hard_failures": [],
            },
        }
        row = report_row("gru_ir_96_64", "controls", Path("best.pt"), report)
        self.assertEqual(row["regressed_groups"], 1)
        self.assertEqual(row["worst_group_median"], 1.08)
        self.assertAlmostEqual(row["latency_p95_ratio"], 1.1)

    def test_promotion_keeps_baseline_when_best_stage_changes_reverb(self) -> None:
        row = {
            "model_id": "gru_ir_96_64",
            "stage": "controls",
            "checkpoint": "/run/best.pt",
            "composite_median": 0.96,
            "regressed_groups": 0,
            "wet_dry_rms_ratio": {"baseline": 2.0, "candidate": 1.0},
            "latency_p95_ratio": 1.0,
            "numerical_allclose": True,
            "hard_failures": [],
        }
        decision = promotion_decisions([row])["gru_ir_96_64"]
        self.assertEqual(decision["decision"], "baseline")
        self.assertIn("wet_dry_drift", decision["best_rejected"]["reasons"])

    def test_promotion_selects_best_stage_that_passes_all_gates(self) -> None:
        base = {
            "model_id": "film_fdn_128_96",
            "checkpoint": "/run/controls.pt",
            "regressed_groups": 0,
            "wet_dry_rms_ratio": {"baseline": 0.5, "candidate": 0.48},
            "latency_p95_ratio": 1.01,
            "numerical_allclose": True,
            "hard_failures": [],
        }
        controls = {**base, "stage": "controls", "composite_median": 0.97}
        refine = {
            **base,
            "stage": "refine",
            "checkpoint": "/run/refine.pt",
            "composite_median": 0.95,
        }
        decision = promotion_decisions([controls, refine])["film_fdn_128_96"]
        self.assertEqual(decision["decision"], "candidate")
        self.assertEqual(decision["stage"], "refine")

    def test_pilot_gate_can_allow_bounded_group_regressions(self) -> None:
        row = {
            "model_id": "gru_ir_96_64",
            "stage": "pilot",
            "checkpoint": "/run/pilot.pt",
            "composite_median": 0.94,
            "worst_group_median": 1.10,
            "regressed_groups": 3,
            "wet_dry_rms_ratio": {"baseline": 2.0, "candidate": 2.05},
            "latency_p95_ratio": 1.0,
            "numerical_allclose": True,
            "hard_failures": [],
        }
        strict = promotion_decisions([row])["gru_ir_96_64"]
        pilot = promotion_decisions(
            [row], max_composite_median=1.0, max_regressed_groups=3,
            max_worst_group_median=1.11,
        )["gru_ir_96_64"]
        self.assertEqual(strict["decision"], "baseline")
        self.assertEqual(pilot["decision"], "candidate")


if __name__ == "__main__":
    unittest.main()
