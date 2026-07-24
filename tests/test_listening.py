from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from ddsp_piano.evaluation import LISTENING_SCHEMA, read_json, write_json
from ddsp_piano.listening import DIMENSIONS, create_listening_package, defer_review, finalize_review


class ListeningTest(unittest.TestCase):
    def _item(self) -> dict:
        baseline = np.linspace(-0.1, 0.1, 1600, dtype=np.float32)
        candidate = baseline * 0.9
        return {
            "id": "excerpt",
            "title": "Excerpt",
            "sample_rate": 16_000,
            "baseline_full_peak": 0.1,
            "candidate_full_peak": 0.09,
            "baseline": {name: baseline for name in ("harmonic", "noise", "dry", "wet")},
            "candidate": {name: candidate for name in ("harmonic", "noise", "dry", "wet")},
        }

    def test_deferred_review_can_be_finalized_later(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary)
            review = create_listening_package(
                output, "evaluation-1", [self._item()], 0, -23.0, -3.0
            )
            report = {
                "evaluation_id": "evaluation-1",
                "human_review": review,
                "verdict": {"objective_eligible": True, "human_status": "pending", "promotion_eligible": False},
            }
            write_json(output / "report.json", report)
            deferred = defer_review(output)
            self.assertEqual(deferred["human_review"]["status"], "deferred")

            mapping = read_json(output / "private" / "blind_mapping.json")["mapping"]
            answers = []
            for trial_id, sides in mapping.items():
                candidate_side = "A" if sides["A"] == "candidate" else "B"
                ratings = {
                    name: {"A": 3, "B": 3}
                    for name in DIMENSIONS
                }
                ratings["timbre"][candidate_side] = 4
                answers.append(
                    {
                        "trial_id": trial_id,
                        "preference": candidate_side,
                        "ratings": ratings,
                        "severe_artifact": {"A": False, "B": False},
                        "notes": "",
                    }
                )
            scores = output / "scores.json"
            write_json(
                scores,
                {
                    "schema": LISTENING_SCHEMA,
                    "evaluation_id": "evaluation-1",
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                    "trials": answers,
                },
            )
            config = {
                "gates": {
                    "human_preference_rate": 0.6,
                    "human_dimension_regression": 0.25,
                    "human_repeated_artifacts": 2,
                }
            }
            finalized = finalize_review(output, scores, config)
            self.assertEqual(finalized["human_review"]["status"], "passed")
            self.assertTrue(finalized["human_review"]["submitted_after_deadline"])
            self.assertTrue(finalized["verdict"]["promotion_eligible"])
            self.assertTrue((output / "listening" / "index.html").is_file())


if __name__ == "__main__":
    unittest.main()
