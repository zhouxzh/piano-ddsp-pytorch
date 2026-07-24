from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from scripts.run_quality_cycle import rank_candidates, review_deadline_expired


class QualityCycleTest(unittest.TestCase):
    def test_ranking_prefers_objective_gate_then_metric(self) -> None:
        records = [
            {"candidate_id": "failed", "hard_gates_passed": False, "objective_eligible": True, "composite_median": 0.5},
            {"candidate_id": "eligible", "hard_gates_passed": True, "objective_eligible": True, "composite_median": 0.97},
            {"candidate_id": "lower_but_ineligible", "hard_gates_passed": True, "objective_eligible": False, "composite_median": 0.9},
        ]
        ranked = rank_candidates(records)
        self.assertEqual([item["candidate_id"] for item in ranked], ["eligible", "lower_but_ineligible"])

    def test_review_deadline(self) -> None:
        now = datetime.now(timezone.utc)
        report = {"human_review": {"deadline": (now - timedelta(seconds=1)).isoformat()}}
        self.assertTrue(review_deadline_expired(report, now))
        report["human_review"]["deadline"] = (now + timedelta(seconds=1)).isoformat()
        self.assertFalse(review_deadline_expired(report, now))


if __name__ == "__main__":
    unittest.main()
