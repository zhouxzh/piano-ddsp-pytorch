from __future__ import annotations

import unittest
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

from scripts.run_quality_cycle import QualityCycle, rank_candidates, review_deadline_expired


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

    def test_all_finalists_finish_before_review_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cycle = QualityCycle.__new__(QualityCycle)
            cycle.config = {
                "candidates": [
                    {"id": "first", "route": "a"},
                    {"id": "fallback", "route": "b"},
                ],
                "stages": [
                    {"name": "semifinal", "profile": "dev", "keep": 2},
                    {"name": "final", "profile": "release"},
                ],
            }
            cycle.evaluation_config = {}
            cycle.human_review_policy = "after_training_manual"
            cycle.runtime_evaluation_config = Path(temporary) / "evaluation.json"
            cycle.state = {
                "active_candidates": ["first", "fallback"],
                "candidates": {
                    "first": {"stages": {}},
                    "fallback": {"stages": {}},
                },
            }
            events = []
            cycle.prepare = lambda profile: events.append(("prepare", profile))

            def execute(candidate, stage, **kwargs):
                events.append(("execute", stage["name"], candidate["id"], kwargs))
                record = {
                    "candidate_id": candidate["id"],
                    "hard_gates_passed": True,
                    "objective_eligible": True,
                    "composite_median": 0.9 if candidate["id"] == "first" else 0.95,
                    "human_status": "prepared",
                }
                key = stage["name"] + kwargs.get("suffix", "")
                cycle.state["candidates"][candidate["id"]]["stages"][key] = record
                return record

            cycle.execute_stage = execute
            cycle.collect_review_status = lambda record: events.append(
                ("collect", record["candidate_id"])
            ) or "deferred"
            cycle.save = lambda *args, **kwargs: None
            cycle.summarize = lambda: None

            cycle.run()

        final_executes = [
            index
            for index, event in enumerate(events)
            if event[:2] == ("execute", "final")
        ]
        collections = [
            index for index, event in enumerate(events) if event[0] == "collect"
        ]
        self.assertEqual(len(final_executes), 2)
        self.assertLess(max(final_executes), min(collections))
        self.assertFalse(any(event[0] in {"activate", "wait"} for event in events))
        self.assertEqual(cycle.state["status"], "awaiting_human_review")


if __name__ == "__main__":
    unittest.main()
