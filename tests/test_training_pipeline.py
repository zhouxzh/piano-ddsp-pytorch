import json
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from scripts.train_model_suite import (
    benchmark_memory,
    benchmark_report_path,
    load_json,
    resolve_stage_batch_size,
    run,
    stage_plan,
)


class TrainingPipelineTest(unittest.TestCase):
    def setUp(self) -> None:
        self.registry = load_json(Path("ddsp_piano/model-suite-v1.1.0-rc1.json"))

    def test_only_high_memory_refine_stages_use_batch_six(self) -> None:
        batches = {
            (model_id, stage): resolve_stage_batch_size(settings, 8)
            for model_id, stage, settings, _ in stage_plan(self.registry)
        }
        self.assertEqual(batches[("film_fdn", "refine")], 6)
        self.assertEqual(batches[("calibrated_film_ir", "refine")], 6)
        self.assertTrue(
            all(
                batch_size == 8
                for key, batch_size in batches.items()
                if key
                not in {
                    ("film_fdn", "refine"),
                    ("calibrated_film_ir", "refine"),
                }
            )
        )

    def test_benchmark_report_is_stage_specific(self) -> None:
        self.assertEqual(
            benchmark_report_path(Path("run"), "film_fdn", "refine", 6),
            Path("run/benchmarks/film_fdn-refine-batch-6.json"),
        )

    def test_memory_gate_uses_reserved_memory(self) -> None:
        limit = 26 * 1024**3
        self.assertTrue(
            benchmark_memory(
                {"peak_cuda_memory_reserved_bytes": limit}, 26.0
            )["passed"]
        )
        self.assertFalse(
            benchmark_memory(
                {"peak_cuda_memory_reserved_bytes": limit + 1}, 26.0
            )["passed"]
        )

    def test_failed_subprocess_is_recorded_in_pipeline_state(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            state_path = Path(temporary) / "pipeline-state.json"
            state = {"status": "running", "completed_stages": []}
            error = subprocess.CalledProcessError(7, ["train.py"])
            with mock.patch(
                "scripts.train_model_suite.subprocess.run", side_effect=error
            ):
                with self.assertRaises(subprocess.CalledProcessError):
                    run(
                        ["train.py"],
                        state,
                        state_path,
                        False,
                        operation="train",
                        stage_key="film_fdn/refine",
                        stage_batch_size=6,
                    )
            persisted = json.loads(state_path.read_text(encoding="utf-8"))
        self.assertEqual(persisted["status"], "failed")
        self.assertEqual(persisted["failed_stage"], "film_fdn/refine")
        self.assertEqual(persisted["return_code"], 7)
        self.assertEqual(persisted["failure_reason"], "command_failed")


if __name__ == "__main__":
    unittest.main()
