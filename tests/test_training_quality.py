from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from ddsp_piano.training_quality import (
    MixedCurriculumSampler,
    build_quality_manifest,
    load_quality_manifest,
    write_quality_manifest,
)
from scripts.run_quality_finetune import objective_gate, reverb_delta_gate
from scripts.run_v3_quality_cycle import final_gate, screen_gate, severe_regression


class TinyQualityDataset:
    def __init__(self, root: Path) -> None:
        self.config = SimpleNamespace(segment_samples=4096, segment_frames=64)
        self.index = []
        for index in range(8):
            cache = root / str(index)
            cache.mkdir(parents=True)
            audio = (
                np.sin(np.linspace(0, 20 + index, 4096)) * (0.01 + index * 0.002)
            ).astype(np.float32)
            conditioning = np.zeros((64, 2, 2), dtype=np.float32)
            conditioning[: 16 + index, 0, 0] = 48 + index
            conditioning[0, 0, 1] = 0.2 + index * 0.08
            np.save(cache / "audio.npy", audio)
            np.save(cache / "conditioning.npy", conditioning)
            np.save(cache / "polyphony.npy", (conditioning[..., 0] > 0).sum(axis=-1))
            self.index.append((str(cache), 0, 0, index % 2))

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, item: int):
        audio = torch.from_numpy(np.load(Path(self.index[item][0]) / "audio.npy"))
        conditioning = torch.zeros(64, 2, 2)
        conditioning[: 16 + item, 0, 0] = 48 + item
        conditioning[0, 0, 1] = 0.2 + item * 0.08
        pedal = torch.zeros(64, 4)
        return audio, conditioning, pedal, torch.tensor(item % 2)


def fake_report(candidate=None, baseline=None, composite=0.95):
    candidate = candidate or {
        "loudness_error_lu": {"p95": 5.0},
        "spectral_centroid_error": {"p95": 0.015},
        "tail_decay_error_db_per_second": {"p95": 7.5},
    }
    baseline = baseline or {
        "loudness_error_lu": {"p95": 5.0},
        "spectral_centroid_error": {"p95": 0.017},
        "tail_decay_error_db_per_second": {"p95": 8.5},
    }
    return {
        "verdict": {"hard_gates_passed": True, "hard_failures": []},
        "comparison": {
            "composite_ratio": {"median": composite},
            "groups": {"year:2018": {"median": 1.0}},
            "metric_ratios": {"mrstft": {"median": 1.0}},
        },
        "summary": {"candidate": candidate, "baseline": baseline},
        "models": {
            "baseline": {"latency_ms": {"p95": 0.10}},
            "candidate": {"latency_ms": {"p95": 0.10}},
        },
    }


class TrainingQualityTest(unittest.TestCase):
    def test_manifest_is_train_only_deterministic_and_validated(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            dataset = TinyQualityDataset(root / "cache")
            first = build_quality_manifest(dataset, 16_000, 250, seed=7)
            second = build_quality_manifest(dataset, 16_000, 250, seed=7)
            self.assertEqual(first, second)
            self.assertEqual(first["split"], "train")
            self.assertEqual(len(first["entries"]), len(dataset))
            path = root / "manifest.json"
            write_quality_manifest(path, first)
            loaded = load_quality_manifest(path, dataset)
        self.assertEqual(loaded["dataset_index_sha256"], first["dataset_index_sha256"])

    def test_mixed_curriculum_sampler_is_reproducible(self):
        first = list(MixedCurriculumSampler([1, 2, 3, 4], torch.Generator().manual_seed(5)))
        second = list(MixedCurriculumSampler([1, 2, 3, 4], torch.Generator().manual_seed(5)))
        self.assertEqual(first, second)
        self.assertEqual(len(first), 4)
        self.assertTrue(all(0 <= index < 4 for index in first))

    def test_objective_and_reverb_gates(self):
        objective_thresholds = {
            "composite_median": 0.98,
            "group_median": 1.02,
            "mrstft_median_ratio": 1.005,
            "latency_p95_ratio": 1.05,
            "loudness_p95": 5.5,
            "centroid_p95": 0.0165,
            "tail_p95": 8.1,
        }
        passed, failures = objective_gate(fake_report(), objective_thresholds)
        self.assertTrue(passed, failures)
        failed, failures = objective_gate(fake_report(composite=1.01), objective_thresholds)
        self.assertFalse(failed)
        self.assertTrue(any("composite" in failure for failure in failures))

        reverb_thresholds = {
            "maximum_composite_ratio": 1.005,
            "maximum_loudness_regression_lu": 0.25,
            "required_tail_or_centroid_improvement": 0.05,
        }
        passed, failures = reverb_delta_gate(fake_report(composite=1.0), reverb_thresholds)
        self.assertTrue(passed, failures)

    def test_v3_relative_screen_and_final_gates(self):
        report = fake_report(
            candidate={
                "loudness_error_lu": {"p95": 4.5},
                "spectral_centroid_error": {"p95": 0.015},
                "tail_decay_error_db_per_second": {"p95": 7.8},
            },
            baseline={
                "loudness_error_lu": {"p95": 5.0},
                "spectral_centroid_error": {"p95": 0.017},
                "tail_decay_error_db_per_second": {"p95": 8.0},
            },
            composite=0.99,
        )
        screen, failures = screen_gate(
            report,
            {
                "composite_max": 1.0,
                "mrstft_max": 1.005,
                "year_max": 1.02,
                "loudness_p95_ratio_max": 0.95,
                "centroid_p95_ratio_max": 0.95,
            },
        )
        self.assertTrue(screen, failures)
        self.assertFalse(
            severe_regression(report, {"composite_max": 1.02, "year_max": 1.08})
        )
        final, failures = final_gate(
            report,
            fake_report(composite=0.92),
            {
                "base_composite_max": 1.0,
                "v1_composite_max": 0.9339,
                "loudness_p95_ratio_max": 0.90,
                "centroid_p95_ratio_max": 0.90,
                "tail_p95_ratio_max": 1.05,
            },
        )
        self.assertTrue(final, failures)


if __name__ == "__main__":
    unittest.main()
