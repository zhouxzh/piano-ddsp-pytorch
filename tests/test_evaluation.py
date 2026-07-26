from __future__ import annotations

import math
import unittest

import numpy as np

from ddsp_piano.evaluation import audio_metrics, compare_models, signal_metrics


class EvaluationTest(unittest.TestCase):
    def test_identical_errors_have_neutral_ratio(self) -> None:
        metrics = {
            "mrstft": 0.0,
            "loudness_error_lu": 0.0,
            "onset_envelope_l1": 0.0,
            "spectral_centroid_error": 0.0,
            "tail_decay_error_db_per_second": 0.0,
        }
        segment = {
            "id": "same",
            "year": 2004,
            "category": "quiet",
            "audio_metrics": metrics,
        }
        weights = {name: 0.2 for name in metrics}
        comparison = compare_models([segment], [segment], weights)
        self.assertAlmostEqual(comparison["composite_ratio"]["median"], 1.0)

    def test_metrics_are_finite_for_silence_and_signal(self) -> None:
        sample_rate = 16_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        signal = 0.1 * np.sin(2 * math.pi * 440.0 * time)
        for prediction in (signal, np.zeros_like(signal)):
            metrics = audio_metrics(prediction, signal, sample_rate, [256, 128, 64])
            for value in metrics.values():
                if isinstance(value, float):
                    self.assertTrue(math.isfinite(value))

    def test_signal_metrics_report_all_stems(self) -> None:
        signals = {
            name: np.ones(64, dtype=np.float32)
            for name in ("harmonic", "noise", "dry", "wet")
        }
        result = signal_metrics(signals)
        self.assertTrue(all(result[f"{name}_finite"] for name in signals))
        self.assertEqual(result["wet_dry_rms_ratio"], 1.0)

    def test_timbre_metrics_are_loudness_matched(self) -> None:
        sample_rate = 16_000
        time = np.arange(sample_rate, dtype=np.float32) / sample_rate
        target = 0.1 * np.sin(2 * math.pi * 440.0 * time)
        quiet = target * 0.25
        metrics = audio_metrics(quiet, target, sample_rate, [256, 128, 64])
        self.assertGreater(metrics["loudness_error_lu"], 6.0)
        self.assertLess(metrics["mrstft"], 0.02)
        self.assertAlmostEqual(metrics["timbre_match_gain_db"], 12.0, delta=0.5)


if __name__ == "__main__":
    unittest.main()
