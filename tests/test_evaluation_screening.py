from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from scripts.evaluate_model import _evaluate_model_segments


class EvaluationScreeningTest(unittest.TestCase):
    def test_quick_profile_renders_bounded_context_window(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            cache = Path(temporary)
            np.save(cache / "audio.npy", np.zeros(12_000, dtype=np.float32))
            np.save(cache / "conditioning.npy", np.zeros((120, 2, 2), dtype=np.float32))
            np.save(cache / "pedal.npy", np.zeros((120, 4), dtype=np.float32))
            corpus = {
                "profile": "quick",
                "preprocess": {"segment_seconds": 3.0},
                "entries": [
                    {
                        "id": "segment",
                        "cache_path": str(cache),
                        "track_id": "track",
                        "year": 2004,
                        "category": "onset",
                        "frame_start": 70,
                        "sample_start": 7000,
                        "piano_model": 0,
                    }
                ],
            }
            metadata = {"sample_rate": 1000, "frame_rate": 10}
            config = {
                "screening": {"context_seconds": 2.5},
                "metrics": {"fft_sizes": [64]},
            }

            def render(*args):
                conditioning = args[2]
                samples = conditioning.shape[0] * 100
                signal = np.arange(samples, dtype=np.float32)
                return {name: signal for name in ("harmonic", "noise", "dry", "wet")}

            with (
                patch("scripts.evaluate_model._render_track", side_effect=render) as renderer,
                patch("scripts.evaluate_model.audio_metrics", return_value={}) as metrics,
                patch("scripts.evaluate_model.signal_metrics", return_value={}),
            ):
                result = _evaluate_model_segments(
                    Path("model.onnx"), metadata, corpus, config
                )

            rendered_conditioning = renderer.call_args.args[2]
            self.assertEqual(rendered_conditioning.shape[0], 55)
            prediction = metrics.call_args.args[0]
            self.assertEqual(prediction.shape[0], 3000)
            self.assertEqual(float(prediction[0]), 2500.0)
            self.assertEqual(result[0]["id"], "segment")


if __name__ == "__main__":
    unittest.main()
