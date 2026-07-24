from __future__ import annotations

import unittest

import numpy as np

from scripts.render_onnx import _StreamingDrySynthesizer


class StreamingRenderTest(unittest.TestCase):
    def test_dry_synthesis_is_invariant_to_chunk_size(self) -> None:
        frames = 10
        voices = 2
        metadata = {
            "outputs": {
                "amplitudes": [1, 1, voices, 1],
                "harmonic_distribution": [1, 1, voices, 8],
                "inharmonicity": [1, 1, voices, 1],
                "f0_hz": [1, 1, voices, 2],
                "noise_magnitudes": [1, 1, voices, 8],
            }
        }
        controls = {
            "amplitudes": np.zeros((frames, voices, 1), dtype=np.float32),
            "harmonic_distribution": np.zeros((frames, voices, 8), dtype=np.float32),
            "inharmonicity": np.zeros((frames, voices, 1), dtype=np.float32),
            "f0_hz": np.full((frames, voices, 2), 440.0, dtype=np.float32),
            "noise_magnitudes": np.zeros((frames, voices, 8), dtype=np.float32),
        }

        whole = _StreamingDrySynthesizer(metadata, 16_000, 64, 1234)
        expected_harmonic, expected_noise = whole.render(controls)

        chunked = _StreamingDrySynthesizer(metadata, 16_000, 64, 1234)
        harmonic_parts = []
        noise_parts = []
        for start, end in ((0, 4), (4, frames)):
            harmonic, noise = chunked.render(
                {name: value[start:end] for name, value in controls.items()}
            )
            harmonic_parts.append(harmonic)
            noise_parts.append(noise)

        np.testing.assert_allclose(np.concatenate(harmonic_parts), expected_harmonic, atol=1e-4)
        np.testing.assert_allclose(np.concatenate(noise_parts), expected_noise, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
