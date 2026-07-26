from __future__ import annotations

import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np
import onnxruntime as ort

from scripts.render_onnx import (
    CONTROL_OUTPUT_NAMES,
    INPUT_NAMES,
    _StreamingDrySynthesizer,
    _StreamingOnnxRunner,
)


class StreamingRenderTest(unittest.TestCase):
    def test_realtime_runner_uses_bounded_sequential_onnx_threads(self) -> None:
        metadata = {
            "release_frames": 250,
            "inputs": {
                "conditioning": [1, 1, 16, 2],
                "context_state": [1, 1, 64],
                "monophonic_state": [1, 16, 192],
            },
        }
        session = mock.Mock()
        session.get_inputs.return_value = [
            SimpleNamespace(name=name) for name in INPUT_NAMES
        ]
        session.get_outputs.return_value = [
            SimpleNamespace(name=name)
            for name in CONTROL_OUTPUT_NAMES
            + ["reverb_ir", "next_context_state", "next_monophonic_state"]
        ]

        with mock.patch(
            "scripts.render_onnx.ort.InferenceSession", return_value=session
        ) as inference_session:
            runner = _StreamingOnnxRunner(
                "model.onnx",
                metadata,
                piano_model=0,
                reverb_output_name="reverb_ir",
                intra_op_threads=1,
                inter_op_threads=1,
            )

        options = inference_session.call_args.kwargs["sess_options"]
        self.assertEqual(options.intra_op_num_threads, 1)
        self.assertEqual(options.inter_op_num_threads, 1)
        self.assertEqual(options.execution_mode, ort.ExecutionMode.ORT_SEQUENTIAL)
        self.assertEqual(runner.intra_op_threads, 1)
        self.assertEqual(runner.inter_op_threads, 1)

    def test_realtime_runner_rejects_nonpositive_thread_counts(self) -> None:
        with self.assertRaisesRegex(ValueError, "intra_op_threads must be positive"):
            _StreamingOnnxRunner(
                "model.onnx",
                {},
                piano_model=0,
                reverb_output_name="reverb_ir",
                intra_op_threads=0,
            )

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

    def test_voice_envelope_gates_harmonic_and_noise_outputs(self) -> None:
        frames = 2
        voices = 1
        samples_per_frame = 64
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
            "amplitudes": np.full((frames, voices, 1), 0.2, dtype=np.float32),
            "harmonic_distribution": np.full(
                (frames, voices, 8), 0.125, dtype=np.float32
            ),
            "inharmonicity": np.zeros((frames, voices, 1), dtype=np.float32),
            "f0_hz": np.full((frames, voices, 2), 440.0, dtype=np.float32),
            "noise_magnitudes": np.full(
                (frames, voices, 8), 0.1, dtype=np.float32
            ),
        }
        synthesizer = _StreamingDrySynthesizer(
            metadata, 16_000, samples_per_frame, 1234
        )
        zero_envelope = np.zeros(
            (voices, frames * samples_per_frame), dtype=np.float32
        )

        harmonic, noise = synthesizer.render(controls, zero_envelope)

        np.testing.assert_array_equal(harmonic, 0.0)
        np.testing.assert_array_equal(noise, 0.0)
        with self.assertRaisesRegex(ValueError, "voice_envelopes must have shape"):
            synthesizer.render(controls, np.zeros((voices, 1), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
