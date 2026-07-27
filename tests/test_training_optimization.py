import argparse
import random
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ddsp_piano.ddsp_pytorch.noise import Noise
from ddsp_piano.default_model import build_configurable_model, build_paper_model
from ddsp_piano.modules.inharm_synth import MultiInharmonic
from ddsp_piano.modules.loss import SSSLoss
from ddsp_piano.modules.piano_model import PianoModel
from train import (
    capture_rng_state,
    load_finetune_initialization,
    restore_rng_state,
    save_checkpoint,
    set_trainable_scope,
    velocity_counterfactual_loss,
)


class CountingControlModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.calls = 0

    def predict_controls(self, conditioning, pedal, piano_model):
        self.calls += 1
        amplitudes = conditioning[..., 1:2].permute(2, 0, 1, 3)
        return (amplitudes,)


class TrainingOptimizationTest(unittest.TestCase):
    def _dsp_model(self, layout: str) -> PianoModel:
        return PianoModel(
            n_synths=2,
            harmonic_synthesizer=MultiInharmonic(
                n_samples=64,
                sample_rate=16_000,
                n_harmonics=4,
            ),
            noise_synthesizer=Noise(),
            synthesis_layout=layout,
        )

    def _controls(self):
        torch.manual_seed(11)
        amplitudes = torch.randn(2, 1, 4, 1, requires_grad=True)
        harmonics = torch.randn(2, 1, 4, 4, requires_grad=True)
        inharmonicity = (torch.rand(2, 1, 4, 1) * 1e-4).requires_grad_(True)
        f0 = torch.full((2, 1, 4, 2), 220.0, requires_grad=True)
        noise = torch.randn(2, 1, 4, 8, requires_grad=True)
        return amplitudes, harmonics, inharmonicity, f0, noise

    def test_vectorized_synthesis_matches_serial_output_and_gradients(self):
        serial_controls = self._controls()
        vector_controls = tuple(
            value.detach().clone().requires_grad_(True) for value in serial_controls
        )
        torch.manual_seed(77)
        serial = self._dsp_model("serial").synthesize_voices(*serial_controls)
        serial.square().mean().backward()
        torch.manual_seed(77)
        vectorized = self._dsp_model("vectorized").synthesize_voices(*vector_controls)
        vectorized.square().mean().backward()

        torch.testing.assert_close(vectorized, serial, rtol=2e-5, atol=2e-6)
        for expected, actual in zip(serial_controls, vector_controls):
            self.assertIsNotNone(expected.grad)
            self.assertIsNotNone(actual.grad)
            torch.testing.assert_close(actual.grad, expected.grad, rtol=5e-5, atol=2e-6)

    def test_combined_stft_matches_separate_reference(self):
        torch.manual_seed(5)
        target = torch.randn(2, 1024)
        prediction = torch.randn(2, 1024)
        loss = SSSLoss(n_fft=256, transform_layout="combined")
        actual = loss(target, prediction)["loss"]

        window = torch.hann_window(256)
        target_spectrum = torch.stft(
            target, n_fft=256, hop_length=64, window=window, return_complex=True
        ).abs()
        prediction_spectrum = torch.stft(
            prediction, n_fft=256, hop_length=64, window=window, return_complex=True
        ).abs()
        expected = F.l1_loss(prediction_spectrum, target_spectrum) + F.l1_loss(
            (target_spectrum + 1e-7).log2(),
            (prediction_spectrum + 1e-7).log2(),
        )
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)

    def test_vectorized_training_forward_covers_all_architectures(self):
        common = {
            "n_synths": 16,
            "n_piano_models": 1,
            "duration": 4 / 250,
            "frame_rate": 250,
            "sample_rate": 16_000,
            "reverb_duration": 4 / 250,
            "synthesis_layout": "vectorized",
        }
        models = [
            build_paper_model(**common),
            build_configurable_model(
                **common,
                n_harmonics=96,
                n_noise_filter_banks=64,
                reverb_type="ir",
                context_type="legacy",
                monophonic_type="legacy",
                inharmonicity_type="legacy",
            ),
            build_configurable_model(
                **common,
                n_harmonics=96,
                n_noise_filter_banks=64,
                reverb_type="ir",
                context_type="film",
                monophonic_type="deep",
                inharmonicity_type="joint",
            ),
        ]
        conditioning = torch.zeros(1, 4, 16, 2)
        conditioning[..., 0] = torch.arange(48.0, 64.0)
        conditioning[..., 1] = 0.7
        pedal = torch.zeros(1, 4, 4)
        piano_model = torch.zeros(1, dtype=torch.int64)
        for model in models:
            torch.manual_seed(41)
            wet, reverb, dry = model(conditioning, pedal, piano_model)
            self.assertEqual(tuple(wet.shape), (1, 256))
            self.assertEqual(tuple(dry.shape), (1, 256))
            self.assertTrue(torch.isfinite(wet).all())
            self.assertTrue(torch.isfinite(reverb).all())
            wet.square().mean().backward()

    def test_velocity_counterfactual_uses_one_control_forward(self):
        model = CountingControlModel()
        conditioning = torch.tensor([[[[60.0, 0.8], [64.0, 0.6]]]])
        pedal = torch.zeros(1, 1, 4)
        piano_model = torch.zeros(1, dtype=torch.int64)
        loss = velocity_counterfactual_loss(
            model,
            conditioning,
            pedal,
            piano_model,
            combined=True,
        )
        self.assertEqual(model.calls, 1)
        self.assertTrue(torch.isfinite(loss))

    def test_rng_state_restores_all_training_generators(self):
        random.seed(3)
        np.random.seed(3)
        torch.manual_seed(3)
        generator = torch.Generator().manual_seed(3)
        state = capture_rng_state(generator)
        expected = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
            torch.rand(3, generator=generator),
        )
        restore_rng_state(state, generator)
        actual = (
            random.random(),
            float(np.random.rand()),
            torch.rand(3),
            torch.rand(3, generator=generator),
        )
        self.assertEqual(actual[0], expected[0])
        self.assertEqual(actual[1], expected[1])
        torch.testing.assert_close(actual[2], expected[2])
        torch.testing.assert_close(actual[3], expected[3])

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is required")
    def test_rng_state_restores_after_checkpoint_is_mapped_to_cuda(self):
        random.seed(11)
        np.random.seed(11)
        torch.manual_seed(11)
        torch.cuda.manual_seed_all(11)
        generator = torch.Generator().manual_seed(11)
        state = capture_rng_state(generator)
        expected = (
            torch.rand(3),
            torch.rand(3, device="cuda"),
            torch.rand(3, generator=generator),
        )
        relocated_state = {
            **state,
            "torch": state["torch"].to("cuda"),
            "cuda": [cuda_state.to("cuda") for cuda_state in state["cuda"]],
            "train_loader_generator": state["train_loader_generator"].to("cuda"),
        }

        restore_rng_state(relocated_state, generator)

        torch.testing.assert_close(torch.rand(3), expected[0])
        torch.testing.assert_close(torch.rand(3, device="cuda"), expected[1])
        torch.testing.assert_close(torch.rand(3, generator=generator), expected[2])

    def test_checkpoint_records_resume_and_performance_state(self):
        model = torch.nn.Linear(2, 1)
        optimizer = torch.optim.Adam(model.parameters())
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        generator = torch.Generator().manual_seed(9)
        args = argparse.Namespace(experiment_dir=Path("unused"), batch_size=2)
        with tempfile.TemporaryDirectory() as temporary:
            destination = Path(temporary) / "checkpoint.pt"
            save_checkpoint(
                destination,
                model,
                optimizer,
                scaler,
                epoch=2,
                global_step=30,
                best_validation=1.5,
                args=args,
                piano_models=[2018],
                loss_calibration={},
                initialization=None,
                validation_corpus_sha256="abc",
                examples_seen=60,
                train_generator=generator,
                training_performance={"steps_per_second": 4.0},
            )
            checkpoint = torch.load(destination, weights_only=False)
        self.assertEqual(checkpoint["schema"], "ddsp-piano-training-checkpoint/v2")
        self.assertEqual(checkpoint["examples_seen"], 60)
        self.assertIn("rng_state", checkpoint)
        self.assertEqual(checkpoint["training_performance"]["steps_per_second"], 4.0)

    def test_finetune_initialization_is_strict_and_resets_parent_state(self):
        source = torch.nn.Linear(2, 1)
        target = torch.nn.Linear(2, 1)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint_path = Path(temporary) / "parent.pt"
            torch.save(
                {"model": source.state_dict(), "global_step": 40, "examples_seen": 160},
                checkpoint_path,
            )
            report = load_finetune_initialization(target, checkpoint_path, torch.device("cpu"))
        torch.testing.assert_close(target.weight, source.weight)
        self.assertEqual(report["parent_global_step"], 40)
        self.assertTrue(report["optimizer_reset"])
        self.assertTrue(report["step_reset"])

    def test_trainable_scopes_only_enable_expected_modules(self):
        common = {
            "n_synths": 2,
            "n_piano_models": 1,
            "duration": 4 / 250,
            "frame_rate": 250,
            "sample_rate": 16_000,
            "reverb_duration": 4 / 250,
            "n_harmonics": 96,
            "n_noise_filter_banks": 64,
            "reverb_type": "ir",
            "context_type": "legacy",
            "monophonic_type": "legacy",
            "inharmonicity_type": "legacy",
        }
        controls = build_configurable_model(**common)
        controls.alternate_training(first_phase=True)
        set_trainable_scope(controls, "controls", phase=1)
        self.assertFalse(any(p.requires_grad for p in controls.reverb_model.parameters()))
        self.assertTrue(any(p.requires_grad for p in controls.monophonic_network.parameters()))

        reverb = build_configurable_model(**common)
        reverb.alternate_training(first_phase=True)
        set_trainable_scope(reverb, "reverb", phase=1)
        trainable = {
            name for name, parameter in reverb.named_parameters() if parameter.requires_grad
        }
        self.assertTrue(trainable)
        self.assertTrue(all(name.startswith("reverb_model.") for name in trainable))


if __name__ == "__main__":
    unittest.main()
