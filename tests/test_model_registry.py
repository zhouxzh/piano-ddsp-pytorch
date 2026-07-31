import unittest
from pathlib import Path

from ddsp_piano.model_registry import load_model_registry, model_output_label


class ModelRegistryTest(unittest.TestCase):
    def test_release_contains_four_stable_models(self) -> None:
        registry = load_model_registry()
        self.assertEqual(registry.release, "model-suite-v1.0.1")
        self.assertEqual(registry.default_model_id, "gru_ir_96_64")
        self.assertEqual(
            list(registry.models),
            [
                "gru_ir_96_64",
                "film_fdn_128_96",
                "gru_ir_fullwet_96_64",
                "film_ir_fullwet_96_64",
            ],
        )

    def test_default_model_must_be_published(self) -> None:
        registry = load_model_registry()
        self.assertIn(registry.default_model_id, registry.models)

    def test_legacy_names_fail_with_migration_hint(self) -> None:
        with self.assertRaisesRegex(ValueError, "use 'gru_ir_96_64'"):
            load_model_registry().require("v1")
        migrations = {
            "paper_ir": "gru_ir_96_64",
            "film_fdn": "film_fdn_128_96",
            "calibrated_ir": "gru_ir_fullwet_96_64",
            "calibrated_film_ir": "film_ir_fullwet_96_64",
        }
        registry = load_model_registry()
        for legacy_name, replacement in migrations.items():
            with self.subTest(legacy_name=legacy_name):
                with self.assertRaisesRegex(ValueError, f"use '{replacement}'"):
                    registry.require(legacy_name)

    def test_metadata_model_id_is_the_output_label(self) -> None:
        path = Path("diagnostic.onnx")
        self.assertEqual(model_output_label(path, {"model_id": "film_fdn_128_96"}), "film_fdn_128_96")
        self.assertEqual(model_output_label(path, {}), "diagnostic")

    def test_quality_first_candidate_registry_keeps_four_public_ids(self) -> None:
        registry = load_model_registry(
            Path("ddsp_piano/model-suite-v1.1.0-rc1.json")
        )
        self.assertEqual(registry.release, "model-suite-v1.1.0-rc1")
        self.assertEqual(list(registry.models), list(load_model_registry().models))
        for spec in registry.models.values():
            self.assertEqual(spec.training["sampling_mode"], "coverage")
            self.assertEqual(spec.training["batch_size"], 8)
        self.assertEqual(
            registry.require("film_fdn_128_96").training["stage_schedule"]["refine"][
                "batch_size"
            ],
            6,
        )
        self.assertEqual(
            registry.require("film_ir_fullwet_96_64")
            .training["stage_schedule"]["refine"]["batch_size"],
            6,
        )


if __name__ == "__main__":
    unittest.main()
