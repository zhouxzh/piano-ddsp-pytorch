import unittest
from pathlib import Path

from ddsp_piano.model_registry import load_model_registry, model_output_label


class ModelRegistryTest(unittest.TestCase):
    def test_release_contains_four_stable_models(self) -> None:
        registry = load_model_registry()
        self.assertEqual(registry.release, "model-suite-v1.0.0")
        self.assertEqual(
            list(registry.models),
            ["paper_ir", "film_fdn", "calibrated_ir", "calibrated_film_ir"],
        )

    def test_legacy_names_fail_with_migration_hint(self) -> None:
        with self.assertRaisesRegex(ValueError, "use 'paper_ir'"):
            load_model_registry().require("v1")

    def test_metadata_model_id_is_the_output_label(self) -> None:
        path = Path("diagnostic.onnx")
        self.assertEqual(model_output_label(path, {"model_id": "film_fdn"}), "film_fdn")
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


if __name__ == "__main__":
    unittest.main()
