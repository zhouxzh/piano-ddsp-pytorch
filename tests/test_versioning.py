from pathlib import Path
import unittest

from ddsp_piano.versioning import (
    model_output_label,
    release_version_for_variant,
)


class VersioningTest(unittest.TestCase):
    def test_release_versions_hide_internal_current_name(self) -> None:
        self.assertEqual(release_version_for_variant("current"), "v1")
        self.assertEqual(release_version_for_variant("v2"), "v2")

    def test_metadata_release_version_takes_precedence(self) -> None:
        path = Path("exports/legacy-name.onnx")
        self.assertEqual(model_output_label(path, {"release_version": "v1"}), "v1")
        self.assertEqual(model_output_label(path, {"model_variant": "v2"}), "v2")
        self.assertEqual(model_output_label(path, {}), "legacy-name")

    def test_unknown_internal_variant_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unknown model variant"):
            release_version_for_variant("v3")


if __name__ == "__main__":
    unittest.main()
