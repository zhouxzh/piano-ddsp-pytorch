from __future__ import annotations

import json
import re
import unittest
from pathlib import Path

from ddsp_piano.model_registry import ROOT, load_model_registry


class ReleaseManifestTest(unittest.TestCase):
    def test_tracked_release_matches_registry(self) -> None:
        registry = load_model_registry()
        path = ROOT / "releases" / f"{registry.release}.json"
        text = path.read_text(encoding="utf-8")
        self.assertNotIn("/home/", text)
        manifest = json.loads(text)
        self.assertEqual(list(manifest["models"]), list(registry.models))
        for model_id, spec in registry.models.items():
            assets = manifest["models"][model_id]["assets"]
            self.assertEqual(
                set(assets),
                {
                    f"{spec.asset_basename}.pt",
                    f"{spec.asset_basename}.onnx",
                    f"{spec.asset_basename}.json",
                },
            )
            for values in assets.values():
                self.assertGreater(values["bytes"], 0)
                self.assertRegex(values["sha256"], r"^[0-9a-f]{64}$")

    def test_local_markdown_links_exist(self) -> None:
        pattern = re.compile(r"\[[^]]*\]\(([^)]+)\)")
        markdown_files = [ROOT / "README.md", ROOT / "README.zh-CN.md"]
        markdown_files.extend(sorted((ROOT / "docs").glob("*.md")))
        markdown_files.extend(sorted((ROOT / "releases").glob("*.md")))
        for source in markdown_files:
            for target in pattern.findall(source.read_text(encoding="utf-8")):
                if "://" in target or target.startswith("#"):
                    continue
                destination = (source.parent / target.split("#", 1)[0]).resolve()
                self.assertTrue(destination.exists(), f"{source}: missing {target}")


if __name__ == "__main__":
    unittest.main()
