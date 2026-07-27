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

if __name__ == "__main__":
    unittest.main()
