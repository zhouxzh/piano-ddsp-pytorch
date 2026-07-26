"""Training-only quality statistics and deterministic curriculum sampling."""

from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
from torch.utils.data import Sampler


MANIFEST_SCHEMA = "ddsp-piano-quality-manifest/v1"
PITCH_BANDS = ((21, 47), (48, 71), (72, 108))


def dataset_index_sha256(index: list[tuple[str, int, int, int]]) -> str:
    digest = hashlib.sha256()
    for cache_path, sample_start, frame_start, piano_id in index:
        digest.update(
            f"{Path(cache_path).resolve()}\0{sample_start}\0{frame_start}\0{piano_id}\n".encode(
                "utf-8"
            )
        )
    return digest.hexdigest()


def _quantile_bins(values: np.ndarray, bins: int = 5) -> np.ndarray:
    if values.size == 0:
        return np.zeros(0, dtype=np.int64)
    edges = np.quantile(values, np.linspace(0.0, 1.0, bins + 1)[1:-1])
    return np.searchsorted(edges, values, side="right").astype(np.int64)


def _robust_slope(points: list[tuple[float, float]]) -> float | None:
    if len(points) < 32:
        return None
    values = np.asarray(points, dtype=np.float64)
    x = values[:, 0]
    y = values[:, 1]
    design = np.stack((x, np.ones_like(x)), axis=1)
    weights = np.ones_like(x)
    coefficient = np.zeros(2, dtype=np.float64)
    for _ in range(8):
        weighted = design * np.sqrt(weights[:, None])
        target = y * np.sqrt(weights)
        coefficient = np.linalg.lstsq(weighted, target, rcond=None)[0]
        residual = y - design @ coefficient
        scale = 1.4826 * np.median(np.abs(residual - np.median(residual))) + 1e-6
        normalized = np.abs(residual) / (1.345 * scale)
        weights = np.where(normalized <= 1.0, 1.0, 1.0 / normalized)
    return float(np.clip(coefficient[0], 0.5, 2.0))


def _pitch_band(pitch: float) -> int:
    if pitch <= PITCH_BANDS[0][1]:
        return 0
    if pitch <= PITCH_BANDS[1][1]:
        return 1
    return 2


def build_quality_manifest(dataset, sample_rate: int, frame_rate: int, seed: int) -> dict:
    """Describe train-only quality strata and estimate velocity response slopes."""
    entries: list[dict | None] = [None] * len(dataset)
    grouped_points: dict[tuple[int, int], list[tuple[float, float]]] = {}
    all_points: list[tuple[float, float]] = []
    response_start = max(1, int(round(0.032 * sample_rate)))
    response_end = max(response_start + 1, int(round(0.160 * sample_rate)))
    samples_per_frame = sample_rate // frame_rate
    release_frames = max(1, int(round(0.5 * frame_rate)))

    by_track: dict[str, list[tuple[int, int, int, int]]] = defaultdict(list)
    for index, (cache_path, sample_start, frame_start, piano_id) in enumerate(dataset.index):
        by_track[cache_path].append((index, sample_start, frame_start, piano_id))

    for cache_path, segments in by_track.items():
        path = Path(cache_path)
        audio = np.load(path / "audio.npy", mmap_mode="r")
        conditioning = np.load(path / "conditioning.npy", mmap_mode="r")
        polyphony = np.load(path / "polyphony.npy", mmap_mode="r")
        audio32 = np.asarray(audio, dtype=np.float32)
        squared = np.square(audio32, dtype=np.float32)
        squared_prefix = np.concatenate(([0.0], np.cumsum(squared, dtype=np.float64)))
        derivative_prefix = np.concatenate(
            (
                [0.0],
                np.cumsum(
                    np.square(np.diff(audio32, prepend=audio32[:1]), dtype=np.float32),
                    dtype=np.float64,
                ),
            )
        )
        onset = np.asarray(conditioning[..., 1])
        onset_sum_prefix = np.concatenate(([0.0], np.cumsum(onset.sum(axis=-1))))
        onset_count_prefix = np.concatenate(
            ([0], np.cumsum((onset > 0).sum(axis=-1), dtype=np.int64))
        )
        active_count = np.asarray(polyphony, dtype=np.int16)
        global_active = active_count > 0
        release_event = np.zeros(global_active.size, dtype=np.int32)
        release_indices = np.flatnonzero(global_active[:-1] & ~global_active[1:]) + 1
        active_prefix = np.concatenate(
            ([0], np.cumsum(global_active, dtype=np.int64))
        )
        bounded = release_indices[release_indices + release_frames <= global_active.size]
        quiet = (
            active_prefix[bounded + release_frames] - active_prefix[bounded]
        ) == 0
        release_event[bounded[quiet]] = 1
        release_prefix = np.concatenate(([0], np.cumsum(release_event, dtype=np.int64)))

        for index, sample_start, frame_start, piano_id in segments:
            sample_end = sample_start + dataset.config.segment_samples
            frame_end = frame_start + dataset.config.segment_frames
            energy = squared_prefix[sample_end] - squared_prefix[sample_start]
            derivative_energy = derivative_prefix[sample_end] - derivative_prefix[sample_start]
            rms = math.sqrt(energy / dataset.config.segment_samples + 1e-12)
            # The RMS spectral moment is computed from Parseval's derivative
            # identity. It is used only as a cheap brightness stratum proxy.
            spectral_moment = min(
                sample_rate / 2,
                sample_rate
                / (2.0 * math.pi)
                * math.sqrt(derivative_energy / max(energy, 1e-12)),
            )
            onset_sum = onset_sum_prefix[frame_end] - onset_sum_prefix[frame_start]
            onset_count = onset_count_prefix[frame_end] - onset_count_prefix[frame_start]
            entries[index] = {
                "index": index,
                "rms": rms,
                "spectral_centroid_hz": spectral_moment,
                "mean_onset_velocity": (
                    float(onset_sum / onset_count) if onset_count else 0.0
                ),
                "max_polyphony": int(active_count[frame_start:frame_end].max(initial=0)),
                "release_rich": bool(
                    release_prefix[frame_end] - release_prefix[frame_start]
                ),
            }

        onset_frames, onset_slots = np.nonzero(onset > 0)
        for frame, slot in zip(onset_frames.tolist(), onset_slots.tolist()):
            if int(active_count[frame]) != 1:
                continue
            velocity = float(onset[frame, slot])
            pitch = float(conditioning[frame, slot, 0])
            start = frame * samples_per_frame + response_start
            end = min(audio.size, frame * samples_per_frame + response_end)
            if velocity <= 0.0 or end <= start:
                continue
            response_energy = squared_prefix[end] - squared_prefix[start]
            response_rms = math.sqrt(response_energy / (end - start) + 1e-12)
            point = (math.log(velocity), math.log(response_rms))
            grouped_points.setdefault((int(segments[0][3]), _pitch_band(pitch)), []).append(point)
            all_points.append(point)

    if any(entry is None for entry in entries):
        raise RuntimeError("Quality manifest construction missed one or more dataset entries")
    entries = list(entries)

    rms_bins = _quantile_bins(np.asarray([entry["rms"] for entry in entries]))
    centroid_bins = _quantile_bins(
        np.asarray([entry["spectral_centroid_hz"] for entry in entries])
    )
    strata = []
    for entry, rms_bin, centroid_bin in zip(entries, rms_bins, centroid_bins):
        velocity = float(entry["mean_onset_velocity"])
        velocity_bin = 0 if velocity == 0 else 1 if velocity < 0.35 else 3 if velocity > 0.8 else 2
        polyphony = int(entry["max_polyphony"])
        polyphony_bin = 0 if polyphony <= 1 else 1 if polyphony <= 4 else 2 if polyphony <= 8 else 3
        stratum = f"r{rms_bin}:c{centroid_bin}:v{velocity_bin}:p{polyphony_bin}:t{int(entry['release_rich'])}"
        entry["stratum"] = stratum
        strata.append(stratum)
    counts = Counter(strata)
    mean_count = len(entries) / max(len(counts), 1)
    for entry in entries:
        entry["curriculum_weight"] = float(
            np.clip(mean_count / counts[entry["stratum"]], 0.25, 4.0)
        )

    global_slope = _robust_slope(all_points) or 1.0
    piano_count = max((int(entry[3]) for entry in dataset.index), default=-1) + 1
    slopes = []
    sample_counts = []
    for piano_id in range(piano_count):
        piano_slopes = []
        piano_counts = []
        for band in range(len(PITCH_BANDS)):
            points = grouped_points.get((piano_id, band), [])
            piano_slopes.append(_robust_slope(points) or global_slope)
            piano_counts.append(len(points))
        slopes.append(piano_slopes)
        sample_counts.append(piano_counts)

    return {
        "schema": MANIFEST_SCHEMA,
        "split": "train",
        "seed": int(seed),
        "dataset_index_sha256": dataset_index_sha256(dataset.index),
        "entry_count": len(entries),
        "entries": entries,
        "velocity_response": {
            "pitch_bands": [list(value) for value in PITCH_BANDS],
            "global_slope": global_slope,
            "slopes": slopes,
            "sample_counts": sample_counts,
            "minimum_group_samples": 32,
            "response_window_ms": [32, 160],
        },
        "spectral_stratum_method": "rms-first-derivative-spectral-moment",
    }


def write_quality_manifest(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(path)


def load_quality_manifest(path: Path, dataset) -> dict:
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != MANIFEST_SCHEMA:
        raise ValueError(f"Unsupported quality manifest schema: {manifest.get('schema')}")
    if manifest.get("split") != "train":
        raise ValueError("Quality curriculum manifests must be built from the train split")
    if int(manifest.get("entry_count", -1)) != len(dataset):
        raise ValueError("Quality manifest entry count does not match the training dataset")
    if manifest.get("dataset_index_sha256") != dataset_index_sha256(dataset.index):
        raise ValueError("Quality manifest does not match the training dataset index")
    return manifest


class MixedCurriculumSampler(Sampler[int]):
    """Yield half uniform and half inverse-stratum samples per epoch."""

    def __init__(self, weights: list[float], generator: torch.Generator) -> None:
        if not weights or any(weight <= 0 for weight in weights):
            raise ValueError("Curriculum weights must be positive and non-empty")
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.generator = generator

    def __len__(self) -> int:
        return int(self.weights.numel())

    def __iter__(self) -> Iterator[int]:
        count = len(self)
        uniform_count = count // 2
        uniform = torch.randperm(count, generator=self.generator)[:uniform_count]
        curriculum = torch.multinomial(
            self.weights,
            count - uniform_count,
            replacement=True,
            generator=self.generator,
        )
        order = torch.randperm(count, generator=self.generator)
        mixed = torch.cat((uniform, curriculum))[order]
        return iter(mixed.tolist())


def velocity_slopes_tensor(manifest: dict) -> torch.Tensor:
    slopes = manifest.get("velocity_response", {}).get("slopes")
    if not slopes:
        raise ValueError("Quality manifest has no velocity response slopes")
    return torch.tensor(slopes, dtype=torch.float32)
