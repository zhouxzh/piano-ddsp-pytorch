"""Deterministic quality-suite data selection, metrics, and reporting."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import platform
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torchaudio

from ddsp_piano.maestro import MaestroSegmentDataset, PreprocessConfig


EVALUATION_SCHEMA = "ddsp-piano-eval/v1"
CORPUS_SCHEMA = "ddsp-piano-corpus/v1"
LISTENING_SCHEMA = "ddsp-piano-listening/v1"
LOWER_IS_BETTER = (
    "mrstft",
    "loudness_error_lu",
    "onset_envelope_l1",
    "spectral_centroid_error",
    "tail_decay_error_db_per_second",
)
DEFAULT_METRIC_WEIGHTS = {
    "mrstft": 0.50,
    "loudness_error_lu": 0.20,
    "onset_envelope_l1": 0.15,
    "spectral_centroid_error": 0.10,
    "tail_decay_error_db_per_second": 0.05,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _stable_key(*parts: object) -> str:
    return hashlib.sha256("\0".join(str(part) for part in parts).encode("utf-8")).hexdigest()


def _cache_metadata(cache_path: Path) -> dict:
    return read_json(cache_path / "metadata.json")


def _segment_features(cache_path: Path, frame_start: int, config: PreprocessConfig) -> dict:
    sample_start = frame_start * (config.sample_rate // config.frame_rate)
    audio = np.load(cache_path / "audio.npy", mmap_mode="r")
    conditioning = np.load(cache_path / "conditioning.npy", mmap_mode="r")
    pedal = np.load(cache_path / "pedal.npy", mmap_mode="r")
    audio_block = np.asarray(
        audio[sample_start : sample_start + config.segment_samples], dtype=np.float32
    )
    conditioning_block = conditioning[frame_start : frame_start + config.segment_frames]
    pedal_block = pedal[frame_start : frame_start + config.segment_frames]
    active = conditioning_block[..., 0] > 0
    onset = conditioning_block[..., 1] > 0
    return {
        "rms": float(np.sqrt(np.mean(np.square(audio_block, dtype=np.float64)))),
        "polyphony": float(active.sum(axis=-1).mean()),
        "onsets": int(onset.sum()),
        "sustain": float(pedal_block[:, 0].mean()),
    }


def _select_track_segments(
    items: list[tuple[str, int, int, int]],
    cache_path: Path,
    config: PreprocessConfig,
    per_category: int,
) -> list[dict]:
    candidates = []
    for _, sample_start, frame_start, piano_id in items:
        candidates.append(
            {
                "sample_start": int(sample_start),
                "frame_start": int(frame_start),
                "piano_model": int(piano_id),
                "features": _segment_features(cache_path, frame_start, config),
                "stable_key": _stable_key(cache_path.name, frame_start),
            }
        )
    category_orders = {
        "quiet": lambda item: (item["features"]["rms"], item["stable_key"]),
        "loud": lambda item: (-item["features"]["rms"], item["stable_key"]),
        "dense": lambda item: (-item["features"]["polyphony"], item["stable_key"]),
        "onset": lambda item: (-item["features"]["onsets"], item["stable_key"]),
        "sustain": lambda item: (-item["features"]["sustain"], item["stable_key"]),
    }
    selected: list[dict] = []
    used: set[int] = set()
    for category, order in category_orders.items():
        added = 0
        for item in sorted(candidates, key=order):
            if item["frame_start"] in used:
                continue
            chosen = dict(item)
            chosen["category"] = category
            selected.append(chosen)
            used.add(item["frame_start"])
            added += 1
            if added == per_category:
                break
    if len(selected) < per_category * len(category_orders):
        for item in sorted(candidates, key=lambda value: value["stable_key"]):
            if item["frame_start"] in used:
                continue
            chosen = dict(item)
            chosen["category"] = "fallback"
            selected.append(chosen)
            used.add(item["frame_start"])
            if len(selected) == per_category * len(category_orders):
                break
    return selected


def build_corpus(
    maestro_root: Path,
    cache_root: Path,
    profile: str,
    config: PreprocessConfig,
) -> dict:
    """Build a stable quick/dev/release corpus from cached MAESTRO tracks."""
    if profile not in {"quick", "dev", "release"}:
        raise ValueError("profile must be quick, dev, or release")
    split = "test" if profile == "release" else "validation"
    dataset = MaestroSegmentDataset(
        maestro_root,
        cache_root,
        split,
        config,
        require_cache=True,
    )
    by_track: dict[str, list[tuple[str, int, int, int]]] = defaultdict(list)
    year_by_track: dict[str, int] = {}
    for item in dataset.index:
        cache_path = Path(item[0])
        by_track[str(cache_path)].append(item)
        if str(cache_path) not in year_by_track:
            year_by_track[str(cache_path)] = int(_cache_metadata(cache_path)["year"])

    entries: list[dict] = []
    if profile == "release":
        for cache_name, items in sorted(by_track.items()):
            cache_path = Path(cache_name)
            for _, sample_start, frame_start, piano_id in items:
                if frame_start % config.segment_frames:
                    continue
                entries.append(
                    {
                        "id": _stable_key(cache_path.name, frame_start)[:20],
                        "cache_path": str(cache_path.resolve()),
                        "track_id": cache_path.name,
                        "year": year_by_track[cache_name],
                        "piano_model": int(piano_id),
                        "sample_start": int(sample_start),
                        "frame_start": int(frame_start),
                        "category": "all",
                    }
                )
    else:
        tracks_by_year: dict[int, list[str]] = defaultdict(list)
        for cache_name, year in year_by_track.items():
            tracks_by_year[year].append(cache_name)
        tracks_per_year = 1 if profile == "quick" else 2
        per_category = 1 if profile == "quick" else 2
        for year in sorted(tracks_by_year):
            track_names = sorted(
                tracks_by_year[year], key=lambda name: _stable_key(year, Path(name).name)
            )[:tracks_per_year]
            for cache_name in track_names:
                cache_path = Path(cache_name)
                for selected in _select_track_segments(
                    by_track[cache_name], cache_path, config, per_category
                ):
                    entries.append(
                        {
                            "id": _stable_key(cache_path.name, selected["frame_start"])[:20],
                            "cache_path": str(cache_path.resolve()),
                            "track_id": cache_path.name,
                            "year": year,
                            "piano_model": selected["piano_model"],
                            "sample_start": selected["sample_start"],
                            "frame_start": selected["frame_start"],
                            "category": selected["category"],
                            "features": selected["features"],
                        }
                    )

    contract = {
        "schema": CORPUS_SCHEMA,
        "profile": profile,
        "split": split,
        "preprocess": config.__dict__,
        "entries": entries,
    }
    contract["corpus_sha256"] = canonical_sha256(contract)
    return contract


def _safe_loudness(audio: np.ndarray, sample_rate: int) -> float:
    tensor = torch.from_numpy(np.array(audio, dtype=np.float32, copy=True)).unsqueeze(0)
    try:
        value = float(torchaudio.functional.loudness(tensor, sample_rate).item())
    except (RuntimeError, ValueError):
        value = math.nan
    if not math.isfinite(value):
        rms = float(np.sqrt(np.mean(np.square(audio, dtype=np.float64))))
        value = 20.0 * math.log10(max(rms, 1e-8))
    return value


def _frame_rms(audio: np.ndarray, window: int, hop: int) -> np.ndarray:
    tensor = torch.from_numpy(np.array(audio, dtype=np.float32, copy=True)).reshape(1, 1, -1)
    if tensor.shape[-1] < window:
        tensor = torch.nn.functional.pad(tensor, (0, window - tensor.shape[-1]))
    power = torch.nn.functional.avg_pool1d(
        tensor.square(), window, stride=hop, ceil_mode=True
    )
    return torch.sqrt(power.clamp_min(1e-10)).flatten().numpy()


def _spectral_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    fft_sizes: Iterable[int],
) -> tuple[float, float, float]:
    pred_tensor = torch.from_numpy(np.array(prediction, dtype=np.float32, copy=True))
    target_tensor = torch.from_numpy(np.array(target, dtype=np.float32, copy=True))
    convergence = []
    log_l1 = []
    for size in fft_sizes:
        hop = max(1, size // 4)
        window = torch.hann_window(size)
        pred_mag = torch.stft(
            pred_tensor,
            size,
            hop_length=hop,
            window=window,
            return_complex=True,
        ).abs()
        target_mag = torch.stft(
            target_tensor,
            size,
            hop_length=hop,
            window=window,
            return_complex=True,
        ).abs()
        convergence.append(
            float(torch.linalg.vector_norm(pred_mag - target_mag) / torch.linalg.vector_norm(target_mag).clamp_min(1e-7))
        )
        log_l1.append(float(torch.mean(torch.abs(torch.log1p(pred_mag) - torch.log1p(target_mag)))))
    spectral_convergence = float(np.mean(convergence))
    log_magnitude_l1 = float(np.mean(log_l1))
    return spectral_convergence + log_magnitude_l1, spectral_convergence, log_magnitude_l1


def _spectral_centroid(audio: np.ndarray, sample_rate: int) -> float:
    magnitude = np.abs(np.fft.rfft(np.asarray(audio, dtype=np.float64)))
    frequencies = np.fft.rfftfreq(audio.size, 1.0 / sample_rate)
    return float(np.sum(magnitude * frequencies) / max(float(np.sum(magnitude)), 1e-8))


def _tail_slope(audio: np.ndarray, sample_rate: int) -> float:
    window = max(1, int(round(0.05 * sample_rate)))
    rms = _frame_rms(audio[-sample_rate:], window, window)
    db = 20.0 * np.log10(np.maximum(rms, 1e-8))
    if db.size < 2:
        return 0.0
    return float(np.polyfit(np.arange(db.size) * 0.05, db, 1)[0])


def audio_metrics(
    prediction: np.ndarray,
    target: np.ndarray,
    sample_rate: int,
    fft_sizes: Iterable[int],
) -> dict[str, float | bool]:
    size = min(prediction.size, target.size)
    prediction = np.asarray(prediction[:size], dtype=np.float32)
    target = np.asarray(target[:size], dtype=np.float32)
    finite = bool(np.isfinite(prediction).all())
    pred_peak = float(np.max(np.abs(prediction), initial=0.0))
    pred_rms = float(np.sqrt(np.mean(np.square(prediction, dtype=np.float64))))
    target_rms = float(np.sqrt(np.mean(np.square(target, dtype=np.float64))))
    pred_loudness = _safe_loudness(prediction, sample_rate)
    target_loudness = _safe_loudness(target, sample_rate)
    timbre_gain_db = float(np.clip(target_loudness - pred_loudness, -24.0, 24.0))
    matched_prediction = prediction * math.pow(10.0, timbre_gain_db / 20.0)
    mrstft, convergence, log_l1 = _spectral_metrics(
        matched_prediction, target, fft_sizes
    )
    onset_window = max(1, int(round(0.016 * sample_rate)))
    onset_hop = max(1, int(round(0.004 * sample_rate)))
    pred_envelope = _frame_rms(matched_prediction, onset_window, onset_hop)
    target_envelope = _frame_rms(target, onset_window, onset_hop)
    pred_envelope /= max(float(pred_envelope.mean()), 1e-7)
    target_envelope /= max(float(target_envelope.mean()), 1e-7)
    onset_error = float(np.mean(np.abs(np.diff(pred_envelope) - np.diff(target_envelope))))
    pred_centroid = _spectral_centroid(matched_prediction, sample_rate)
    target_centroid = _spectral_centroid(target, sample_rate)
    pred_crest = 20.0 * math.log10(max(pred_peak, 1e-8) / max(pred_rms, 1e-8))
    target_peak = float(np.max(np.abs(target), initial=0.0))
    target_crest = 20.0 * math.log10(max(target_peak, 1e-8) / max(target_rms, 1e-8))
    return {
        "finite": finite,
        "silent": pred_rms <= 1e-8,
        "peak": pred_peak,
        "rms": pred_rms,
        "target_rms": target_rms,
        "rms_error_db": abs(20.0 * math.log10(max(pred_rms, 1e-8) / max(target_rms, 1e-8))),
        "loudness_lufs": pred_loudness,
        "target_loudness_lufs": target_loudness,
        "loudness_error_lu": abs(pred_loudness - target_loudness),
        "timbre_match_gain_db": timbre_gain_db,
        "crest_factor_db": pred_crest,
        "crest_factor_error_db": abs(pred_crest - target_crest),
        "mrstft": mrstft,
        "spectral_convergence": convergence,
        "log_magnitude_l1": log_l1,
        "onset_envelope_l1": onset_error,
        "spectral_centroid_hz": pred_centroid,
        "spectral_centroid_error": abs(pred_centroid - target_centroid) / (sample_rate / 2),
        "tail_decay_db_per_second": _tail_slope(prediction, sample_rate),
        "tail_decay_error_db_per_second": abs(
            _tail_slope(prediction, sample_rate) - _tail_slope(target, sample_rate)
        ),
    }


def signal_metrics(signals: dict[str, np.ndarray]) -> dict[str, float | bool]:
    result: dict[str, float | bool] = {}
    for name in ("harmonic", "noise", "dry", "wet"):
        value = np.asarray(signals[name], dtype=np.float32)
        result[f"{name}_finite"] = bool(np.isfinite(value).all())
        result[f"{name}_rms"] = float(np.sqrt(np.mean(np.square(value, dtype=np.float64))))
        result[f"{name}_peak"] = float(np.max(np.abs(value), initial=0.0))
    result["noise_harmonic_rms_ratio"] = float(result["noise_rms"]) / max(
        float(result["harmonic_rms"]), 1e-8
    )
    result["wet_dry_rms_ratio"] = float(result["wet_rms"]) / max(
        float(result["dry_rms"]), 1e-8
    )
    return result


def inspect_onnx_model(
    model_path: Path,
    allowed_operators: set[str],
    min_stateful_steps: int,
    latency_iterations: int,
) -> dict:
    metadata_path = model_path.with_suffix(".json")
    metadata = read_json(metadata_path)
    graph = onnx.load(str(model_path))
    onnx.checker.check_model(graph)
    operators = sorted({node.op_type for node in graph.graph.node})
    unexpected = sorted(set(operators) - allowed_operators)
    comparison = metadata.get("onnx_runtime_comparison", {})
    numerical_ok = bool(comparison) and all(
        bool(value.get("allclose")) for value in comparison.values()
    )
    stateful_steps = int(metadata.get("onnx_runtime_stateful_steps", 0))

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    feed: dict[str, np.ndarray] = {}
    for value in session.get_inputs():
        shape = tuple(int(size) for size in value.shape)
        dtype = np.int32 if value.name == "piano_model" else np.float32
        feed[value.name] = np.zeros(shape, dtype=dtype)
    feed["conditioning"][..., 0, 0] = 60.0
    feed["conditioning"][..., 0, 1] = 0.8
    feed["extended_pitch"][..., 0, 0] = 60.0
    output_names = [value.name for value in session.get_outputs()]
    for _ in range(20):
        outputs = session.run(output_names, feed)
        feed["context_state"] = outputs[-2]
        feed["monophonic_state"] = outputs[-1]
    samples = []
    import time

    for _ in range(max(1, latency_iterations)):
        started = time.perf_counter_ns()
        outputs = session.run(output_names, feed)
        samples.append((time.perf_counter_ns() - started) / 1_000_000.0)
        feed["context_state"] = outputs[-2]
        feed["monophonic_state"] = outputs[-1]
    latency = np.asarray(samples, dtype=np.float64)
    dynamic_dimensions = any(
        not isinstance(size, int)
        for value in list(session.get_inputs()) + list(session.get_outputs())
        for size in value.shape
    )
    input_shapes = {value.name: list(value.shape) for value in session.get_inputs()}
    output_shapes = {value.name: list(value.shape) for value in session.get_outputs()}
    input_types = {value.name: value.type for value in session.get_inputs()}
    output_types = {value.name: value.type for value in session.get_outputs()}
    expected_inputs = {
        "conditioning": [1, 1, 16, 2],
        "pedal": [1, 1, 4],
        "piano_model": [1],
        "extended_pitch": [1, 1, 16, 1],
        "context_state": [1, 1, 64],
        "monophonic_state": [1, 16, 192],
    }
    violations = []
    if int(metadata.get("opset", -1)) != 13:
        violations.append("opset")
    if metadata.get("dtype") != "FP32":
        violations.append("dtype")
    if int(metadata.get("sample_rate", -1)) != 16_000:
        violations.append("sample_rate")
    if int(metadata.get("frame_rate", -1)) != 250:
        violations.append("frame_rate")
    if int(metadata.get("frames_per_call", -1)) != 1:
        violations.append("frames_per_call")
    if int(metadata.get("audio_samples_per_call", -1)) != 64:
        violations.append("audio_samples_per_call")
    if input_shapes != expected_inputs:
        violations.append("input_shapes")
    expected_input_types = {
        name: "tensor(int32)" if name == "piano_model" else "tensor(float)"
        for name in expected_inputs
    }
    if input_types != expected_input_types:
        violations.append("input_dtypes")
    expected_state_outputs = {
        "next_context_state": [1, 1, 64],
        "next_monophonic_state": [1, 16, 192],
    }
    if any(output_shapes.get(name) != shape for name, shape in expected_state_outputs.items()):
        violations.append("output_state_shapes")
    control_prefixes = {
        "amplitudes": ([1, 1, 16], {1}),
        "harmonic_distribution": ([1, 1, 16], None),
        "inharmonicity": ([1, 1, 16], {1}),
        "f0_hz": ([1, 1, 16], {1, 2}),
        "noise_magnitudes": ([1, 1, 16], None),
    }
    for name, (prefix, allowed_last) in control_prefixes.items():
        shape = output_shapes.get(name, [])
        if len(shape) != 4 or shape[:3] != prefix or not isinstance(shape[-1], int):
            violations.append(f"output_shape:{name}")
        elif allowed_last is not None and shape[-1] not in allowed_last:
            violations.append(f"output_shape:{name}")
    reverb_name = metadata.get("reverb_output")
    expected_reverb = [1, 9] if reverb_name == "reverb_controls" else [1, 24_000]
    if output_shapes.get(reverb_name) != expected_reverb:
        violations.append("reverb_output_shape")
    if any(value != "tensor(float)" for value in output_types.values()):
        violations.append("output_dtypes")
    metadata_inputs = metadata.get("inputs", {})
    metadata_outputs = metadata.get("outputs", {})
    if metadata_inputs != input_shapes or metadata_outputs != output_shapes:
        violations.append("metadata_graph_shapes")
    return {
        "model": str(model_path.resolve()),
        "metadata": str(metadata_path.resolve()),
        "sha256": sha256_file(model_path),
        "metadata_sha256": sha256_file(metadata_path),
        "file_bytes": model_path.stat().st_size,
        "model_id": metadata.get("model_id"),
        "architecture": metadata.get("architecture"),
        "model_config": metadata.get("model_config", {}),
        "operators": operators,
        "unexpected_operators": unexpected,
        "dynamic_dimensions": dynamic_dimensions,
        "deployment_contract_ok": not violations,
        "deployment_contract_violations": violations,
        "input_shapes": input_shapes,
        "output_shapes": output_shapes,
        "input_types": input_types,
        "output_types": output_types,
        "numerical_allclose": numerical_ok,
        "stateful_comparison_steps": stateful_steps,
        "minimum_stateful_steps": min_stateful_steps,
        "stateful_steps_ok": stateful_steps >= min_stateful_steps,
        "latency_ms": {
            "iterations": int(latency.size),
            "mean": float(latency.mean()),
            "p50": float(np.percentile(latency, 50)),
            "p95": float(np.percentile(latency, 95)),
            "p99": float(np.percentile(latency, 99)),
        },
        "metadata_contract": metadata,
    }


def _numeric_summary(values: list[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "p05": float(np.percentile(array, 5)),
        "p95": float(np.percentile(array, 95)),
    }


def summarize_segments(segments: list[dict]) -> dict:
    metrics: dict[str, list[float]] = defaultdict(list)
    for segment in segments:
        for name, value in segment["audio_metrics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[name].append(float(value))
        for name, value in segment["signal_metrics"].items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                metrics[name].append(float(value))
    return {name: _numeric_summary(values) for name, values in sorted(metrics.items())}


def compare_models(
    baseline_segments: list[dict],
    candidate_segments: list[dict],
    weights: dict[str, float],
) -> dict:
    baseline_by_id = {entry["id"]: entry for entry in baseline_segments}
    candidate_ids = {entry["id"] for entry in candidate_segments}
    if candidate_ids != set(baseline_by_id):
        raise ValueError("Baseline and candidate segment ids differ")
    if not candidate_segments:
        raise ValueError("No segments were provided for comparison")
    if not math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-6):
        raise ValueError("Metric weights must sum to 1.0")
    ratio_floors = {
        "mrstft": 0.01,
        "loudness_error_lu": 0.1,
        "onset_envelope_l1": 0.001,
        "spectral_centroid_error": 0.001,
        "tail_decay_error_db_per_second": 0.5,
    }
    ratios = []
    groups: dict[str, list[float]] = defaultdict(list)
    metric_ratios: dict[str, list[float]] = defaultdict(list)
    matched = []
    for candidate in candidate_segments:
        baseline = baseline_by_id[candidate["id"]]
        weighted = 0.0
        per_metric = {}
        for name, weight in weights.items():
            baseline_value = float(baseline["audio_metrics"][name])
            candidate_value = float(candidate["audio_metrics"][name])
            floor = ratio_floors.get(name, 1e-6)
            ratio = (candidate_value + floor) / (baseline_value + floor)
            per_metric[name] = ratio
            metric_ratios[name].append(ratio)
            weighted += weight * ratio
        ratios.append(weighted)
        groups[f"year:{candidate['year']}"].append(weighted)
        groups[f"category:{candidate['category']}"].append(weighted)
        matched.append({"id": candidate["id"], "composite_ratio": weighted, **per_metric})
    group_summary = {
        name: _numeric_summary(values) for name, values in sorted(groups.items())
    }
    return {
        "composite_ratio": _numeric_summary(ratios),
        "metric_ratios": {
            name: _numeric_summary(values) for name, values in metric_ratios.items()
        },
        "groups": group_summary,
        "segments": matched,
        "ratio_floors": {
            name: ratio_floors.get(name, 1e-6) for name in weights
        },
    }


def environment_report(root: Path) -> dict:
    def command_output(command: list[str]) -> str:
        result = subprocess.run(
            command,
            cwd=root,
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return result.stdout.strip()

    return {
        "created_at": utc_now(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "torchaudio": torchaudio.__version__,
        "onnxruntime": ort.__version__,
        "git_commit": command_output(["git", "rev-parse", "HEAD"]),
        "git_dirty": bool(command_output(["git", "status", "--porcelain"])),
    }


def evaluate_gates(report: dict, config: dict) -> dict:
    candidate_contract = report["models"]["candidate"]
    baseline_contract = report["models"]["baseline"]
    hard_failures = []
    if baseline_contract["unexpected_operators"]:
        hard_failures.append("baseline_unexpected_operators")
    if baseline_contract["dynamic_dimensions"]:
        hard_failures.append("baseline_dynamic_dimensions")
    if not baseline_contract["numerical_allclose"]:
        hard_failures.append("baseline_onnx_numerical_comparison")
    if not baseline_contract["stateful_steps_ok"]:
        hard_failures.append("baseline_stateful_comparison_steps")
    if not baseline_contract["deployment_contract_ok"]:
        hard_failures.append("baseline_deployment_contract")
    if baseline_contract["file_bytes"] > int(config["gates"]["maximum_onnx_bytes"]):
        hard_failures.append("baseline_onnx_size")
    if candidate_contract["unexpected_operators"]:
        hard_failures.append("unexpected_operators")
    if candidate_contract["dynamic_dimensions"]:
        hard_failures.append("dynamic_dimensions")
    if not candidate_contract["numerical_allclose"]:
        hard_failures.append("onnx_numerical_comparison")
    if not candidate_contract["stateful_steps_ok"]:
        hard_failures.append("stateful_comparison_steps")
    if not candidate_contract["deployment_contract_ok"]:
        hard_failures.append("deployment_contract")
    if candidate_contract["file_bytes"] > int(config["gates"]["maximum_onnx_bytes"]):
        hard_failures.append("onnx_size")
    for segment in report["segments"]["candidate"]:
        metrics = segment["audio_metrics"]
        signal = segment["signal_metrics"]
        if not metrics["finite"] or metrics["silent"]:
            hard_failures.append(f"signal_health:{segment['id']}")
            break
        if not all(value for key, value in signal.items() if key.endswith("_finite")):
            hard_failures.append(f"stem_health:{segment['id']}")
            break
    chunk = report.get("chunk_consistency", {})
    if chunk and float(chunk.get("max_abs", math.inf)) > float(config["gates"]["chunk_max_abs"]):
        hard_failures.append("chunk_consistency")

    comparison = report["comparison"]
    objective_failures = []
    if comparison["composite_ratio"]["median"] > float(config["gates"]["composite_median"]):
        objective_failures.append("composite_median")
    bad_groups = [
        name
        for name, value in comparison["groups"].items()
        if value["median"] > float(config["gates"]["group_median"])
    ]
    if bad_groups:
        objective_failures.append("group_regression")
    candidate_loudness_p95 = report["summary"]["candidate"]["loudness_error_lu"]["p95"]
    baseline_loudness_p95 = report["summary"]["baseline"]["loudness_error_lu"]["p95"]
    if candidate_loudness_p95 > baseline_loudness_p95 + float(
        config["gates"]["loudness_p95_regression_lu"]
    ):
        objective_failures.append("loudness_p95")
    candidate_latency = candidate_contract["latency_ms"]["p95"]
    baseline_latency = baseline_contract["latency_ms"]["p95"]
    if candidate_latency > baseline_latency * float(config["gates"]["latency_p95_ratio"]):
        objective_failures.append("latency_p95")
    hard_passed = not hard_failures
    objective_passed = hard_passed and not objective_failures
    return {
        "hard_gates_passed": hard_passed,
        "hard_failures": hard_failures,
        "objective_eligible": objective_passed,
        "objective_failures": objective_failures,
        "human_status": report.get("human_review", {}).get("status", "not_generated"),
        "promotion_eligible": False,
    }


def write_metrics_csv(path: Path, report: dict) -> None:
    rows = []
    for role in ("baseline", "candidate"):
        for segment in report["segments"][role]:
            row = {
                "role": role,
                "segment_id": segment["id"],
                "year": segment["year"],
                "category": segment["category"],
            }
            row.update(segment["audio_metrics"])
            row.update(segment["signal_metrics"])
            rows.append(row)
    fieldnames = sorted({name for row in rows for name in row})
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_report(report: dict) -> str:
    verdict = report["verdict"]
    comparison = report["comparison"]["composite_ratio"]
    human = report.get("human_review", {"status": "not_generated"})
    lines = [
        "# DDSP-Piano 标准化测试报告",
        "",
        f"- Schema：`{report['schema']}`",
        f"- Profile：`{report['profile']}`",
        f"- 测试集：`{report['corpus']['corpus_sha256']}`",
        f"- Baseline：`{report['models']['baseline']['sha256']}`",
        f"- Candidate：`{report['models']['candidate']['sha256']}`",
        "",
        "## 结论",
        "",
        f"- 正确性门禁：{'通过' if verdict['hard_gates_passed'] else '失败'}",
        f"- 客观音质门禁：{'通过' if verdict['objective_eligible'] else '失败'}",
        f"- 人工评审：`{human['status']}`",
        f"- 可晋级：{'是' if verdict['promotion_eligible'] else '否'}",
        "",
        "## 核心指标",
        "",
        "| 指标 | 结果 | 门槛 |",
        "| --- | ---: | ---: |",
        f"| 综合比值中位数 | {comparison['median']:.4f} | <= {report['config']['gates']['composite_median']:.4f} |",
        f"| 综合比值 P95 | {comparison['p95']:.4f} | 记录 |",
        f"| Candidate ONNX P95 | {report['models']['candidate']['latency_ms']['p95']:.4f} ms | 相对 baseline |",
        "",
        "## 失败项",
        "",
    ]
    failures = verdict["hard_failures"] + verdict["objective_failures"]
    lines.extend([f"- `{failure}`" for failure in failures] or ["- 无"])
    lines.extend(["", "## 产物", "", "- 逐片段数据：`metrics.csv`", "- 机器报告：`report.json`"])
    if human["status"] != "not_generated":
        lines.append("- 盲听入口：`listening/index.html`")
        lines.append(f"- 人工评审截止：`{human['deadline']}`")
    lines.append("")
    return "\n".join(lines)
