#!/usr/bin/env python3
"""Prepare, run, and finalize the DDSP-Piano quality-v1 evaluation suite."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.evaluation import (
    EVALUATION_SCHEMA,
    audio_metrics,
    build_corpus,
    canonical_sha256,
    compare_models,
    environment_report,
    evaluate_gates,
    inspect_onnx_model,
    markdown_report,
    read_json,
    sha256_file,
    signal_metrics,
    summarize_segments,
    write_json,
    write_metrics_csv,
)
from ddsp_piano.listening import (
    create_listening_package,
    defer_review,
    finalize_review,
    select_excerpt_frames,
)
from ddsp_piano.maestro import PreprocessConfig, load_midi_conditioning, prepare_split
from scripts.render_onnx import (
    _render_streaming_conditioning,
    _shape,
    _validate_contract,
)


DEFAULT_CONFIG = ROOT / "configs" / "evaluation_v1.json"


def _implementation_hashes() -> dict[str, str]:
    return {
        "evaluation": sha256_file(ROOT / "ddsp_piano" / "evaluation.py"),
        "listening": sha256_file(ROOT / "ddsp_piano" / "listening.py"),
        "renderer": sha256_file(ROOT / "scripts" / "render_onnx.py"),
        "cli": sha256_file(Path(__file__).resolve()),
    }


def _preprocess(config: dict) -> PreprocessConfig:
    return PreprocessConfig(**config["preprocess"])


def _metadata(model_path: Path) -> dict:
    path = model_path.with_suffix(".json")
    if not path.is_file():
        raise FileNotFoundError(f"Deployment JSON not found: {path}")
    return read_json(path)


def _validate_pair(baseline: dict, candidate: dict, corpus: dict) -> None:
    for name in ("sample_rate", "frame_rate"):
        if int(baseline[name]) != int(candidate[name]):
            raise ValueError(f"Model {name} values differ")
        if int(baseline[name]) != int(corpus["preprocess"][name]):
            raise ValueError(f"Corpus and model {name} values differ")
    baseline_polyphony = _shape(baseline, "inputs", "conditioning")[2]
    candidate_polyphony = _shape(candidate, "inputs", "conditioning")[2]
    if baseline_polyphony != candidate_polyphony:
        raise ValueError("Model polyphony contracts differ")
    if baseline_polyphony != int(corpus["preprocess"]["max_polyphony"]):
        raise ValueError("Corpus and model polyphony contracts differ")
    if baseline.get("piano_model_index_to_maestro_year") != candidate.get(
        "piano_model_index_to_maestro_year"
    ):
        raise ValueError("Model piano-year embedding mappings differ")


def _render_arguments(metadata: dict) -> tuple[str, str, float]:
    return (
        str(metadata.get("reverb_output", "reverb_ir")),
        str(metadata.get("reverb_ir_postprocess", {}).get("type", "ir")),
        float(metadata.get("reverb_wet_gain", 1.0)),
    )


def _render_track(
    model_path: Path,
    metadata: dict,
    conditioning: np.ndarray,
    pedal: np.ndarray,
    piano_model: int,
    config: dict,
) -> dict[str, np.ndarray]:
    sample_rate, frame_rate, _, _ = _validate_contract(metadata)
    reverb_output, reverb_type, reverb_wet_gain = _render_arguments(metadata)
    warm_up_frames = int(round(float(config["render"]["warm_up_seconds"]) * frame_rate))
    chunk_frames = max(1, int(round(float(config["render"]["chunk_seconds"]) * frame_rate)))
    return _render_streaming_conditioning(
        model_path,
        metadata,
        conditioning,
        pedal,
        piano_model,
        warm_up_frames,
        chunk_frames,
        int(config["render"]["noise_seed"]),
        reverb_output,
        reverb_type,
        reverb_wet_gain,
    )


def _evaluate_model_segments(
    model_path: Path,
    metadata: dict,
    corpus: dict,
    config: dict,
) -> list[dict]:
    sample_rate = int(metadata["sample_rate"])
    frame_rate = int(metadata["frame_rate"])
    samples_per_frame = sample_rate // frame_rate
    segment_frames = int(corpus["preprocess"]["segment_seconds"] * frame_rate)
    segment_samples = segment_frames * samples_per_frame
    by_track: dict[str, list[dict]] = defaultdict(list)
    for entry in corpus["entries"]:
        by_track[entry["cache_path"]].append(entry)

    results = []
    for track_number, (cache_name, entries) in enumerate(sorted(by_track.items()), start=1):
        cache_path = Path(cache_name)
        max_frame = max(int(entry["frame_start"]) + segment_frames for entry in entries)
        audio = np.load(cache_path / "audio.npy", mmap_mode="r")
        conditioning = np.asarray(
            np.load(cache_path / "conditioning.npy", mmap_mode="r")[:max_frame],
            dtype=np.float32,
        )
        pedal = np.asarray(
            np.load(cache_path / "pedal.npy", mmap_mode="r")[:max_frame],
            dtype=np.float32,
        )
        print(
            f"Evaluation render {track_number}/{len(by_track)}: {cache_path.name}",
            flush=True,
        )
        signals = _render_track(
            model_path,
            metadata,
            conditioning,
            pedal,
            int(entries[0]["piano_model"]),
            config,
        )
        for entry in entries:
            sample_start = int(entry["sample_start"])
            sample_end = sample_start + segment_samples
            target = np.asarray(audio[sample_start:sample_end], dtype=np.float32)
            signal_slice = {
                name: value[sample_start:sample_end]
                for name, value in signals.items()
            }
            result = {
                "id": entry["id"],
                "track_id": entry["track_id"],
                "year": entry["year"],
                "category": entry["category"],
                "frame_start": entry["frame_start"],
                "audio_metrics": audio_metrics(
                    signal_slice["wet"],
                    target,
                    sample_rate,
                    config["metrics"]["fft_sizes"],
                ),
                "signal_metrics": signal_metrics(signal_slice),
            }
            results.append(result)
    return sorted(results, key=lambda entry: entry["id"])


def _chunk_consistency(model_path: Path, metadata: dict, corpus: dict, config: dict) -> dict:
    entry = corpus["entries"][0]
    cache_path = Path(entry["cache_path"])
    frame_rate = int(metadata["frame_rate"])
    segment_frames = int(corpus["preprocess"]["segment_seconds"] * frame_rate)
    conditioning = np.asarray(
        np.load(cache_path / "conditioning.npy", mmap_mode="r")[:segment_frames],
        dtype=np.float32,
    )
    pedal = np.asarray(
        np.load(cache_path / "pedal.npy", mmap_mode="r")[:segment_frames],
        dtype=np.float32,
    )
    values = []
    for chunk_seconds in (1.0, 4.0):
        local_config = json.loads(json.dumps(config))
        local_config["render"]["chunk_seconds"] = chunk_seconds
        values.append(
            _render_track(
                model_path,
                metadata,
                conditioning,
                pedal,
                int(entry["piano_model"]),
                local_config,
            )["wet"]
        )
    difference = np.abs(values[0] - values[1])
    return {
        "chunk_seconds": [1.0, 4.0],
        "max_abs": float(difference.max(initial=0.0)),
        "mean_abs": float(difference.mean()),
    }


def _midi_paths(midi_dir: Path) -> list[Path]:
    paths = sorted(
        path
        for path in midi_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )
    if not paths:
        raise FileNotFoundError(f"No MIDI files found in: {midi_dir}")
    return paths


def _listening_items(
    baseline_path: Path,
    candidate_path: Path,
    baseline_metadata: dict,
    candidate_metadata: dict,
    midi_dir: Path,
    config: dict,
) -> list[dict]:
    sample_rate = int(baseline_metadata["sample_rate"])
    frame_rate = int(baseline_metadata["frame_rate"])
    max_polyphony = _shape(baseline_metadata, "inputs", "conditioning")[2]
    tail_seconds = float(config["render"]["tail_seconds"])
    excerpt_seconds = float(config["listening"]["excerpt_seconds"])
    items = []
    for midi_path in _midi_paths(midi_dir):
        midi = load_midi_conditioning(
            midi_path,
            frame_rate=frame_rate,
            max_polyphony=max_polyphony,
            tail_seconds=tail_seconds,
        )
        start_frame, end_frame = select_excerpt_frames(
            midi.conditioning,
            midi.pedal,
            frame_rate,
            excerpt_seconds,
        )
        start_sample = start_frame * (sample_rate // frame_rate)
        end_sample = end_frame * (sample_rate // frame_rate)
        rendered = {}
        full_peaks = {}
        for role, model_path, metadata in (
            ("baseline", baseline_path, baseline_metadata),
            ("candidate", candidate_path, candidate_metadata),
        ):
            piano_models = metadata["piano_model_index_to_maestro_year"]
            piano_model = min(9, len(piano_models) - 1)
            signals = _render_track(
                model_path,
                metadata,
                midi.conditioning,
                midi.pedal,
                piano_model,
                config,
            )
            full_peaks[role] = float(np.max(np.abs(signals["wet"]), initial=0.0))
            rendered[role] = {
                name: value[start_sample:end_sample] for name, value in signals.items()
            }
        items.append(
            {
                "id": midi_path.stem,
                "title": midi_path.stem.replace("-", " "),
                "sample_rate": sample_rate,
                "start_seconds": start_frame / frame_rate,
                "duration_seconds": (end_frame - start_frame) / frame_rate,
                "baseline_full_peak": full_peaks["baseline"],
                "candidate_full_peak": full_peaks["candidate"],
                "baseline": rendered["baseline"],
                "candidate": rendered["candidate"],
            }
        )
    return items


def _default_corpus_path(config: dict, profile: str) -> Path:
    return ROOT / "evaluations" / "corpora" / f"{config['suite_id']}-{profile}.json"


def _cached_segments(
    model_path: Path,
    metadata: dict,
    model_sha256: str,
    corpus: dict,
    config: dict,
) -> tuple[list[dict], Path, bool]:
    cache_contract = {
        "schema": "ddsp-piano-segment-cache/v1",
        "model_sha256": model_sha256,
        "model_metadata_sha256": sha256_file(model_path.with_suffix(".json")),
        "corpus_sha256": corpus["corpus_sha256"],
        "render": config["render"],
        "metrics": config["metrics"],
        "implementation": {
            name: value
            for name, value in _implementation_hashes().items()
            if name in {"evaluation", "renderer", "cli"}
        },
    }
    cache_key = canonical_sha256(cache_contract)
    cache_path = ROOT / "evaluations" / "cache" / cache_key / "segments.json"
    if cache_path.is_file():
        cached = read_json(cache_path)
        if cached.get("contract") != cache_contract:
            raise RuntimeError(f"Segment cache contract mismatch: {cache_path}")
        return cached["segments"], cache_path.resolve(), True
    segments = _evaluate_model_segments(model_path, metadata, corpus, config)
    write_json(cache_path, {"contract": cache_contract, "segments": segments})
    return segments, cache_path.resolve(), False


def prepare_command(args: argparse.Namespace, config: dict) -> int:
    if args.profile == "release" and args.prepare_missing:
        minimum = int(config["release_cache"]["minimum_free_gb"]) * 1024**3
        free = shutil.disk_usage(args.cache_dir).free
        if free < minimum:
            raise RuntimeError(
                f"Release cache requires at least {minimum / 1024**3:.0f} GiB free; "
                f"found {free / 1024**3:.1f} GiB"
            )
        prepare_split(
            args.maestro_root,
            args.cache_dir,
            "test",
            _preprocess(config),
            workers=int(config["release_cache"]["prepare_workers"]),
        )
    corpus = build_corpus(
        args.maestro_root,
        args.cache_dir,
        args.profile,
        _preprocess(config),
    )
    destination = args.output or _default_corpus_path(config, args.profile)
    if destination.exists() and not args.refresh:
        existing = read_json(destination)
        if existing.get("corpus_sha256") != corpus["corpus_sha256"]:
            raise RuntimeError(f"Corpus is locked and differs from regenerated data: {destination}")
        print(json.dumps({"corpus": str(destination.resolve()), "status": "unchanged"}, indent=2))
        return 0
    write_json(destination, corpus)
    print(
        json.dumps(
            {
                "corpus": str(destination.resolve()),
                "sha256": corpus["corpus_sha256"],
                "segments": len(corpus["entries"]),
            },
            indent=2,
        )
    )
    return 0


def run_command(args: argparse.Namespace, config: dict) -> int:
    baseline_path = args.baseline.resolve()
    candidate_path = args.candidate.resolve()
    corpus_path = args.corpus or _default_corpus_path(config, args.profile)
    if not corpus_path.is_file():
        raise FileNotFoundError(
            f"Evaluation corpus not found: {corpus_path}. Run the prepare subcommand first."
        )
    corpus = read_json(corpus_path)
    if corpus.get("profile") != args.profile:
        raise ValueError(
            f"Corpus profile {corpus.get('profile')!r} does not match {args.profile!r}"
        )
    if corpus.get("preprocess") != config["preprocess"]:
        raise ValueError("Corpus preprocessing does not match the evaluation config")
    baseline_metadata = _metadata(baseline_path)
    candidate_metadata = _metadata(candidate_path)
    _validate_pair(baseline_metadata, candidate_metadata, corpus)
    allowed = set(config["allowed_operators"])
    iterations = int(config["profiles"][args.profile]["latency_iterations"])
    baseline_contract = inspect_onnx_model(
        baseline_path,
        allowed,
        int(config["gates"]["minimum_stateful_steps"]),
        iterations,
    )
    candidate_contract = inspect_onnx_model(
        candidate_path,
        allowed,
        int(config["gates"]["minimum_stateful_steps"]),
        iterations,
    )
    implementation = _implementation_hashes()
    evaluation_id = canonical_sha256(
        {
            "schema": EVALUATION_SCHEMA,
            "config": config,
            "implementation": implementation,
            "corpus": corpus["corpus_sha256"],
            "baseline": baseline_contract["sha256"],
            "baseline_metadata": baseline_contract["metadata_sha256"],
            "candidate": candidate_contract["sha256"],
            "candidate_metadata": candidate_contract["metadata_sha256"],
        }
    )
    output_dir = args.output_dir or (
        ROOT
        / "evaluations"
        / config["suite_id"]
        / corpus["corpus_sha256"][:12]
        / f"{baseline_contract['sha256'][:12]}__{candidate_contract['sha256'][:12]}"
        / args.profile
    )
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    existing_report_path = output_dir / "report.json"
    if existing_report_path.is_file():
        existing = read_json(existing_report_path)
        if existing.get("evaluation_id") != evaluation_id:
            raise RuntimeError(
                f"Refusing to overwrite a different evaluation: {existing_report_path}"
            )
        write_metrics_csv(output_dir / "metrics.csv", existing)
        (output_dir / "report.md").write_text(
            markdown_report(existing), encoding="utf-8"
        )
        print(
            json.dumps(
                {
                    "report_directory": str(output_dir),
                    "report": str(existing_report_path),
                    "status": "unchanged",
                    "objective_eligible": existing["verdict"]["objective_eligible"],
                    "human_status": existing["human_review"]["status"],
                },
                indent=2,
            )
        )
        return 0
    write_json(output_dir / "corpus.json", corpus)

    baseline_segments, baseline_cache, baseline_cache_hit = _cached_segments(
        baseline_path,
        baseline_metadata,
        baseline_contract["sha256"],
        corpus,
        config,
    )
    candidate_segments, candidate_cache, candidate_cache_hit = _cached_segments(
        candidate_path,
        candidate_metadata,
        candidate_contract["sha256"],
        corpus,
        config,
    )
    report = {
        "schema": EVALUATION_SCHEMA,
        "evaluation_id": evaluation_id,
        "profile": args.profile,
        "implementation": implementation,
        "environment": environment_report(ROOT),
        "config": config,
        "corpus": corpus,
        "models": {
            "baseline": baseline_contract,
            "candidate": candidate_contract,
        },
        "segment_cache": {
            "baseline": {"path": str(baseline_cache), "hit": baseline_cache_hit},
            "candidate": {"path": str(candidate_cache), "hit": candidate_cache_hit},
        },
        "segments": {
            "baseline": baseline_segments,
            "candidate": candidate_segments,
        },
        "summary": {
            "baseline": summarize_segments(baseline_segments),
            "candidate": summarize_segments(candidate_segments),
        },
        "comparison": compare_models(
            baseline_segments,
            candidate_segments,
            config["metrics"]["weights"],
        ),
        "chunk_consistency": _chunk_consistency(
            candidate_path, candidate_metadata, corpus, config
        ),
    }
    listening_enabled = (
        not args.skip_listening
        and args.profile in config["listening"]["enabled_profiles"]
    )
    if listening_enabled:
        items = _listening_items(
            baseline_path,
            candidate_path,
            baseline_metadata,
            candidate_metadata,
            args.midi_dir.resolve(),
            config,
        )
        report["human_review"] = create_listening_package(
            output_dir,
            evaluation_id,
            items,
            int(config["listening"]["human_review_timeout_minutes"]),
            float(config["listening"]["target_lufs"]),
            float(config["listening"]["fixed_target_peak_dbfs"]),
        )
    else:
        report["human_review"] = {"status": "not_generated"}
    report["verdict"] = evaluate_gates(report, config)
    if report["human_review"].get("clipped_samples", 0):
        report["verdict"]["hard_gates_passed"] = False
        report["verdict"]["objective_eligible"] = False
        report["verdict"]["hard_failures"].append("fixed_gain_clipping")
    write_json(output_dir / "report.json", report)
    write_metrics_csv(output_dir / "metrics.csv", report)
    (output_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(
        json.dumps(
            {
                "report_directory": str(output_dir),
                "report": str((output_dir / "report.json").resolve()),
                "objective_eligible": report["verdict"]["objective_eligible"],
                "human_status": report["human_review"]["status"],
            },
            indent=2,
        )
    )
    return 0


def finalize_command(args: argparse.Namespace, config: dict) -> int:
    report_dir = args.report_dir.resolve()
    existing = read_json(report_dir / "report.json")
    report = finalize_review(
        report_dir,
        args.scores.resolve(),
        existing.get("config", config),
    )
    (args.report_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2, ensure_ascii=False))
    return 0


def defer_command(args: argparse.Namespace) -> int:
    report = defer_review(args.report_dir.resolve())
    (args.report_dir / "report.md").write_text(markdown_report(report), encoding="utf-8")
    print(json.dumps(report["verdict"], indent=2, ensure_ascii=False))
    return 0


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    commands = root.add_subparsers(dest="command", required=True)

    prepare = commands.add_parser("prepare", help="Build and lock an evaluation corpus")
    prepare.add_argument("--profile", choices=("quick", "dev", "release"), default="dev")
    prepare.add_argument("--maestro-root", type=Path, required=True)
    prepare.add_argument("--cache-dir", type=Path, required=True)
    prepare.add_argument("--output", type=Path)
    prepare.add_argument("--refresh", action="store_true")
    prepare.add_argument("--prepare-missing", action="store_true")

    run = commands.add_parser("run", help="Evaluate one ONNX candidate against a baseline")
    run.add_argument("--baseline", type=Path, required=True)
    run.add_argument("--candidate", type=Path, required=True)
    run.add_argument("--profile", choices=("quick", "dev", "release"), default="dev")
    run.add_argument("--corpus", type=Path)
    run.add_argument("--output-dir", type=Path)
    run.add_argument("--midi-dir", type=Path, default=ROOT / "midi")
    run.add_argument("--skip-listening", action="store_true")

    finalize = commands.add_parser("finalize", help="Import blind-listening scores")
    finalize.add_argument("--report-dir", type=Path, required=True)
    finalize.add_argument("--scores", type=Path, required=True)

    defer = commands.add_parser("defer", help="Mark an expired listening task as deferred")
    defer.add_argument("--report-dir", type=Path, required=True)
    return root


def main() -> int:
    args = parser().parse_args()
    config = read_json(args.config.resolve())
    if args.command == "prepare":
        return prepare_command(args, config)
    if args.command == "run":
        return run_command(args, config)
    if args.command == "finalize":
        return finalize_command(args, config)
    if args.command == "defer":
        return defer_command(args)
    raise AssertionError(args.command)


if __name__ == "__main__":
    raise SystemExit(main())
