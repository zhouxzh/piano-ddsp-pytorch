#!/usr/bin/env python3
"""Render the same MIDI test set through every compatible exported ONNX model."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import onnx
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.versioning import model_output_label
from scripts.render_onnx import CONTROL_OUTPUT_NAMES, INPUT_NAMES, _validate_contract


def _midi_files(midi_dir: Path) -> list[Path]:
    files = sorted(
        path
        for path in midi_dir.iterdir()
        if path.is_file() and path.suffix.lower() in {".mid", ".midi"}
    )
    if not files:
        raise FileNotFoundError(f"No .mid or .midi files found in: {midi_dir}")
    return files


def _test_set_signature(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _model_sha256(model_path: Path) -> str:
    digest = hashlib.sha256()
    with model_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _inspect_model(model_path: Path) -> tuple[dict, list[str]]:
    reasons: list[str] = []
    metadata_path = model_path.with_suffix(".json")
    metadata: dict = {}
    if not metadata_path.is_file():
        reasons.append("adjacent deployment JSON is missing")
        return metadata, reasons
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        _validate_contract(metadata)
    except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError) as error:
        reasons.append(f"invalid fixed deployment metadata: {error}")

    try:
        onnx.checker.check_model(onnx.load(str(model_path)))
        session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
        inputs = [value.name for value in session.get_inputs()]
        outputs = [value.name for value in session.get_outputs()]
        expected_outputs = CONTROL_OUTPUT_NAMES + [
            str(metadata.get("reverb_output", "reverb_ir")),
            "next_context_state",
            "next_monophonic_state",
        ]
        if inputs != INPUT_NAMES:
            reasons.append(f"input contract mismatch: {inputs}")
        if outputs != expected_outputs:
            reasons.append(f"output contract mismatch: {outputs}")
    except Exception as error:
        reasons.append(f"ONNX validation failed: {error}")

    piano_models = metadata.get("piano_model_index_to_maestro_year", [])
    if not isinstance(piano_models, list) or not piano_models:
        reasons.append("piano embedding index mapping is missing")
    return metadata, reasons


def _write_index(path: Path, index: dict) -> None:
    path.write_text(json.dumps(index, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-dir", type=Path, default=Path("exports"))
    parser.add_argument("--midi-dir", type=Path, default=Path("midi"))
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("exports/midi_tests/all_models"),
    )
    parser.add_argument("--piano-model", type=int, default=9)
    parser.add_argument("--warm-up-seconds", type=float, default=0.5)
    parser.add_argument("--tail-seconds", type=float, default=2.5)
    parser.add_argument("--chunk-seconds", type=float, default=4.0)
    args = parser.parse_args()

    model_dir = args.model_dir.resolve()
    midi_dir = args.midi_dir.resolve()
    if not model_dir.is_dir():
        raise FileNotFoundError(f"ONNX model directory not found: {model_dir}")
    if not midi_dir.is_dir():
        raise FileNotFoundError(f"MIDI directory not found: {midi_dir}")
    if args.piano_model < 0:
        raise ValueError("--piano-model must be non-negative")

    midi_files = _midi_files(midi_dir)
    signature = _test_set_signature(midi_files)
    run_dir = (args.output_root / f"midi-{signature[:12]}").resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    index_path = run_dir / "index.json"
    index: dict[str, object] = {
        "status": "running",
        "midi_directory": str(midi_dir),
        "midi_set_sha256": signature,
        "midi_files": [path.name for path in midi_files],
        "model_directory": str(model_dir),
        "models": [],
        "excluded_models": [],
    }
    _write_index(index_path, index)

    compatible: list[tuple[Path, dict]] = []
    for model_path in sorted(model_dir.glob("*.onnx")):
        if model_path.is_symlink():
            index["excluded_models"].append(
                {
                    "model": str(model_path),
                    "reasons": ["compatibility alias; canonical model is rendered instead"],
                }
            )
            continue
        metadata, reasons = _inspect_model(model_path)
        if reasons:
            index["excluded_models"].append(
                {"model": str(model_path), "reasons": reasons}
            )
        else:
            compatible.append((model_path, metadata))
    _write_index(index_path, index)
    if not compatible:
        index["status"] = "failed"
        _write_index(index_path, index)
        raise RuntimeError(f"No compatible stateful ONNX models found in: {model_dir}")

    failures = 0
    used_output_labels: set[str] = set()
    renderer = ROOT / "scripts" / "render_onnx.py"
    for model_path, metadata in compatible:
        piano_models = metadata["piano_model_index_to_maestro_year"]
        piano_model = min(args.piano_model, len(piano_models) - 1)
        release_version = model_output_label(model_path, metadata)
        output_label = release_version
        if output_label in used_output_labels:
            output_label = f"{release_version}-{model_path.stem}"
        used_output_labels.add(output_label)
        output_dir = run_dir / output_label
        entry: dict[str, object] = {
            "model": str(model_path),
            "model_sha256": _model_sha256(model_path),
            "metadata": str(model_path.with_suffix(".json")),
            "model_variant": metadata.get("model_variant"),
            "release_version": release_version,
            "output_label": output_label,
            "training_phase": metadata.get("training_phase"),
            "piano_model": piano_model,
            "maestro_year": piano_models[piano_model],
            "output_directory": str(output_dir),
            "status": "running",
        }
        index["models"].append(entry)
        _write_index(index_path, index)
        command = [
            sys.executable,
            str(renderer),
            "--model",
            str(model_path),
            "--midi-dir",
            str(midi_dir),
            "--output-dir",
            str(output_dir),
            "--piano-model",
            str(piano_model),
            "--warm-up-seconds",
            str(args.warm_up_seconds),
            "--tail-seconds",
            str(args.tail_seconds),
            "--chunk-seconds",
            str(args.chunk_seconds),
        ]
        try:
            subprocess.run(command, cwd=ROOT, check=True)
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
            entry["wav_files"] = len(manifest["files"])
            entry["status"] = "complete"
        except (OSError, ValueError, KeyError, subprocess.CalledProcessError) as error:
            entry["status"] = "failed"
            entry["error"] = str(error)
            failures += 1
        _write_index(index_path, index)

    index["status"] = "complete" if failures == 0 else "partial_failure"
    _write_index(index_path, index)
    print(json.dumps({"index": str(index_path), "status": index["status"]}, indent=2))
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
