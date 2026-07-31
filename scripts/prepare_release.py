#!/usr/bin/env python3
"""Build the four verified assets for a DDSP-Piano model-suite release."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.model_registry import load_model_registry


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def state_dict_sha256(state_dict: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(state_dict.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(str(tuple(value.shape)).encode("ascii"))
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def parse_sources(values: list[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for value in values:
        model_id, separator, path = value.partition("=")
        if not separator or not model_id or not path:
            raise ValueError("--source values must use MODEL_ID=CHECKPOINT")
        result[model_id] = Path(path).resolve()
    return result


def public_checkpoint(source: Path, destination: Path, model_spec, release: str) -> dict[str, Any]:
    checkpoint = torch.load(source, map_location="cpu", weights_only=False)
    if "model" not in checkpoint or "piano_models" not in checkpoint:
        raise ValueError(f"Not a DDSP-Piano training checkpoint: {source}")

    args = dict(checkpoint.get("args", {}))
    args.pop("model_variant", None)
    args.update(model_spec.model)
    args["model_id"] = model_spec.model_id
    args["architecture"] = model_spec.architecture
    for source_name, destination_name in {"learning_rate": "lr"}.items():
        if source_name in model_spec.training:
            args[destination_name] = model_spec.training[source_name]
    for name, value in model_spec.training.items():
        if name != "learning_rate":
            args[name] = value
    for name in (
        "resume",
        "weights",
        "finetune_from",
        "init_checkpoint",
        "quality_manifest",
    ):
        args[name] = None
    args["registry"] = f"ddsp_piano/{release}.json"
    args["maestro_root"] = "data/maestro-v3.0.0"
    args["cache_dir"] = "cache/maestro-v3.0.0"
    args["experiment_dir"] = f"runs/{model_spec.model_id}"

    checkpoint["args"] = args
    checkpoint["model_id"] = model_spec.model_id
    checkpoint["architecture"] = model_spec.architecture
    checkpoint["model_suite_release"] = release
    checkpoint["source_checkpoint_sha256"] = checkpoint.get(
        "source_checkpoint_sha256", sha256_file(source)
    )
    checkpoint["model_state_sha256"] = state_dict_sha256(checkpoint["model"])
    if isinstance(checkpoint.get("initialization"), dict):
        initialization = checkpoint["initialization"]
        checkpoint["initialization"] = {
            name: initialization[name]
            for name in ("checkpoint_sha256", "loaded_tensors", "target_tensors")
            if name in initialization
        }
    if isinstance(checkpoint.get("loss_calibration"), dict):
        checkpoint["loss_calibration"] = dict(checkpoint["loss_calibration"])
        checkpoint["loss_calibration"].pop("quality_manifest", None)

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(checkpoint, temporary)
    temporary.replace(destination)
    return {
        "source_checkpoint_sha256": checkpoint["source_checkpoint_sha256"],
        "model_state_sha256": checkpoint["model_state_sha256"],
        "published_checkpoint_sha256": sha256_file(destination),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", action="append", required=True, metavar="MODEL_ID=CHECKPOINT")
    parser.add_argument("--registry", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--verify-steps", type=int, default=100)
    args = parser.parse_args()

    registry = load_model_registry(args.registry) if args.registry is not None else load_model_registry()
    sources = parse_sources(args.source)
    missing = set(registry.models) - set(sources)
    extra = set(sources) - set(registry.models)
    if missing or extra:
        raise ValueError(f"Sources must match registry; missing={sorted(missing)}, extra={sorted(extra)}")
    if any(not path.is_file() for path in sources.values()):
        missing_paths = [str(path) for path in sources.values() if not path.is_file()]
        raise FileNotFoundError(f"Checkpoint files not found: {missing_paths}")

    output_dir = (args.output_dir or Path("artifacts") / registry.release).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": "ddsp-piano-release/v1",
        "release": registry.release,
        "default_model_id": registry.default_model_id,
        "deployment_contract": registry.deployment_contract,
        "models": {},
    }
    report_lines = [
        f"# {registry.release} validation",
        "",
        "All models passed PyTorch CPU, ONNX checker, and stateful ONNX Runtime comparison.",
        "OM/CANN validation was not run in this repository.",
        "",
        "| Model | Architecture | Reverb output | Stateful steps |",
        "| --- | --- | --- | ---: |",
    ]

    for model_id, model_spec in registry.models.items():
        checkpoint_path = model_spec.asset_path(output_dir, ".pt")
        provenance = public_checkpoint(
            sources[model_id], checkpoint_path, model_spec, registry.release
        )
        onnx_path = model_spec.asset_path(output_dir, ".onnx")
        subprocess.run(
            [
                sys.executable,
                str(ROOT / "scripts" / "export_onnx.py"),
                "--model-id",
                model_id,
                "--registry",
                str(registry.source),
                "--checkpoint",
                str(checkpoint_path),
                "--output",
                str(onnx_path),
                "--verify-steps",
                str(args.verify_steps),
            ],
            cwd=ROOT,
            check=True,
        )
        metadata_path = onnx_path.with_suffix(".json")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        manifest["models"][model_id] = {
            "display_name": model_spec.display_name,
            "architecture": model_spec.architecture,
            "lineage": model_spec.lineage,
            "quality_status": model_spec.quality_status,
            "assets": {
                path.name: {"bytes": path.stat().st_size, "sha256": sha256_file(path)}
                for path in (checkpoint_path, onnx_path, metadata_path)
            },
            **provenance,
        }
        report_lines.append(
            f"| `{model_id}` | `{model_spec.architecture}` | "
            f"`{metadata['reverb_output']}` | {metadata['onnx_runtime_stateful_steps']} |"
        )

    manifest_path = output_dir / "model-suite.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    validation_path = output_dir / "VALIDATION.md"
    validation_path.write_text("\n".join(report_lines) + "\n", encoding="utf-8")
    checksum_paths = sorted(
        path for path in output_dir.iterdir() if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(f"{sha256_file(path)}  {path.name}\n" for path in checksum_paths),
        encoding="ascii",
    )
    release_index = ROOT / "releases"
    release_index.mkdir(parents=True, exist_ok=True)
    (release_index / f"{registry.release}.json").write_text(
        manifest_path.read_text(encoding="utf-8"), encoding="utf-8"
    )
    (release_index / f"{registry.release}.sha256").write_text(
        (output_dir / "SHA256SUMS").read_text(encoding="ascii"), encoding="ascii"
    )
    print(json.dumps({"release": registry.release, "output_dir": str(output_dir)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
