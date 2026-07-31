#!/usr/bin/env python3
"""Run baseline-anchored pilot training and continue only promoted models."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.model_registry import load_model_registry


DEFAULT_CONFIG = ROOT / "configs" / "quality-recovery.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def resolve_repo_path(value: str | Path) -> Path:
    path = Path(value)
    return (path if path.is_absolute() else ROOT / path).resolve()


def run(command: list[str], *, dry_run: bool) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def checkpoint_path(run_root: Path, model_id: str, phase: str, name: str = "best.pt") -> Path:
    return run_root / "training" / model_id / phase / "checkpoints" / name


def training_command(
    *,
    python: str,
    registry: Path,
    model_id: str,
    source_checkpoint: Path,
    experiment_dir: Path,
    maestro_root: Path,
    cache_dir: Path,
    quality_manifest: Path,
    phase_settings: dict[str, Any],
    model_settings: dict[str, Any],
    common: dict[str, Any],
) -> list[str]:
    command = [
        python,
        str(ROOT / "train.py"),
        "--registry",
        str(registry),
        "--model-id",
        model_id,
        "--stage",
        str(common["stage"]),
        "--maestro-root",
        str(maestro_root),
        "--cache-dir",
        str(cache_dir),
        "--quality-manifest",
        str(quality_manifest),
        "--experiment-dir",
        str(experiment_dir),
        "--epochs",
        str(int(phase_settings["epochs"])),
        "--steps-per-epoch",
        str(int(phase_settings["steps_per_epoch"])),
        "--batch-size",
        str(int(model_settings["batch_size"])),
        "--lr",
        str(float(model_settings["learning_rate"])),
        "--sampling-mode",
        str(common["sampling_mode"]),
        "--curriculum-tail-fraction",
        str(float(common["curriculum_tail_fraction"])),
        "--balanced-validation",
        "--validation-batches",
        str(int(phase_settings["validation_batches"])),
        "--device",
        "cuda",
        "--amp",
        "--synthesis-layout",
        "vectorized",
        "--optimizer-implementation",
        "fused",
        "--spectral-layout",
        "combined",
        "--velocity-counterfactual-layout",
        "combined",
        "--train-workers",
        str(int(common["train_workers"])),
        "--validation-workers",
        str(int(common["validation_workers"])),
        "--log-every",
        str(int(common["log_every"])),
        "--reverb-regularizer-reduction",
        str(common["reverb_regularizer_reduction"]),
        "--finetune-from",
        str(source_checkpoint),
    ]
    if common["freeze_reverb"]:
        command.append("--freeze-reverb")
    return command


def promoted_models(summary: dict[str, Any]) -> list[str]:
    return sorted(
        model_id
        for model_id, result in summary["promotions"].items()
        if result["decision"] == "candidate"
    )


def replace_finetune_with_resume(command: list[str], checkpoint: Path) -> list[str]:
    updated = list(command)
    index = updated.index("--finetune-from")
    updated[index : index + 2] = ["--resume", str(checkpoint)]
    return updated


def run_screening(
    *,
    python: str,
    run_root: Path,
    registry: Path,
    baseline_dir: Path,
    maestro_root: Path,
    cache_dir: Path,
    promotion: dict[str, Any],
    dry_run: bool,
) -> Path:
    output_root = run_root / "screening" / "quick"
    command = [
        python,
        str(ROOT / "scripts" / "sweep_stage_checkpoints.py"),
        "--run-root",
        str(run_root),
        "--registry",
        str(registry),
        "--baseline-dir",
        str(baseline_dir),
        "--maestro-root",
        str(maestro_root),
        "--cache-dir",
        str(cache_dir),
        "--output-root",
        str(output_root),
        "--profile",
        "quick",
        "--max-composite-median",
        str(float(promotion["max_composite_median"])),
        "--max-wet-dry-factor",
        str(float(promotion["max_wet_dry_factor"])),
        "--max-latency-ratio",
        str(float(promotion["max_latency_ratio"])),
        "--max-regressed-groups",
        str(int(promotion.get("max_regressed_groups", 0))),
        "--max-worst-group-median",
        str(float(promotion.get("max_worst_group_median", 1.05))),
    ]
    run(command, dry_run=dry_run)
    return output_root / "summary.json"


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--maestro-root", type=Path, default=ROOT / "data/maestro-v3.0.0")
    result.add_argument("--cache-dir", type=Path, default=ROOT / "cache/maestro-v3.0.0")
    result.add_argument("--run-root", type=Path)
    result.add_argument("--python", default=sys.executable)
    result.add_argument("--skip-full", action="store_true")
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    config = read_json(args.config.resolve())
    if config.get("schema") != "ddsp-piano-quality-recovery/v1":
        raise ValueError("unsupported quality recovery config schema")
    registry_path = resolve_repo_path(config["baseline_registry"])
    baseline_dir = resolve_repo_path(config["baseline_dir"])
    run_root = (args.run_root or resolve_repo_path(config["run_root"])).resolve()
    quality_manifest = resolve_repo_path(config["quality_manifest"])
    registry = load_model_registry(registry_path)
    model_ids = list(config["models"])
    unknown = set(model_ids) - set(registry.models)
    if unknown:
        raise ValueError(f"unknown recovery models: {sorted(unknown)}")
    if not quality_manifest.is_file() and not args.dry_run:
        raise FileNotFoundError(
            f"quality manifest is missing: {quality_manifest}; run build_quality_manifest.py"
        )

    state_path = run_root / "recovery-state.json"
    state = read_json(state_path) if state_path.is_file() else {
        "schema": "ddsp-piano-quality-recovery-state/v1",
        "created_at": utc_now(),
        "completed_phases": [],
    }
    state["status"] = "pilot_training"
    state["updated_at"] = utc_now()
    write_json(state_path, state)

    common = config["training"]
    for model_id in model_ids:
        experiment_dir = run_root / "training" / model_id / "pilot"
        best = checkpoint_path(run_root, model_id, "pilot")
        if best.is_file():
            continue
        source = registry.require(model_id).asset_path(baseline_dir, ".pt")
        if not source.is_file() and not args.dry_run:
            raise FileNotFoundError(f"stable baseline checkpoint is missing: {source}")
        model_settings = {
            **config["models"][model_id],
            "learning_rate": config["models"][model_id]["pilot_learning_rate"],
        }
        command = training_command(
            python=args.python,
            registry=registry_path,
            model_id=model_id,
            source_checkpoint=source,
            experiment_dir=experiment_dir,
            maestro_root=args.maestro_root.resolve(),
            cache_dir=args.cache_dir.resolve(),
            quality_manifest=quality_manifest,
            phase_settings=config["pilot"],
            model_settings=model_settings,
            common=common,
        )
        last = checkpoint_path(run_root, model_id, "pilot", "last.pt")
        if last.is_file():
            command = replace_finetune_with_resume(command, last)
        run(command, dry_run=args.dry_run)

    if args.dry_run:
        state["status"] = "dry_run_complete"
        state["updated_at"] = utc_now()
        write_json(state_path, state)
        return 0

    pilot_summary_path = run_screening(
        python=args.python,
        run_root=run_root,
        registry=registry_path,
        baseline_dir=baseline_dir,
        maestro_root=args.maestro_root.resolve(),
        cache_dir=args.cache_dir.resolve(),
        promotion=config["pilot_promotion"],
        dry_run=False,
    )
    pilot_summary = read_json(pilot_summary_path)
    selected = promoted_models(pilot_summary)
    state["pilot_promotions"] = selected
    state["pilot_summary"] = str(pilot_summary_path)
    state["status"] = "full_training" if selected else "screening_complete"
    state["updated_at"] = utc_now()
    write_json(state_path, state)

    if selected and config["full"]["enabled"] and not args.skip_full:
        for model_id in selected:
            experiment_dir = run_root / "training" / model_id / "full"
            best = checkpoint_path(run_root, model_id, "full")
            if best.is_file():
                continue
            model_settings = {
                **config["models"][model_id],
                "learning_rate": config["models"][model_id]["full_learning_rate"],
            }
            command = training_command(
                python=args.python,
                registry=registry_path,
                model_id=model_id,
                source_checkpoint=checkpoint_path(run_root, model_id, "pilot"),
                experiment_dir=experiment_dir,
                maestro_root=args.maestro_root.resolve(),
                cache_dir=args.cache_dir.resolve(),
                quality_manifest=quality_manifest,
                phase_settings=config["full"],
                model_settings=model_settings,
                common=common,
            )
            last = checkpoint_path(run_root, model_id, "full", "last.pt")
            if last.is_file():
                command = replace_finetune_with_resume(command, last)
            run(command, dry_run=False)
        final_summary_path = run_screening(
            python=args.python,
            run_root=run_root,
            registry=registry_path,
            baseline_dir=baseline_dir,
            maestro_root=args.maestro_root.resolve(),
            cache_dir=args.cache_dir.resolve(),
            promotion=config["final_promotion"],
            dry_run=False,
        )
        state["final_summary"] = str(final_summary_path)
        state["final_promotions"] = promoted_models(read_json(final_summary_path))

    state["status"] = "screening_complete"
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    print(json.dumps(state, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
