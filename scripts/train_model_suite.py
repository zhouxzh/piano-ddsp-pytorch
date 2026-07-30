#!/usr/bin/env python3
"""Run the resumable quality-first training and evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY = ROOT / "ddsp_piano" / "model-suite-v1.1.0-rc1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def run(
    command: list[str],
    state: dict,
    state_path: Path,
    dry_run: bool,
    *,
    operation: str = "command",
    stage_key: str | None = None,
    stage_batch_size: int | None = None,
) -> None:
    printable = " ".join(command)
    state["status"] = "running"
    state["last_command"] = printable
    state["active_operation"] = operation
    state.pop("return_code", None)
    state.pop("failure_reason", None)
    state.pop("failed_stage", None)
    if stage_key is not None:
        state["active_stage"] = stage_key
    if stage_batch_size is not None:
        state["stage_batch_size"] = stage_batch_size
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    print(f"$ {printable}", flush=True)
    if not dry_run:
        try:
            subprocess.run(command, cwd=ROOT, check=True)
        except subprocess.CalledProcessError as error:
            state["status"] = "failed"
            state["return_code"] = int(error.returncode)
            state["failure_reason"] = "command_failed"
            if stage_key is not None:
                state["failed_stage"] = stage_key
            state["updated_at"] = utc_now()
            write_json(state_path, state)
            raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maestro-root", type=Path, default=ROOT / "data/maestro-v3.0.0")
    parser.add_argument("--cache-dir", type=Path, default=ROOT / "cache/maestro-v3.0.0")
    parser.add_argument("--midi-dir", type=Path, default=ROOT / "midi")
    parser.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--run-root", type=Path, default=ROOT / "runs/model-suite-v1.1.0-rc1")
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "artifacts/model-suite-v1.0.0")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--memory-limit-gib", type=float, default=26.0)
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-evaluation", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def stage_plan(registry: dict) -> list[tuple[str, str, dict, tuple[str, str] | None]]:
    models = registry["models"]
    plan: list[tuple[str, str, dict, tuple[str, str] | None]] = []
    for model_id in ("paper_ir", "film_fdn", "calibrated_film_ir"):
        parent = None
        for stage, settings in models[model_id]["training"]["stage_schedule"].items():
            plan.append((model_id, stage, settings, parent))
            parent = (model_id, stage)
    calibrated = models["calibrated_ir"]["training"]["stage_schedule"]["calibrate"]
    paper_refine = ("paper_ir", "refine")
    paper_end = plan.index(next(item for item in plan if item[:2] == paper_refine)) + 1
    plan.insert(paper_end, ("calibrated_ir", "calibrate", calibrated, paper_refine))
    return plan


def checkpoint_path(run_root: Path, model_id: str, stage: str, name: str = "best.pt") -> Path:
    return run_root / "training" / model_id / stage / "checkpoints" / name


def resolve_stage_batch_size(settings: dict, default_batch_size: int) -> int:
    batch_size = int(settings.get("batch_size", default_batch_size))
    if batch_size <= 0:
        raise ValueError("stage batch_size must be positive")
    return batch_size


def benchmark_report_path(
    run_root: Path, model_id: str, stage: str, batch_size: int
) -> Path:
    return run_root / "benchmarks" / f"{model_id}-{stage}-batch-{batch_size}.json"


def benchmark_memory(report: dict, memory_limit_gib: float) -> dict:
    reserved_bytes = int(report["peak_cuda_memory_reserved_bytes"])
    limit_bytes = int(memory_limit_gib * 1024**3)
    return {
        "reserved_bytes": reserved_bytes,
        "reserved_gib": reserved_bytes / 1024**3,
        "limit_gib": float(memory_limit_gib),
        "passed": reserved_bytes <= limit_bytes,
    }


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    registry = load_json(args.registry.resolve())
    release = str(registry["release"])
    if release != "model-suite-v1.1.0-rc1":
        raise ValueError(f"This pipeline expects model-suite-v1.1.0-rc1, got {release}")
    run_root = args.run_root.resolve()
    state_path = run_root / "pipeline-state.json"
    state = (
        load_json(state_path)
        if state_path.is_file()
        else {
            "schema": "ddsp-piano-training-pipeline/v1",
            "release": release,
            "created_at": utc_now(),
            "status": "running",
            "completed_stages": [],
        }
    )

    quality_manifest = args.cache_dir.resolve() / "quality-v1.1.json"
    if not quality_manifest.is_file():
        run(
            [
                args.python,
                str(ROOT / "scripts/build_quality_manifest.py"),
                "--maestro-root",
                str(args.maestro_root.resolve()),
                "--cache-dir",
                str(args.cache_dir.resolve()),
                "--output",
                str(quality_manifest),
                "--seed",
                "20260727",
            ],
            state,
            state_path,
            args.dry_run,
        )
    if args.dry_run and not quality_manifest.is_file():
        manifest_entries = 348657
    else:
        manifest_entries = int(load_json(quality_manifest)["entry_count"])
    validation_every_examples = int(math.ceil(manifest_entries / 4))

    state["batch_size"] = args.batch_size
    state["dataset_segments"] = manifest_entries
    state["validation_every_examples"] = validation_every_examples
    write_json(state_path, state)

    for model_id, stage, settings, parent in stage_plan(registry):
        stage_key = f"{model_id}/{stage}"
        experiment_dir = run_root / "training" / model_id / stage
        complete_path = experiment_dir / "complete.json"
        if complete_path.is_file():
            if stage_key not in state["completed_stages"]:
                state["completed_stages"].append(stage_key)
            continue
        stage_batch_size = resolve_stage_batch_size(settings, args.batch_size)
        state.setdefault("stage_batch_sizes", {})[stage_key] = stage_batch_size
        if not args.skip_benchmark:
            report = benchmark_report_path(run_root, model_id, stage, stage_batch_size)
            if not report.is_file():
                run(
                    [
                        args.python,
                        str(ROOT / "scripts/benchmark_training.py"),
                        "--registry",
                        str(args.registry.resolve()),
                        "--model-id",
                        model_id,
                        "--stage",
                        stage,
                        "--batch-size",
                        str(stage_batch_size),
                        "--warmup-steps",
                        "3",
                        "--timed-steps",
                        "8",
                        "--synthesis-layout",
                        "vectorized",
                        "--optimizer-implementation",
                        "fused",
                        "--spectral-layout",
                        "combined",
                        "--velocity-counterfactual-layout",
                        "combined",
                        "--output",
                        str(report),
                    ],
                    state,
                    state_path,
                    args.dry_run,
                    operation="benchmark",
                    stage_key=stage_key,
                    stage_batch_size=stage_batch_size,
                )
            if report.is_file():
                memory = benchmark_memory(load_json(report), args.memory_limit_gib)
                memory["report"] = str(report)
                memory["batch_size"] = stage_batch_size
                state.setdefault("stage_benchmarks", {})[stage_key] = memory
                write_json(state_path, state)
                if not memory["passed"]:
                    state["status"] = "failed"
                    state["failed_stage"] = stage_key
                    state["failure_reason"] = "memory_preflight_exceeded"
                    state["updated_at"] = utc_now()
                    write_json(state_path, state)
                    raise RuntimeError(
                        f"{stage_key} reserves {memory['reserved_gib']:.2f} GiB, "
                        f"above the {args.memory_limit_gib:.2f} GiB limit"
                    )
        command = [
            args.python,
            str(ROOT / "train.py"),
            "--registry",
            str(args.registry.resolve()),
            "--model-id",
            model_id,
            "--stage",
            stage,
            "--maestro-root",
            str(args.maestro_root.resolve()),
            "--cache-dir",
            str(args.cache_dir.resolve()),
            "--quality-manifest",
            str(quality_manifest),
            "--experiment-dir",
            str(experiment_dir),
            "--epochs",
            str(int(settings["epochs"])),
            "--steps-per-epoch",
            "0",
            "--batch-size",
            str(stage_batch_size),
            "--lr",
            str(float(settings["learning_rate"])),
            "--sampling-mode",
            "coverage",
            "--curriculum-tail-fraction",
            "0.2",
            "--balanced-validation",
            "--validation-batches",
            "0",
            "--validation-every-examples",
            str(validation_every_examples),
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
            "8",
            "--validation-workers",
            "2",
            "--log-every",
            "100",
        ]
        last_checkpoint = checkpoint_path(run_root, model_id, stage, "last.pt")
        if last_checkpoint.is_file():
            command.extend(("--resume", str(last_checkpoint)))
        elif parent is not None:
            parent_checkpoint = checkpoint_path(run_root, parent[0], parent[1])
            if not parent_checkpoint.is_file() and not args.dry_run:
                raise FileNotFoundError(f"parent checkpoint is missing: {parent_checkpoint}")
            command.extend(("--finetune-from", str(parent_checkpoint)))
        run(
            command,
            state,
            state_path,
            args.dry_run,
            operation="train",
            stage_key=stage_key,
            stage_batch_size=stage_batch_size,
        )
        best_checkpoint = checkpoint_path(run_root, model_id, stage)
        if not best_checkpoint.is_file() and not args.dry_run:
            raise FileNotFoundError(f"training completed without best checkpoint: {best_checkpoint}")
        if not args.dry_run:
            write_json(
                complete_path,
                {
                    "schema": "ddsp-piano-training-stage-completion/v1",
                    "model_id": model_id,
                    "stage": stage,
                    "batch_size": stage_batch_size,
                    "best_checkpoint": str(best_checkpoint),
                    "completed_at": utc_now(),
                },
            )
        if stage_key not in state["completed_stages"]:
            state["completed_stages"].append(stage_key)
        state["active_stage"] = None
        state["active_operation"] = None
        state.pop("stage_batch_size", None)
        write_json(state_path, state)

    sources = {
        "paper_ir": checkpoint_path(run_root, "paper_ir", "refine"),
        "film_fdn": checkpoint_path(run_root, "film_fdn", "refine"),
        "calibrated_ir": checkpoint_path(run_root, "calibrated_ir", "calibrate"),
        "calibrated_film_ir": checkpoint_path(run_root, "calibrated_film_ir", "refine"),
    }
    candidate_dir = ROOT / "artifacts" / release
    release_manifest = candidate_dir / "model-suite.json"
    if not release_manifest.is_file():
        command = [
            args.python,
            str(ROOT / "scripts/prepare_release.py"),
            "--registry",
            str(args.registry.resolve()),
            "--output-dir",
            str(candidate_dir),
        ]
        for model_id, checkpoint in sources.items():
            command.extend(("--source", f"{model_id}={checkpoint}"))
        run(command, state, state_path, args.dry_run)

    if not args.skip_evaluation:
        render_index_root = run_root / "listening" / "all-models-all-timbres"
        if not any(render_index_root.glob("midi-*/index.json")):
            run(
                [
                    args.python,
                    str(ROOT / "scripts/render_all_onnx_models.py"),
                    "--model-dir",
                    str(candidate_dir),
                    "--midi-dir",
                    str(args.midi_dir.resolve()),
                    "--output-root",
                    str(render_index_root),
                    "--all-piano-models",
                ],
                state,
                state_path,
                args.dry_run,
            )
        corpus = run_root / "evaluation" / "release-corpus.json"
        if not corpus.is_file():
            run(
                [
                    args.python,
                    str(ROOT / "scripts/evaluate_model.py"),
                    "prepare",
                    "--profile",
                    "release",
                    "--maestro-root",
                    str(args.maestro_root.resolve()),
                    "--cache-dir",
                    str(args.cache_dir.resolve()),
                    "--output",
                    str(corpus),
                ],
                state,
                state_path,
                args.dry_run,
            )
        for model_id in sources:
            report_dir = run_root / "evaluation" / model_id
            if (report_dir / "report.json").is_file():
                continue
            run(
                [
                    args.python,
                    str(ROOT / "scripts/evaluate_model.py"),
                    "run",
                    "--profile",
                    "release",
                    "--baseline-id",
                    model_id,
                    "--candidate-id",
                    model_id,
                    "--baseline-artifacts-dir",
                    str(args.baseline_dir.resolve()),
                    "--candidate-artifacts-dir",
                    str(candidate_dir),
                    "--corpus",
                    str(corpus),
                    "--midi-dir",
                    str(args.midi_dir.resolve()),
                    "--output-dir",
                    str(report_dir),
                    "--prepare-listening-only",
                ],
                state,
                state_path,
                args.dry_run,
            )

    state["status"] = "dry_run_complete" if args.dry_run else "human_review_pending"
    state["candidate_artifacts"] = str(candidate_dir)
    state["updated_at"] = utc_now()
    write_json(state_path, state)
    print(json.dumps(state, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
