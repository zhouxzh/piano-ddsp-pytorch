#!/usr/bin/env python3
"""Run the staged, resumable v2 Q1 quality fine-tuning cycle."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.evaluation import read_json, sha256_file, write_json


DEFAULT_CONFIG = ROOT / "configs" / "v2_quality_q1.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def checkpoint_step(path: Path) -> int:
    if not path.is_file():
        return 0
    return int(torch.load(path, map_location="cpu", weights_only=False).get("global_step", 0))


def metric_value(report: dict, metric: str, statistic: str) -> float:
    return float(report["summary"]["candidate"][metric][statistic])


def objective_gate(report: dict, thresholds: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    if not report["verdict"]["hard_gates_passed"]:
        failures.extend(report["verdict"].get("hard_failures", ["hard_gates"]))
    composite = float(report["comparison"]["composite_ratio"]["median"])
    if composite > float(thresholds["composite_median"]):
        failures.append(f"composite_median={composite:.6f}")
    group_limit = float(thresholds["group_median"])
    for group, values in report["comparison"]["groups"].items():
        if group.startswith("year:") and float(values["median"]) > group_limit:
            failures.append(f"{group}_median={float(values['median']):.6f}")
    mrstft_ratio = float(report["comparison"]["metric_ratios"]["mrstft"]["median"])
    if mrstft_ratio > float(thresholds["mrstft_median_ratio"]):
        failures.append(f"mrstft_median_ratio={mrstft_ratio:.6f}")
    baseline_latency = float(report["models"]["baseline"]["latency_ms"]["p95"])
    candidate_latency = float(report["models"]["candidate"]["latency_ms"]["p95"])
    latency_ratio = candidate_latency / max(baseline_latency, 1e-9)
    if latency_ratio > float(thresholds["latency_p95_ratio"]):
        failures.append(f"latency_p95_ratio={latency_ratio:.6f}")
    absolute_limits = {
        "loudness_error_lu": "loudness_p95",
        "spectral_centroid_error": "centroid_p95",
        "tail_decay_error_db_per_second": "tail_p95",
    }
    for metric, threshold_name in absolute_limits.items():
        value = metric_value(report, metric, "p95")
        if value > float(thresholds[threshold_name]):
            failures.append(f"{metric}_p95={value:.6f}")
    return not failures, failures


def reverb_delta_gate(report: dict, thresholds: dict) -> tuple[bool, list[str]]:
    failures: list[str] = []
    composite = float(report["comparison"]["composite_ratio"]["median"])
    if composite > float(thresholds["maximum_composite_ratio"]):
        failures.append(f"composite_median={composite:.6f}")
    loudness = metric_value(report, "loudness_error_lu", "p95")
    baseline_loudness = float(report["summary"]["baseline"]["loudness_error_lu"]["p95"])
    if loudness > baseline_loudness + float(thresholds["maximum_loudness_regression_lu"]):
        failures.append(f"loudness_p95_regression={loudness - baseline_loudness:.6f}")
    improvement = float(thresholds["required_tail_or_centroid_improvement"])
    tail_ratio = metric_value(report, "tail_decay_error_db_per_second", "p95") / max(
        float(report["summary"]["baseline"]["tail_decay_error_db_per_second"]["p95"]),
        1e-9,
    )
    centroid_ratio = metric_value(report, "spectral_centroid_error", "p95") / max(
        float(report["summary"]["baseline"]["spectral_centroid_error"]["p95"]),
        1e-9,
    )
    if min(tail_ratio, centroid_ratio) > 1.0 - improvement:
        failures.append(
            f"tail_or_centroid_improvement=tail:{tail_ratio:.6f},centroid:{centroid_ratio:.6f}"
        )
    return not failures, failures


class QualityFineTuneCycle:
    def __init__(self, config_path: Path, args: argparse.Namespace) -> None:
        self.config_path = config_path.resolve()
        self.config = read_json(self.config_path)
        if self.config.get("schema") != "ddsp-piano-quality-finetune/v1":
            raise ValueError("Unsupported quality fine-tune schema")
        self.args = args
        self.python = Path(sys.executable)
        self.device = args.device
        self.dry_run = args.dry_run
        self.cycle_id = self.config["cycle_id"]
        self.run_root = resolve_path(self.config["run_root"]) / self.cycle_id
        self.export_root = resolve_path(self.config["export_root"]) / self.cycle_id
        self.evaluation_root = resolve_path(self.config["evaluation_root"]) / self.cycle_id
        self.listening_root = resolve_path(self.config["listening_root"]) / self.cycle_id
        self.log_root = self.run_root / "logs"
        self.state_path = self.run_root / "state.json"
        self.manifest = resolve_path(self.config["quality_manifest"])
        self.maestro_root = resolve_path(args.maestro_root or self.config["maestro_root"])
        self.cache_dir = resolve_path(args.cache_dir or self.config["cache_dir"])
        self.midi_dir = resolve_path(args.midi_dir or self.config["midi_dir"])
        self.base_checkpoint = resolve_path(self.config["base_checkpoint"])
        self.base_onnx = resolve_path(self.config["base_onnx"])
        self.anchor_onnx = resolve_path(self.config["anchor_onnx"])
        self.evaluation_config = resolve_path(self.config["evaluation_config"])
        if self.state_path.is_file():
            self.state = read_json(self.state_path)
            if self.state.get("cycle_id") != self.cycle_id:
                raise ValueError("Existing Q1 state belongs to a different cycle")
            if self.base_checkpoint.is_file():
                current_base_sha256 = sha256_file(self.base_checkpoint)
                recorded_base_sha256 = self.state.get("base_checkpoint_sha256")
                if recorded_base_sha256 and recorded_base_sha256 != current_base_sha256:
                    raise ValueError("The Q1 base checkpoint changed after the cycle started")
        else:
            self.state = {
                "schema": "ddsp-piano-quality-finetune-state/v1",
                "cycle_id": self.cycle_id,
                "status": "created",
                "base_checkpoint": str(self.base_checkpoint.resolve()),
                "base_checkpoint_sha256": (
                    sha256_file(self.base_checkpoint) if self.base_checkpoint.is_file() else None
                ),
                "records": [],
                "events": [],
            }

    def save(self, event: str, **details) -> None:
        self.state["updated_at"] = utc_now()
        self.state["events"].append({"at": self.state["updated_at"], "event": event, **details})
        write_json(self.state_path, self.state)

    def command(self, command: list[str], label: str) -> None:
        self.log_root.mkdir(parents=True, exist_ok=True)
        if self.dry_run:
            print("DRY RUN:", " ".join(command), flush=True)
            return
        log_path = self.log_root / f"{label}.log"
        with log_path.open("a", encoding="utf-8") as log:
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def prepare(self) -> None:
        required = [self.base_checkpoint, self.base_onnx, self.anchor_onnx, self.evaluation_config]
        missing = [str(path) for path in required if not path.is_file()]
        if missing and not self.dry_run:
            raise FileNotFoundError("Missing Q1 input(s): " + ", ".join(missing))
        if not self.manifest.is_file():
            self.command(
                [
                    str(self.python),
                    "scripts/build_quality_manifest.py",
                    "--maestro-root", str(self.maestro_root),
                    "--cache-dir", str(self.cache_dir),
                    "--output", str(self.manifest),
                    "--seed", str(self.config["training"]["seed"]),
                ],
                "build_quality_manifest",
            )
        current_manifest_sha256 = sha256_file(self.manifest) if self.manifest.is_file() else None
        recorded_manifest_sha256 = self.state.get("quality_manifest_sha256")
        if recorded_manifest_sha256 and recorded_manifest_sha256 != current_manifest_sha256:
            raise ValueError("The Q1 quality manifest changed after the cycle started")
        self.state["quality_manifest"] = str(self.manifest.resolve())
        self.state["quality_manifest_sha256"] = current_manifest_sha256
        prepared_profiles = self.state.setdefault("prepared_profiles", [])
        profiles = sorted(
            {phase["profile"] for phase in self.config["phases"].values()} | {"release"}
        )
        for profile in profiles:
            if profile in prepared_profiles:
                continue
            command = [
                str(self.python),
                "scripts/evaluate_model.py",
                "--config", str(self.evaluation_config),
                "prepare",
                "--profile", profile,
                "--maestro-root", str(self.maestro_root),
                "--cache-dir", str(self.cache_dir),
            ]
            if profile == "release":
                command.append("--prepare-missing")
            self.command(command, f"prepare_{profile}_corpus")
            prepared_profiles.append(profile)
            self.save("evaluation_corpus_prepared", profile=profile)
        self.save("prepared")

    def phase_config(self, name: str) -> dict:
        return dict(self.config["phases"][name])

    def train(
        self,
        run_id: str,
        phase: dict,
        parent_checkpoint: Path,
        target_examples: int,
        sampling_mode: str,
    ) -> Path:
        training = self.config["training"]
        batch_size = int(training["batch_size"])
        if target_examples % batch_size:
            raise ValueError("Target examples must be divisible by batch size")
        target_steps = target_examples // batch_size
        first_milestone_steps = int(phase["milestones_examples"][0]) // batch_size
        if target_steps % first_milestone_steps:
            raise ValueError("Every phase milestone must be divisible by the first milestone")
        epochs = target_steps // first_milestone_steps
        run_dir = self.run_root / "candidates" / run_id
        last = run_dir / "checkpoints" / "last.pt"
        if last.is_file():
            checkpoint = torch.load(last, map_location="cpu", weights_only=False)
            initialization = checkpoint.get("initialization") or {}
            if initialization.get("checkpoint_sha256") != sha256_file(parent_checkpoint):
                raise ValueError(
                    f"Existing {run_id} checkpoint was initialized from a different parent"
                )
        if checkpoint_step(last) < target_steps:
            weights = phase["loss_weights"]
            command = [
                str(self.python),
                "train.py",
                "--maestro-root", str(self.maestro_root),
                "--cache-dir", str(self.cache_dir),
                "--experiment-dir", str(run_dir),
                "--model-variant", "v2",
                "--batch-size", str(batch_size),
                "--epochs", str(epochs),
                "--steps-per-epoch", str(first_milestone_steps),
                "--validation-batches", "0",
                "--validation-every-epochs", "1",
                "--balanced-validation",
                "--num-workers", "0",
                "--train-workers", str(training["train_workers"]),
                "--validation-workers", "0",
                "--lr", str(phase["learning_rate"]),
                "--phase", "1",
                "--device", self.device,
                "--grad-clip", str(training["grad_clip"]),
                "--synthesis-layout", "vectorized",
                "--optimizer-implementation", "fused",
                "--compile-training",
                "--compile-mode", "reduce-overhead",
                "--spectral-layout", "combined",
                "--velocity-counterfactual-layout", "combined",
                "--loss-version", "perceptual_v2",
                "--dry-loss-weight", "0.0",
                "--wet-loss-weight", str(weights["wet"]),
                "--energy-loss-weight", str(weights["energy"]),
                "--onset-loss-weight", str(weights["onset"]),
                "--centroid-loss-weight", str(weights["centroid"]),
                "--tail-loss-weight", str(weights["tail"]),
                "--velocity-loss-weight", str(weights["velocity"]),
                "--energy-hard-fraction", "0.2",
                "--velocity-loss-every", str(training["velocity_loss_every"]),
                "--velocity-response-ms", "125",
                "--loss-calibration-batches", str(training["loss_calibration_batches"]),
                "--quality-manifest", str(self.manifest),
                "--sampling-mode", sampling_mode,
                "--trainable-scope", phase["trainable_scope"],
                "--n-harmonics", "96",
                "--n-noise-bands", "64",
                "--reverb-type", "ir",
                "--reverb-wet-gain", "1.0",
                "--context-type", "legacy",
                "--monophonic-type", "legacy",
                "--inharmonicity-type", "legacy",
                "--reverb-regularizer-weight", "0.01",
                "--log-every", str(training["log_every"]),
                "--save-every", "1",
                "--seed", str(training["seed"]),
                "--amp",
            ]
            if last.is_file():
                command.extend(["--resume", str(last)])
            else:
                command.extend(["--finetune-from", str(parent_checkpoint)])
            self.command(command, f"{run_id}_train_to_{target_examples}")
        epoch = target_steps // first_milestone_steps
        checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        if not checkpoint.is_file() and not self.dry_run:
            raise RuntimeError(f"Missing milestone checkpoint: {checkpoint}")
        return checkpoint

    def export(self, run_id: str, checkpoint: Path, target_examples: int) -> Path:
        output = self.export_root / run_id / f"examples_{target_examples}.onnx"
        metadata_path = output.with_suffix(".json")
        reusable = output.is_file() and metadata_path.is_file()
        if reusable:
            metadata = read_json(metadata_path)
            reusable = (
                metadata.get("checkpoint_sha256") == sha256_file(checkpoint)
                and int(metadata.get("onnx_runtime_stateful_steps", 0)) >= 100
            )
        if not reusable:
            self.command(
                [
                    str(self.python),
                    "scripts/export_onnx.py",
                    "--checkpoint", str(checkpoint),
                    "--output", str(output),
                    "--model-variant", "v2",
                    "--verify-steps", "100",
                ],
                f"{run_id}_export_{target_examples}",
            )
        return output

    def evaluate(
        self,
        run_id: str,
        onnx_path: Path,
        target_examples: int,
        profile: str,
        baseline: Path | None = None,
        listening: bool = False,
        suffix: str = "",
    ) -> tuple[dict, Path]:
        report_dir = self.evaluation_root / run_id / f"examples_{target_examples}_{profile}{suffix}"
        report_path = report_dir / "report.json"
        if not report_path.is_file():
            command = [
                str(self.python),
                "scripts/evaluate_model.py",
                "--config", str(self.evaluation_config),
                "run",
                "--baseline", str(baseline or self.base_onnx),
                "--candidate", str(onnx_path),
                "--profile", profile,
                "--output-dir", str(report_dir),
                "--midi-dir", str(self.midi_dir),
            ]
            if listening:
                command.append("--prepare-listening-only")
            else:
                command.append("--skip-listening")
            self.command(command, f"{run_id}_evaluate_{target_examples}_{profile}{suffix}")
        if self.dry_run:
            return {
                "verdict": {"hard_gates_passed": True, "hard_failures": []},
                "comparison": {
                    "composite_ratio": {"median": 0.95},
                    "groups": {},
                    "metric_ratios": {"mrstft": {"median": 0.99}},
                },
                "summary": {
                    side: {
                        "loudness_error_lu": {"p95": 5.0},
                        "spectral_centroid_error": {"p95": 0.015},
                        "tail_decay_error_db_per_second": {"p95": 7.5},
                    }
                    for side in ("baseline", "candidate")
                },
                "models": {
                    side: {"latency_ms": {"p95": 0.1}}
                    for side in ("baseline", "candidate")
                },
                "human_review": {"status": "prepared"},
            }, report_path
        return read_json(report_path), report_path

    def record(
        self,
        run_id: str,
        phase_name: str,
        examples: int,
        checkpoint: Path,
        onnx_path: Path,
        report: dict,
        report_path: Path,
        passed: bool,
        failures: list[str],
    ) -> dict:
        record = {
            "run_id": run_id,
            "phase": phase_name,
            "examples_seen": examples,
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_sha256": sha256_file(checkpoint) if checkpoint.is_file() else None,
            "onnx": str(onnx_path.resolve()),
            "report": str(report_path.resolve()),
            "composite_median": float(report["comparison"]["composite_ratio"]["median"]),
            "objective_passed": passed,
            "failures": failures,
        }
        self.state["records"].append(record)
        self.save("milestone_complete", run_id=run_id, phase=phase_name, examples=examples)
        return record

    def existing_record(self, run_id: str, examples: int) -> dict | None:
        return next(
            (
                item
                for item in self.state["records"]
                if item["run_id"] == run_id and int(item["examples_seen"]) == examples
            ),
            None,
        )

    def run_milestones(
        self,
        run_id: str,
        phase_name: str,
        parent_checkpoint: Path,
        parent_onnx: Path,
        sampling_mode: str,
    ) -> list[dict]:
        phase = self.phase_config(phase_name)
        records = []
        for examples in phase["milestones_examples"]:
            existing = self.existing_record(run_id, int(examples))
            if existing is not None:
                records.append(existing)
                continue
            checkpoint = self.train(
                run_id, phase, parent_checkpoint, int(examples), sampling_mode
            )
            onnx_path = self.export(run_id, checkpoint, int(examples))
            if phase_name == "reverb":
                parent_report, _ = self.evaluate(
                    run_id,
                    onnx_path,
                    int(examples),
                    phase["profile"],
                    baseline=parent_onnx,
                    suffix="_parent_delta",
                )
                delta_passed, failures = reverb_delta_gate(
                    parent_report, self.config["reverb_delta_gate"]
                )
                report, report_path = self.evaluate(
                    run_id,
                    onnx_path,
                    int(examples),
                    phase["profile"],
                    baseline=self.base_onnx,
                    suffix="_base",
                )
                passed = delta_passed and bool(report["verdict"]["hard_gates_passed"])
                if not report["verdict"]["hard_gates_passed"]:
                    failures.extend(report["verdict"].get("hard_failures", ["hard_gates"]))
            else:
                report, report_path = self.evaluate(
                    run_id,
                    onnx_path,
                    int(examples),
                    phase["profile"],
                    baseline=self.base_onnx,
                )
                passed, failures = objective_gate(report, self.config["objective_gate"])
            records.append(
                self.record(
                    run_id,
                    phase_name,
                    int(examples),
                    checkpoint,
                    onnx_path,
                    report,
                    report_path,
                    passed,
                    failures,
                )
            )
        return records

    @staticmethod
    def best(records: list[dict], require_pass: bool = True) -> dict | None:
        choices = [record for record in records if record["objective_passed"] or not require_pass]
        return min(choices, key=lambda record: record["composite_median"], default=None)

    def render_final(self, onnx_path: Path) -> None:
        report_path = self.listening_root / "manifest.json"
        if report_path.is_file():
            return
        self.command(
            [
                str(self.python),
                "scripts/render_onnx.py",
                "--model", str(onnx_path),
                "--midi-dir", str(self.midi_dir),
                "--output-dir", str(self.listening_root),
                "--warm-up-seconds", "0.5",
                "--tail-seconds", "2.5",
                "--chunk-seconds", "4.0",
                "--seed", str(self.config["training"]["seed"]),
            ],
            "render_all_midi",
        )

    def run(self) -> None:
        self.prepare()
        self.state["status"] = "training_controls"
        self.save("controls_started")
        screen_records = []
        for candidate in self.config["screen_candidates"]:
            screen_records.extend(
                self.run_milestones(
                    candidate["id"],
                    "controls",
                    self.base_checkpoint,
                    self.base_onnx,
                    candidate["sampling_mode"],
                )
            )
        controls_winner = self.best(screen_records)
        if controls_winner is None:
            self.state["status"] = "no_improvement"
            self.save("cycle_complete", reason="no_controls_candidate_passed")
            return

        parent_checkpoint = Path(controls_winner["checkpoint"])
        parent_onnx = Path(controls_winner["onnx"])
        self.state["controls_winner"] = controls_winner
        self.state["status"] = "training_reverb"
        self.save("controls_selected", run_id=controls_winner["run_id"])
        reverb_records = self.run_milestones(
            "q1_reverb",
            "reverb",
            parent_checkpoint,
            parent_onnx,
            "curriculum",
        )
        reverb_winner = self.best(reverb_records)
        if reverb_winner is not None:
            parent_checkpoint = Path(reverb_winner["checkpoint"])
            parent_onnx = Path(reverb_winner["onnx"])
            self.state["reverb_winner"] = reverb_winner
        else:
            self.state["reverb_rollback"] = controls_winner
        self.state["status"] = "training_joint"
        self.save("reverb_complete", accepted=reverb_winner is not None)

        joint_records = self.run_milestones(
            "q1_joint",
            "joint",
            parent_checkpoint,
            parent_onnx,
            "curriculum",
        )
        eligible = [controls_winner]
        if reverb_winner is not None:
            eligible.append(reverb_winner)
        eligible.extend(record for record in joint_records if record["objective_passed"])
        best_local = min(eligible, key=lambda record: record["composite_median"])
        best_onnx = Path(best_local["onnx"])

        release_report, release_path = self.evaluate(
            best_local["run_id"],
            best_onnx,
            int(best_local["examples_seen"]),
            "release",
            baseline=self.base_onnx,
            listening=True,
            suffix="_final",
        )
        release_passed, release_failures = objective_gate(
            release_report, self.config["objective_gate"]
        )
        anchor_report, anchor_path = self.evaluate(
            best_local["run_id"],
            best_onnx,
            int(best_local["examples_seen"]),
            "release",
            baseline=self.anchor_onnx,
            suffix="_v1_anchor",
        )
        anchor_limit = float(self.config["anchor_gate"]["composite_median"])
        anchor_composite = float(anchor_report["comparison"]["composite_ratio"]["median"])
        anchor_passed = (
            anchor_report["verdict"]["hard_gates_passed"]
            and anchor_composite <= anchor_limit
        )
        self.render_final(best_onnx)
        self.state["selected_candidate"] = {
            **best_local,
            "release_report": str(release_path.resolve()),
            "release_passed": release_passed,
            "release_failures": release_failures,
            "v1_anchor_report": str(anchor_path.resolve()),
            "v1_anchor_composite": anchor_composite,
            "v1_anchor_passed": anchor_passed,
            "human_status": release_report["human_review"]["status"],
        }
        self.state["status"] = (
            "objective_candidate" if release_passed and anchor_passed else "no_improvement"
        )
        self.save("cycle_complete", status=self.state["status"])


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--device", default="cuda")
    result.add_argument("--maestro-root", type=Path)
    result.add_argument("--cache-dir", type=Path)
    result.add_argument("--midi-dir", type=Path)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    cycle = QualityFineTuneCycle(args.config, args)
    cycle.run_root.mkdir(parents=True, exist_ok=True)
    lock_path = cycle.run_root / ".lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Q1 fine-tune cycle is already running: {lock_path}") from error
        try:
            cycle.run()
        except Exception as error:
            cycle.state["status"] = "interrupted"
            cycle.save("cycle_interrupted", error=f"{type(error).__name__}: {error}")
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
