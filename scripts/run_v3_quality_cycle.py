#!/usr/bin/env python3
"""Run the resumable v3 factorized-decoder quality cycle."""

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

from ddsp_piano.evaluation import canonical_sha256, read_json, sha256_file, write_json


DEFAULT_CONFIG = ROOT / "configs" / "v3_quality_cycle.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def resolve_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def checkpoint_step(path: Path) -> int:
    if not path.is_file():
        return 0
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    return int(checkpoint.get("global_step", 0))


def metric_p95(report: dict, side: str, metric: str) -> float:
    return float(report["summary"][side][metric]["p95"])


def report_ratios(report: dict) -> dict[str, float]:
    baseline = report["summary"]["baseline"]
    candidate = report["summary"]["candidate"]

    def ratio(metric: str) -> float:
        return float(candidate[metric]["p95"]) / max(
            float(baseline[metric]["p95"]), 1e-9
        )

    return {
        "composite": float(report["comparison"]["composite_ratio"]["median"]),
        "mrstft": float(
            report["comparison"]["metric_ratios"]["mrstft"]["median"]
        ),
        "loudness_p95": ratio("loudness_error_lu"),
        "centroid_p95": ratio("spectral_centroid_error"),
        "tail_p95": ratio("tail_decay_error_db_per_second"),
        "maximum_year": max(
            (
                float(values["median"])
                for name, values in report["comparison"]["groups"].items()
                if name.startswith("year:")
            ),
            default=0.0,
        ),
    }


def screen_gate(report: dict, thresholds: dict) -> tuple[bool, list[str]]:
    ratios = report_ratios(report)
    failures = list(report["verdict"].get("hard_failures", []))
    if not report["verdict"]["hard_gates_passed"] and not failures:
        failures.append("hard_gates")
    limits = {
        "composite": "composite_max",
        "mrstft": "mrstft_max",
        "maximum_year": "year_max",
        "loudness_p95": "loudness_p95_ratio_max",
        "centroid_p95": "centroid_p95_ratio_max",
    }
    for name, threshold_name in limits.items():
        if ratios[name] > float(thresholds[threshold_name]):
            failures.append(f"{name}={ratios[name]:.6f}")
    return not failures, failures


def severe_regression(report: dict, thresholds: dict) -> bool:
    ratios = report_ratios(report)
    return (
        ratios["composite"] > float(thresholds["composite_max"])
        or ratios["maximum_year"] > float(thresholds["year_max"])
    )


def final_gate(
    base_report: dict,
    v1_report: dict,
    thresholds: dict,
) -> tuple[bool, list[str]]:
    ratios = report_ratios(base_report)
    failures = list(base_report["verdict"].get("hard_failures", []))
    if not base_report["verdict"]["hard_gates_passed"] and not failures:
        failures.append("base_hard_gates")
    checks = {
        "composite": "base_composite_max",
        "loudness_p95": "loudness_p95_ratio_max",
        "centroid_p95": "centroid_p95_ratio_max",
        "tail_p95": "tail_p95_ratio_max",
    }
    for name, threshold_name in checks.items():
        if ratios[name] > float(thresholds[threshold_name]):
            failures.append(f"{name}={ratios[name]:.6f}")
    if not v1_report["verdict"]["hard_gates_passed"]:
        failures.extend(v1_report["verdict"].get("hard_failures", ["v1_hard_gates"]))
    v1_composite = float(v1_report["comparison"]["composite_ratio"]["median"])
    if v1_composite > float(thresholds["v1_composite_max"]):
        failures.append(f"v1_composite={v1_composite:.6f}")
    return not failures, failures


class V3QualityCycle:
    def __init__(self, config_path: Path, args: argparse.Namespace) -> None:
        self.config_path = config_path.resolve()
        self.config = read_json(self.config_path)
        if self.config.get("schema") != "ddsp-piano-v3-quality-cycle/v1":
            raise ValueError("Unsupported v3 quality-cycle schema")
        self.device = args.device
        self.dry_run = args.dry_run
        self.python = Path(sys.executable).resolve()
        self.cycle_id = self.config["cycle_id"]
        self.run_root = resolve_path(self.config["run_root"]) / self.cycle_id
        self.export_root = resolve_path(self.config["export_root"]) / self.cycle_id
        self.evaluation_root = resolve_path(self.config["evaluation_root"]) / self.cycle_id
        self.listening_root = resolve_path(self.config["listening_root"]) / self.cycle_id
        self.logs_root = self.run_root / "logs"
        self.state_path = self.run_root / "state.json"
        self.maestro_root = resolve_path(args.maestro_root or self.config["maestro_root"])
        self.cache_dir = resolve_path(args.cache_dir or self.config["cache_dir"])
        self.midi_dir = resolve_path(args.midi_dir or self.config["midi_dir"])
        self.parent_checkpoint = resolve_path(self.config["parent_checkpoint"])
        self.parent_onnx = resolve_path(self.config["parent_onnx"])
        self.v1_onnx = resolve_path(self.config["v1_onnx"])
        self.evaluation_config = resolve_path(self.config["evaluation_config"])
        self.state = self._load_state()

    def _fingerprint(self) -> str:
        inputs = {
            "config": self.config,
            "parent_checkpoint": (
                sha256_file(self.parent_checkpoint)
                if self.parent_checkpoint.is_file()
                else "missing"
            ),
            "parent_onnx": (
                sha256_file(self.parent_onnx) if self.parent_onnx.is_file() else "missing"
            ),
            "v1_onnx": sha256_file(self.v1_onnx) if self.v1_onnx.is_file() else "missing",
        }
        return canonical_sha256(inputs)

    def _load_state(self) -> dict:
        fingerprint = self._fingerprint()
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if state.get("fingerprint") != fingerprint:
                raise RuntimeError("Existing v3 cycle state does not match its inputs")
            return state
        return {
            "schema": "ddsp-piano-v3-quality-cycle-state/v1",
            "cycle_id": self.cycle_id,
            "fingerprint": fingerprint,
            "status": "created",
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "candidates": {
                candidate["id"]: {"config": candidate, "records": []}
                for candidate in self.config["candidates"]
            },
            "events": [],
        }

    def save(self, event: str, **details: object) -> None:
        self.state["updated_at"] = utc_now()
        self.state["events"].append(
            {"at": self.state["updated_at"], "event": event, **details}
        )
        if not self.dry_run:
            write_json(self.state_path, self.state)

    def command(self, command: list[str], label: str) -> None:
        printable = " ".join(command)
        print(f"$ {printable}", flush=True)
        if self.dry_run:
            return
        self.logs_root.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_root / f"{label}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] $ {printable}\n")
            subprocess.run(
                command,
                cwd=ROOT,
                stdout=log,
                stderr=subprocess.STDOUT,
                check=True,
            )

    def prepare(self) -> None:
        required = [
            self.parent_checkpoint,
            self.parent_onnx,
            self.parent_onnx.with_suffix(".json"),
            self.v1_onnx,
            self.v1_onnx.with_suffix(".json"),
            self.evaluation_config,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing and not self.dry_run:
            raise FileNotFoundError("Missing v3 cycle input(s): " + ", ".join(missing))
        for candidate in self.config["candidates"]:
            report = self.run_root / "preflight" / f"{candidate['id']}.json"
            if report.is_file():
                if not read_json(report).get("passed"):
                    raise RuntimeError(f"Failed v3 initialization preflight: {report}")
                continue
            self.command(
                [
                    str(self.python),
                    "scripts/verify_v3_initialization.py",
                    "--parent-checkpoint", str(self.parent_checkpoint),
                    "--conditioning-gate", candidate["conditioning_gate"],
                    "--output", str(report),
                    "--steps", "100",
                    "--atol", "0.00001",
                ],
                f"{candidate['id']}_preflight",
            )
            self.save("initialization_verified", candidate=candidate["id"])
        for profile in ("dev", "release"):
            if profile in self.state.setdefault("prepared_profiles", []):
                continue
            command = [
                str(self.python),
                "scripts/evaluate_model.py",
                "--config",
                str(self.evaluation_config),
                "prepare",
                "--profile",
                profile,
                "--maestro-root",
                str(self.maestro_root),
                "--cache-dir",
                str(self.cache_dir),
            ]
            if profile == "release":
                command.append("--prepare-missing")
            self.command(command, f"prepare_{profile}")
            self.state["prepared_profiles"].append(profile)
            self.save("profile_prepared", profile=profile)

    def train(self, candidate: dict, target_examples: int) -> Path:
        training = self.config["training"]
        batch_size = int(training["batch_size"])
        first_examples = int(self.config["screen_milestones_examples"][0])
        if target_examples % batch_size or first_examples % batch_size:
            raise ValueError("Training milestones must be divisible by batch size")
        steps_per_epoch = first_examples // batch_size
        target_steps = target_examples // batch_size
        if target_steps % steps_per_epoch:
            raise ValueError("Training milestone is not aligned to the first milestone")
        run_dir = self.run_root / "candidates" / candidate["id"]
        last = run_dir / "checkpoints" / "last.pt"
        if checkpoint_step(last) < target_steps:
            command = [
                str(self.python),
                "train.py",
                "--maestro-root", str(self.maestro_root),
                "--cache-dir", str(self.cache_dir),
                "--experiment-dir", str(run_dir),
                "--model-variant", "v3",
                "--decoder-type", "factorized",
                "--conditioning-gate", candidate["conditioning_gate"],
                "--batch-size", str(batch_size),
                "--epochs", str(target_steps // steps_per_epoch),
                "--steps-per-epoch", str(steps_per_epoch),
                "--validation-batches", "0",
                "--validation-every-epochs", "1",
                "--balanced-validation",
                "--num-workers", "0",
                "--train-workers", str(training["train_workers"]),
                "--validation-workers", "0",
                "--lr", str(training["learning_rate"]),
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
                "--wet-loss-weight", "0.70",
                "--energy-loss-weight", "0.12",
                "--onset-loss-weight", "0.10",
                "--centroid-loss-weight", "0.03",
                "--tail-loss-weight", "0.0",
                "--velocity-loss-weight", "0.05",
                "--energy-hard-fraction", "0.2",
                "--velocity-loss-every", str(training["velocity_loss_every"]),
                "--velocity-response-ms", "125",
                "--loss-calibration-batches", str(training["loss_calibration_batches"]),
                "--loss-calibration-max-scale", "20",
                "--sampling-mode", "uniform",
                "--trainable-scope", "controls",
                "--n-harmonics", "96",
                "--n-noise-bands", "64",
                "--reverb-type", "ir",
                "--reverb-wet-gain", "1.0",
                "--reverb-regularizer-weight", "0.01",
                "--control-anchor-checkpoint", str(self.parent_checkpoint),
                "--control-anchor-weight", "0.10",
                "--log-every", str(training["log_every"]),
                "--save-every", "1",
                "--seed", str(training["seed"]),
                "--amp",
            ]
            if last.is_file():
                command.extend(["--resume", str(last)])
            else:
                command.extend(["--init-checkpoint", str(self.parent_checkpoint)])
            self.command(command, f"{candidate['id']}_train_{target_examples}")
        epoch = target_steps // steps_per_epoch
        checkpoint = run_dir / "checkpoints" / f"epoch_{epoch:04d}.pt"
        if not checkpoint.is_file() and not self.dry_run:
            raise RuntimeError(f"Missing milestone checkpoint: {checkpoint}")
        return checkpoint

    def export(self, candidate: dict, checkpoint: Path, target_examples: int) -> Path:
        output = self.export_root / candidate["id"] / f"examples_{target_examples}.onnx"
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
                    "--model-variant", "v3",
                    "--verify-steps", "100",
                ],
                f"{candidate['id']}_export_{target_examples}",
            )
        return output

    def evaluate(
        self,
        candidate: dict,
        onnx_path: Path,
        target_examples: int,
        profile: str,
        baseline: Path,
        suffix: str = "",
        listening: bool = False,
    ) -> tuple[dict, Path]:
        report_dir = (
            self.evaluation_root
            / candidate["id"]
            / f"examples_{target_examples}_{profile}{suffix}"
        )
        report_path = report_dir / "report.json"
        if not report_path.is_file():
            command = [
                str(self.python),
                "scripts/evaluate_model.py",
                "--config", str(self.evaluation_config),
                "run",
                "--baseline", str(baseline),
                "--candidate", str(onnx_path),
                "--profile", profile,
                "--output-dir", str(report_dir),
                "--midi-dir", str(self.midi_dir),
            ]
            command.append("--prepare-listening-only" if listening else "--skip-listening")
            self.command(command, f"{candidate['id']}_evaluate_{target_examples}_{profile}{suffix}")
        if self.dry_run:
            report = {
                "verdict": {"hard_gates_passed": True, "hard_failures": []},
                "comparison": {
                    "composite_ratio": {"median": 0.95},
                    "metric_ratios": {"mrstft": {"median": 0.99}},
                    "groups": {"year:2018": {"median": 0.99}},
                },
                "summary": {
                    "baseline": {
                        "loudness_error_lu": {"p95": 10.0},
                        "spectral_centroid_error": {"p95": 0.03},
                        "tail_decay_error_db_per_second": {"p95": 7.5},
                    },
                    "candidate": {
                        "loudness_error_lu": {"p95": 8.5},
                        "spectral_centroid_error": {"p95": 0.025},
                        "tail_decay_error_db_per_second": {"p95": 7.5},
                    },
                },
                "human_review": {"status": "prepared" if listening else "not_generated"},
            }
            return report, report_path
        return read_json(report_path), report_path

    def record(
        self,
        candidate: dict,
        examples: int,
        checkpoint: Path,
        onnx_path: Path,
        report: dict,
        report_path: Path,
        passed: bool,
        failures: list[str],
    ) -> dict:
        existing = next(
            (
                record
                for record in self.state["candidates"][candidate["id"]]["records"]
                if int(record["examples_seen"]) == examples
            ),
            None,
        )
        if existing is not None:
            return existing
        record = {
            "candidate_id": candidate["id"],
            "examples_seen": examples,
            "checkpoint": str(checkpoint.resolve()),
            "onnx": str(onnx_path.resolve()),
            "report": str(report_path.resolve()),
            "ratios": report_ratios(report),
            "objective_passed": passed,
            "failures": failures,
        }
        self.state["candidates"][candidate["id"]]["records"].append(record)
        self.save("milestone_complete", candidate=candidate["id"], examples=examples)
        return record

    def run_milestone(self, candidate: dict, examples: int) -> dict:
        existing = next(
            (
                record
                for record in self.state["candidates"][candidate["id"]]["records"]
                if int(record["examples_seen"]) == examples
            ),
            None,
        )
        if existing is not None:
            return existing
        checkpoint = self.train(candidate, examples)
        onnx_path = self.export(candidate, checkpoint, examples)
        report, report_path = self.evaluate(
            candidate, onnx_path, examples, "dev", self.parent_onnx
        )
        passed, failures = screen_gate(report, self.config["screen_gate"])
        return self.record(
            candidate,
            examples,
            checkpoint,
            onnx_path,
            report,
            report_path,
            passed,
            failures,
        )

    def render_wet_ablation(self, onnx_path: Path) -> None:
        for wet_gain in self.config["listening_wet_gains"]:
            output_dir = self.listening_root / f"wet_{float(wet_gain):.2f}"
            if (output_dir / "manifest.json").is_file():
                continue
            self.command(
                [
                    str(self.python),
                    "scripts/render_onnx.py",
                    "--model", str(onnx_path),
                    "--midi-dir", str(self.midi_dir),
                    "--output-dir", str(output_dir),
                    "--reverb-wet-gain", str(wet_gain),
                    "--warm-up-seconds", "0.5",
                    "--tail-seconds", "2.5",
                    "--chunk-seconds", "4.0",
                    "--seed", str(self.config["training"]["seed"]),
                ],
                f"render_wet_{float(wet_gain):.2f}",
            )

    def summarize(self) -> None:
        summary = {
            "schema": "ddsp-piano-v3-quality-cycle-summary/v1",
            "cycle_id": self.cycle_id,
            "status": self.state["status"],
            "parent_onnx": str(self.parent_onnx.resolve()),
            "official_models_unchanged": True,
            "selected_candidate": self.state.get("selected_candidate"),
            "candidates": self.state["candidates"],
        }
        write_json(self.run_root / "cycle_summary.json", summary)
        rows = [
            "# DDSP-Piano v3 Candidate Quality Cycle",
            "",
            f"- Status: `{summary['status']}`",
            "- Official v1/v2 overwritten: no",
            "",
            "| Candidate | Examples | Composite | Loudness p95 | Centroid p95 | Pass |",
            "| --- | ---: | ---: | ---: | ---: | --- |",
        ]
        for candidate_id, candidate in self.state["candidates"].items():
            for record in candidate["records"]:
                ratios = record["ratios"]
                rows.append(
                    f"| {candidate_id} | {record['examples_seen']} | "
                    f"{ratios['composite']:.4f} | {ratios['loudness_p95']:.4f} | "
                    f"{ratios['centroid_p95']:.4f} | "
                    f"{'yes' if record['objective_passed'] else 'no'} |"
                )
        (self.run_root / "cycle_summary.md").write_text(
            "\n".join(rows) + "\n", encoding="utf-8"
        )

    def run(self) -> None:
        self.prepare()
        candidates = self.config["candidates"]
        self.state["status"] = "screening"
        self.save("screening_started")
        finalists = []
        for candidate in candidates:
            severe_streak = 0
            last_record = None
            for examples in self.config["screen_milestones_examples"]:
                record = self.run_milestone(candidate, int(examples))
                last_record = record
                report = read_json(record["report"]) if not self.dry_run else None
                is_severe = (
                    severe_regression(report, self.config["early_stop_gate"])
                    if report is not None
                    else False
                )
                severe_streak = severe_streak + 1 if is_severe else 0
                if severe_streak >= 2:
                    self.save(
                        "candidate_stopped",
                        candidate=candidate["id"],
                        reason="two_consecutive_severe_regressions",
                    )
                    break
            if last_record and int(last_record["examples_seen"]) == int(
                self.config["screen_milestones_examples"][-1]
            ) and last_record["objective_passed"]:
                finalists.append(last_record)

        if not finalists:
            self.state["status"] = "no_improvement"
            self.save("cycle_complete", reason="no_screen_candidate_passed")
            self.summarize()
            return
        winner_record = min(finalists, key=lambda record: record["ratios"]["composite"])
        winner = next(
            candidate
            for candidate in candidates
            if candidate["id"] == winner_record["candidate_id"]
        )
        self.state["screen_winner"] = winner_record
        self.state["status"] = "full_training"
        self.save("screen_winner_selected", candidate=winner["id"])
        final_record = winner_record
        for examples in self.config["full_milestones_examples"]:
            final_record = self.run_milestone(winner, int(examples))

        final_onnx = Path(final_record["onnx"])
        final_examples = int(final_record["examples_seen"])
        base_report, base_path = self.evaluate(
            winner,
            final_onnx,
            final_examples,
            "release",
            self.parent_onnx,
            suffix="_base",
            listening=True,
        )
        v1_report, v1_path = self.evaluate(
            winner,
            final_onnx,
            final_examples,
            "release",
            self.v1_onnx,
            suffix="_v1",
        )
        passed, failures = final_gate(
            base_report, v1_report, self.config["final_gate"]
        )
        self.render_wet_ablation(final_onnx)
        self.state["selected_candidate"] = {
            **final_record,
            "release_base_report": str(base_path.resolve()),
            "release_v1_report": str(v1_path.resolve()),
            "release_passed": passed,
            "release_failures": failures,
            "human_status": base_report["human_review"]["status"],
        }
        self.state["status"] = "objective_candidate" if passed else "no_improvement"
        self.save("cycle_complete", status=self.state["status"])
        self.summarize()


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
    cycle = V3QualityCycle(args.config, args)
    cycle.run_root.mkdir(parents=True, exist_ok=True)
    lock_path = cycle.run_root / ".lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"v3 quality cycle is already running: {lock_path}") from error
        cycle.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
