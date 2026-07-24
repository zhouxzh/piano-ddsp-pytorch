#!/usr/bin/env python3
"""Run the resumable v2 train/export/evaluate/listen quality cycle."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.evaluation import canonical_sha256, read_json, sha256_file, write_json


DEFAULT_CONFIG = ROOT / "configs" / "v2_quality_cycle.json"


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


def rank_candidates(records: list[dict]) -> list[dict]:
    """Rank valid candidates, preferring gate passes and then lower objective ratios."""
    valid = [record for record in records if record["hard_gates_passed"]]
    return sorted(
        valid,
        key=lambda record: (
            not record["objective_eligible"],
            float(record["composite_median"]),
            record["candidate_id"],
        ),
    )


def review_deadline_expired(report: dict, now: datetime | None = None) -> bool:
    deadline = report.get("human_review", {}).get("deadline")
    if not deadline:
        return False
    current = now or datetime.now(timezone.utc)
    parsed = datetime.fromisoformat(deadline.replace("Z", "+00:00"))
    return current >= parsed


class QualityCycle:
    def __init__(self, config_path: Path, args: argparse.Namespace) -> None:
        self.config_path = config_path.resolve()
        self.config = read_json(self.config_path)
        if self.config.get("schema") != "ddsp-piano-quality-cycle/v1":
            raise ValueError("Unsupported quality-cycle config schema")
        candidates = self.config.get("candidates", [])
        if not 1 <= len(candidates) <= 4:
            raise ValueError("The quality cycle requires between one and four candidates")
        identifiers = [candidate["id"] for candidate in candidates]
        if len(set(identifiers)) != len(identifiers):
            raise ValueError("Candidate ids must be unique")

        self.device = args.device
        self.amp = self.config["training"]["amp"] if args.amp is None else args.amp
        self.baseline = resolve_path(args.baseline or self.config["baseline_onnx"]).resolve()
        self.maestro_root = resolve_path(args.maestro_root or self.config["maestro_root"]).resolve()
        self.cache_dir = resolve_path(args.cache_dir or self.config["cache_dir"]).resolve()
        self.midi_dir = resolve_path(args.midi_dir or self.config["midi_dir"]).resolve()
        self.evaluation_config_path = resolve_path(self.config["evaluation_config"]).resolve()
        self.evaluation_config = read_json(self.evaluation_config_path)
        if args.review_timeout_minutes is not None:
            if args.review_timeout_minutes < 0:
                raise ValueError("--review-timeout-minutes must be non-negative")
            self.evaluation_config["listening"]["human_review_timeout_minutes"] = (
                args.review_timeout_minutes
            )

        cycle_id = self.config["cycle_id"]
        self.run_root = resolve_path(self.config["run_root"]) / cycle_id
        self.export_root = resolve_path(self.config["export_root"]) / cycle_id
        self.evaluation_root = resolve_path(self.config["evaluation_root"]) / cycle_id
        self.logs_dir = self.run_root / "logs"
        self.state_path = self.run_root / "state.json"
        self.runtime_evaluation_config = self.run_root / "evaluation_config.json"
        self.dry_run = args.dry_run
        self.python = Path(sys.executable).resolve()
        self.state = self._load_or_create_state()

    def _load_or_create_state(self) -> dict:
        fingerprint = canonical_sha256(
            {
                "cycle": self.config,
                "evaluation": self.evaluation_config,
                "baseline_sha256": sha256_file(self.baseline),
                "baseline_metadata_sha256": sha256_file(
                    self.baseline.with_suffix(".json")
                ),
            }
        )
        if self.state_path.is_file():
            state = read_json(self.state_path)
            if state.get("fingerprint") != fingerprint:
                raise RuntimeError(
                    f"Existing cycle state does not match the current config: {self.state_path}"
                )
            return state
        return {
            "schema": "ddsp-piano-quality-cycle-state/v1",
            "cycle_id": self.config["cycle_id"],
            "fingerprint": fingerprint,
            "created_at": utc_now(),
            "updated_at": utc_now(),
            "status": "running",
            "prepared_profiles": [],
            "active_candidates": [item["id"] for item in self.config["candidates"]],
            "candidates": {
                item["id"]: {"config": item, "stages": {}}
                for item in self.config["candidates"]
            },
            "history": [],
        }

    def save(self, event: str | None = None, **details: object) -> None:
        self.state["updated_at"] = utc_now()
        if event:
            self.state["history"].append({"at": utc_now(), "event": event, **details})
        if not self.dry_run:
            write_json(self.state_path, self.state)

    def command(self, command: list[str], log_name: str) -> None:
        printable = " ".join(command)
        print(f"$ {printable}", flush=True)
        if self.dry_run:
            return
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.logs_dir / f"{log_name}.log"
        with log_path.open("a", encoding="utf-8") as log:
            log.write(f"\n[{utc_now()}] $ {printable}\n")
            process = subprocess.Popen(
                command,
                cwd=ROOT,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            assert process.stdout is not None
            for line in process.stdout:
                print(line, end="", flush=True)
                log.write(line)
                log.flush()
            return_code = process.wait()
        if return_code:
            raise subprocess.CalledProcessError(return_code, command)

    def prepare(self, profile: str) -> None:
        if profile in self.state["prepared_profiles"]:
            return
        write_json(self.runtime_evaluation_config, self.evaluation_config)
        training = self.config["training"]
        if "training" not in self.state["prepared_profiles"]:
            self.command(
                [
                    str(self.python),
                    "train.py",
                    "--maestro-root", str(self.maestro_root),
                    "--cache-dir", str(self.cache_dir),
                    "--prepare-only",
                    "--prepare-splits", "train,validation",
                    "--prepare-workers", str(training["prepare_workers"]),
                    "--device", "cpu",
                ],
                "prepare_training_cache",
            )
            self.state["prepared_profiles"].append("training")
            self.save("prepared", profile="training")

        command = [
            str(self.python),
            "scripts/evaluate_model.py",
            "--config", str(self.runtime_evaluation_config),
            "prepare",
            "--profile", profile,
            "--maestro-root", str(self.maestro_root),
            "--cache-dir", str(self.cache_dir),
        ]
        if profile == "release":
            command.append("--prepare-missing")
        self.command(command, f"prepare_{profile}_corpus")
        self.state["prepared_profiles"].append(profile)
        self.save("prepared", profile=profile)

    def train(self, candidate: dict, target_steps: int, stage_name: str) -> Path:
        training = self.config["training"]
        steps_per_epoch = int(training["steps_per_epoch"])
        if target_steps % steps_per_epoch:
            raise ValueError("Stage target_steps must be divisible by steps_per_epoch")
        run_dir = self.run_root / "candidates" / candidate["id"]
        checkpoint = run_dir / "checkpoints" / "last.pt"
        existing_steps = checkpoint_step(checkpoint)
        if existing_steps >= target_steps:
            return checkpoint
        epochs = target_steps // steps_per_epoch
        command = [
            str(self.python),
            "train.py",
            "--maestro-root", str(self.maestro_root),
            "--cache-dir", str(self.cache_dir),
            "--experiment-dir", str(run_dir),
            "--model-variant", "v2",
            "--batch-size", str(training["batch_size"]),
            "--epochs", str(epochs),
            "--steps-per-epoch", str(steps_per_epoch),
            "--validation-batches", str(training["validation_batches"]),
            "--num-workers", str(training["num_workers"]),
            "--lr", str(training["learning_rate"]),
            "--phase", str(training["phase"]),
            "--device", self.device,
            "--grad-clip", str(training["grad_clip"]),
            "--dry-loss-weight", str(training["dry_loss_weight"]),
            "--wet-loss-weight", str(training["wet_loss_weight"]),
            "--reverb-regularizer-weight", str(training["reverb_regularizer_weight"]),
            "--n-harmonics", str(candidate["n_harmonics"]),
            "--n-noise-bands", str(candidate["n_noise_bands"]),
            "--reverb-type", candidate["reverb_type"],
            "--energy-loss-weight", str(candidate["energy_loss_weight"]),
            "--onset-loss-weight", str(candidate["onset_loss_weight"]),
            "--seed", str(training["seed"]),
            "--amp" if self.amp else "--no-amp",
        ]
        if checkpoint.is_file():
            command.extend(["--resume", str(checkpoint)])
        self.command(command, f"{candidate['id']}_{stage_name}_train")
        if not self.dry_run and checkpoint_step(checkpoint) < target_steps:
            raise RuntimeError(f"Training stopped before {target_steps} steps: {checkpoint}")
        return checkpoint

    def export(self, candidate: dict, checkpoint: Path, target_steps: int, stage_name: str) -> Path:
        output = self.export_root / candidate["id"] / f"step_{target_steps}.onnx"
        if output.is_file() and output.with_suffix(".json").is_file():
            metadata = read_json(output.with_suffix(".json"))
            if (
                metadata.get("checkpoint_sha256") == sha256_file(checkpoint)
                and int(metadata.get("onnx_runtime_stateful_steps", 0)) >= 100
            ):
                return output
        self.command(
            [
                str(self.python),
                "scripts/export_onnx.py",
                "--checkpoint", str(checkpoint),
                "--output", str(output),
                "--model-variant", "v2",
                "--verify-steps", "100",
            ],
            f"{candidate['id']}_{stage_name}_export",
        )
        return output

    def evaluate(
        self,
        candidate: dict,
        onnx_path: Path,
        stage: dict,
        listening: bool,
        suffix: str = "",
    ) -> dict:
        profile = stage["profile"]
        target_steps = int(stage["target_steps"])
        report_dir = (
            self.evaluation_root
            / candidate["id"]
            / f"{stage['name']}{suffix}_step_{target_steps}_{profile}"
        )
        report_path = report_dir / "report.json"
        if not report_path.is_file():
            command = [
                str(self.python),
                "scripts/evaluate_model.py",
                "--config", str(self.runtime_evaluation_config),
                "run",
                "--baseline", str(self.baseline),
                "--candidate", str(onnx_path),
                "--profile", profile,
                "--output-dir", str(report_dir),
                "--midi-dir", str(self.midi_dir),
            ]
            if not listening:
                command.append("--skip-listening")
            self.command(command, f"{candidate['id']}_{stage['name']}{suffix}_evaluate")
        if self.dry_run:
            return {
                "verdict": {"hard_gates_passed": True, "objective_eligible": True},
                "comparison": {"composite_ratio": {"median": 1.0}},
                "human_review": {"status": "not_generated"},
            }
        return read_json(report_path)

    def execute_stage(self, candidate: dict, stage: dict, listening: bool = False, suffix: str = "") -> dict:
        stage_key = stage["name"] + suffix
        record = self.state["candidates"][candidate["id"]]["stages"].get(stage_key)
        if record and record.get("complete"):
            return record
        checkpoint = self.train(candidate, int(stage["target_steps"]), stage_key)
        onnx_path = self.export(candidate, checkpoint, int(stage["target_steps"]), stage_key)
        report = self.evaluate(candidate, onnx_path, stage, listening, suffix)
        verdict = report["verdict"]
        record = {
            "candidate_id": candidate["id"],
            "complete": True,
            "target_steps": int(stage["target_steps"]),
            "checkpoint": str(checkpoint.resolve()),
            "checkpoint_steps": 0 if self.dry_run else checkpoint_step(checkpoint),
            "onnx": str(onnx_path.resolve()),
            "report": str(
                (
                    self.evaluation_root
                    / candidate["id"]
                    / f"{stage['name']}{suffix}_step_{stage['target_steps']}_{stage['profile']}"
                    / "report.json"
                ).resolve()
            ),
            "hard_gates_passed": bool(verdict["hard_gates_passed"]),
            "objective_eligible": bool(verdict["objective_eligible"]),
            "composite_median": float(report["comparison"]["composite_ratio"]["median"]),
            "human_status": report["human_review"]["status"],
        }
        self.state["candidates"][candidate["id"]]["stages"][stage_key] = record
        self.save("stage_complete", candidate=candidate["id"], stage=stage_key)
        return record

    def wait_for_review(self, record: dict) -> str:
        report_dir = Path(record["report"]).parent
        report_path = report_dir / "report.json"
        if self.dry_run:
            return "deferred"
        poll_seconds = max(1, int(self.evaluation_config["listening"]["poll_seconds"]))
        while True:
            report = read_json(report_path)
            status = report["human_review"]["status"]
            scores_path = report_dir / "listening" / "listening_scores.json"
            if scores_path.is_file() and status in {"pending", "deferred"}:
                self.command(
                    [
                        str(self.python),
                        "scripts/evaluate_model.py",
                        "--config", str(self.runtime_evaluation_config),
                        "finalize",
                        "--report-dir", str(report_dir),
                        "--scores", str(scores_path),
                    ],
                    f"review_{report['evaluation_id'][:12]}_finalize",
                )
                status = read_json(report_path)["human_review"]["status"]
            if status in {"passed", "failed"}:
                record["human_status"] = status
                self.save("human_review_complete", report=str(report_path), status=status)
                return status
            if status == "deferred" or review_deadline_expired(report):
                if status != "deferred":
                    self.command(
                        [
                            str(self.python),
                            "scripts/evaluate_model.py",
                            "--config", str(self.runtime_evaluation_config),
                            "defer",
                            "--report-dir", str(report_dir),
                        ],
                        f"review_{report['evaluation_id'][:12]}_defer",
                    )
                record["human_status"] = "deferred"
                self.save("human_review_deferred", report=str(report_path))
                return "deferred"
            page = report["human_review"]["page"]
            deadline = report["human_review"]["deadline"]
            print(
                f"Waiting for blind review until {deadline}. Open {page} and place "
                f"listening_scores.json in {scores_path.parent}",
                flush=True,
            )
            time.sleep(poll_seconds)

    def summarize(self) -> None:
        summaries = []
        for candidate_id, candidate in self.state["candidates"].items():
            for stage, record in candidate["stages"].items():
                summaries.append(
                    {
                        "candidate": candidate_id,
                        "stage": stage,
                        "target_steps": record["target_steps"],
                        "objective_eligible": record["objective_eligible"],
                        "composite_median": record["composite_median"],
                        "human_status": record["human_status"],
                        "report": record["report"],
                    }
                )
        summary = {
            "schema": "ddsp-piano-quality-cycle-summary/v1",
            "cycle_id": self.config["cycle_id"],
            "status": self.state["status"],
            "baseline": str(self.baseline),
            "official_v1_unchanged": True,
            "candidates": summaries,
            "deferred_reviews": [
                item["report"] for item in summaries if item["human_status"] == "deferred"
            ],
        }
        write_json(self.run_root / "cycle_summary.json", summary)
        rows = [
            "# DDSP-Piano v2 Quality Cycle",
            "",
            f"- Status: `{summary['status']}`",
            f"- Baseline: `{summary['baseline']}`",
            "- Official v1 overwritten: no",
            "",
            "| Candidate | Stage | Steps | Objective | Composite | Human |",
            "| --- | --- | ---: | --- | ---: | --- |",
        ]
        for item in summaries:
            rows.append(
                f"| {item['candidate']} | {item['stage']} | {item['target_steps']} | "
                f"{'pass' if item['objective_eligible'] else 'fail'} | "
                f"{item['composite_median']:.4f} | {item['human_status']} |"
            )
        (self.run_root / "cycle_summary.md").write_text("\n".join(rows) + "\n", encoding="utf-8")

    def run(self) -> None:
        write_json(self.runtime_evaluation_config, self.evaluation_config)
        candidates = {item["id"]: item for item in self.config["candidates"]}
        stages = self.config["stages"]
        active = list(self.state["active_candidates"])

        for stage in stages[:-1]:
            self.prepare(stage["profile"])
            records = [self.execute_stage(candidates[candidate_id], stage) for candidate_id in active]
            ranked = rank_candidates(records)
            if not ranked:
                raise RuntimeError(f"Every candidate failed hard gates at stage {stage['name']}")
            keep = min(int(stage["keep"]), len(ranked))
            active = [record["candidate_id"] for record in ranked[:keep]]
            self.state["active_candidates"] = active
            self.save("stage_ranked", stage=stage["name"], selected=active)

        final_stage = stages[-1]
        self.prepare(final_stage["profile"])
        semifinal_stage = stages[-2]
        semifinal_records = [
            self.state["candidates"][candidate_id]["stages"][semifinal_stage["name"]]
            for candidate_id in active
        ]
        finalists = [record["candidate_id"] for record in rank_candidates(semifinal_records)]
        if not finalists:
            raise RuntimeError("No candidate passed hard gates for the final stage")

        reviewed = []
        for index, candidate_id in enumerate(finalists[:2]):
            suffix = "" if index == 0 else "_fallback"
            record = self.execute_stage(
                candidates[candidate_id], final_stage, listening=True, suffix=suffix
            )
            reviewed.append(candidate_id)
            if not record["objective_eligible"]:
                continue
            status = self.wait_for_review(record)
            if status == "passed":
                self.state["status"] = "promotion_ready"
                self.state["promotion_candidate"] = candidate_id
                break
            # Failed or deferred human review does not block the next trained candidate.
        else:
            self.state["status"] = "awaiting_deferred_review" if any(
                self.state["candidates"][candidate_id]["stages"][
                    final_stage["name"] + ("" if index == 0 else "_fallback")
                ]["human_status"] == "deferred"
                for index, candidate_id in enumerate(reviewed)
            ) else "no_promotion"
        self.state["reviewed_candidates"] = reviewed
        self.save("cycle_complete", status=self.state["status"])
        self.summarize()


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    result.add_argument("--device", default="cuda")
    result.add_argument("--amp", action=argparse.BooleanOptionalAction, default=None)
    result.add_argument("--baseline", type=Path)
    result.add_argument("--maestro-root", type=Path)
    result.add_argument("--cache-dir", type=Path)
    result.add_argument("--midi-dir", type=Path)
    result.add_argument("--review-timeout-minutes", type=int)
    result.add_argument("--dry-run", action="store_true")
    return result


def main() -> int:
    args = parser().parse_args()
    cycle = QualityCycle(args.config, args)
    cycle.run_root.mkdir(parents=True, exist_ok=True)
    lock_path = cycle.run_root / ".lock"
    with lock_path.open("w", encoding="utf-8") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Quality cycle is already running: {lock_path}") from error
        cycle.save("cycle_started", pid=os.getpid())
        try:
            cycle.run()
        except Exception as error:
            cycle.state["status"] = "interrupted"
            cycle.save("cycle_interrupted", error=f"{type(error).__name__}: {error}")
            cycle.summarize()
            raise
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
