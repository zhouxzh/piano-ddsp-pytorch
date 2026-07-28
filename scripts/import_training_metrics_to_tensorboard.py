#!/usr/bin/env python3
"""Import historical JSONL metrics and pipeline logs into TensorBoard."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter


COMMAND_RE = re.compile(
    r"^\$ .*?train\.py .*?--model-id (?P<model>\S+) .*?--stage (?P<stage>\S+)"
)
TRAIN_RE = re.compile(
    r"epoch=(?P<epoch>\d+)/(?:\d+)\s+step=(?P<step>\d+)\s+"
    r"loss=(?P<loss>[-+0-9.eE]+)\s+wet=(?P<spectral>[-+0-9.eE]+)\s+"
    r"dry=(?P<dry>[-+0-9.eE]+)\s+reverb=(?P<reverb>[-+0-9.eE]+)\s+"
    r"energy=(?P<energy>[-+0-9.eE]+)\s+onset=(?P<onset>[-+0-9.eE]+)\s+"
    r"centroid=(?P<centroid>[-+0-9.eE]+)\s+tail=(?P<tail>[-+0-9.eE]+)\s+"
    r"velocity=(?P<velocity>[-+0-9.eE]+)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        required=True,
        help="Previous run root containing training/<model>/<stage>/metrics.jsonl",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="TensorBoard logdir to create; existing event files are rejected",
    )
    parser.add_argument(
        "--pipeline-log",
        type=Path,
        help="Optional pipeline.log containing per-step training loss lines",
    )
    return parser.parse_args()


def _stage_from_metrics(path: Path, source_root: Path) -> tuple[str, str]:
    relative = path.relative_to(source_root / "training")
    if len(relative.parts) != 3 or relative.parts[-1] != "metrics.jsonl":
        raise ValueError(f"expected training/<model>/<stage>/metrics.jsonl, got {path}")
    return relative.parts[0], relative.parts[1]


def _reject_existing_events(output_root: Path) -> None:
    if any(output_root.rglob("events.out.tfevents.*")):
        raise FileExistsError(
            f"TensorBoard event files already exist under {output_root}; "
            "choose a new --output-root to avoid duplicate history"
        )


def import_metrics(source_root: Path, output_root: Path) -> tuple[dict[tuple[str, str], SummaryWriter], int]:
    writers: dict[tuple[str, str], SummaryWriter] = {}
    imported = 0
    metrics_paths = sorted((source_root / "training").glob("*/*/metrics.jsonl"))
    for metrics_path in metrics_paths:
        model_id, stage = _stage_from_metrics(metrics_path, source_root)
        writer = writers.setdefault(
            (model_id, stage),
            SummaryWriter(
                log_dir=str(output_root / "training" / model_id / stage),
                flush_secs=1,
            ),
        )
        writer.add_text("history/source", str(metrics_path), global_step=0)
        for line_number, line in enumerate(metrics_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            payload = json.loads(line)
            step = int(payload["global_step"])
            epoch = payload.get("epoch")
            if epoch is not None:
                writer.add_scalar("train/epoch", float(epoch), step)
            validation_loss = payload.get("validation_loss")
            if validation_loss is not None:
                writer.add_scalar("validation/loss", float(validation_loss), step)
            learning_rate = payload.get("learning_rate")
            if learning_rate is not None:
                writer.add_scalar("train/learning_rate", float(learning_rate), step)
            examples_seen = payload.get("examples_seen")
            if examples_seen is not None:
                writer.add_scalar("train/examples_seen", float(examples_seen), step)
            performance = payload.get("train_performance", {})
            if performance.get("peak_cuda_memory_bytes") is not None:
                writer.add_scalar(
                    "system/peak_cuda_memory_gib",
                    float(performance["peak_cuda_memory_bytes"]) / 2**30,
                    step,
                )
            seconds = float(performance.get("seconds", 0.0))
            if seconds > 0 and performance.get("examples") is not None:
                writer.add_scalar(
                    "system/examples_per_second",
                    float(performance["examples"]) / seconds,
                    step,
                )
            imported += 1
            if payload.get("event") != "validation":
                writer.add_text(
                    "history/unknown_event",
                    f"{metrics_path}:{line_number}: {payload.get('event')}",
                    global_step=step,
                )
    return writers, imported


def import_pipeline_log(
    pipeline_log: Path,
    output_root: Path,
    writers: dict[tuple[str, str], SummaryWriter],
) -> int:
    current_stage: tuple[str, str] | None = None
    imported = 0
    for line in pipeline_log.read_text(encoding="utf-8", errors="replace").splitlines():
        command = COMMAND_RE.search(line)
        if command:
            current_stage = (command.group("model"), command.group("stage"))
            writer = writers.get(current_stage)
            if writer is None:
                writer = SummaryWriter(
                    log_dir=str(output_root / "training" / current_stage[0] / current_stage[1]),
                    flush_secs=1,
                )
                writers[current_stage] = writer
            writer.add_text("history/source", str(pipeline_log), global_step=0)
            continue
        match = TRAIN_RE.search(line)
        if not match or current_stage is None:
            continue
        step = int(match.group("step"))
        writer = writers[current_stage]
        for name in (
            "loss",
            "spectral",
            "dry",
            "reverb",
            "energy",
            "onset",
            "centroid",
            "tail",
            "velocity",
        ):
            writer.add_scalar(
                f"train/{name}",
                float(match.group(name)),
                step,
            )
        writer.add_scalar("train/epoch", float(match.group("epoch")), step)
        imported += 1
    return imported


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    output_root = args.output_root.resolve()
    if not (source_root / "training").is_dir():
        raise FileNotFoundError(f"source training directory is missing: {source_root / 'training'}")
    _reject_existing_events(output_root)
    writers, metric_records = import_metrics(source_root, output_root)
    log_records = 0
    if args.pipeline_log is not None:
        log_records = import_pipeline_log(args.pipeline_log.resolve(), output_root, writers)
    for writer in writers.values():
        writer.flush()
        writer.close()
    print(
        json.dumps(
            {
                "source_root": str(source_root),
                "output_root": str(output_root),
                "metric_records": metric_records,
                "pipeline_log_records": log_records,
                "runs": [f"{model}/{stage}" for model, stage in sorted(writers)],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
