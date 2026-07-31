#!/usr/bin/env python3
"""Export and rank intermediate training checkpoints against release baselines."""

from __future__ import annotations

import argparse
import json
import math
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.model_registry import load_model_registry


DEFAULT_REGISTRY = ROOT / "ddsp_piano" / "model-suite-v1.1.0-rc1.json"
DEFAULT_RUN_ROOT = ROOT / "runs" / "model-suite-v1.1.0-tb"
DEFAULT_BASELINE_DIR = ROOT / "artifacts" / "model-suite-v1.0.1"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def run(command: list[str]) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def candidate_stages(run_root: Path, model_ids: set[str]) -> list[tuple[str, str, Path]]:
    candidates = []
    training_root = run_root / "training"
    for checkpoint in sorted(training_root.glob("*/*/checkpoints/best.pt")):
        stage = checkpoint.parents[1].name
        model_id = checkpoint.parents[2].name
        if not model_ids or model_id in model_ids:
            candidates.append((model_id, stage, checkpoint.resolve()))
    return candidates


def report_row(model_id: str, stage: str, checkpoint: Path, report: dict) -> dict[str, Any]:
    comparison = report["comparison"]
    groups = [float(group["median"]) for group in comparison["groups"].values()]
    gate = float(report["config"]["gates"]["group_median"])
    baseline_latency = float(report["models"]["baseline"]["latency_ms"]["p95"])
    candidate_latency = float(report["models"]["candidate"]["latency_ms"]["p95"])
    return {
        "model_id": model_id,
        "stage": stage,
        "checkpoint": str(checkpoint),
        "composite_median": float(comparison["composite_ratio"]["median"]),
        "composite_p95": float(comparison["composite_ratio"]["p95"]),
        "worst_group_median": max(groups),
        "regressed_groups": sum(value > gate for value in groups),
        "metric_ratio_medians": {
            name: float(values["median"])
            for name, values in comparison["metric_ratios"].items()
        },
        "wet_dry_rms_ratio": {
            "baseline": float(report["summary"]["baseline"]["wet_dry_rms_ratio"]["median"]),
            "candidate": float(report["summary"]["candidate"]["wet_dry_rms_ratio"]["median"]),
        },
        "unexpected_operators": list(report["models"]["candidate"]["unexpected_operators"]),
        "numerical_allclose": bool(report["models"]["candidate"]["numerical_allclose"]),
        "latency_p95_ms": candidate_latency,
        "latency_p95_ratio": candidate_latency / max(baseline_latency, 1e-9),
        "objective_eligible": bool(report["verdict"]["objective_eligible"]),
        "objective_failures": list(report["verdict"]["objective_failures"]),
        "hard_failures": list(report["verdict"]["hard_failures"]),
    }


def promotion_decisions(
    rows: list[dict[str, Any]],
    *,
    max_composite_median: float = 0.98,
    max_wet_dry_factor: float = 1.25,
    max_latency_ratio: float = 1.05,
    max_regressed_groups: int = 0,
    max_worst_group_median: float = 1.05,
) -> dict[str, dict[str, Any]]:
    """Select a stage only when it beats the stable model without regressions."""
    decisions = {}
    for model_id in sorted({row["model_id"] for row in rows}):
        ranked = sorted(
            (row for row in rows if row["model_id"] == model_id),
            key=lambda row: (row["composite_median"], row["stage"]),
        )
        accepted = []
        rejected = []
        for row in ranked:
            baseline_wet_dry = max(row["wet_dry_rms_ratio"]["baseline"], 1e-9)
            wet_dry_factor = row["wet_dry_rms_ratio"]["candidate"] / baseline_wet_dry
            reasons = []
            if row["composite_median"] > max_composite_median:
                reasons.append("composite_not_improved")
            if row["regressed_groups"] > max_regressed_groups:
                reasons.append("group_regression")
            if row.get("worst_group_median", 1.0) > max_worst_group_median:
                reasons.append("worst_group_regression")
            if not 1.0 / max_wet_dry_factor <= wet_dry_factor <= max_wet_dry_factor:
                reasons.append("wet_dry_drift")
            if row["latency_p95_ratio"] > max_latency_ratio:
                reasons.append("latency_regression")
            if not row["numerical_allclose"]:
                reasons.append("onnx_numerical_mismatch")
            if row["hard_failures"]:
                reasons.append("deployment_hard_failure")
            diagnostic = {
                "stage": row["stage"],
                "checkpoint": row["checkpoint"],
                "composite_median": row["composite_median"],
                "wet_dry_factor": wet_dry_factor,
                "reasons": reasons,
            }
            (accepted if not reasons else rejected).append(diagnostic)
        if accepted:
            winner = accepted[0]
            decisions[model_id] = {
                "decision": "candidate",
                "stage": winner["stage"],
                "checkpoint": winner["checkpoint"],
                "composite_median": winner["composite_median"],
                "wet_dry_factor": winner["wet_dry_factor"],
                "rejected": rejected,
            }
        else:
            decisions[model_id] = {
                "decision": "baseline",
                "stage": None,
                "checkpoint": None,
                "reason": "no_candidate_passed_promotion_gates",
                "best_rejected": rejected[0] if rejected else None,
                "rejected": rejected,
            }
    return decisions


def markdown_summary(
    rows: list[dict[str, Any]],
    profile: str,
    promotions: dict[str, dict[str, Any]],
) -> str:
    lines = [
        "# Intermediate checkpoint sweep",
        "",
        f"Profile: `{profile}`. Ratios below 1.0 favor the candidate.",
        "",
        "| Model | Stage | Composite median | P95 | Worst group | Regressed groups | Wet/dry |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(rows, key=lambda item: (item["model_id"], item["composite_median"])):
        lines.append(
            f"| `{row['model_id']}` | `{row['stage']}` | "
            f"{row['composite_median']:.4f} | {row['composite_p95']:.4f} | "
            f"{row['worst_group_median']:.4f} | {row['regressed_groups']} | "
            f"{row['wet_dry_rms_ratio']['candidate']:.4f} |"
        )
    lines.extend(("", "## Promotion decisions", ""))
    for model_id, decision in promotions.items():
        if decision["decision"] == "candidate":
            lines.append(
                f"- `{model_id}`: promote `{decision['stage']}` "
                f"(composite {decision['composite_median']:.4f})."
            )
        else:
            best = decision.get("best_rejected")
            detail = (
                f" Best candidate `{best['stage']}` was {best['composite_median']:.4f}."
                if best is not None
                else ""
            )
            lines.append(f"- `{model_id}`: retain stable baseline.{detail}")
    return "\n".join(lines) + "\n"


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description=__doc__)
    root.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    root.add_argument("--registry", type=Path, default=DEFAULT_REGISTRY)
    root.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    root.add_argument("--maestro-root", type=Path, default=ROOT / "data" / "maestro-v3.0.0")
    root.add_argument("--cache-dir", type=Path, default=ROOT / "cache" / "maestro-v3.0.0")
    root.add_argument("--output-root", type=Path)
    root.add_argument("--profile", choices=("quick", "dev"), default="quick")
    root.add_argument("--model-id", action="append", default=[])
    root.add_argument("--verify-steps", type=int, default=100)
    root.add_argument("--max-composite-median", type=float, default=0.98)
    root.add_argument("--max-wet-dry-factor", type=float, default=1.25)
    root.add_argument("--max-latency-ratio", type=float, default=1.05)
    root.add_argument("--max-regressed-groups", type=int, default=0)
    root.add_argument("--max-worst-group-median", type=float, default=1.05)
    root.add_argument("--python", default=sys.executable)
    return root


def main() -> int:
    args = parser().parse_args()
    if not 0.0 < args.max_composite_median <= 1.0:
        raise ValueError("--max-composite-median must be in (0, 1]")
    if args.max_wet_dry_factor < 1.0 or not math.isfinite(args.max_wet_dry_factor):
        raise ValueError("--max-wet-dry-factor must be finite and at least 1")
    if args.max_latency_ratio < 1.0 or not math.isfinite(args.max_latency_ratio):
        raise ValueError("--max-latency-ratio must be finite and at least 1")
    if args.max_regressed_groups < 0:
        raise ValueError("--max-regressed-groups must be non-negative")
    if args.max_worst_group_median < 1.0 or not math.isfinite(
        args.max_worst_group_median
    ):
        raise ValueError("--max-worst-group-median must be finite and at least 1")
    run_root = args.run_root.resolve()
    registry = load_model_registry(args.registry)
    requested = set(args.model_id)
    unknown = requested - set(registry.models)
    if unknown:
        raise ValueError(f"Unknown model IDs: {sorted(unknown)}")
    candidates = candidate_stages(run_root, requested)
    if not candidates:
        raise FileNotFoundError("No intermediate best checkpoints were found")
    output_root = (
        args.output_root or run_root / "checkpoint-sweep" / args.profile
    ).resolve()
    corpus = output_root / f"{args.profile}-corpus.json"
    run(
        [
            args.python,
            str(ROOT / "scripts" / "evaluate_model.py"),
            "prepare",
            "--profile",
            args.profile,
            "--maestro-root",
            str(args.maestro_root.resolve()),
            "--cache-dir",
            str(args.cache_dir.resolve()),
            "--output",
            str(corpus),
        ]
    )

    rows = []
    for model_id, stage, checkpoint in candidates:
        spec = registry.require(model_id)
        artifact_dir = output_root / "artifacts" / model_id / stage
        onnx_path = spec.asset_path(artifact_dir, ".onnx")
        metadata_path = onnx_path.with_suffix(".json")
        if not onnx_path.is_file() or not metadata_path.is_file():
            run(
                [
                    args.python,
                    str(ROOT / "scripts" / "export_onnx.py"),
                    "--model-id",
                    model_id,
                    "--registry",
                    str(args.registry.resolve()),
                    "--checkpoint",
                    str(checkpoint),
                    "--output",
                    str(onnx_path),
                    "--verify-steps",
                    str(args.verify_steps),
                ]
            )
        report_dir = output_root / "reports" / model_id / stage
        if not (report_dir / "report.json").is_file():
            run(
                [
                    args.python,
                    str(ROOT / "scripts" / "evaluate_model.py"),
                    "run",
                    "--profile",
                    args.profile,
                    "--baseline-id",
                    model_id,
                    "--candidate-id",
                    model_id,
                    "--baseline-artifacts-dir",
                    str(args.baseline_dir.resolve()),
                    "--candidate-artifacts-dir",
                    str(artifact_dir),
                    "--corpus",
                    str(corpus),
                    "--output-dir",
                    str(report_dir),
                    "--skip-listening",
                ]
            )
        rows.append(report_row(model_id, stage, checkpoint, read_json(report_dir / "report.json")))

    promotions = promotion_decisions(
        rows,
        max_composite_median=args.max_composite_median,
        max_wet_dry_factor=args.max_wet_dry_factor,
        max_latency_ratio=args.max_latency_ratio,
        max_regressed_groups=args.max_regressed_groups,
        max_worst_group_median=args.max_worst_group_median,
    )
    summary = {
        "schema": "ddsp-piano-checkpoint-sweep/v1",
        "profile": args.profile,
        "baseline_dir": str(args.baseline_dir.resolve()),
        "corpus": str(corpus),
        "promotion_gates": {
            "max_composite_median": args.max_composite_median,
            "max_wet_dry_factor": args.max_wet_dry_factor,
            "max_latency_ratio": args.max_latency_ratio,
            "max_regressed_groups": args.max_regressed_groups,
            "max_worst_group_median": args.max_worst_group_median,
            "require_numerical_allclose": True,
            "require_no_deployment_hard_failures": True,
        },
        "promotions": promotions,
        "candidates": sorted(rows, key=lambda item: (item["model_id"], item["composite_median"])),
    }
    write_json(output_root / "summary.json", summary)
    (output_root / "summary.md").write_text(
        markdown_summary(rows, args.profile, promotions), encoding="utf-8"
    )
    print(json.dumps({"summary": str(output_root / "summary.json"), "candidates": len(rows)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
