#!/usr/bin/env python3
"""Benchmark representative DDSP-Piano training steps on synthetic fixed-shape input."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ddsp_piano.default_model import get_model, get_v2_model
from ddsp_piano.modules.loss import HybridLoss
from train import velocity_counterfactual_loss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--architecture", choices=("v1", "v2a", "v2b"), default="v2a")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--polyphony", type=int, default=16)
    parser.add_argument("--warmup-steps", type=int, default=20)
    parser.add_argument("--timed-steps", type=int, default=200)
    parser.add_argument("--synthesis-layout", choices=("serial", "vectorized"), default="serial")
    parser.add_argument("--optimizer-implementation", choices=("standard", "fused"), default="standard")
    parser.add_argument("--compile-training", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--compile-mode", choices=("default", "reduce-overhead", "max-autotune"), default="reduce-overhead")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--velocity-loss-every", type=int, default=4)
    parser.add_argument("--spectral-layout", choices=("separate", "combined"), default="combined")
    parser.add_argument("--velocity-counterfactual-layout", choices=("separate", "combined"), default="combined")
    parser.add_argument("--freeze-reverb", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--seed", type=int, default=20260725)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def build_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    common = {
        "n_synths": args.polyphony,
        "n_piano_models": 10,
        "duration": args.segment_seconds,
        "frame_rate": 250,
        "sample_rate": 16_000,
        "reverb_wet_gain": 1.0,
        "synthesis_layout": args.synthesis_layout,
    }
    if args.architecture == "v1":
        model = get_model(**common)
    elif args.architecture == "v2a":
        model = get_v2_model(
            **common,
            n_harmonics=96,
            n_noise_filter_banks=64,
            reverb_type="ir",
            context_type="legacy",
            monophonic_type="legacy",
            inharmonicity_type="legacy",
        )
    else:
        model = get_v2_model(
            **common,
            n_harmonics=96,
            n_noise_filter_banks=64,
            reverb_type="ir",
            context_type="film",
            monophonic_type="deep",
            inharmonicity_type="joint",
        )
    model.alternate_training(first_phase=True)
    if args.freeze_reverb:
        for parameter in model.reverb_model.parameters():
            parameter.requires_grad = False
    return model.to(device)


def synthetic_batch(args: argparse.Namespace, device: torch.device):
    frames = int(round(args.segment_seconds * 250))
    conditioning = torch.zeros(
        args.batch_size,
        frames,
        args.polyphony,
        2,
        device=device,
    )
    active = min(args.polyphony, 8)
    pitches = torch.arange(48, 48 + active, device=device, dtype=torch.float32)
    conditioning[:, :, :active, 0] = pitches
    conditioning[:, :, :active, 1] = 0.7
    pedal = torch.zeros(args.batch_size, frames, 4, device=device)
    piano_model = torch.zeros(args.batch_size, dtype=torch.int64, device=device)
    audio = torch.zeros(
        args.batch_size,
        int(round(args.segment_seconds * 16_000)),
        device=device,
    )
    return audio, conditioning, pedal, piano_model


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    args = parse_args()
    if args.batch_size <= 0 or args.warmup_steps < 0 or args.timed_steps <= 0:
        raise ValueError("batch and step counts must be positive")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA benchmark requested but CUDA is unavailable")
    if args.optimizer_implementation == "fused" and device.type != "cuda":
        raise ValueError("fused Adam requires CUDA")

    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = True
    model = build_model(args, device)
    forward_model = (
        torch.compile(model, mode=args.compile_mode) if args.compile_training else model
    )
    optimizer_kwargs = {"lr": 3e-4}
    if args.optimizer_implementation == "fused":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        **optimizer_kwargs,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    loss_fn = HybridLoss(
        [2048, 1024, 512, 256, 128, 64],
        model.inharm_model,
        phase=True,
        weight=0.01,
        dry_weight=0.0,
        wet_weight=0.75,
        reverb_mode="ir",
        energy_weight=0.1,
        onset_weight=0.1,
        sample_rate=16_000,
        loss_version="perceptual_v2",
        spectral_layout=args.spectral_layout,
    ).to(device)
    audio, conditioning, pedal, piano_model = synthetic_batch(args, device)
    autocast = torch.amp.autocast if device.type == "cuda" else nullcontext

    def step(step_index: int) -> torch.Tensor:
        model.train()
        optimizer.zero_grad(set_to_none=True)
        with autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else autocast():
            signal, reverb_ir, dry_signal = forward_model(
                conditioning, pedal, piano_model
            )
            signal = signal[..., : audio.shape[-1]]
            loss = loss_fn.components(
                signal,
                audio,
                reverb_ir,
                dry_pred=dry_signal[..., : audio.shape[-1]],
            )[0]
            if step_index % args.velocity_loss_every == 0:
                loss = loss + 0.05 * velocity_counterfactual_loss(
                    model,
                    conditioning,
                    pedal,
                    piano_model,
                    combined=args.velocity_counterfactual_layout == "combined",
                )
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        return loss.detach()

    for step_index in range(args.warmup_steps):
        step(step_index)
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    final_loss = None
    for step_index in range(args.timed_steps):
        final_loss = step(step_index + args.warmup_steps)
    synchronize(device)
    elapsed = time.perf_counter() - started

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    )
    report = {
        "schema": "ddsp-piano-training-benchmark/v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "architecture": args.architecture,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "batch_size": args.batch_size,
        "segment_seconds": args.segment_seconds,
        "polyphony": args.polyphony,
        "warmup_steps": args.warmup_steps,
        "timed_steps": args.timed_steps,
        "synthesis_layout": args.synthesis_layout,
        "compile_training": args.compile_training,
        "compile_mode": args.compile_mode,
        "optimizer_implementation": args.optimizer_implementation,
        "spectral_layout": args.spectral_layout,
        "velocity_counterfactual_layout": args.velocity_counterfactual_layout,
        "amp": amp_enabled,
        "elapsed_seconds": elapsed,
        "steps_per_second": args.timed_steps / elapsed,
        "examples_per_second": args.timed_steps * args.batch_size / elapsed,
        "milliseconds_per_step": elapsed * 1000.0 / args.timed_steps,
        "peak_cuda_memory_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else 0
        ),
        "peak_cuda_memory_reserved_bytes": (
            int(torch.cuda.max_memory_reserved(device)) if device.type == "cuda" else 0
        ),
        "final_loss": float(final_loss.cpu()) if final_loss is not None else None,
    }
    output = args.output
    if output is None:
        stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
        output = ROOT / "runs" / "benchmarks" / "training" / f"{stamp}.json"
    elif not output.is_absolute():
        output = ROOT / output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "output": str(output)}, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
