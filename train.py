#!/usr/bin/env python3
"""Train the PyTorch DDSP-Piano model from cached MAESTRO audio/MIDI segments."""

from __future__ import annotations

import argparse
import json
import random
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ddsp_piano.default_model import get_model
from ddsp_piano.maestro import MaestroSegmentDataset, PreprocessConfig, prepare_split
from ddsp_piano.modules.loss import HybridLoss


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("runs/piano_16k"))
    parser.add_argument("--prepare", action="store_true", help="Build missing track caches before training")
    parser.add_argument("--prepare-only", action="store_true", help="Build caches then exit")
    parser.add_argument("--prepare-splits", default="train,validation")
    parser.add_argument("--limit-tracks", type=int, help="Limit tracks for a smoke run")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--frame-rate", type=int, default=250)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-polyphony", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps-per-epoch", type=int, default=0, help="0 uses every cached segment")
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--phase", choices=(1, 2), type=int, default=1)
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=20260720)
    return parser.parse_args()


def select_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def config_from_args(args: argparse.Namespace) -> PreprocessConfig:
    config = PreprocessConfig(
        sample_rate=args.sample_rate,
        frame_rate=args.frame_rate,
        segment_seconds=args.segment_seconds,
        overlap=args.overlap,
        max_polyphony=args.max_polyphony,
    )
    config.validate()
    return config


def make_loader(dataset: MaestroSegmentDataset, args: argparse.Namespace, shuffle: bool) -> DataLoader:
    worker_count = max(0, args.num_workers)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=worker_count > 0,
    )


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in batch)


def run_validation(
    model: torch.nn.Module,
    loss_fn: HybridLoss,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            audio, conditioning, pedal, piano_model = move_batch(batch, device)
            with autocast(enabled=amp_enabled) if device.type == "cuda" else autocast():
                signal, reverb_ir, _ = model(conditioning, pedal, piano_model)
                signal = signal[..., : audio.shape[-1]]
                loss, _, _ = loss_fn(signal, audio, reverb_ir)
            losses.append(float(loss.detach().cpu()))
    if not losses:
        raise RuntimeError("Validation loader produced no batches")
    return float(np.mean(losses))


def save_checkpoint(
    destination: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.cuda.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation: float,
    args: argparse.Namespace,
    piano_models: list[int],
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "global_step": global_step,
            "best_validation": best_validation,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "args": vars(args),
            "piano_models": piano_models,
        },
        destination,
    )


def main() -> int:
    args = parse_args()
    config = config_from_args(args)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    requested_splits = [value.strip() for value in args.prepare_splits.split(",") if value.strip()]
    if args.prepare or args.prepare_only:
        for split in requested_splits:
            result = prepare_split(args.maestro_root, args.cache_dir, split, config, args.limit_tracks)
            print(f"cache {split}: {result}")
    if args.prepare_only:
        return 0

    train_dataset = MaestroSegmentDataset(
        args.maestro_root,
        args.cache_dir,
        "train",
        config,
        require_cache=not args.prepare,
        limit_tracks=args.limit_tracks,
    )
    validation_dataset = MaestroSegmentDataset(
        args.maestro_root,
        args.cache_dir,
        "validation",
        config,
        require_cache=not args.prepare,
        limit_tracks=args.limit_tracks,
    )
    if train_dataset.piano_models != validation_dataset.piano_models:
        raise RuntimeError("Training and validation MAESTRO piano-year mappings differ")

    train_loader = make_loader(train_dataset, args, shuffle=True)
    validation_loader = make_loader(validation_dataset, args, shuffle=False)
    model = get_model(
        n_synths=config.max_polyphony,
        n_piano_models=len(train_dataset.piano_models),
        sample_rate=config.sample_rate,
        duration=config.segment_seconds,
        frame_rate=config.frame_rate,
    ).to(device)
    model.alternate_training(first_phase=args.phase == 1)
    loss_fn = HybridLoss(
        [2048, 1024, 512, 256, 128, 64],
        model.inharm_model,
        phase=args.phase == 1,
    ).to(device)
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
    )
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.cuda.amp.GradScaler(enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    best_validation = float("inf")
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint["best_validation"])
        print(f"resumed {args.resume} at epoch={start_epoch} step={global_step}")

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "config.json").write_text(
        json.dumps({"args": vars(args), "preprocess": config.__dict__}, default=str, indent=2),
        encoding="utf-8",
    )
    metrics_path = args.experiment_dir / "metrics.jsonl"
    autocast = torch.cuda.amp.autocast if device.type == "cuda" else nullcontext

    print(
        f"device={device} train_segments={len(train_dataset)} "
        f"validation_segments={len(validation_dataset)} piano_models={train_dataset.piano_models}"
    )
    for epoch in range(start_epoch, args.epochs):
        model.train()
        train_losses: list[float] = []
        for batch_index, batch in enumerate(train_loader):
            if args.steps_per_epoch and batch_index >= args.steps_per_epoch:
                break
            audio, conditioning, pedal, piano_model = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast(enabled=amp_enabled) if device.type == "cuda" else autocast():
                signal, reverb_ir, _ = model(conditioning, pedal, piano_model)
                signal = signal[..., : audio.shape[-1]]
                loss, spectral_loss, reverb_loss = loss_fn(signal, audio, reverb_ir)
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            train_losses.append(float(loss.detach().cpu()))
            if global_step % 20 == 0:
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                    f"loss={loss.item():.5f} spectral={spectral_loss.item():.5f} "
                    f"reverb={reverb_loss.item():.5f}"
                )

        validation_loss = run_validation(
            model,
            loss_fn,
            validation_loader,
            device,
            amp_enabled,
            args.validation_batches,
        )
        mean_train_loss = float(np.mean(train_losses))
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "train_loss": mean_train_loss,
            "validation_loss": validation_loss,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics))

        save_checkpoint(
            args.experiment_dir / "checkpoints" / "last.pt",
            model,
            optimizer,
            scaler,
            epoch,
            global_step,
            best_validation,
            args,
            train_dataset.piano_models,
        )
        if validation_loss < best_validation:
            best_validation = validation_loss
            save_checkpoint(
                args.experiment_dir / "checkpoints" / "best.pt",
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_validation,
                args,
                train_dataset.piano_models,
            )
        if args.save_every and (epoch + 1) % args.save_every == 0:
            save_checkpoint(
                args.experiment_dir / "checkpoints" / f"epoch_{epoch + 1:04d}.pt",
                model,
                optimizer,
                scaler,
                epoch,
                global_step,
                best_validation,
                args,
                train_dataset.piano_models,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
