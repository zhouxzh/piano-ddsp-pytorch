#!/usr/bin/env python3
"""Train the PyTorch DDSP-Piano model from cached MAESTRO audio/MIDI segments."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - only minimal installs lack TensorBoard
    SummaryWriter = None

from ddsp_piano.default_model import build_configurable_model, build_paper_model
from ddsp_piano.evaluation import build_corpus
from ddsp_piano.maestro import MaestroSegmentDataset, PreprocessConfig, prepare_split
from ddsp_piano.modules.loss import HybridLoss
from ddsp_piano.model_registry import load_model_registry
from ddsp_piano.training_quality import (
    CoverageCurriculumSampler,
    MixedCurriculumSampler,
    build_quality_manifest,
    dataset_index_sha256,
    load_quality_manifest,
    velocity_slopes_tensor,
    write_quality_manifest,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--maestro-root", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path, default=Path("cache"))
    parser.add_argument("--experiment-dir", type=Path, default=Path("runs/piano_16k"))
    parser.add_argument("--model-id", default="paper_ir")
    parser.add_argument("--registry", type=Path, help="Model-suite registry; defaults to the stable release")
    parser.add_argument("--prepare", action="store_true", help="Build missing track caches before training")
    parser.add_argument("--prepare-only", action="store_true", help="Build caches then exit")
    parser.add_argument("--prepare-splits", default="train,validation")
    parser.add_argument("--prepare-workers", type=int, default=4)
    parser.add_argument("--limit-tracks", type=int, help="Limit tracks for a smoke run")
    parser.add_argument("--sample-rate", type=int, default=16_000)
    parser.add_argument("--frame-rate", type=int, default=250)
    parser.add_argument("--segment-seconds", type=float, default=3.0)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--max-polyphony", type=int, default=16)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--steps-per-epoch", type=int, help="0 uses every cached segment")
    parser.add_argument("--validation-batches", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument(
        "--train-workers",
        type=int,
        help="Training loader workers; defaults to --num-workers",
    )
    parser.add_argument(
        "--validation-workers",
        type=int,
        help="Validation loader workers; defaults to --num-workers",
    )
    parser.add_argument(
        "--validation-every-epochs",
        type=int,
        default=1,
        help="Validate every N epochs; 0 validates only at the end of this invocation",
    )
    parser.add_argument("--lr", type=float)
    parser.add_argument(
        "--stage",
        choices=("controls", "pitch", "refine", "calibrate"),
        help="Explicit trainable parameter and detuning stage",
    )
    parser.add_argument(
        "--phase",
        choices=(1, 2),
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--device", default="auto", help="auto, cuda, cuda:N, or cpu")
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--warmup-fraction", type=float, default=0.02)
    parser.add_argument("--minimum-lr-ratio", type=float, default=0.01)
    parser.add_argument(
        "--synthesis-layout",
        choices=("serial", "vectorized"),
        default="serial",
        help="Training DSP layout; exported control-only ONNX is unaffected",
    )
    parser.add_argument(
        "--compile-training",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--compile-mode",
        choices=("default", "reduce-overhead", "max-autotune"),
        default="reduce-overhead",
    )
    parser.add_argument(
        "--optimizer-implementation",
        choices=("standard", "fused"),
        default="standard",
    )
    parser.add_argument("--log-every", type=int, default=20)
    parser.add_argument(
        "--tensorboard",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Write TensorBoard event files alongside metrics.jsonl",
    )
    parser.add_argument(
        "--tensorboard-logdir",
        type=Path,
        help="TensorBoard directory; defaults to <experiment-dir>/tensorboard",
    )
    parser.add_argument(
        "--spectral-layout",
        choices=("separate", "combined"),
        default="separate",
    )
    parser.add_argument(
        "--velocity-counterfactual-layout",
        choices=("separate", "combined"),
        default="separate",
    )
    parser.add_argument("--dry-loss-weight", type=float, default=0.7)
    parser.add_argument("--wet-loss-weight", type=float, default=0.3)
    parser.add_argument("--reverb-regularizer-weight", type=float, default=0.05)
    parser.add_argument("--n-harmonics", type=int)
    parser.add_argument("--n-noise-bands", type=int)
    parser.add_argument("--reverb-type", choices=("auto", "ir", "fdn"), default="auto")
    parser.add_argument("--reverb-wet-gain", type=float, default=0.25)
    parser.add_argument(
        "--context-type", choices=("legacy", "residual_film", "film"), default="film"
    )
    parser.add_argument(
        "--monophonic-type", choices=("legacy", "residual_deep", "deep"), default="deep"
    )
    parser.add_argument(
        "--inharmonicity-type",
        choices=("legacy", "residual_joint", "joint"),
        default="joint",
    )
    parser.add_argument("--freeze-reverb", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--adapter-only", action="store_true")
    parser.add_argument(
        "--trainable-scope",
        choices=("phase_default", "controls", "reverb", "joint"),
        default="phase_default",
    )
    parser.add_argument(
        "--loss-version", choices=("legacy", "perceptual_v2"), default="legacy"
    )
    parser.add_argument("--energy-loss-weight", type=float, default=0.0)
    parser.add_argument("--onset-loss-weight", type=float, default=0.0)
    parser.add_argument("--centroid-loss-weight", type=float, default=0.0)
    parser.add_argument("--tail-loss-weight", type=float, default=0.0)
    parser.add_argument("--energy-hard-fraction", type=float, default=0.0)
    parser.add_argument("--velocity-loss-weight", type=float, default=0.0)
    parser.add_argument("--velocity-loss-every", type=int, default=4)
    parser.add_argument("--velocity-response-ms", type=float, default=125.0)
    parser.add_argument("--loss-calibration-batches", type=int, default=0)
    parser.add_argument(
        "--sampling-mode",
        choices=("uniform", "curriculum", "coverage"),
        default="uniform",
    )
    parser.add_argument("--curriculum-tail-fraction", type=float, default=0.2)
    parser.add_argument("--quality-manifest", type=Path)
    parser.add_argument(
        "--balanced-validation",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--validation-every-examples",
        type=int,
        default=0,
        help="Validate and checkpoint after this many stage examples; 0 uses epoch boundaries",
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--weights", type=Path, help="Load model weights only, for phase changes")
    parser.add_argument(
        "--finetune-from",
        type=Path,
        help="Strictly load model weights and reset optimizer, steps, and RNG state",
    )
    parser.add_argument(
        "--init-checkpoint",
        type=Path,
        help="Partially initialize matching weights and reset optimizer/step state",
    )
    parser.add_argument("--seed", type=int)
    return parser.parse_args()


def select_device(spec: str) -> torch.device:
    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(spec)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    return device


def explicit_cli_destinations(arguments: list[str]) -> set[str]:
    """Return argparse destination names explicitly present on the command line."""
    result = set()
    for argument in arguments:
        if not argument.startswith("--"):
            continue
        name = argument[2:].split("=", 1)[0]
        if name.startswith("no-"):
            name = name[3:]
        result.add(name.replace("-", "_"))
    return result


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


def make_loader(
    dataset: Dataset,
    args: argparse.Namespace,
    shuffle: bool,
    worker_count: int | None = None,
    generator: torch.Generator | None = None,
    sampler=None,
) -> DataLoader:
    if worker_count is None:
        worker_count = args.num_workers
    worker_count = max(0, worker_count)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle if sampler is None else False,
        sampler=sampler,
        num_workers=worker_count,
        pin_memory=True,
        persistent_workers=worker_count > 0,
        generator=generator,
    )


def capture_rng_state(train_generator: torch.Generator | None = None) -> dict:
    state = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    if train_generator is not None:
        state["train_loader_generator"] = train_generator.get_state()
    return state


def restore_rng_state(
    state: dict | None,
    train_generator: torch.Generator | None = None,
) -> None:
    if not state:
        return
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    # A checkpoint loaded with map_location="cuda" also relocates these CPU
    # RNG tensors. PyTorch's RNG restore APIs require CPU ByteTensors.
    torch.set_rng_state(state["torch"].detach().cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(
            [cuda_state.detach().cpu() for cuda_state in state["cuda"]]
        )
    if train_generator is not None and "train_loader_generator" in state:
        train_generator.set_state(
            state["train_loader_generator"].detach().cpu()
        )


def move_batch(batch: tuple[torch.Tensor, ...], device: torch.device) -> tuple[torch.Tensor, ...]:
    return tuple(value.to(device, non_blocking=True) for value in batch)


def balanced_validation_subset(
    dataset: MaestroSegmentDataset,
    maestro_root: Path,
    cache_dir: Path,
    config: PreprocessConfig,
) -> tuple[Subset, str]:
    corpus = build_corpus(maestro_root, cache_dir, "dev", config)
    lookup = {
        (str(Path(cache_path).resolve()), int(sample_start), int(frame_start), int(piano_id)): index
        for index, (cache_path, sample_start, frame_start, piano_id) in enumerate(dataset.index)
    }
    indices = []
    for entry in corpus["entries"]:
        key = (
            str(Path(entry["cache_path"]).resolve()),
            int(entry["sample_start"]),
            int(entry["frame_start"]),
            int(entry["piano_model"]),
        )
        if key not in lookup:
            raise RuntimeError(f"Balanced validation entry is absent from dataset: {key}")
        indices.append(lookup[key])
    return Subset(dataset, indices), str(corpus["corpus_sha256"])


def fixed_calibration_subset(
    dataset: MaestroSegmentDataset, count: int, seed: int
) -> tuple[Subset, str]:
    ranked = sorted(
        range(len(dataset.index)),
        key=lambda index: hashlib.sha256(
            f"{seed}\0{dataset.index[index]}".encode("utf-8")
        ).hexdigest(),
    )
    selected = ranked[: min(count, len(ranked))]
    digest = hashlib.sha256(
        "\n".join(str(dataset.index[index]) for index in selected).encode("utf-8")
    ).hexdigest()
    return Subset(dataset, selected), digest


def load_partial_initialization(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    source = checkpoint["model"]
    target = model.state_dict()
    loaded: dict[str, str] = {}
    skipped: list[str] = []
    wrapper_prefixes = (
        "context_network.",
        "monophonic_network.",
        "inharm_model.",
    )
    for source_name, value in source.items():
        candidates = []
        for prefix in wrapper_prefixes:
            if source_name.startswith(prefix):
                candidates.append(prefix + "base." + source_name[len(prefix) :])
        candidates.append(source_name)
        destination = next(
            (
                name
                for name in candidates
                if name in target and target[name].shape == value.shape
            ),
            None,
        )
        if destination is None:
            skipped.append(source_name)
            continue
        target[destination] = value
        loaded[destination] = source_name
    model.load_state_dict(target)
    digest = hashlib.sha256()
    with checkpoint_path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    report = {
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": digest.hexdigest(),
        "loaded_tensors": len(loaded),
        "target_tensors": len(target),
        "skipped_source_tensors": skipped,
        "new_target_tensors": sorted(set(target) - set(loaded)),
        "mapping": loaded,
    }
    if not loaded:
        raise RuntimeError(f"No compatible tensors found in {checkpoint_path}")
    return report


def checkpoint_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_finetune_initialization(
    model: torch.nn.Module,
    checkpoint_path: Path,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    return {
        "mode": "finetune",
        "checkpoint": str(checkpoint_path.resolve()),
        "checkpoint_sha256": checkpoint_sha256(checkpoint_path),
        "parent_global_step": int(checkpoint.get("global_step", 0)),
        "parent_examples_seen": int(checkpoint.get("examples_seen", 0)),
        "optimizer_reset": True,
        "step_reset": True,
    }


def set_adapter_only(model: torch.nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    adapter_parameters = []
    context = model.context_network
    monophonic = model.monophonic_network
    inharmonicity = model.inharm_model
    if hasattr(context, "film") and hasattr(context, "base"):
        adapter_parameters.extend(context.film.parameters())
    if hasattr(monophonic, "residual") and hasattr(monophonic, "base"):
        adapter_parameters.extend(monophonic.residual.parameters())
    if hasattr(inharmonicity, "base"):
        adapter_parameters.extend(
            [inharmonicity.slopes_modifier, inharmonicity.offsets_modifier]
        )
    if not adapter_parameters:
        raise ValueError("--adapter-only requires at least one residual v2A module")
    for parameter in adapter_parameters:
        parameter.requires_grad = True


def set_trainable_scope(
    model: torch.nn.Module,
    scope: str,
    phase: int | None = None,
) -> None:
    """Apply training-only parameter scopes without changing model structure."""
    if scope == "phase_default":
        return
    if phase is not None and phase != 1:
        raise ValueError("Explicit Q1 trainable scopes are only supported in phase 1")
    if scope == "joint":
        return
    if scope == "controls":
        for parameter in model.reverb_model.parameters():
            parameter.requires_grad = False
        return
    if scope == "reverb":
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.reverb_model.parameters():
            parameter.requires_grad = True
        return
    raise ValueError(f"Unsupported trainable scope: {scope}")


def velocity_counterfactual_loss(
    model: torch.nn.Module,
    conditioning: torch.Tensor,
    pedal: torch.Tensor,
    piano_model: torch.Tensor,
    combined: bool = False,
    response_slopes: torch.Tensor | None = None,
    response_frames: int = 31,
) -> torch.Tensor:
    onset = conditioning[..., 1:2] > 0
    low = conditioning.clone()
    high = conditioning.clone()
    low[..., 1:2] = torch.where(onset, conditioning[..., 1:2] * 0.5, conditioning[..., 1:2])
    high[..., 1:2] = torch.where(
        onset,
        (conditioning[..., 1:2] * 1.25).clamp_max(1.0),
        conditioning[..., 1:2],
    )
    if combined:
        batch_size = conditioning.shape[0]
        amplitudes = model.predict_controls(
            torch.cat((low, high), dim=0),
            torch.cat((pedal, pedal), dim=0),
            torch.cat((piano_model, piano_model), dim=0),
        )[0]
        low_amplitudes, high_amplitudes = amplitudes.split(batch_size, dim=1)
    else:
        low_amplitudes = model.predict_controls(low, pedal, piano_model)[0]
        high_amplitudes = model.predict_controls(high, pedal, piano_model)[0]
    batch, frames, voices, _ = conditioning.shape
    frame_indices = torch.arange(frames, device=conditioning.device).view(1, frames, 1)
    onset_btv = onset.squeeze(-1)
    last_onset = torch.cummax(
        torch.where(
            onset_btv,
            frame_indices.expand(batch, frames, voices),
            torch.full((batch, frames, voices), -frames, device=conditioning.device),
        ),
        dim=1,
    ).values
    gather_indices = last_onset.clamp_min(0)
    held_velocity = torch.gather(conditioning[..., 1], 1, gather_indices)
    low_velocity = held_velocity * 0.5
    high_velocity = (held_velocity * 1.25).clamp_max(1.0)
    expected_log_ratio = torch.log(
        high_velocity.clamp_min(1e-5) / low_velocity.clamp_min(1e-5)
    )
    if response_slopes is not None:
        pitches = conditioning[..., 0]
        pitch_band = torch.where(pitches <= 47, 0, torch.where(pitches <= 71, 1, 2)).long()
        per_piano = response_slopes[piano_model.long()]
        expanded = per_piano[:, None, None, :].expand(batch, frames, voices, 3)
        expected_log_ratio = expected_log_ratio * expanded.gather(
            -1, pitch_band.unsqueeze(-1)
        ).squeeze(-1)
    age = frame_indices - last_onset
    active_mask = (
        conditioning[..., 0].ne(0)
        & (last_onset >= 0)
        & (age >= 0)
        & (age < max(1, int(response_frames)))
    )
    return HybridLoss.velocity_response_loss(
        low_amplitudes,
        high_amplitudes,
        active_mask.permute(2, 0, 1).unsqueeze(-1),
        expected_log_ratio.permute(2, 0, 1).unsqueeze(-1),
    )


def calibrate_perceptual_loss(
    model: torch.nn.Module,
    loss_fn: HybridLoss,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
    velocity_every: int,
    combined_velocity: bool = False,
    response_slopes: torch.Tensor | None = None,
    response_frames: int = 31,
) -> dict:
    if max_batches <= 0:
        return {"batches": 0, "component_scales": dict(loss_fn.component_scales)}
    values: dict[str, list[torch.Tensor]] = {
        "wet": [],
        "energy": [],
        "onset": [],
        "centroid": [],
        "tail": [],
        "velocity": [],
    }
    autocast = torch.amp.autocast if device.type == "cuda" else nullcontext
    model.eval()
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if batch_index >= max_batches:
                break
            audio, conditioning, pedal, piano_model = move_batch(batch, device)
            with autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else autocast():
                signal, reverb_ir, dry_signal = model(conditioning, pedal, piano_model)
                signal = signal[..., : audio.shape[-1]]
                components = loss_fn.components(
                    signal,
                    audio,
                    reverb_ir,
                    dry_pred=dry_signal[..., : audio.shape[-1]],
                    conditioning=conditioning,
                )
                raw = {
                    "wet": components[1],
                    "energy": components[4],
                    "onset": components[5],
                    "centroid": components[6],
                    "tail": components[7],
                }
                for name, value in raw.items():
                    values[name].append(value.detach())
                if batch_index % velocity_every == 0:
                    velocity = velocity_counterfactual_loss(
                        model,
                        conditioning,
                        pedal,
                        piano_model,
                        combined=combined_velocity,
                        response_slopes=response_slopes,
                        response_frames=response_frames,
                    )
                    values["velocity"].append(velocity.detach())
    host_values = {
        name: torch.stack(component_values).cpu().numpy().tolist()
        if component_values
        else []
        for name, component_values in values.items()
    }
    medians = {
        name: float(np.median(component_values)) if component_values else 0.0
        for name, component_values in host_values.items()
    }
    positive_velocity = [value for value in host_values["velocity"] if value > 1e-7]
    velocity_reference = (
        float(np.median(positive_velocity)) if positive_velocity else 0.01
    )
    for name in ("wet", "energy", "onset", "centroid", "tail"):
        loss_fn.component_scales[name] = min(100.0, max(0.1, 1.0 / max(medians[name], 1e-3)))
    return {
        "batches": min(max_batches, len(loader)),
        "raw_medians": medians,
        "component_scales": dict(loss_fn.component_scales),
        "velocity_scale": min(100.0, 1.0 / max(velocity_reference, 0.01)),
    }


def run_validation(
    model: torch.nn.Module,
    loss_fn: HybridLoss,
    loader: DataLoader,
    device: torch.device,
    amp_enabled: bool,
    max_batches: int,
    velocity_weight: float = 0.0,
    velocity_every: int = 4,
    velocity_scale: float = 1.0,
    combined_velocity: bool = False,
    response_slopes: torch.Tensor | None = None,
    response_frames: int = 31,
) -> float:
    model.eval()
    loss_sum = torch.zeros((), device=device)
    batch_count = 0
    autocast = torch.amp.autocast if device.type == "cuda" else nullcontext
    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches and batch_index >= max_batches:
                break
            audio, conditioning, pedal, piano_model = move_batch(batch, device)
            with autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else autocast():
                signal, reverb_ir, dry_signal = model(conditioning, pedal, piano_model)
                signal = signal[..., : audio.shape[-1]]
                dry_signal = dry_signal[..., : audio.shape[-1]]
                loss, _, _, _, _, _, _, _ = loss_fn.components(
                    signal,
                    audio,
                    reverb_ir,
                    dry_pred=dry_signal,
                    conditioning=conditioning,
                )
                if velocity_weight and batch_index % velocity_every == 0:
                    velocity_loss = velocity_counterfactual_loss(
                        model,
                        conditioning,
                        pedal,
                        piano_model,
                        combined=combined_velocity,
                        response_slopes=response_slopes,
                        response_frames=response_frames,
                    )
                    loss = loss + velocity_weight * velocity_scale * velocity_loss
            loss_sum = loss_sum + loss.detach()
            batch_count += 1
    if not batch_count:
        raise RuntimeError("Validation loader produced no batches")
    return float((loss_sum / batch_count).cpu())


def build_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_fraction: float,
    minimum_ratio: float,
) -> torch.optim.lr_scheduler.LambdaLR:
    """Linear warmup followed by a cosine decay for one training stage."""
    if total_steps <= 0:
        raise ValueError("total_steps must be positive")
    if not 0.0 <= warmup_fraction < 1.0:
        raise ValueError("warmup_fraction must be in [0, 1)")
    if not 0.0 <= minimum_ratio <= 1.0:
        raise ValueError("minimum_ratio must be in [0, 1]")
    warmup_steps = int(round(total_steps * warmup_fraction))

    def scale(step: int) -> float:
        if warmup_steps and step < warmup_steps:
            return max(1, step + 1) / warmup_steps
        decay_steps = max(1, total_steps - warmup_steps)
        progress = min(1.0, max(0.0, (step - warmup_steps) / decay_steps))
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum_ratio + (1.0 - minimum_ratio) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, scale)


def training_stage_metadata(
    model: torch.nn.Module,
    args: argparse.Namespace,
    dataset: MaestroSegmentDataset,
) -> dict:
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    piano_counts: dict[str, int] = {}
    for _, _, _, piano_id in dataset.index:
        key = str(int(piano_id))
        piano_counts[key] = piano_counts.get(key, 0) + 1
    return {
        "stage": args.stage,
        "detune_enabled": bool(model.detuner.use_detune),
        "trainable_parameters": trainable,
        "trainable_parameter_count": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "dataset_segments": len(dataset),
        "dataset_index_sha256": dataset_index_sha256(dataset.index),
        "piano_model_segment_counts": piano_counts,
        "sampling_mode": args.sampling_mode,
        "curriculum_tail_fraction": (
            args.curriculum_tail_fraction if args.sampling_mode == "coverage" else 0.0
        ),
    }


def stage_progress_metadata(
    base: dict,
    epoch: int,
    epoch_examples: int,
    examples_seen: int,
) -> dict:
    result = dict(base)
    dataset_segments = int(result["dataset_segments"])
    result.update(
        {
            "stage_examples_seen": int(examples_seen),
            "coverage_epoch": int(epoch),
            "coverage_epoch_examples": int(epoch_examples),
            "coverage_passes_completed": float(
                epoch + min(epoch_examples, dataset_segments) / dataset_segments
            ),
            "curriculum_examples_seen_in_epoch": max(
                0, int(epoch_examples) - dataset_segments
            ),
        }
    )
    return result


def save_checkpoint(
    destination: Path,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scaler: torch.amp.GradScaler,
    epoch: int,
    global_step: int,
    best_validation: float,
    args: argparse.Namespace,
    piano_models: list[int],
    loss_calibration: dict,
    initialization: dict | None,
    validation_corpus_sha256: str | None,
    examples_seen: int,
    train_generator: torch.Generator,
    training_performance: dict,
    lr_scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    stage_metadata: dict | None = None,
    sampler_state: dict | None = None,
    epoch_complete: bool = True,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    serialized_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    payload = {
        "schema": "ddsp-piano-training-checkpoint/v3",
        "epoch": epoch,
        "global_step": global_step,
        "examples_seen": examples_seen,
        "best_validation": best_validation,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scaler": scaler.state_dict(),
        "args": serialized_args,
        "piano_models": piano_models,
        "loss_calibration": loss_calibration,
        "initialization": initialization,
        "validation_corpus_sha256": validation_corpus_sha256,
        "rng_state": capture_rng_state(train_generator),
        "training_performance": training_performance,
        "lr_scheduler": lr_scheduler.state_dict() if lr_scheduler is not None else None,
        "training_stage": stage_metadata,
        "sampler_state": sampler_state,
        "epoch_complete": bool(epoch_complete),
    }
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(
        payload,
        temporary,
    )
    temporary.replace(destination)


def main() -> int:
    args = parse_args()
    explicit_cli = explicit_cli_destinations(sys.argv[1:])
    registry = load_model_registry(args.registry) if args.registry is not None else load_model_registry()
    model_spec = registry.require(args.model_id)
    args.architecture = model_spec.architecture
    for name, value in model_spec.model.items():
        setattr(args, name, value)
    training_names = {"learning_rate": "lr"}
    for name, value in model_spec.training.items():
        destination = training_names.get(name, name)
        if destination not in explicit_cli:
            setattr(args, destination, value)
    if args.phase is not None:
        if args.stage is not None:
            raise ValueError("--phase and --stage are mutually exclusive")
        args.stage = "controls" if args.phase == 1 else "pitch"
        print(
            f"warning: --phase is deprecated; mapped phase {args.phase} to --stage {args.stage}",
            flush=True,
        )
    if args.stage is None:
        args.stage = "controls"
    initialization_options = [
        args.resume is not None,
        args.weights is not None,
        args.init_checkpoint is not None,
        args.finetune_from is not None,
    ]
    if sum(initialization_options) > 1:
        raise ValueError(
            "--resume, --weights, --init-checkpoint, and --finetune-from are mutually exclusive"
        )
    if args.n_harmonics <= 0 or args.n_noise_bands <= 0:
        raise ValueError("--n-harmonics and --n-noise-bands must be positive")
    if min(
        args.energy_loss_weight,
        args.onset_loss_weight,
        args.centroid_loss_weight,
        args.tail_loss_weight,
        args.velocity_loss_weight,
    ) < 0:
        raise ValueError("quality loss weights must be non-negative")
    if args.velocity_loss_every <= 0:
        raise ValueError("--velocity-loss-every must be positive")
    if args.loss_calibration_batches < 0:
        raise ValueError("--loss-calibration-batches must be non-negative")
    if not 0.0 <= args.energy_hard_fraction <= 1.0:
        raise ValueError("--energy-hard-fraction must be in [0, 1]")
    if args.velocity_response_ms <= 0:
        raise ValueError("--velocity-response-ms must be positive")
    if args.sampling_mode in {"curriculum", "coverage"} and args.quality_manifest is None:
        raise ValueError(f"--sampling-mode {args.sampling_mode} requires --quality-manifest")
    if not 0.0 <= args.curriculum_tail_fraction <= 1.0:
        raise ValueError("--curriculum-tail-fraction must be in [0, 1]")
    if args.validation_every_examples < 0:
        raise ValueError("--validation-every-examples must be non-negative")
    if args.adapter_only and args.trainable_scope != "phase_default":
        raise ValueError("--adapter-only cannot be combined with --trainable-scope")
    if args.freeze_reverb and args.trainable_scope == "reverb":
        raise ValueError("--freeze-reverb conflicts with --trainable-scope reverb")
    if args.validation_every_epochs < 0:
        raise ValueError("--validation-every-epochs must be non-negative")
    if args.log_every <= 0:
        raise ValueError("--log-every must be positive")
    if args.train_workers is None:
        args.train_workers = args.num_workers
    if args.validation_workers is None:
        args.validation_workers = args.num_workers
    if args.train_workers < 0 or args.validation_workers < 0:
        raise ValueError("loader worker counts must be non-negative")
    if args.reverb_wet_gain < 0:
        raise ValueError("--reverb-wet-gain must be non-negative")
    if not 0.0 <= args.warmup_fraction < 1.0:
        raise ValueError("--warmup-fraction must be in [0, 1)")
    if not 0.0 <= args.minimum_lr_ratio <= 1.0:
        raise ValueError("--minimum-lr-ratio must be in [0, 1]")
    if args.loss_version == "perceptual_v2" and args.dry_loss_weight:
        print("perceptual_v2 ignores --dry-loss-weight", flush=True)
    if args.architecture == "paper" and (
        args.n_harmonics != 96
        or args.n_noise_bands != 64
        or args.reverb_type != "ir"
    ):
        raise ValueError(
            "The paper architecture requires 96 harmonics, 64 noise bands, and IR reverb"
        )
    config = config_from_args(args)
    device = select_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
    if args.optimizer_implementation == "fused" and device.type != "cuda":
        raise ValueError("--optimizer-implementation fused requires CUDA")
    train_generator = torch.Generator()
    train_generator.manual_seed(args.seed)

    requested_splits = [value.strip() for value in args.prepare_splits.split(",") if value.strip()]
    if args.prepare or args.prepare_only:
        for split in requested_splits:
            result = prepare_split(
                args.maestro_root,
                args.cache_dir,
                split,
                config,
                args.limit_tracks,
                args.prepare_workers,
            )
            print(f"cache {split}: {result}", flush=True)
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

    quality_manifest = None
    quality_manifest_sha256 = None
    response_slopes = None
    if args.quality_manifest is not None:
        if args.quality_manifest.is_file():
            quality_manifest = load_quality_manifest(args.quality_manifest, train_dataset)
        else:
            quality_manifest = build_quality_manifest(
                train_dataset, config.sample_rate, config.frame_rate, args.seed
            )
            write_quality_manifest(args.quality_manifest, quality_manifest)
        quality_manifest_sha256 = checkpoint_sha256(args.quality_manifest)
        response_slopes = velocity_slopes_tensor(quality_manifest).to(device)
    curriculum_sampler = None
    if args.sampling_mode == "curriculum":
        curriculum_sampler = MixedCurriculumSampler(
            [entry["curriculum_weight"] for entry in quality_manifest["entries"]],
            train_generator,
        )
    elif args.sampling_mode == "coverage":
        curriculum_sampler = CoverageCurriculumSampler(
            quality_manifest["entries"],
            seed=args.seed,
            tail_fraction=args.curriculum_tail_fraction,
        )
    train_loader = make_loader(
        train_dataset,
        args,
        shuffle=curriculum_sampler is None,
        worker_count=args.train_workers,
        generator=train_generator if curriculum_sampler is None else None,
        sampler=curriculum_sampler,
    )
    validation_corpus_sha256 = None
    validation_source: Dataset = validation_dataset
    if args.balanced_validation:
        validation_source, validation_corpus_sha256 = balanced_validation_subset(
            validation_dataset, args.maestro_root, args.cache_dir, config
        )
    validation_loader = make_loader(
        validation_source,
        args,
        shuffle=False,
        worker_count=args.validation_workers,
    )
    model_builder = (
        build_configurable_model
        if args.architecture == "configurable"
        else build_paper_model
    )
    model_kwargs = dict(
        n_synths=config.max_polyphony,
        n_piano_models=len(train_dataset.piano_models),
        sample_rate=config.sample_rate,
        duration=config.segment_seconds,
        frame_rate=config.frame_rate,
        reverb_wet_gain=args.reverb_wet_gain,
        synthesis_layout=args.synthesis_layout,
    )
    if args.architecture == "configurable":
        model_kwargs.update(
            n_harmonics=args.n_harmonics,
            n_noise_filter_banks=args.n_noise_bands,
            reverb_type=args.reverb_type,
            context_type=args.context_type,
            monophonic_type=args.monophonic_type,
            inharmonicity_type=args.inharmonicity_type,
        )
    model = model_builder(**model_kwargs).to(device)
    model.configure_training_stage(args.stage)
    set_trainable_scope(model, args.trainable_scope)
    if args.freeze_reverb:
        for parameter in model.reverb_model.parameters():
            parameter.requires_grad = False
    if args.adapter_only:
        set_adapter_only(model)
    loss_fn = HybridLoss(
        [2048, 1024, 512, 256, 128, 64],
        model.inharm_model,
        phase=args.stage != "pitch",
        weight=args.reverb_regularizer_weight,
        dry_weight=args.dry_loss_weight,
        wet_weight=args.wet_loss_weight,
        reverb_mode=args.reverb_type,
        energy_weight=args.energy_loss_weight,
        onset_weight=args.onset_loss_weight,
        centroid_weight=args.centroid_loss_weight,
        tail_weight_audio=args.tail_loss_weight,
        energy_hard_fraction=args.energy_hard_fraction,
        sample_rate=config.sample_rate,
        frame_rate=config.frame_rate,
        loss_version=args.loss_version,
        spectral_layout=args.spectral_layout,
    ).to(device)
    amp_enabled = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    examples_seen = 0
    best_validation = float("inf")
    initialization = None
    loss_calibration = {
        "batches": 0,
        "component_scales": dict(loss_fn.component_scales),
        "velocity_scale": 1.0,
        "quality_manifest": (
            str(args.quality_manifest.resolve()) if args.quality_manifest is not None else None
        ),
        "quality_manifest_sha256": quality_manifest_sha256,
    }
    if args.init_checkpoint is not None:
        initialization = load_partial_initialization(model, args.init_checkpoint, device)
        print(
            f"initialized {initialization['loaded_tensors']}/{initialization['target_tensors']} "
            f"tensors from {args.init_checkpoint}; step reset to 0",
            flush=True,
        )
    if args.weights is not None:
        checkpoint = torch.load(args.weights, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        global_step = int(checkpoint.get("global_step", 0))
        print(f"loaded model weights from {args.weights} at step={global_step}", flush=True)
    if args.finetune_from is not None:
        initialization = load_finetune_initialization(model, args.finetune_from, device)
        print(
            f"strictly loaded fine-tune parent {args.finetune_from}; "
            "optimizer, step, examples, and RNG reset",
            flush=True,
        )
    optimizer_kwargs = {"lr": args.lr}
    if args.optimizer_implementation == "fused":
        optimizer_kwargs["fused"] = True
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        **optimizer_kwargs,
    )
    full_steps_per_epoch = len(train_loader)
    if args.steps_per_epoch:
        full_steps_per_epoch = min(full_steps_per_epoch, args.steps_per_epoch)
    lr_scheduler = build_lr_scheduler(
        optimizer,
        total_steps=max(1, args.epochs * full_steps_per_epoch),
        warmup_fraction=args.warmup_fraction,
        minimum_ratio=args.minimum_lr_ratio,
    )
    resume_epoch_offset = 0
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        checkpoint_stage = checkpoint.get("training_stage") or {}
        if checkpoint_stage.get("stage") not in {None, args.stage}:
            raise ValueError(
                f"checkpoint stage {checkpoint_stage.get('stage')!r} does not match {args.stage!r}"
            )
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint.get("scaler", {}))
        epoch_complete = bool(checkpoint.get("epoch_complete", True))
        start_epoch = int(checkpoint["epoch"]) + int(epoch_complete)
        global_step = int(checkpoint["global_step"])
        examples_seen = int(checkpoint.get("examples_seen", global_step * args.batch_size))
        best_validation = float(checkpoint["best_validation"])
        loss_calibration = checkpoint.get("loss_calibration", loss_calibration)
        loss_fn.component_scales.update(loss_calibration.get("component_scales", {}))
        initialization = checkpoint.get("initialization")
        validation_corpus_sha256 = checkpoint.get(
            "validation_corpus_sha256", validation_corpus_sha256
        )
        restore_rng_state(checkpoint.get("rng_state"), train_generator)
        if checkpoint.get("lr_scheduler") is not None:
            lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        sampler_state = checkpoint.get("sampler_state")
        if sampler_state is not None:
            if not isinstance(curriculum_sampler, CoverageCurriculumSampler):
                raise ValueError("checkpoint requires --sampling-mode coverage")
            curriculum_sampler.load_state_dict(sampler_state)
            resume_epoch_offset = int(sampler_state["start_offset"])
        print(
            f"resumed {args.resume} at epoch={start_epoch} offset={resume_epoch_offset} "
            f"step={global_step}",
            flush=True,
        )
    elif args.loss_version == "perceptual_v2":
        calibration_source, calibration_sha256 = fixed_calibration_subset(
            train_dataset, args.loss_calibration_batches, args.seed
        )
        calibration_loader = make_loader(
            calibration_source,
            args,
            shuffle=False,
            worker_count=0,
        )
        loss_calibration = calibrate_perceptual_loss(
            model,
            loss_fn,
            calibration_loader,
            device,
            amp_enabled,
            args.loss_calibration_batches,
            args.velocity_loss_every,
            args.velocity_counterfactual_layout == "combined",
            response_slopes,
            max(1, int(round(config.frame_rate * args.velocity_response_ms / 1000.0))),
        )
        loss_calibration["corpus_sha256"] = calibration_sha256
        loss_calibration["quality_manifest"] = (
            str(args.quality_manifest.resolve()) if args.quality_manifest is not None else None
        )
        loss_calibration["quality_manifest_sha256"] = quality_manifest_sha256
        print(f"loss calibration: {json.dumps(loss_calibration)}", flush=True)

    args.experiment_dir.mkdir(parents=True, exist_ok=True)
    (args.experiment_dir / "config.json").write_text(
        json.dumps(
            {
                "args": vars(args),
                "preprocess": config.__dict__,
                "loss_calibration": loss_calibration,
                "initialization": initialization,
                "validation_corpus_sha256": validation_corpus_sha256,
            },
            default=str,
            indent=2,
        ),
        encoding="utf-8",
    )
    metrics_path = args.experiment_dir / "metrics.jsonl"
    if args.tensorboard and SummaryWriter is None:
        raise RuntimeError(
            "TensorBoard logging is enabled but tensorboard is not installed; "
            "install requirements-cuda.txt or pass --no-tensorboard"
        )
    tensorboard_logdir = args.tensorboard_logdir or (args.experiment_dir / "tensorboard")
    writer = (
        SummaryWriter(log_dir=str(tensorboard_logdir), flush_secs=30)
        if args.tensorboard
        else None
    )
    if writer is not None:
        writer.add_text(
            "run/config",
            json.dumps(
                {
                    "model_id": args.model_id,
                    "stage": args.stage,
                    "detune_enabled": bool(model.detuner.use_detune),
                    "args": vars(args),
                    "preprocess": config.__dict__,
                },
                default=str,
                indent=2,
            ),
            global_step=0,
        )
    autocast = torch.amp.autocast if device.type == "cuda" else nullcontext
    training_model = model
    if args.compile_training:
        training_model = torch.compile(model, mode=args.compile_mode)
        print(f"compiled training forward with mode={args.compile_mode}", flush=True)
    training_performance: dict = {}
    base_stage_metadata = training_stage_metadata(model, args, train_dataset)

    print(
        f"device={device} train_segments={len(train_dataset)} "
        f"validation_segments={len(validation_source)} piano_models={train_dataset.piano_models} "
        f"stage={args.stage} detune={model.detuner.use_detune} "
        f"trainable_parameters={base_stage_metadata['trainable_parameter_count']}",
        flush=True,
    )
    for epoch in range(start_epoch, args.epochs):
        epoch_start_offset = resume_epoch_offset if epoch == start_epoch else 0
        if isinstance(curriculum_sampler, CoverageCurriculumSampler):
            curriculum_sampler.set_epoch(epoch, epoch_start_offset)
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        epoch_started = time.perf_counter()
        train_loss_sum = torch.zeros((), device=device)
        train_batches = 0
        epoch_examples = epoch_start_offset
        next_validation_example = (
            ((epoch_examples // args.validation_every_examples) + 1)
            * args.validation_every_examples
            if args.validation_every_examples
            else 0
        )
        for batch_index, batch in enumerate(train_loader):
            if args.steps_per_epoch and batch_index >= args.steps_per_epoch:
                break
            audio, conditioning, pedal, piano_model = move_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=amp_enabled) if device.type == "cuda" else autocast():
                signal, reverb_ir, dry_signal = training_model(
                    conditioning, pedal, piano_model
                )
                signal = signal[..., : audio.shape[-1]]
                dry_signal = dry_signal[..., : audio.shape[-1]]
                (
                    loss,
                    spectral_loss,
                    reverb_loss,
                    dry_loss,
                    energy_loss,
                    onset_loss,
                    centroid_loss,
                    tail_loss,
                ) = loss_fn.components(
                    signal,
                    audio,
                    reverb_ir,
                    dry_pred=dry_signal,
                    conditioning=conditioning,
                )
                velocity_loss = signal.new_zeros(())
                if (
                    args.velocity_loss_weight
                    and global_step % args.velocity_loss_every == 0
                ):
                    velocity_loss = velocity_counterfactual_loss(
                        model,
                        conditioning,
                        pedal,
                        piano_model,
                        combined=args.velocity_counterfactual_layout == "combined",
                        response_slopes=response_slopes,
                        response_frames=max(
                            1,
                            int(
                                round(
                                    config.frame_rate
                                    * args.velocity_response_ms
                                    / 1000.0
                                )
                            ),
                        ),
                    )
                    loss = loss + (
                        args.velocity_loss_weight
                        * float(loss_calibration.get("velocity_scale", 1.0))
                        * velocity_loss
                    )
            scaler.scale(loss).backward()
            if args.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()
            global_step += 1
            batch_examples = int(audio.shape[0])
            examples_seen += batch_examples
            epoch_examples += batch_examples
            train_loss_sum = train_loss_sum + loss.detach()
            train_batches += 1
            if global_step % args.log_every == 0:
                logged = torch.stack(
                    (
                        loss,
                        spectral_loss,
                        dry_loss,
                        reverb_loss,
                        energy_loss,
                        onset_loss,
                        centroid_loss,
                        tail_loss,
                        velocity_loss,
                    )
                ).detach().float().cpu().tolist()
                print(
                    f"epoch={epoch + 1}/{args.epochs} step={global_step} "
                    f"loss={logged[0]:.5f} wet={logged[1]:.5f} "
                    f"dry={logged[2]:.5f} reverb={logged[3]:.5f} "
                    f"energy={logged[4]:.5f} onset={logged[5]:.5f}",
                    f"centroid={logged[6]:.5f} tail={logged[7]:.5f} "
                    f"velocity={logged[8]:.5f}",
                    flush=True,
                )
                if writer is not None:
                    for name, value in zip(
                        (
                            "loss",
                            "spectral",
                            "dry",
                            "reverb",
                            "energy",
                            "onset",
                            "centroid",
                            "tail",
                            "velocity",
                        ),
                        logged,
                    ):
                        writer.add_scalar(f"train/{name}", value, global_step)
                    writer.add_scalar(
                        "train/learning_rate",
                        lr_scheduler.get_last_lr()[0],
                        global_step,
                    )
                    writer.add_scalar("train/examples_seen", examples_seen, global_step)
            if (
                args.validation_every_examples
                and epoch_examples >= next_validation_example
            ):
                validation_loss = run_validation(
                    model,
                    loss_fn,
                    validation_loader,
                    device,
                    amp_enabled,
                    args.validation_batches,
                    args.velocity_loss_weight,
                    args.velocity_loss_every,
                    float(loss_calibration.get("velocity_scale", 1.0)),
                    args.velocity_counterfactual_layout == "combined",
                    response_slopes,
                    max(
                        1,
                        int(
                            round(
                                config.frame_rate
                                * args.velocity_response_ms
                                / 1000.0
                            )
                        ),
                    ),
                )
                partial_performance = {
                    "epoch": epoch,
                    "steps": train_batches,
                    "seconds": time.perf_counter() - epoch_started,
                    "examples": epoch_examples,
                    "partial_epoch": True,
                    "peak_cuda_memory_bytes": (
                        int(torch.cuda.max_memory_allocated(device))
                        if device.type == "cuda"
                        else 0
                    ),
                }
                progress = stage_progress_metadata(
                    base_stage_metadata, epoch, epoch_examples, examples_seen
                )
                sampler_state = (
                    curriculum_sampler.state_dict(epoch_examples)
                    if isinstance(curriculum_sampler, CoverageCurriculumSampler)
                    else None
                )
                partial_metrics = {
                    "event": "validation",
                    "epoch": epoch,
                    "global_step": global_step,
                    "examples_seen": examples_seen,
                    "coverage_epoch_examples": epoch_examples,
                    "validation_loss": validation_loss,
                    "learning_rate": lr_scheduler.get_last_lr()[0],
                    "train_performance": partial_performance,
                }
                with metrics_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(partial_metrics) + "\n")
                print(json.dumps(partial_metrics), flush=True)
                if writer is not None:
                    writer.add_scalar("validation/loss", validation_loss, global_step)
                    writer.add_scalar(
                        "validation/coverage_passes",
                        progress["coverage_passes_completed"],
                        global_step,
                    )
                    writer.add_scalar(
                        "system/peak_cuda_memory_gib",
                        partial_performance["peak_cuda_memory_bytes"] / 2**30,
                        global_step,
                    )
                    writer.flush()
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
                        loss_calibration,
                        initialization,
                        validation_corpus_sha256,
                        examples_seen,
                        train_generator,
                        partial_performance,
                        lr_scheduler,
                        progress,
                        sampler_state,
                        False,
                    )
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
                    loss_calibration,
                    initialization,
                    validation_corpus_sha256,
                    examples_seen,
                    train_generator,
                    partial_performance,
                    lr_scheduler,
                    progress,
                    sampler_state,
                    False,
                )
                next_validation_example += args.validation_every_examples
                model.train()

        if not train_batches:
            raise RuntimeError("Training loader produced no batches")
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        train_seconds = time.perf_counter() - epoch_started
        mean_train_loss = float((train_loss_sum / train_batches).cpu())
        training_performance = {
            "epoch": epoch,
            "steps": train_batches,
            "seconds": train_seconds,
            "steps_per_second": train_batches / max(train_seconds, 1e-9),
            "examples_per_second": (
                (epoch_examples - epoch_start_offset) / max(train_seconds, 1e-9)
            ),
            "peak_cuda_memory_bytes": (
                int(torch.cuda.max_memory_allocated(device))
                if device.type == "cuda"
                else 0
            ),
        }
        should_validate = (
            epoch + 1 == args.epochs
            or (
                args.validation_every_epochs > 0
                and (epoch + 1) % args.validation_every_epochs == 0
            )
        )
        validation_loss = None
        validation_seconds = 0.0
        if should_validate:
            validation_started = time.perf_counter()
            validation_loss = run_validation(
                model,
                loss_fn,
                validation_loader,
                device,
                amp_enabled,
                args.validation_batches,
                args.velocity_loss_weight,
                args.velocity_loss_every,
                float(loss_calibration.get("velocity_scale", 1.0)),
                args.velocity_counterfactual_layout == "combined",
                response_slopes,
                max(1, int(round(config.frame_rate * args.velocity_response_ms / 1000.0))),
            )
            validation_seconds = time.perf_counter() - validation_started
        metrics = {
            "epoch": epoch,
            "global_step": global_step,
            "examples_seen": examples_seen,
            "train_loss": mean_train_loss,
            "validation_loss": validation_loss,
            "learning_rate": lr_scheduler.get_last_lr()[0],
            "train_performance": training_performance,
            "validation_seconds": validation_seconds,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(metrics) + "\n")
        print(json.dumps(metrics), flush=True)

        progress = stage_progress_metadata(
            base_stage_metadata, epoch, epoch_examples, examples_seen
        )
        if writer is not None:
            if validation_loss is not None:
                writer.add_scalar("validation/loss", validation_loss, global_step)
            writer.add_scalar(
                "validation/coverage_passes",
                progress["coverage_passes_completed"],
                global_step,
            )
            writer.add_scalar(
                "system/examples_per_second",
                training_performance["examples_per_second"],
                global_step,
            )
            writer.add_scalar(
                "system/peak_cuda_memory_gib",
                training_performance["peak_cuda_memory_bytes"] / 2**30,
                global_step,
            )
            writer.flush()
        sampler_state = (
            curriculum_sampler.state_dict(epoch_examples)
            if isinstance(curriculum_sampler, CoverageCurriculumSampler)
            else None
        )

        if validation_loss is not None and validation_loss < best_validation:
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
                loss_calibration,
                initialization,
                validation_corpus_sha256,
                examples_seen,
                train_generator,
                training_performance,
                lr_scheduler,
                progress,
                sampler_state,
                True,
            )
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
            loss_calibration,
            initialization,
            validation_corpus_sha256,
            examples_seen,
            train_generator,
            training_performance,
            lr_scheduler,
            progress,
            sampler_state,
            True,
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
                loss_calibration,
                initialization,
                validation_corpus_sha256,
                examples_seen,
                train_generator,
                training_performance,
                lr_scheduler,
                progress,
                sampler_state,
                True,
            )
        resume_epoch_offset = 0
    if writer is not None:
        writer.flush()
        writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
