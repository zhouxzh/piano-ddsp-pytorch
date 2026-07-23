#!/usr/bin/env python3
"""Report the PyTorch CUDA environment and the lightweight package prerequisites."""

from __future__ import annotations

import importlib
import shutil
import sys

import torch


def main() -> int:
    failures = 0
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
        capability = torch.cuda.get_device_capability(0)
        print(f"compute_capability={capability[0]}.{capability[1]}")
    for name in ("torchaudio", "mido", "numpy", "tqdm", "onnx", "onnxruntime"):
        try:
            module = importlib.import_module(name)
        except ImportError as exc:
            print(f"{name}=MISSING ({exc})")
            failures += 1
        else:
            print(f"{name}={getattr(module, '__version__', 'installed')}")
    try:
        module = importlib.import_module("torchcodec")
    except Exception as exc:
        print(f"torchcodec=MISSING ({exc})")
        failures += 1
    else:
        print(f"torchcodec={getattr(module, '__version__', 'installed')}")
    ffmpeg = shutil.which("ffmpeg")
    print(f"ffmpeg={ffmpeg or 'MISSING'}")
    if ffmpeg is None:
        failures += 1
    hf = shutil.which("hf")
    print(f"hf={hf or 'MISSING'}")
    if hf is None:
        failures += 1
    if not torch.cuda.is_available():
        print("ERROR: CUDA is required for practical training", file=sys.stderr)
        failures += 1
    return int(failures > 0)


if __name__ == "__main__":
    raise SystemExit(main())
