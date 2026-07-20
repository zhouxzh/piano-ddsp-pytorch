#!/usr/bin/env python3
"""Report the PyTorch CUDA environment and the lightweight package prerequisites."""

from __future__ import annotations

import importlib

import torch


def main() -> int:
    print(f"torch={torch.__version__}")
    print(f"cuda_available={torch.cuda.is_available()}")
    print(f"torch_cuda={torch.version.cuda}")
    if torch.cuda.is_available():
        print(f"gpu={torch.cuda.get_device_name(0)}")
    for name in ("torchaudio", "mido", "numpy", "tqdm"):
        module = importlib.import_module(name)
        print(f"{name}={getattr(module, '__version__', 'installed')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
