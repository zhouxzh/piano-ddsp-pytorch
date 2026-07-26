"""Stable public version labels for exported DDSP-Piano models."""

from __future__ import annotations

from pathlib import Path


RELEASE_VERSION_BY_VARIANT = {
    "current": "v1",
    "v2": "v2",
    "v3": "v3-candidate",
}


def release_version_for_variant(model_variant: str) -> str:
    """Map an internal architecture name to its public release version."""
    try:
        return RELEASE_VERSION_BY_VARIANT[model_variant]
    except KeyError as error:
        raise ValueError(f"Unknown model variant: {model_variant!r}") from error


def model_output_label(model_path: Path, metadata: dict) -> str:
    """Return the stable version label, falling back to the ONNX file stem."""
    release_version = metadata.get("release_version")
    if isinstance(release_version, str) and release_version.strip():
        return release_version.strip()

    model_variant = metadata.get("model_variant")
    if model_variant in RELEASE_VERSION_BY_VARIANT:
        return RELEASE_VERSION_BY_VARIANT[model_variant]
    return model_path.stem
