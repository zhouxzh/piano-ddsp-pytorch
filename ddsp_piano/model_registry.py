"""Stable model identities and release asset contracts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = Path(__file__).with_name("model-suite-v1.0.1.json")


@dataclass(frozen=True)
class ModelSpec:
    model_id: str
    display_name: str
    description: str
    asset_basename: str
    architecture: str
    lineage: str
    quality_status: str
    model: Mapping[str, Any]
    training: Mapping[str, Any]

    def asset_path(self, root: Path, suffix: str) -> Path:
        return root / f"{self.asset_basename}{suffix}"


class ModelRegistry:
    def __init__(self, payload: Mapping[str, Any], source: Path) -> None:
        if payload.get("schema") != "ddsp-piano-model-suite/v1":
            raise ValueError(f"Unsupported model registry schema in {source}")
        self.source = source
        self.release = str(payload["release"])
        self.deployment_contract = dict(payload["deployment_contract"])
        self.legacy_names = dict(payload.get("legacy_names", {}))
        self.models = {
            model_id: ModelSpec(model_id=model_id, **values)
            for model_id, values in payload["models"].items()
        }
        if len({spec.asset_basename for spec in self.models.values()}) != len(self.models):
            raise ValueError("Model asset basenames must be unique")
        self.default_model_id = str(
            payload.get("default_model_id", next(iter(self.models)))
        )
        if self.default_model_id not in self.models:
            raise ValueError(
                f"Default model ID {self.default_model_id!r} is not in the registry"
            )

    def require(self, model_id: str) -> ModelSpec:
        if model_id in self.legacy_names:
            replacement = self.legacy_names[model_id]
            raise ValueError(
                f"Legacy model name {model_id!r} is no longer accepted; "
                f"use {replacement!r}"
            )
        try:
            return self.models[model_id]
        except KeyError as error:
            choices = ", ".join(self.models)
            raise ValueError(f"Unknown model ID {model_id!r}; choose one of: {choices}") from error


def load_model_registry(path: Path = DEFAULT_REGISTRY_PATH) -> ModelRegistry:
    resolved = path.resolve()
    return ModelRegistry(json.loads(resolved.read_text(encoding="utf-8")), resolved)


def model_output_label(model_path: Path, metadata: Mapping[str, Any]) -> str:
    """Return a public model ID, falling back only for non-release diagnostics."""
    model_id = metadata.get("model_id")
    if isinstance(model_id, str) and model_id.strip():
        return model_id.strip()
    return model_path.stem
