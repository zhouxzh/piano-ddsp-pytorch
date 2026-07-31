#!/usr/bin/env python3
"""Stage a verified model-suite release for Hugging Face upload."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = "model-suite-v1.0.1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_release(release_dir: Path) -> None:
    checksum_path = release_dir / "SHA256SUMS"
    if not checksum_path.is_file():
        raise FileNotFoundError(f"Missing release checksum file: {checksum_path}")
    for line in checksum_path.read_text(encoding="ascii").splitlines():
        expected, separator, relative_name = line.partition("  ")
        if not separator or not expected or not relative_name:
            raise ValueError(f"Invalid SHA256SUMS line: {line!r}")
        asset = release_dir / relative_name
        if not asset.is_file():
            raise FileNotFoundError(f"Missing release asset: {asset}")
        actual = sha256_file(asset)
        if actual != expected:
            raise ValueError(f"Checksum mismatch for {relative_name}: {actual} != {expected}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", required=True, help="HF namespace/repository")
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=Path("artifacts") / DEFAULT_RELEASE,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/hf-upload") / DEFAULT_RELEASE,
    )
    args = parser.parse_args()

    if "/" not in args.repo_id or args.repo_id.endswith("/"):
        raise ValueError("--repo-id must use NAMESPACE/REPOSITORY format")

    release_dir = args.release_dir.resolve()
    output_dir = args.output_dir.resolve()
    verify_release(release_dir)
    manifest = json.loads((release_dir / "model-suite.json").read_text(encoding="utf-8"))
    release = str(manifest["release"])
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Staging directory is not empty: {output_dir}. Use a new --output-dir."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    for source in release_dir.iterdir():
        if source.is_file():
            shutil.copy2(source, output_dir / source.name)

    model_card = (ROOT / "releases/huggingface-model-card.md").read_text(encoding="utf-8")
    model_card = model_card.replace("zhouxzh/piano-ddsp-ascend310", args.repo_id)
    model_card = model_card.replace(DEFAULT_RELEASE, release)
    (output_dir / "README.md").write_text(model_card, encoding="utf-8")
    license_dir = output_dir / "LICENSES"
    license_dir.mkdir(exist_ok=True)
    shutil.copy2(ROOT / "LICENSES/CC-BY-NC-SA-4.0.txt", output_dir / "LICENSE")
    shutil.copy2(
        ROOT / "LICENSES/Apache-2.0.txt", license_dir / "Apache-2.0.txt"
    )
    shutil.copy2(
        ROOT / "LICENSES/CC-BY-NC-SA-4.0.txt",
        license_dir / "CC-BY-NC-SA-4.0.txt",
    )
    shutil.copy2(ROOT / "THIRD_PARTY_NOTICES.md", output_dir / "THIRD_PARTY_NOTICES.md")

    staged_files = sorted(
        path
        for path in output_dir.rglob("*")
        if path.is_file() and path.name != "SHA256SUMS"
    )
    (output_dir / "SHA256SUMS").write_text(
        "".join(
            f"{sha256_file(path)}  {path.relative_to(output_dir).as_posix()}\n"
            for path in staged_files
        ),
        encoding="ascii",
    )

    print(f"Staged {args.repo_id} at {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
