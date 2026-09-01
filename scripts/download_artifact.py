#!/usr/bin/env python3
"""Download and validate published selector artifacts from a GitHub release.

Downloads occur only when this script is explicitly called. The library itself
never contacts the network when an artifact is missing.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST_PATH = PROJECT_ROOT / "artifacts" / "manifest.json"


class ArtifactDownloadError(RuntimeError):
    """Raised when a published artifact cannot be downloaded or verified."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--domain",
        choices=("eye_tracking", "traffic", "weather"),
        help="Download one domain selector.",
    )
    selection.add_argument("--all", action="store_true", help="Download all published selectors.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing local selector artifact.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=DEFAULT_MANIFEST_PATH,
        help=argparse.SUPPRESS,
    )
    return parser.parse_args()


def load_manifest(path: Path) -> dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise ArtifactDownloadError(f"Artifact manifest does not exist: {path}") from error
    except json.JSONDecodeError as error:
        raise ArtifactDownloadError(f"Artifact manifest is not valid JSON: {path}") from error

    if manifest.get("schema_version") != 1 or not isinstance(manifest.get("artifacts"), dict):
        raise ArtifactDownloadError("Artifact manifest has an unsupported schema.")
    return manifest


def sha256_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def checked_destination(relative_path: str) -> Path:
    project_root = PROJECT_ROOT.resolve()
    destination = (project_root / relative_path).resolve()
    try:
        destination.relative_to(project_root)
    except ValueError as error:
        raise ArtifactDownloadError(f"Artifact destination is outside the project: {relative_path}") from error
    return destination


def verify_file(path: Path, specification: dict[str, Any]) -> None:
    observed_hash, observed_size = sha256_and_size(path)
    expected_hash = specification.get("sha256")
    expected_size = specification.get("size_bytes")
    if observed_size != expected_size:
        raise ArtifactDownloadError(
            f"Downloaded {specification.get('asset_name')} has {observed_size} bytes; "
            f"expected {expected_size}."
        )
    if observed_hash != expected_hash:
        raise ArtifactDownloadError(
            f"SHA-256 mismatch for {specification.get('asset_name')}; the file was not installed."
        )


def validate_selector(path: Path, domain: str) -> None:
    source_dir = PROJECT_ROOT / "src"
    if str(source_dir) not in sys.path:
        sys.path.insert(0, str(source_dir))
    try:
        from gap_imputation_benchmark.artifacts.selector import load_selector_artifact
    except (ImportError, ModuleNotFoundError) as error:
        raise ArtifactDownloadError(
            "The artifact was downloaded but cannot be validated because its Python dependencies "
            "are missing. Install the project requirements, then run this command again."
        ) from error
    try:
        load_selector_artifact(path, expected_domain=domain)
    except Exception as error:
        raise ArtifactDownloadError(
            f"Downloaded {path.name} failed selector compatibility validation for {domain}."
        ) from error


def download_one(domain: str, specification: dict[str, Any], *, force: bool) -> Path:
    required = {"asset_name", "url", "destination", "size_bytes", "sha256"}
    missing = sorted(required - set(specification))
    if missing:
        raise ArtifactDownloadError(f"Manifest entry for {domain} is missing: {missing}")

    destination = checked_destination(str(specification["destination"]))
    if destination.exists() and not force:
        verify_file(destination, specification)
        validate_selector(destination, domain)
        print(f"{domain}: already installed and verified at {destination.relative_to(PROJECT_ROOT.resolve())}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{destination.name}.", suffix=".part", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            try:
                with urlopen(str(specification["url"])) as response:
                    shutil.copyfileobj(response, temporary)
            except URLError as error:
                raise ArtifactDownloadError(
                    f"Could not download {specification['asset_name']} from {specification['url']}: {error.reason}"
                ) from error

        verify_file(temporary_path, specification)
        validate_selector(temporary_path, domain)
        temporary_path.replace(destination)
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)

    print(f"{domain}: installed and verified at {destination.relative_to(PROJECT_ROOT.resolve())}")
    return destination


def main() -> None:
    args = parse_args()
    manifest = load_manifest(args.manifest)
    artifacts = manifest["artifacts"]
    domains = sorted(artifacts) if args.all else [args.domain]
    for domain in domains:
        if domain not in artifacts:
            raise ArtifactDownloadError(f"No published selector is configured for domain: {domain}")
        download_one(domain, artifacts[domain], force=args.force)


if __name__ == "__main__":
    try:
        main()
    except ArtifactDownloadError as error:
        raise SystemExit(f"Artifact download failed: {error}") from error
