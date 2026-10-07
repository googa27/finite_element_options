"""Fail-closed pricing-engine identity and validation-artifact admission."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

_FORBIDDEN_ENGINE_TOKENS = ("toy", "synthetic", "polynomial", "fixture")


def _validate_heston_engine_name(pricing_engine: str) -> str:
    normalized = pricing_engine.strip()
    if not normalized:
        raise ValueError("pricing_engine must name a validated Heston pricing engine")
    lowered = normalized.lower()
    if "heston" not in lowered or any(
        token in lowered for token in _FORBIDDEN_ENGINE_TOKENS
    ):
        raise ValueError("pricing_engine must name a validated Heston pricing engine")
    return normalized


def _validated_artifact_sha256(artifact: str, expected_sha256: str) -> str:
    """Load a validation artifact and verify its content-addressed digest."""

    artifact_path = Path(artifact).expanduser()
    if not artifact_path.is_file():
        raise ValueError(
            "pricing_engine_validation validation_artifact must be an existing file"
        )
    actual_sha256 = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if actual_sha256 != expected_sha256:
        raise ValueError(
            "pricing_engine_validation validation_artifact_sha256 does not match artifact"
        )
    return actual_sha256


def _validate_heston_engine_metadata(
    pricing_engine: str,
    pricing_engine_validation: Mapping[str, object],
) -> dict[str, object]:
    """Validate non-lexical evidence for a Heston pricing engine."""

    engine = _validate_heston_engine_name(pricing_engine)
    metadata = dict(pricing_engine_validation)
    if metadata.get("validated") is not True:
        raise ValueError("pricing_engine_validation must mark the engine as validated")
    if str(metadata.get("engine_family", "")).lower() != "heston":
        raise ValueError(
            "pricing_engine_validation must declare engine_family='heston'"
        )
    artifact = str(metadata.get("validation_artifact", "")).strip()
    if not artifact:
        raise ValueError("pricing_engine_validation must include a validation_artifact")
    artifact_sha256 = (
        str(metadata.get("validation_artifact_sha256", "")).strip().lower()
    )
    if len(artifact_sha256) != 64 or any(
        ch not in "0123456789abcdef" for ch in artifact_sha256
    ):
        raise ValueError(
            "pricing_engine_validation must include validation_artifact_sha256"
        )
    artifact_sha256 = _validated_artifact_sha256(artifact, artifact_sha256)
    version = str(metadata.get("version", "")).strip()
    if not version:
        raise ValueError(
            "pricing_engine_validation must include a pricing engine version"
        )
    metadata["pricing_engine"] = engine
    metadata["validation_artifact"] = artifact
    metadata["validation_artifact_sha256"] = artifact_sha256
    metadata["version"] = version
    return metadata
