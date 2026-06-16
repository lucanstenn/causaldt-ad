from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import msgspec

from causaldt_ad.drawings.schema import Config

REGIME_ROOT = Path(__file__).resolve().parents[2] / "regimes"


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as handle:
        return tomllib.load(handle)


def _resolve_path(reference: str, base: Path) -> Path:
    candidate = Path(reference)
    if candidate.suffix != ".toml":
        candidate = candidate.with_suffix(".toml")
    if candidate.is_absolute() and candidate.exists():
        return candidate
    local = (base / candidate).resolve()
    if local.exists():
        return local
    return (REGIME_ROOT / candidate.name).resolve()


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        existing = merged.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged[key] = _deep_merge(existing, value)
        else:
            merged[key] = value
    return merged


def _load_layers(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    if path in seen:
        raise ValueError(f"cyclic extends chain at {path.name}")
    raw = _read_toml(path)
    parent_ref = raw.pop("extends", None)
    if parent_ref is None:
        return raw
    parent_path = _resolve_path(str(parent_ref), path.parent)
    parent = _load_layers(parent_path, seen | {path})
    return _deep_merge(parent, raw)


def _coerce(token: str) -> Any:
    lowered = token.lower()
    if lowered in {"true", "false"}:
        return lowered == "true"
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def _apply_override(tree: dict[str, Any], dotted: str) -> None:
    key, _, value = dotted.partition("=")
    if not _:
        raise ValueError(f"override must be key=value, got {dotted!r}")
    parts = key.split(".")
    cursor = tree
    for part in parts[:-1]:
        nxt = cursor.get(part)
        if not isinstance(nxt, dict):
            nxt = {}
            cursor[part] = nxt
        cursor = nxt
    cursor[parts[-1]] = _coerce(value)


def load(reference: str, overrides: tuple[str, ...] = ()) -> Config:
    path = _resolve_path(reference, REGIME_ROOT)
    tree = _load_layers(path, frozenset())
    for dotted in overrides:
        _apply_override(tree, dotted)
    return msgspec.convert(tree, Config)
