# type: ignore[attr-defined]
"""Awesome `motor_defects_classification` project."""

from importlib import metadata as importlib_metadata


def get_version() -> str:  # noqa: D103
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


__version__: str = get_version()
