"""Utility helpers shared across the project."""

import os

# ---------------------------------------------------------------------------
# Environment-tag helper — controls per-environment data/output segregation
# ---------------------------------------------------------------------------

ENV_TAG: str = os.getenv("ENV_TAG", "").upper()


def env_suffix() -> str:
    """Return '_<TAG>' when ENV_TAG is set, otherwise empty string."""
    return f"_{ENV_TAG}" if ENV_TAG else ""
