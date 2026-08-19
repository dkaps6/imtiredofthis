"""Canonical modeling layer for the v2 projection stack.

This package is the migration target for the older ``scripts.models`` and
``scripts.model`` implementations. Production wiring is introduced in small,
tested migrations so legacy behavior is preserved until parity is verified.
"""

from .bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from .contracts import (
    ComponentPrediction,
    PlayerContext,
    PlayerProjectionInput,
    ProjectionDistribution,
    TeamContext,
)

__all__ = [
    "ComponentPrediction",
    "PlayerContext",
    "PlayerProjectionInput",
    "ProjectionDistribution",
    "TeamContext",
    "apply_bayesian_to_metrics",
    "build_bayesian_baseline",
]
