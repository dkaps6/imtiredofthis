"""Canonical modeling layer for the v2 projection stack."""

from .bayesian_v2 import apply_bayesian_to_metrics, build_bayesian_baseline
from .contracts import ComponentPrediction, PlayerContext, PlayerProjectionInput, ProjectionDistribution, TeamContext
from .ensemble_v2 import apply_ensemble, fit_market_weights
from .ml_v2 import apply_ml_to_metrics, build_and_train
from .state_v2 import apply_state_to_metrics, build_state_predictions, train_state_model

__all__ = [
    "ComponentPrediction","PlayerContext","PlayerProjectionInput","ProjectionDistribution","TeamContext",
    "apply_bayesian_to_metrics","build_bayesian_baseline","apply_ml_to_metrics","build_and_train",
    "apply_state_to_metrics","build_state_predictions","train_state_model","apply_ensemble","fit_market_weights",
]
