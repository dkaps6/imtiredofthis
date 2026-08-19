"""Retired legacy ML placeholder.

The old implementation merely consumed an externally supplied ``p_ml`` and
returned 0.5 when absent. That is not a trained ML model and must not be used as
an independent ensemble voter. Use ``scripts.modeling.ml_v2`` instead.
"""


def run(*args, **kwargs):
    raise RuntimeError(
        "scripts.models.ml_ensemble is retired; use scripts.modeling.ml_v2 "
        "for leakage-safe trained supervised ML projections"
    )
