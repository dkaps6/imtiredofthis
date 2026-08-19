"""Deprecated legacy Bayesian adapter.

The old implementation only applied a Normal CDF to pre-supplied ``mu``/``sd``
and therefore was not a hierarchical Bayesian model. Production Bayesian
shrinkage now lives in ``scripts.modeling.bayesian_v2`` and is applied before
rules + joint Monte Carlo.
"""
from .shared_types import Leg, LegResult


def run(leg: Leg) -> LegResult:
    raise RuntimeError(
        "scripts.models.bayes_hier is retired: use scripts.modeling.bayesian_v2 "
        "through the canonical production model stack. The legacy 25% ensemble "
        "must not silently treat this placeholder as an independent model."
    )
