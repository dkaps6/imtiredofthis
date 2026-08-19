"""Retired legacy pseudo-Markov predictor.

This module never implemented a Markov chain: it multiplied adjusted attempts by
an efficiency mean, applied a Normal CDF, and silently returned 0.5 when inputs
were absent. Production state modeling now lives in ``scripts.modeling.state_v2``.
"""


def run(*args, **kwargs):
    raise RuntimeError(
        "scripts.models.markov is retired; use scripts.modeling.state_v2. "
        "Do not reconnect the legacy fixed-weight ensemble."
    )
