"""Retired legacy ensemble.

The former implementation hard-coded 25/25/25/25 component weights and then
blended 65% model / 35% sportsbook market probability. Several component models
were placeholders, so this path could silently dilute valid predictions.

Use ``scripts.modeling.ensemble_v2``. Canonical ensemble weights must be learned
from out-of-sample walk-forward component predictions and sportsbook market
probabilities are not model inputs.
"""


def blend(*args, **kwargs):
    raise RuntimeError(
        "Legacy fixed-weight ensemble is retired. Use scripts.modeling.ensemble_v2 with calibrated OOS weights."
    )
