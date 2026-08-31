"""M95H mechanical compatibility wrapper.

Pandas 1.5 Index does not expose .ne(); the base evaluator used it only for
index inequality while building within-team competitor sets. Patch that method
to the equivalent vectorized != operation. No model, feature, split, selection,
or validation logic changes.
"""
from __future__ import annotations

import pandas as pd

if not hasattr(pd.Index, "ne"):
    pd.Index.ne = lambda self, other: self != other  # type: ignore[attr-defined]

import scripts.backtest.evaluate_rb_lead_role_entitlement as h

if __name__ == "__main__":
    raise SystemExit(h.main())
