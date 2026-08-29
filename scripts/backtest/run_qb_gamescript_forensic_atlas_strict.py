#!/usr/bin/env python3
"""Strict M69 entrypoint.

The forensic atlas module supports fallbacks for general diagnostics. M69's
frozen matchup-conditioning target is stricter: opening deviation must be
measured against verified playcaller prior first-15 history only. Rows lacking
that history remain missing for the opening-deviation screen.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import scripts.backtest.audit_qb_gamescript_forensic_atlas as m


def strict_playcaller_baseline(row, *cols):
    col = "playcaller_opening_first15_dbr_mean8"
    v = pd.to_numeric(pd.Series([row.get(col, np.nan)]), errors="coerce").iloc[0]
    return float(v) if np.isfinite(v) else np.nan


m.choose_baseline = strict_playcaller_baseline

if __name__ == "__main__":
    raise SystemExit(m.main())
