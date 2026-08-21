#!/usr/bin/env python3
"""Migration 37: focused WR hierarchy boundary calibration.

Reuses the Migration 36 leakage-safe canonical calibration harness but narrows
candidate search around the winning rank_strong region and slightly beyond it.
No production football logic is changed.
"""
from scripts.backtest import calibrate_wr_target_hierarchy as base

base.CANDIDATES = {
    "current": ("power", 1.00),
    "rank_medium": ("rank", (1.25, 1.10, 0.97, 0.88)),
    "rank_midstrong": ("rank", (1.30, 1.11, 0.955, 0.85)),
    "rank_strong": ("rank", (1.35, 1.12, 0.94, 0.82)),
    "rank_stronger_1": ("rank", (1.40, 1.14, 0.91, 0.78)),
    "rank_stronger_2": ("rank", (1.45, 1.16, 0.88, 0.74)),
    "rank_stronger_3": ("rank", (1.50, 1.18, 0.85, 0.70)),
}

if __name__ == "__main__":
    raise SystemExit(base.main())
