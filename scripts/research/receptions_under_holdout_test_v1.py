#!/usr/bin/env python3
"""Genuine holdout test of the receptions-UNDER candidate from
SITUATIONAL_EDGE_HUNT_V1/V2.

The V1/V2 "positive in both seasons" check is weaker than it looks: the
prob_edge quartile cutoff that DEFINES the candidate was computed on the
POOLED 2024+2025 sample, then each season was checked individually for
consistency. That is not a prospective holdout -- 2025 data influenced the
threshold used to grade 2025.

This script fits the top-quartile prob_edge cutoff on ONE season only,
freezes it, and grades the OTHER season blind against that frozen threshold
-- in both directions, as a symmetry check. This is a stricter test and is
expected to show a smaller, noisier edge than the pooled estimate. Research
only; no production change.
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

DETAIL = Path("docs/research/overnight/non_qb_detail_wr_r15_te_r5p_applied.csv")
OUT = Path("docs/research/overnight/RECEPTIONS_UNDER_HOLDOUT_TEST_V1_RESULT.md")


def frozen_threshold_test(rec: pd.DataFrame, fit_season: int, test_season: int) -> dict:
    fit = rec.loc[rec.season.eq(fit_season)]
    cutoff = float(fit["prob_edge"].clip(lower=0).quantile(0.75))
    test = rec.loc[rec.season.eq(test_season) & rec["prob_edge"].ge(cutoff) & rec.side.eq("UNDER")]
    n = int(len(test))
    roi = float(test.unit_result.mean()) if n else float("nan")
    wr = float((test.bet_result == "WIN").mean()) if n else float("nan")
    se = (0.5 * 0.5 / n) ** 0.5 if n else float("nan")
    return {"fit_season": fit_season, "test_season": test_season, "cutoff": cutoff, "n": n, "win_rate": wr, "roi_per_unit": roi, "win_rate_se_vs_50pct": se}


def main() -> int:
    df = pd.read_csv(DETAIL, low_memory=False)
    rec = df.loc[df.market.eq("receptions") & df.signal.eq("STRONG_EDGE")].copy()

    results = [
        frozen_threshold_test(rec, 2024, 2025),
        frozen_threshold_test(rec, 2025, 2024),
    ]
    out_df = pd.DataFrame(results)
    print(out_df.to_string(index=False))

    lines = []
    lines.append("STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.\n")
    lines.append("# Receptions-UNDER Candidate — Genuine Holdout Test\n")
    lines.append("Direct follow-up to SITUATIONAL_EDGE_HUNT_V1/V2. Those results defined the")
    lines.append("top-quartile prob_edge threshold on the POOLED 2024+2025 sample, then checked")
    lines.append("each season individually -- 2025 data influenced the threshold used to grade")
    lines.append("2025, so that is not a true prospective holdout. This test fits the threshold")
    lines.append("on ONE season only and grades the OTHER season blind against the frozen cutoff,")
    lines.append("in both directions.\n")
    lines.append(out_df.to_markdown(index=False))
    lines.append("")
    lines.append("## Reading")
    lines.append("The edge is directionally consistent (win rate ~52.7-52.9% both directions,")
    lines.append("both ROI positive) but materially weaker than the pooled estimate (+1.50%")
    lines.append("pooled ROI in V2 vs +0.61%/+1.59% under genuine holdout here). At n=577/543,")
    lines.append("a ~52.7% win rate has a binomial standard error of ~2.1 percentage points against")
    lines.append("a 50% null -- this is NOT a result you can call statistically distinguishable")
    lines.append("from noise on win rate alone; the positive ROI leans partly on the odds mix, not")
    lines.append("just the hit rate. This downgrades the candidate from \"validated, modest edge\"")
    lines.append("to \"directionally plausible, unconfirmed at this sample size.\" Do not promote")
    lines.append("or bet on this without more data (a genuine third season, or continued")
    lines.append("2026-forward tracking) to grow the holdout sample.")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
