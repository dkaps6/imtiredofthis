#!/usr/bin/env python3
"""Bridge live Sharp team data with explicit prior-season PBP tendencies.

This module exists for preseason / early-season situations where the active
season has no nflverse play-by-play yet. It does not silently relabel prior data
as current. Instead it fills only fields that are genuinely unavailable from the
live Sharp frame and writes provenance columns describing where the value came
from.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.runtime_context import resolve_prior_season, resolve_season
from scripts.utils.pbp import get_pbp

SHARP_PATH = Path("data/sharp_team_form.csv")


def _team_col(df: pd.DataFrame) -> str:
    for c in ("team_abbr", "team", "posteam"):
        if c in df.columns:
            return c
    raise RuntimeError("sharp_team_form.csv has no team/team_abbr column")


def _prior_proe(prior_season: int) -> pd.DataFrame:
    pbp = get_pbp(prior_season)
    if "posteam" not in pbp.columns:
        raise RuntimeError("Prior-season PBP missing posteam; cannot derive PROE prior")

    x = pbp.copy()
    x["team_abbr"] = x["posteam"].map(canon_team)
    x = x.loc[x["team_abbr"].astype(str).str.len().gt(0)].copy()

    pass_col = None
    for c in ("pass", "pass_attempt"):
        if c in x.columns:
            pass_col = c
            break
    if pass_col is None:
        raise RuntimeError("Prior-season PBP missing pass/pass_attempt indicator")

    x["actual_pass"] = pd.to_numeric(x[pass_col], errors="coerce")

    # nflfastR exposes xpass as expected pass probability. That lets us derive a
    # true play-level pass-rate-over-expected prior instead of using raw pass rate.
    if "xpass" in x.columns:
        x["xpass_num"] = pd.to_numeric(x["xpass"], errors="coerce")
        valid = x["actual_pass"].notna() & x["xpass_num"].notna()
        # Exclude obvious non-standard plays when the flags exist.
        if "qb_kneel" in x.columns:
            valid &= pd.to_numeric(x["qb_kneel"], errors="coerce").fillna(0).eq(0)
        if "qb_spike" in x.columns:
            valid &= pd.to_numeric(x["qb_spike"], errors="coerce").fillna(0).eq(0)
        y = x.loc[valid, ["team_abbr", "actual_pass", "xpass_num"]].copy()
        y["proe_play"] = y["actual_pass"] - y["xpass_num"]
        out = y.groupby("team_abbr", as_index=False).agg(
            pass_rate_over_expected_prior=("proe_play", "mean"),
            prior_proe_plays=("proe_play", "size"),
        )
    else:
        # Fallback only if the maintained PBP schema drops xpass. This is clearly
        # labeled as a league-centered pass-rate proxy rather than true xPass PROE.
        y = x.loc[x["actual_pass"].notna(), ["team_abbr", "actual_pass"]].copy()
        league = float(y["actual_pass"].mean())
        out = y.groupby("team_abbr", as_index=False).agg(
            raw_pass_rate=("actual_pass", "mean"),
            prior_proe_plays=("actual_pass", "size"),
        )
        out["pass_rate_over_expected_prior"] = out["raw_pass_rate"] - league
        out.drop(columns=["raw_pass_rate"], inplace=True)

    if out.empty or out["team_abbr"].nunique() < 30:
        raise RuntimeError(
            f"Prior-season PROE derivation produced only {out['team_abbr'].nunique() if not out.empty else 0} teams"
        )
    out["prior_season"] = int(prior_season)
    return out


def bridge(active_season: int, prior_season: int) -> pd.DataFrame:
    if not SHARP_PATH.exists() or SHARP_PATH.stat().st_size == 0:
        raise RuntimeError(f"Required Sharp artifact missing/empty: {SHARP_PATH}")

    sharp = pd.read_csv(SHARP_PATH)
    sharp.columns = [str(c).strip() for c in sharp.columns]
    tcol = _team_col(sharp)
    sharp["team_abbr"] = sharp[tcol].map(canon_team)

    priors = _prior_proe(prior_season)
    sharp = sharp.merge(priors, on="team_abbr", how="left")

    if "pass_rate_over_expected" not in sharp.columns:
        sharp["pass_rate_over_expected"] = np.nan

    current = pd.to_numeric(sharp["pass_rate_over_expected"], errors="coerce")
    prior = pd.to_numeric(sharp["pass_rate_over_expected_prior"], errors="coerce")
    use_prior = current.isna()
    sharp["pass_rate_over_expected"] = current.where(~use_prior, prior)
    sharp["pass_rate_over_expected_source"] = np.where(
        use_prior,
        f"prior_{prior_season}_pbp",
        f"sharp_{active_season}",
    )
    sharp["team_form_active_season"] = int(active_season)
    sharp["team_form_prior_season"] = int(prior_season)

    missing = sharp.loc[sharp["pass_rate_over_expected"].isna(), "team_abbr"].dropna().unique().tolist()
    if missing:
        raise RuntimeError(f"PROE still unresolved after prior bridge for teams={missing}")

    sharp.to_csv(SHARP_PATH, index=False)
    debug = Path("data/_debug/team_form_prior_bridge.csv")
    debug.parent.mkdir(parents=True, exist_ok=True)
    sharp[[c for c in (
        "team_abbr",
        "pass_rate_over_expected",
        "pass_rate_over_expected_prior",
        "pass_rate_over_expected_source",
        "prior_proe_plays",
        "team_form_active_season",
        "team_form_prior_season",
    ) if c in sharp.columns]].to_csv(debug, index=False)

    print(
        f"[team_form_prior_bridge] rows={len(sharp)} active={active_season} prior={prior_season} "
        f"prior_filled={int(use_prior.sum())}"
    )
    return sharp


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=None)
    parser.add_argument("--prior-season", type=int, default=None)
    args = parser.parse_args()
    active = int(args.season if args.season is not None else resolve_season())
    prior = int(args.prior_season if args.prior_season is not None else resolve_prior_season())
    if prior >= active:
        raise RuntimeError(f"prior season {prior} must be less than active season {active}")
    bridge(active, prior)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
