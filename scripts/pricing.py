#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Price player props using model multipliers + market anchoring and write outputs/props_priced_clean.csv.

Adds:
- Weather merge + weather multiplier
- Strict 2025 handling via merged dataframes from metrics

If any optional file is missing, pipeline continues with safe defaults.
"""

import argparse
import os
import sys
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from math import erf, sqrt


OUT_DIR = "outputs"
OUT_FILE = os.path.join(OUT_DIR, "props_priced_clean.csv")

TEAM_FORM = "metrics/team_form.csv"
PLAYER_FORM = "metrics/player_form.csv"
COVERAGE = "data/coverage.csv"
CB_ASSIGN = "data/cb_assignments.csv"
INJURIES = "data/injuries.csv"
ROLES = "data/roles.csv"
GAME_LINES = "outputs/game_lines.csv"
WEATHER = "data/weather.csv"

PROP_CANDIDATES = [
    "outputs/props_raw.csv",
    "data/props_raw.csv",
    "outputs/props_aggregated.csv",
]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _maybe_csv(path: str) -> pd.DataFrame:
    try:
        df = pd.read_csv(path)
        return df
    except Exception:
        return pd.DataFrame()


def _load_props() -> pd.DataFrame:
    for p in PROP_CANDIDATES:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                return df
            except Exception:
                pass
    return pd.DataFrame()


def _load_weather() -> pd.DataFrame:
    try:
        wx = pd.read_csv(WEATHER)
        wx.columns = [c.lower() for c in wx.columns]
        cols = [c for c in ("event_id", "wind_mph", "temp_f", "precip") if c in wx.columns]
        if not cols:
            return pd.DataFrame(columns=["event_id", "wind_mph", "temp_f", "precip"])
        return wx[["event_id", "wind_mph", "temp_f", "precip"]].copy()
    except Exception:
        return pd.DataFrame(columns=["event_id", "wind_mph", "temp_f", "precip"])


def _inv_norm_cdf(p: float) -> float:
    """Approx inverse CDF for Normal via erfinv-like approach."""
    # Clamp p to (0,1)
    p = min(max(p, 1e-9), 1 - 1e-9)
    # Rational approximation (Beasley-Springer/Moro or simpler).
    # Use scipy if available; here we keep it dependency-free.
    # For pricing we mostly need CDF rather than inv; keep this if you back out mean from line & prob.
    # Placeholder:
    return sqrt(2) * 0.5 * np.log(p / (1 - p))


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + erf(x / sqrt(2.0)))


def _prob_over_at_line(mu: float, sigma: float, line: float) -> float:
    if sigma <= 0:
        return float(mu > line)
    z = (line - mu) / sigma
    return 1.0 - _norm_cdf(z)


def _devig_two_way(p_over: float, p_under: float) -> float:
    """Return fair over prob from two vigged sides (simple normalization)."""
    s = p_over + p_under
    if s <= 0:
        return np.nan
    return p_over / s


def _implied_prob_from_american(odds: float) -> float:
    try:
        o = float(odds)
    except Exception:
        return np.nan
    if o > 0:
        return 100.0 / (o + 100.0)
    else:
        return (-o) / ((-o) + 100.0)


def _weather_multiplier(wind, precip, market_key: str) -> float:
    m = 1.0
    try:
        wind = float(wind)
    except Exception:
        wind = np.nan

    mk = str(market_key or "").lower()
    pc = str(precip or "").lower()

    # wind: hurt deep passing and WR YPR/Yards
    if not pd.isna(wind) and wind >= 15:
        if mk in {"pass_yards", "rec_yards", "rush_rec_yards"}:
            m *= 0.94

    # precip: dampen YAC; nudge run
    if pc in {"rain", "snow"}:
        if mk in {"rec_yards", "rush_rec_yards"}:
            m *= 0.97
        if mk in {"rush_yards", "rush_att"}:
            m *= 1.02

    return m


def _apply_multipliers(row: pd.Series, mk: str, mu_base: float, sigma_base: float) -> (float, float):
    """
    All multiplicative effects roll into mu; volatility flags widen sigma modestly.
    This is intentionally conservative to avoid nuking your prior behavior.
    """
    mu_mult = 1.0
    sig_mult = 1.0

    # --- Opponent pressure / sack elasticity ---
    pr_z = row.get("def_pressure_rate_z")
    if pd.notna(pr_z):
        mu_mult *= (1.0 - 0.35 * float(pr_z)) if mk in {"pass_yards"} else 1.0
        # volatility widening on pressure mismatch:
        sig_mult *= (1.0 + 0.10 * max(0.0, float(pr_z)))

    sack_z = row.get("def_sack_rate_z")
    if pd.notna(sack_z) and mk in {"pass_yards"}:
        mu_mult *= (1.0 - 0.15 * float(sack_z))

    # --- EPA funnel shifts (very light-touch when only one leg present) ---
    pass_epa_z = row.get("def_pass_epa_z")
    rush_epa_z = row.get("def_rush_epa_z")

    if mk in {"rush_yards", "rush_att"} and pd.notna(rush_epa_z) and pd.notna(pass_epa_z):
        # pass-funnel (bad vs pass, good vs run) -> small nudge down for rush volume
        if float(pass_epa_z) >= 0.6 and float(rush_epa_z) <= -0.4:
            mu_mult *= 0.97
        # run-funnel (bad vs run, good vs pass) -> small nudge up
        if float(rush_epa_z) >= 0.6 and float(pass_epa_z) <= -0.4:
            mu_mult *= 1.03

    if mk in {"pass_yards", "rec_yards"} and pd.notna(pass_epa_z):
        mu_mult *= (1.0 - 0.05 * max(0.0, float(-pass_epa_z)))  # slightly harder vs elite pass D

    # --- Box-count leverage (RB efficiency) ---
    light_box_z = row.get("light_box_rate_z")
    heavy_box_z = row.get("heavy_box_rate_z")
    if mk in {"rush_yards"}:
        if pd.notna(light_box_z) and float(light_box_z) >= 0.6:
            mu_mult *= 1.07
        if pd.notna(heavy_box_z) and float(heavy_box_z) >= 0.6:
            mu_mult *= 0.94

    # --- Pace smoothing already in team_form via z; tiny plays multiplier if available ---
    pace_z = row.get("pace_neutral_z") or row.get("pace_z")
    if pd.notna(pace_z):
        mu_mult *= (1.0 + 0.5 * float(pace_z) * 0.02)  # very modest

    # --- Game script (win prob) primary RB escalator ---
    # rows need either team alignment or a direct column with win prob; we try generic:
    wp = row.get("team_win_prob")
    if pd.notna(wp):
        try:
            wpf = float(wp)
            if mk in {"rush_att", "rush_yards"} and wpf >= 0.55:
                mu_mult *= 1.03
            if mk in {"pass_yards"} and wpf >= 0.55:
                mu_mult *= 0.98
        except Exception:
            pass

    # --- Coverage/CB / slot/TE boosts (if your pipeline adds flags) ---
    tag = str(row.get("coverage_tag") or "").lower()
    if tag:
        if mk in {"rec_yards"}:
            if tag in {"top_shadow", "heavy_man"}:
                mu_mult *= 0.92
            if tag in {"heavy_zone"}:
                mu_mult *= 1.04

    cb_pen = row.get("cb_penalty")
    if pd.notna(cb_pen) and mk in {"rec_yards"}:
        try:
            mu_mult *= (1.0 - float(cb_pen))
        except Exception:
            pass

    # --- Air-yards sanity (cap WR1 YPR in low AY/Att environments) ---
    ay_z = row.get("ay_per_att_z")
    role = str(row.get("role") or "").upper()
    if mk in {"rec_yards"} and role in {"WR1", "WR2"} and pd.notna(ay_z):
        if float(ay_z) <= -0.8:  # bottom quintile-ish
            mu_mult *= 0.92

    # --- Weather multiplier (added) ---
    mu_mult *= _weather_multiplier(row.get("wind_mph"), row.get("precip"), mk)

    # --- Final μ, σ ---
    mu = mu_base * mu_mult
    sigma = sigma_base * sig_mult
    return mu, sigma


def _blend_probs(p_model: float, p_market_fair: float) -> float:
    if pd.isna(p_market_fair):
        return p_model
    return 0.65 * p_model + 0.35 * p_market_fair


def _fair_odds_from_prob(p: float) -> float:
    p = min(max(p, 1e-9), 1 - 1e-9)
    if p >= 0.5:
        # negative American
        return - (p * 100.0) / (1.0 - p)
    else:
        return (1.0 - p) * 100.0 / p


def _default_sigma_for_market(market: str) -> float:
    mk = str(market).lower()
    if mk in {"rec_yards"}:
        return 26.0
    if mk in {"receptions"}:
        return 1.8
    if mk in {"rush_yards"}:
        return 23.0
    if mk in {"pass_yards"}:
        return 48.0
    if mk in {"rush_att"}:
        return 3.2
    if mk in {"rush_rec_yards"}:
        return 32.0
    return 20.0


def price(season: int):
    _ensure_dir(OUT_DIR)

    # Base frames
    props = _load_props()
    team = _maybe_csv(TEAM_FORM)
    player = _maybe_csv(PLAYER_FORM)
    cov = _maybe_csv(COVERAGE)
    cba = _maybe_csv(CB_ASSIGN)
    inj = _maybe_csv(INJURIES)
    roles = _maybe_csv(ROLES)
    lines = _maybe_csv(GAME_LINES)
    wx = _load_weather()

    # Normalize columns
    for df in (props, team, player, cov, cba, inj, roles, lines, wx):
        if not df.empty:
            df.columns = [c.strip().lower() for c in df.columns]

    # Clamp season in team/player
    if "season" in team.columns:
        team = team[team["season"].astype(int) == 2025].copy()
    if "season" in player.columns:
        player = player[player["season"].astype(int) == 2025].copy()

    if props.empty:
        print("[pricing] No props file found. Expected one of:", PROP_CANDIDATES, file=sys.stderr)
        # write empty output to keep pipeline predictable
        pd.DataFrame().to_csv(OUT_FILE, index=False)
        return

    # Merge contextuals
    df = props.copy()

    # Optional: standardize keys
    # expected columns in props: player, team, opponent, event_id, market, line, over_odds, under_odds
    for need in ("player", "team", "opponent", "market", "line"):
        if need not in df.columns:
            df[need] = pd.NA

    # Join roles/injuries/player_form
    if not roles.empty:
        df = df.merge(roles, on=["player", "team"], how="left")
    if not inj.empty:
        df = df.merge(inj[["player", "status"]], on="player", how="left")
        # example: clamp WR1 if "Limited" etc. (kept simple here)

    if not player.empty:
        # Bring through anything useful like priors (yprr/ypc etc.) if present; left join on player/team
        df = df.merge(player.drop_duplicates(subset=["player", "team"]), on=["player", "team"], how="left")

    # Join team form by DEF side using opponent mapping if your props rows are offense-centric
    # If your props already carry 'defense_team' column use that instead of opponent
    if not team.empty:
        # opponent defense context
        opp = team.add_prefix("opp_")
        if "opponent" in df.columns and "opp_team" in opp.columns:
            df = df.merge(
                opp,
                left_on="opponent",
                right_on="opp_team",
                how="left",
                suffixes=("", "")
            )

        # own offense context (for plays_est/proe if you want)
        if "team" in df.columns and "team" in team.columns:
            df = df.merge(
                team[["team", "plays_est", "proe"]],
                on="team",
                how="left"
            )

    # coverage tags (defense level)
    if not cov.empty:
        cov = cov.rename(columns={"tag": "coverage_tag"})
        if "defense_team" in cov.columns and "opponent" in df.columns:
            df = df.merge(cov[["defense_team", "coverage_tag"]], left_on="opponent", right_on="defense_team", how="left")

    # CB assignments (receiver-specific)
    if not cba.empty:
        # assume columns: defense_team, receiver, cb, penalty or quality
        # unify to penalty in [0..0.25]
        if "penalty" not in cba.columns and "quality" in cba.columns:
            # map quality->penalty light heuristic
            qmap = {"elite": 0.08, "good": 0.05, "avg": 0.0}
            cba["penalty"] = cba["quality"].map(qmap).fillna(0.0)
        cba = cba.rename(columns={"receiver": "player", "penalty": "cb_penalty"})
        if "opponent" in df.columns and "defense_team" in cba.columns:
            df = df.merge(
                cba[["defense_team", "player", "cb_penalty"]],
                left_on=["opponent", "player"],
                right_on=["defense_team", "player"],
                how="left"
            )

    # game lines (get win prob etc.)
    if not lines.empty and "event_id" in df.columns and "event_id" in lines.columns:
        df = df.merge(lines, on="event_id", how="left")
        # Build a single win-prob column from home/away depending on team alignment if you have sides:
        if "home_team" in lines.columns and "away_team" in lines.columns and "team" in df.columns:
            df["team_win_prob"] = np.where(
                df["team"].eq(df["home_team"]), df.get("home_wp", np.nan),
                np.where(df["team"].eq(df["away_team"]), df.get("away_wp", np.nan), np.nan)
            )

    # Weather (added)
    wx = _load_weather()
    if "event_id" in df.columns and not wx.empty:
        df = df.merge(wx, on="event_id", how="left")
    else:
        if "wind_mph" not in df.columns: df["wind_mph"] = np.nan
        if "temp_f" not in df.columns: df["temp_f"] = np.nan
        if "precip" not in df.columns: df["precip"] = np.nan

    # Pricing
    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        mk = str(row.get("market") or "").lower()
        line = row.get("line")
        try:
            L = float(line)
        except Exception:
            # if no line we can’t price
            continue

        over_odds = row.get("over_odds")
        under_odds = row.get("under_odds")
        p_over_vig = _implied_prob_from_american(over_odds)
        p_under_vig = _implied_prob_from_american(under_odds)

        p_market_fair = np.nan
        if not pd.isna(p_over_vig) and not pd.isna(p_under_vig):
            p_market_fair = _devig_two_way(p_over_vig, p_under_vig)
        elif not pd.isna(p_over_vig):
            p_market_fair = p_over_vig  # one-sided anchor

        # Model base: start at market line with default sigma, then apply multipliers
        sigma0 = _default_sigma_for_market(mk)
        # let base_mu equal line for neutral start; multipliers push it off
        mu0 = float(L)

        mu, sigma = _apply_multipliers(row, mk, mu0, sigma0)
        p_model_over = _prob_over_at_line(mu, sigma, L)
        p_blend = _blend_probs(p_model_over, p_market_fair)
        fair_over_odds = _fair_odds_from_prob(p_blend)

        edge = np.nan
        if not pd.isna(p_market_fair):
            edge = p_blend - p_market_fair

        # Tiering
        tier = "RED"
        abs_edge = abs(edge) if not pd.isna(edge) else 0.0
        if abs_edge >= 0.06:
            tier = "ELITE"
        elif abs_edge >= 0.04:
            tier = "GREEN"
        elif abs_edge >= 0.01:
            tier = "AMBER"

        # Bet side (simple)
        bet_side = "OVER" if p_blend >= 0.5 else "UNDER"

        out = {
            "player": row.get("player"),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "market": mk,
            "vegas_line": L,
            "vegas_over_odds": over_odds,
            "vegas_under_odds": under_odds,
            "vegas_over_fair_pct": p_market_fair,
            "model_proj": mu,
            "model_sd": sigma,
            "model_over_pct": p_model_over,
            "blended_over_pct": p_blend,
            "fair_over_odds": fair_over_odds,
            "edge_abs": abs_edge,
            "bet_side": bet_side,
            "tier": tier,
            # traceability
            "wind_mph": row.get("wind_mph"),
            "precip": row.get("precip"),
            "team_win_prob": row.get("team_win_prob"),
            "def_pressure_rate_z": row.get("opp_def_pressure_rate_z") or row.get("def_pressure_rate_z"),
            "def_pass_epa_z": row.get("opp_def_pass_epa_z") or row.get("def_pass_epa_z"),
            "def_rush_epa_z": row.get("opp_def_rush_epa_z") or row.get("def_rush_epa_z"),
            "light_box_rate_z": row.get("opp_light_box_rate_z") or row.get("light_box_rate_z"),
            "heavy_box_rate_z": row.get("opp_heavy_box_rate_z") or row.get("heavy_box_rate_z"),
            "coverage_tag": row.get("coverage_tag"),
            "cb_penalty": row.get("cb_penalty"),
        }
        out_rows.append(out)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"[pricing] wrote {OUT_FILE} rows={len(out_df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--write", type=str, default="outputs", help="Output directory (kept for compatibility)")
    args = parser.parse_args()

    try:
        price(args.season)
    except Exception as e:
        print(f"[pricing] ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
