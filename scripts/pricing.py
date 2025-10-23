#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Price player props using model multipliers + market anchoring and write outputs/props_priced_clean.csv.

Adds (non-breaking):
- Weather merge + weather multiplier
- Strict 2025 handling via merged dataframes from metrics
- Volume × Efficiency blend via scripts/volume.py with env VOLUME_BLEND (default 0.40)

If any optional file is missing, pipeline continues with safe defaults.
"""

import argparse
import os
import sys
from typing import Dict, Any, List, Optional

import numpy as np
import pandas as pd
from math import erf, sqrt

# --- NEW: volume × efficiency helpers ---
try:
    from scripts.volume import team_volume, player_volume, player_efficiency, volume_mu
except Exception:
    # allow pricing to run without volume.py present
    team_volume = player_volume = player_efficiency = volume_mu = None

OUT_DIR = "outputs"
OUT_FILE = os.path.join(OUT_DIR, "props_priced_clean.csv")

TEAM_FORM = ["metrics/team_form.csv", "data/team_form.csv"]
PLAYER_FORM = ["metrics/player_form.csv", "data/player_form.csv"]
COVERAGE = "data/coverage.csv"
CB_ASSIGN = "data/cb_assignments.csv"
INJURIES = "data/injuries.csv"
ROLES = "data/roles.csv"
GAME_LINES = ["outputs/odds_game.csv", "outputs/game_lines.csv"]  # prefer odds file; fallback to schedule
WEATHER = "data/weather.csv"

PROP_CANDIDATES = [
    'data/metrics_ready.csv', 'outputs/metrics_ready.csv', "outputs/props_raw.csv",
    "data/props_raw.csv",
    "outputs/props_aggregated.csv",
]


def _ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def _maybe_csv(path) -> pd.DataFrame:
    """Return the first readable CSV among one or many candidate paths."""
    paths: List[str]
    if isinstance(path, (list, tuple)):
        paths = [p for p in path if p]
    else:
        paths = [path]

    for pth in paths:
        if not pth:
            continue
        try:
            df = pd.read_csv(pth)
        except Exception:
            continue
        else:
            return df
    return pd.DataFrame()


def _load_props(primary: Optional[str] = None) -> pd.DataFrame:
    candidates: List[str] = []
    if primary:
        candidates.append(primary)
    candidates.extend(PROP_CANDIDATES)

    for p in candidates:
        if not p:
            continue
        if os.path.exists(p):
            try:
                return pd.read_csv(p)
            except Exception:
                continue
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
    p = min(max(p, 1e-9), 1 - 1e-9)
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
        if float(pass_epa_z) >= 0.6 and float(rush_epa_z) <= -0.4:
            mu_mult *= 0.97
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

    # --- Coverage/CB / slot/TE boosts ---
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


def _edge_value(prob: float, market_prob: float) -> float:
    if pd.isna(prob) or pd.isna(market_prob):
        return np.nan
    return float(prob) - float(market_prob)


def _tier_from_edge(edge: float) -> str:
    if pd.isna(edge):
        return "RED"
    abs_edge = abs(edge)
    if abs_edge >= 0.06:
        return "ELITE"
    if abs_edge >= 0.04:
        return "GREEN"
    if abs_edge >= 0.01:
        return "AMBER"
    return "RED"


MARKET_ALIASES = {
    'player_pass_yds':'pass_yards','player_rush_yds':'rush_yards','player_rec_yds':'rec_yards',
    'player_receptions':'receptions','player_rush_att':'rush_att','player_rush_rec_yds':'rush_rec_yards',
    'anytime_touchdown':'anytime_td','atd':'anytime_td'
}

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


def price(season: int, props_path: Optional[str] = None):
    # Anytime TD probability from team totals + red-zone share (Bernoulli)
    def _anytime_td_prob(row: pd.Series, tv: dict) -> float:
        team_pts = row.get('team_total_pts')
        if pd.isna(team_pts):
            # fallback to home/away totals if present on lines
            if 'home_total' in row and 'away_total' in row:
                if row.get('team') == row.get('home_team'):
                    team_pts = row.get('home_total')
                elif row.get('team') == row.get('away_team'):
                    team_pts = row.get('away_total')
        if pd.isna(team_pts):
            return np.nan
        try:
            team_pts = float(team_pts)
        except Exception:
            return np.nan
        # Convert points to expected TDs (FGs exist; 6.7 pts/TD heuristic)
        exp_tds = max(0.0, team_pts / 6.7)
        role = str(row.get('role') or '').upper()
        rz_tgt = row.get('rz_tgt_share'); rz_carry = row.get('rz_carry_share')
        pass_rate = float(tv.get('pass_rate_est', np.nan)); rush_rate = float(tv.get('rush_rate_est', np.nan))
        share = np.nan
        if role.startswith('RB') and not pd.isna(rz_carry):
            share = float(rz_carry) * max(0.30, rush_rate if not pd.isna(rush_rate) else 0.45)
        elif role.startswith('TE') or role.startswith('WR'):
            if not pd.isna(rz_tgt):
                share = float(rz_tgt) * max(0.30, pass_rate if not pd.isna(pass_rate) else 0.55)
        if pd.isna(share) or share <= 0:
            return np.nan
        lam = exp_tds * share
        p = 1.0 - np.exp(-max(0.0, lam))
        return float(np.clip(p, 1e-6, 1-1e-6))

    _ensure_dir(OUT_DIR)

    # Base frames
    props = _load_props(props_path)
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

    # Clamp season in team/player to 2025
    if "season" in team.columns:
        team = team[team["season"].astype(int) == 2025].copy()
    if "season" in player.columns:
        player = player[player["season"].astype(int) == 2025].copy()

    if props.empty:
        cand = [props_path] if props_path else []
        cand.extend(PROP_CANDIDATES)
        print("[pricing] No props file found. Expected one of:", cand, file=sys.stderr)
        # write empty output to keep pipeline predictable
        pd.DataFrame().to_csv(OUT_FILE, index=False)
        return

    # Merge contextuals
    df = props.copy()
    # normalize placeholder strings to real NaN so merges work
    df = df.replace({'NAN': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan})

    # ensure expected columns exist
    for need in ("player", "team", "opponent", "market", "line"):
        if need not in df.columns:
            df[need] = pd.NA

    # Join roles/injuries/player_form
    if not roles.empty:
        df = df.merge(roles, on=["player", "team"], how="left")
    if not inj.empty:
        df = df.merge(inj[["player", "status"]], on="player", how="left")

    if not player.empty:
        df = df.merge(player.drop_duplicates(subset=["player", "team"]), on=["player", "team"], how="left")

        # Backfill from player_form_consensus if team/opponent still missing
        try:
            _pfc_paths = [
                "data/player_form_consensus.csv",
                "outputs/player_form_consensus.csv",
            ]
            for _p in _pfc_paths:
                if os.path.exists(_p):
                    _pfc = pd.read_csv(_p)
                    _pfc.columns = [c.lower() for c in _pfc.columns]
                    # normalize player & team
                    if "player" in _pfc.columns:
                        _pfc["player"] = _pfc["player"].astype(str)
                    keep = [
                        c
                        for c in [
                            "player",
                            "team",
                            "opponent",
                            "position",
                            "role",
                            "target_share",
                            "tgt_share",
                            "route_rate",
                            "rush_share",
                            "rz_tgt_share",
                            "rz_carry_share",
                            "yprr_proxy",
                            "ypt",
                            "ypc",
                            "ypa_prior",
                        ]
                        if c in _pfc.columns
                    ]
                    if keep:
                        df = df.merge(
                            _pfc[keep].drop_duplicates("player"),
                            on="player",
                            how="left",
                            suffixes=("", "_pfc"),
                        )
                    break
        except Exception as _e:
            print(f"[pricing] consensus backfill skipped: {_e}", file=sys.stderr)
    

    # Join team form: opponent defense context (opp_*) and own offense context (plays_est/proe)
    if not team.empty:
        opp = team.add_prefix("opp_")
        if "opponent" in df.columns and "opp_team" in opp.columns:
            df = df.merge(opp, left_on="opponent", right_on="opp_team", how="left", suffixes=("", ""))
        if "team" in df.columns and "team" in team.columns:
            df = df.merge(team[["team", "plays_est", "proe", "pace", "def_pass_epa_z", "def_rush_epa_z",
                                "light_box_rate", "heavy_box_rate", "ay_per_att"]],
                          on="team", how="left")

    # coverage tags (defense level)
    if not cov.empty:
        cov = cov.rename(columns={"tag": "coverage_tag"})
        if "defense_team" in cov.columns and "opponent" in df.columns:
            df = df.merge(cov[["defense_team", "coverage_tag"]], left_on="opponent", right_on="defense_team", how="left")

    # CB assignments (receiver-specific)
    if not cba.empty:
        if "penalty" not in cba.columns and "quality" in cba.columns:
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

    # If opponent is still missing but we have home/away from lines, derive it
    if "opponent" in df.columns and df["opponent"].isna().all():
        if {"event_id","home_team","away_team","team"}.issubset(df.columns):
            try:
                import numpy as _np
                df["opponent"] = _np.where(
                    df["team"].eq(df["home_team"]), df["away_team"],
                    _np.where(df["team"].eq(df["away_team"]), df["home_team"], _np.nan)
                )
            except Exception:
                df["opponent"] = df.apply(
                    lambda r: r["away_team"] if r["team"] == r.get("home_team") else (r["home_team"] if r["team"] == r.get("away_team") else pd.NA),
                    axis=1
                )
        if "home_team" in lines.columns and "away_team" in lines.columns and "team" in df.columns:
            df["team_win_prob"] = np.where(
                df["team"].eq(df["home_team"]), df.get("home_wp", np.nan),
                np.where(df["team"].eq(df["away_team"]), df.get("away_wp", np.nan), np.nan)
            )

    # Weather
    if "event_id" in df.columns and not wx.empty:
        df = df.merge(wx, on="event_id", how="left")
    else:
        if "wind_mph" not in df.columns: df["wind_mph"] = np.nan
        if "temp_f" not in df.columns: df["temp_f"] = np.nan
        if "precip" not in df.columns: df["precip"] = np.nan

    # --- Volume × Efficiency blend weight (0..1), default 0.40 ---
    try:
        VOLUME_BLEND = float(os.getenv("VOLUME_BLEND", "0.40"))
        VOLUME_BLEND = max(0.0, min(1.0, VOLUME_BLEND))
    except Exception:
        VOLUME_BLEND = 0.40

    # Pricing
    out_rows: List[Dict[str, Any]] = []
    for _, row in df.iterrows():
        mk_raw = str(row.get("market") or "").lower()
        mk = MARKET_ALIASES.get(mk_raw, mk_raw)
        line = row.get("line")
        try:
            L = float(line)
        except Exception:
            # Allow Anytime TD (Yes) with a dummy line
            if mk in {"anytime_td","anytime_touchdown","atd"}:
                L = 0.0
            else:
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

        # Apply your existing multipliers (pressure/sack/funnel/coverage/AY/weather/script/etc.)
        # NOTE: We want the opponent context columns to be plain (not opp_*) for the multiplier.
        # If opp_* exist, create pass-throughs so _apply_multipliers can read them.
        # (No change if they already exist without opp_.)
        if pd.isna(row.get("def_pressure_rate_z")) and not pd.isna(row.get("opp_def_pressure_rate_z")):
            row["def_pressure_rate_z"] = row.get("opp_def_pressure_rate_z")
        if pd.isna(row.get("def_sack_rate_z")) and not pd.isna(row.get("opp_def_sack_rate_z")):
            row["def_sack_rate_z"] = row.get("opp_def_sack_rate_z")
        if pd.isna(row.get("def_pass_epa_z")) and not pd.isna(row.get("opp_def_pass_epa_z")):
            row["def_pass_epa_z"] = row.get("opp_def_pass_epa_z")
        if pd.isna(row.get("def_rush_epa_z")) and not pd.isna(row.get("opp_def_rush_epa_z")):
            row["def_rush_epa_z"] = row.get("opp_def_rush_epa_z")
        if pd.isna(row.get("light_box_rate_z")) and not pd.isna(row.get("opp_light_box_rate_z")):
            row["light_box_rate_z"] = row.get("opp_light_box_rate_z")
        if pd.isna(row.get("heavy_box_rate_z")) and not pd.isna(row.get("opp_heavy_box_rate_z")):
            row["heavy_box_rate_z"] = row.get("opp_heavy_box_rate_z")
        if pd.isna(row.get("ay_per_att_z")) and not pd.isna(row.get("opp_ay_per_att_z")):
            row["ay_per_att_z"] = row.get("opp_ay_per_att_z")

        mu_anchor, sigma = _apply_multipliers(row, mk, mu0, sigma0)

        # --- NEW: Volume × Efficiency μ and blend ---
        mu_model = mu_anchor  # default if volume.py not available or returns 0
        if all([team_volume, player_volume, player_efficiency, volume_mu]):
            # Build team/opp dicts for helpers
            team_row = {
                "plays_est": row.get("plays_est"),
                "pace": row.get("pace"),
                "proe": row.get("proe"),
                "def_pass_epa_z": row.get("def_pass_epa_z"),
                "def_rush_epa_z": row.get("def_rush_epa_z"),
                "light_box_rate": row.get("light_box_rate"),
                "heavy_box_rate": row.get("heavy_box_rate"),
                "ay_per_att": row.get("ay_per_att"),
            }
            # Opponent context for efficiency mods: prefer opp_* if present
            opp_row = {
                "def_pass_epa_z": row.get("opp_def_pass_epa_z", row.get("def_pass_epa_z")),
                "def_rush_epa_z": row.get("opp_def_rush_epa_z", row.get("def_rush_epa_z")),
                "light_box_rate": row.get("opp_light_box_rate", row.get("light_box_rate")),
                "heavy_box_rate": row.get("opp_heavy_box_rate", row.get("heavy_box_rate")),
            }
            script_wp = row.get("team_win_prob")
            tv = team_volume(team_row, script_wp=script_wp)

            player_row = {
                "player": row.get("player"),
                "team": row.get("team"),
                "role": row.get("role"),
                "position": row.get("position"),
                "tgt_share": row.get("tgt_share"),
                "route_rate": row.get("route_rate"),
                "rush_share": row.get("rush_share"),
                "yprr": row.get("yprr"),
                "ypt": row.get("ypt"),
                "ypc": row.get("ypc"),
                "ypa": row.get("ypa"),
                "receptions_per_target": row.get("receptions_per_target"),
                "rz_share": row.get("rz_share"),
            }
            eff = player_efficiency(player_row, opp_row, mk)
            mu_vol = volume_mu(mk, row, tv, player_row if isinstance(player_row, dict) else {}, eff)

            # If helper returned 0 (unknown market), keep anchor; else blend
            if isinstance(mu_vol, (int, float)) and not np.isnan(mu_vol) and mu_vol != 0.0:
                mu_model = (1.0 - VOLUME_BLEND) * mu_anchor + VOLUME_BLEND * mu_vol

        # Price at line using blended μ
        if mk in {"anytime_td","anytime_touchdown","atd"}:
            # Build a local team volume context (soft dep on scripts.volume)
            try:
                tv_local = team_volume({
                    'plays_est': row.get('plays_est'),
                    'pace': row.get('pace'),
                    'proe': row.get('proe'),
                    'def_pass_epa_z': row.get('def_pass_epa_z'),
                    'def_rush_epa_z': row.get('def_rush_epa_z'),
                }, script_wp=row.get('team_win_prob')) if team_volume else {}
            except Exception:
                tv_local = {}
            p_yes_model = _anytime_td_prob(row, tv_local) if tv_local else np.nan
            p_model_over = p_market_fair if pd.isna(p_yes_model) else p_yes_model
        else:
            p_model_over = _prob_over_at_line(mu_model, sigma, L)
        p_blend = _blend_probs(p_model_over, p_market_fair)
        p_model_under = np.nan if pd.isna(p_model_over) else 1.0 - float(p_model_over)
        p_blend_under = np.nan if pd.isna(p_blend) else 1.0 - float(p_blend)
        p_market_fair_under = np.nan if pd.isna(p_market_fair) else 1.0 - float(p_market_fair)

        fair_over_odds = np.nan if pd.isna(p_blend) else _fair_odds_from_prob(float(p_blend))
        fair_under_odds = np.nan if pd.isna(p_blend_under) else _fair_odds_from_prob(float(p_blend_under))

        edge_over = _edge_value(p_blend, p_market_fair)
        edge_under = _edge_value(p_blend_under, p_market_fair_under)
        over_strength = abs(edge_over) if not pd.isna(edge_over) else -np.inf
        under_strength = abs(edge_under) if not pd.isna(edge_under) else -np.inf
        bet_side = "OVER" if over_strength >= under_strength else "UNDER"

        common_out = {
            "player": row.get("player"),
            "team": row.get("team"),
            "opponent": row.get("opponent"),
            "market": mk,
            "vegas_line": L,
            "vegas_over_odds": over_odds,
            "vegas_under_odds": under_odds,
            "vegas_over_fair_pct": p_market_fair,
            "vegas_under_fair_pct": p_market_fair_under,
            "model_proj": mu_model,
            "model_sd": sigma,
            "model_over_pct": p_model_over,
            "model_under_pct": p_model_under,
            "blended_over_pct": p_blend,
            "blended_under_pct": p_blend_under,
            "fair_over_odds": fair_over_odds,
            "fair_under_odds": fair_under_odds,
            "bet_side": bet_side,
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
            # key usage/efficiency signals copied through to output
            "tgt_share": row.get("tgt_share"),
            "route_rate": row.get("route_rate"),
            "rush_share": row.get("rush_share"),
            "yprr": row.get("yprr"),
            "ypt": row.get("ypt"),
            "ypc": row.get("ypc"),
            "rz_tgt_share": row.get("rz_tgt_share"),
            "rz_carry_share": row.get("rz_carry_share"),
            "plays_est": row.get("plays_est"),
            "pace": row.get("pace"),
            "proe": row.get("proe"),
            # defensive splits (prefer opponent context when available)
            "def_pass_epa": (
                row.get("opp_def_pass_epa")
                or row.get("def_pass_epa_opp")
                or row.get("def_pass_epa")
            ),
            "def_rush_epa": (
                row.get("opp_def_rush_epa")
                or row.get("def_rush_epa_opp")
                or row.get("def_rush_epa")
            ),
            "def_sack_rate": (
                row.get("opp_def_sack_rate")
                or row.get("def_sack_rate_opp")
                or row.get("def_sack_rate")
            ),
            "light_box_rate": (
                row.get("opp_light_box_rate")
                or row.get("light_box_rate_opp")
                or row.get("light_box_rate")
            ),
            "heavy_box_rate": (
                row.get("opp_heavy_box_rate")
                or row.get("heavy_box_rate_opp")
                or row.get("heavy_box_rate")
            ),
        }

        side_map = {
            "OVER": {
                "prob": p_blend,
                "market_prob": p_market_fair,
                "book_odds": over_odds,
                "fair_odds": fair_over_odds,
            },
            "UNDER": {
                "prob": p_blend_under,
                "market_prob": p_market_fair_under,
                "book_odds": under_odds,
                "fair_odds": fair_under_odds,
            },
        }

        for side, info in side_map.items():
            row_out = dict(common_out)
            side_prob = info["prob"]
            market_prob_side = info["market_prob"]
            fair_odds_side = info["fair_odds"]
            vegas_odds_side = info["book_odds"]
            edge_side = _edge_value(side_prob, market_prob_side)

            row_out.update({
                "side": side,
                "fair_prob": side_prob,
                "market_prob": market_prob_side,
                "vegas_odds": vegas_odds_side,
                "fair_odds": fair_odds_side,
                "edge_pct": edge_side,
                "edge_abs": abs(edge_side) if not pd.isna(edge_side) else np.nan,
                "tier": _tier_from_edge(edge_side),
            })
            out_rows.append(row_out)

    out_df = pd.DataFrame(out_rows)
    out_df.to_csv(OUT_FILE, index=False)
    print(f"[pricing] wrote {OUT_FILE} rows={len(out_df)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    parser.add_argument("--write", type=str, default="outputs", help="Output directory (kept for compatibility)")
    parser.add_argument(
        "--props",
        type=str,
        default="",
        help="Primary props CSV (defaults to outputs/props_raw.csv)",
    )
    args = parser.parse_args()

    try:
        price(args.season, props_path=args.props or None)
    except Exception as e:
        print(f"[pricing] ERROR: {e}", file=sys.stderr)
        raise


if __name__ == "__main__":
    main()
