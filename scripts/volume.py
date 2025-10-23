#!/usr/bin/env python3
# scripts/volume.py
"""
Lightweight helpers for volume × efficiency modeling used by pricing.py,
plus a CLI that generates team- and player-level volume features from your
existing CSVs. This file is backward compatible with your current pipeline.

Exports for pricing.py:
- team_volume(team_row, script_wp) -> dict
- player_volume(team_vol: dict, player_row: dict) -> dict
- player_efficiency(player_row: dict, opp_row: dict, market: str) -> dict
- volume_mu(market: str, row: dict, tv: dict, player_row: dict, eff: dict) -> float
"""

from __future__ import annotations
import argparse
import sys
from typing import List, Dict, Any
import pandas as pd
import numpy as np

# ---------- Existing CLI pieces (kept) ----------
REQ_TEAM_COLS = [
    "team","season","event_id",
    "pace",             # seconds per snap (neutral pace)
    "proe",             # pass rate over expectation (decimal)
    "def_pass_epa_z",   # opponent context for funnel
    "def_rush_epa_z",   # opponent context for funnel
]
REQ_LINES_COLS = ["event_id","home_team","away_team","home_wp","away_wp"]
REQ_ROLES_COLS = ["player","team","role"]  # WR1/WR2/SLOT/RB1/TE1/QB1...
# Optional but recommended player-form priors/shares
OPT_PLAYER_FORM_COLS = [
    "player","team","season",
    "target_share","route_rate","rush_share","rz_share",
    "ypt_prior","yprr_prior","ypc_prior","catch_rate_prior","ypa_prior"
]
REQ_INJURIES_COLS = ["player","team","status"]  # Out/Doubtful/Questionable/Limited/Probable

def fail(msg: str):
    print(f"[volume] ❌ {msg}", file=sys.stderr)
    sys.exit(1)

def _read_csv(path: str) -> pd.DataFrame:
    try:
        return pd.read_csv(path)
    except FileNotFoundError:
        return pd.DataFrame()

def assert_cols(df: pd.DataFrame, cols: List[str], name: str):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        fail(f"{name} is missing required columns: {missing}")

def assert_not_null(df: pd.DataFrame, cols: List[str], name: str):
    bad = {c:int(df[c].isna().sum()) for c in cols if c in df.columns and df[c].isna().any()}
    if bad:
        fail(f"{name} has nulls in required columns: {bad}")

def z_to_shift(z, pos=0.03, neg=-0.03, pos_thresh=0.6, neg_thresh=-0.6):
    """Map opponent EPA z to a small funnel shift (±3%) with thresholds."""
    if pd.isna(z):
        return 0.0
    if z >= pos_thresh:
        return pos
    if z <= neg_thresh:
        return neg
    return 0.0

# ---------- Single-row helpers exposed to pricing.py ----------
def team_volume(team_row: Dict[str, Any], script_wp: float | None = None) -> Dict[str, float]:
    """
    Estimate plays and pass/rush split for a SINGLE team in a SINGLE game.
    Inputs come from merged pricing row:
      - team_row: {"plays_est","pace","proe","def_pass_epa_z","def_rush_epa_z", ...}
      - script_wp: win probability for this team (0..1)

    Returns: {"plays_est","pass_rate_est","rush_rate_est","pass_att_team_est","rush_att_team_est"}
    """
    sec_per_snap = float(team_row.get("pace") or np.nan)
    proe = float(team_row.get("proe") or 0.0)
    pass_epa_z = team_row.get("def_pass_epa_z")
    rush_epa_z = team_row.get("def_rush_epa_z")

    # If plays_est already computed upstream, use it; else derive from pace.
    if not np.isnan(team_row.get("plays_est", np.nan)):
        plays_est = float(team_row["plays_est"])
    else:
        plays_est = 3600.0 / sec_per_snap if (not np.isnan(sec_per_snap) and sec_per_snap > 0) else 64.0

    # Base pass rate from league avg + PROE
    LEAGUE_BASE_PASS = 0.58
    pass_rate = LEAGUE_BASE_PASS + float(proe)

    # Script tilt
    if script_wp is not None:
        if script_wp >= 0.60:
            pass_rate -= 0.02
        elif script_wp <= 0.40:
            pass_rate += 0.02

    # Funnel tilt from opponent EPA splits
    pass_rate += z_to_shift(pass_epa_z, pos=+0.03, neg=-0.03)   # worse pass D -> pass more
    pass_rate += -z_to_shift(rush_epa_z, pos=+0.03, neg=-0.03)  # worse rush D -> rush more -> pass less

    pass_rate = float(np.clip(pass_rate, 0.30, 0.75))
    rush_rate = 1.0 - pass_rate

    # Plays small script multiplier (clock bleed / hurry-up)
    if script_wp is not None:
        plays_est *= (1.0 + (-0.02 if script_wp >= 0.60 else 0.0) + (+0.02 if script_wp <= 0.40 else 0.0))

    return {
        "plays_est": float(plays_est),
        "pass_rate_est": float(pass_rate),
        "rush_rate_est": float(rush_rate),
        "pass_att_team_est": float(plays_est * pass_rate),
        "rush_att_team_est": float(plays_est * (1.0 - pass_rate)),
    }

def player_volume(team_vol: Dict[str, Any], player_row: Dict[str, Any]) -> Dict[str, float]:
    """
    Convert team volume into player opportunities using player shares.
    Accepts both 'target_share' and 'tgt_share' keys; route_rate and rush_share optional.
    """
    pass_att = float(team_vol.get("pass_att_team_est", 0.0))
    rush_att = float(team_vol.get("rush_att_team_est", 0.0))

    tgt_share = player_row.get("tgt_share")
    if tgt_share is None:
        tgt_share = player_row.get("target_share")
    route_rate = player_row.get("route_rate")
    rush_share = player_row.get("rush_share")

    tgt_share = float(tgt_share) if tgt_share is not None else np.nan
    route_rate = float(route_rate) if route_rate is not None else np.nan
    rush_share = float(rush_share) if rush_share is not None else np.nan

    targets_est = pass_att * np.nan_to_num(tgt_share, nan=0.0)
    routes_est  = pass_att * np.nan_to_num(route_rate, nan=0.0)
    rushes_est  = rush_att * np.nan_to_num(rush_share, nan=0.0)

    return {
        "targets_est": float(targets_est),
        "routes_est": float(routes_est),
        "rushes_est": float(rushes_est),
    }

def player_efficiency(player_row: Dict[str, Any], opp_row: Dict[str, Any], market: str) -> Dict[str, float]:
    """
    Derive efficiency priors per market from player priors, lightly adjusted by opponent context if available.
    Keys accepted in player_row: yprr, ypt, ypc, ypa, catch_rate (or *_prior variants).
    """
    def _val(*keys, default=np.nan):
        for k in keys:
            v = player_row.get(k)
            if v is not None and not (isinstance(v, float) and np.isnan(v)):
                try:
                    return float(v)
                except Exception:
                    pass
        return float(default)

    yprr = _val("yprr","yprr_prior")
    ypt  = _val("ypt","ypt_prior")
    ypc  = _val("ypc","ypc_prior")
    ypa  = _val("ypa","ypa_prior")
    cr   = _val("receptions_per_target","catch_rate","catch_rate_prior", default=0.64)

    # small opponent modifiers
    pass_epa_z = opp_row.get("def_pass_epa_z")
    rush_epa_z = opp_row.get("def_rush_epa_z")
    if not pd.isna(pass_epa_z):
        yprr *= (1.0 - 0.04 * max(0.0, float(pass_epa_z)))
        ypt  *= (1.0 - 0.03 * max(0.0, float(pass_epa_z)))
        ypa  *= (1.0 - 0.02 * max(0.0, float(pass_epa_z)))
    if not pd.isna(rush_epa_z):
        ypc  *= (1.0 - 0.03 * max(0.0, float(rush_epa_z)))

    return {"yprr":yprr, "ypt":ypt, "ypc":ypc, "ypa":ypa, "catch_rate":cr}

def volume_mu(market: str, row: Dict[str, Any], tv: Dict[str, Any],
              player_row: Dict[str, Any], eff: Dict[str, float]) -> float:
    """
    Compute μ for the given market using team volume + player shares + efficiency.
    Returns 0.0 if the market is unknown (lets pricing fall back to anchor).
    """
    mk = str(market).lower()
    opps = player_volume(tv, player_row)
    targets = opps["targets_est"]
    routes  = opps["routes_est"]
    rushes  = opps["rushes_est"]

    if mk == "rec_yards":
        # Prefer YPRR if routes present; else Y/Tgt
        if routes > 0 and eff["yprr"] > 0:
            return float(routes * eff["yprr"])
        return float(targets * eff["ypt"])

    if mk == "receptions":
        return float(targets * eff["catch_rate"])

    if mk == "rush_yards":
        return float(rushes * eff["ypc"])

    if mk == "rush_att":
        return float(rushes)

    if mk == "pass_yards":
        # Only meaningful for QB1; approximate via team attempts * ypa if role suggests QB
        role = str(player_row.get("role") or row.get("role") or "").upper()
        if role.startswith("QB") and eff["ypa"] > 0:
            return float(tv.get("pass_att_team_est", 0.0) * eff["ypa"])
        return 0.0

    # Unknown market → let pricing fall back to anchor μ
    return 0.0

# ---------- Batch/CLI functions (unchanged logic from your previous script) ----------
def compute_team_volume(team_form: pd.DataFrame, game_lines: pd.DataFrame, season: int) -> pd.DataFrame:
    # 2025-only
    if "season" in team_form.columns:
        team_form = team_form[team_form["season"] == season]
    if team_form.empty:
        fail("team_form for the target season is empty.")

    # Basic integrity
    assert_cols(team_form, REQ_TEAM_COLS, "team_form")
    assert_cols(game_lines, REQ_LINES_COLS, "game_lines")

    # Plays estimate:
    tf = team_form.copy()
    gl = game_lines[REQ_LINES_COLS].copy()

    # Attach opponent per event
    home = tf.merge(gl, left_on=["event_id","team"], right_on=["event_id","home_team"], how="inner")
    home["opp_team"] = home["away_team"]
    home["our_wp"] = home["home_wp"]
    away = tf.merge(gl, left_on=["event_id","team"], right_on=["event_id","away_team"], how="inner")
    away["opp_team"] = away["home_team"]
    away["our_wp"] = away["away_wp"]
    base = pd.concat([home, away], ignore_index=True)

    # Bring opponent pace/proe/funnels for smoothing
    opp_cols = ["event_id","team","pace","proe","def_pass_epa_z","def_rush_epa_z"]
    opp = tf[opp_cols].rename(columns={
        "team":"opp_team",
        "pace":"opp_pace",
        "proe":"opp_proe",
        "def_pass_epa_z":"opp_def_pass_epa_z",
        "def_rush_epa_z":"opp_def_rush_epa_z"
    })
    base = base.merge(opp, on=["event_id","opp_team"], how="left")

    # Baseline plays from both teams' pace (seconds/snap)
    for c in ["pace","opp_pace"]:
        if c in base.columns:
            base[c] = pd.to_numeric(base[c], errors="coerce")
    base["sec_per_play_blend"] = base[["pace","opp_pace"]].mean(axis=1)
    assert_not_null(base, ["sec_per_play_blend"], "team_volume: sec_per_play_blend")
    base["plays_raw"] = 3600.0 / base["sec_per_play_blend"]  # 60 min regulation

    # Script tilt and funnel shift
    LEAGUE_BASE_PASS = 0.58
    base["proe_blend"] = base[["proe","opp_proe"]].mean(axis=1).fillna(0.0)
    pass_rate = LEAGUE_BASE_PASS + base["proe_blend"].fillna(0.0)

    script_shift = np.where(base["our_wp"] >= 0.55, -0.02, 0.0) + np.where(base["our_wp"] <= 0.45, 0.02, 0.0)
    funnel_shift = z_to_shift(base["opp_def_pass_epa_z"], neg=-0.03, pos=+0.03)
    funnel_shift += -z_to_shift(base["opp_def_rush_epa_z"], neg=-0.03, pos=+0.03)

    base["pass_rate_est"] = (pass_rate + script_shift + funnel_shift).clip(0.30, 0.75)
    base["rush_rate_est"] = 1.0 - base["pass_rate_est"]

    base["plays_mult_script"] = 1.0 + np.where(base["our_wp"] >= 0.60, -0.02, 0.0) + np.where(base["our_wp"] <= 0.40, +0.02, 0.0)
    base["plays_est"] = base["plays_raw"] * base["plays_mult_script"]

    base["pass_att_team_est"] = base["plays_est"] * base["pass_rate_est"]
    base["rush_att_team_est"] = base["plays_est"] * base["rush_rate_est"]

    out_cols = [
        "event_id","team","opp_team","season","plays_est","pass_rate_est","rush_rate_est",
        "pass_att_team_est","rush_att_team_est"
    ]
    return base[out_cols].copy()

def apply_injury_redistribution(player_df: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    # WR1 cap + redistribution 60/30/10 (WR2/SLOT/TE) if Alpha Limited/Out
    if injuries.empty:
        return player_df
    inj = injuries.copy()
    inj["status"] = inj["status"].astype(str).str.lower()
    aff = inj[inj["status"].isin(["out","doubtful","questionable","limited"])][["player","team"]].drop_duplicates()
    if aff.empty:
        return player_df

    df = player_df.copy()
    df["target_share"] = pd.to_numeric(df.get("target_share", np.nan), errors="coerce")

    # Identify WR1 per team
    wr1_mask = df["role"].astype(str).str.upper().eq("WR1")
    # Cap WR1 at 40% of its current value if flagged
    cap_idx = df.merge(aff, on=["player","team"], how="inner").index
    cap_wr1 = wr1_mask & df.index.isin(cap_idx)
    df.loc[cap_wr1, "target_share"] = df.loc[cap_wr1, "target_share"] * 0.4

    # Freed share
    freed = (
        player_df.loc[cap_wr1, ["team","event_id","target_share"]]
        .groupby(["team","event_id"], as_index=False)["target_share"]
        .sum()
        .rename(columns={"target_share":"freed_share"})
    )
    if freed.empty:
        return df

    df = df.merge(freed, on=["team","event_id"], how="left")
    df["freed_share"] = df["freed_share"].fillna(0.0)

    def _redis_group(g):
        freed_share = g["freed_share"].iloc[0]
        if freed_share <= 0:
            return g
        role_map = {"WR2":0.60, "SLOT":0.30}
        g["add_share"] = 0.0
        for role, portion in role_map.items():
            mask = g["role"].astype(str).str.upper().eq(role)
            subtotal = g.loc[mask, "target_share"].sum()
            if subtotal > 0:
                g.loc[mask, "add_share"] += (g.loc[mask, "target_share"] / subtotal) * (freed_share * portion)
        # RBs total 10%
        mask_rb = g["role"].astype(str).str.upper().str.startswith("RB")
        subtotal_rb = g.loc[mask_rb, "target_share"].sum()
        if subtotal_rb > 0:
            g.loc[mask_rb, "add_share"] += (g.loc[mask_rb, "target_share"] / subtotal_rb) * (freed_share * 0.10)
        g["target_share"] = g["target_share"] + g["add_share"]
        return g.drop(columns=["add_share"])

    df = df.groupby(["team","event_id"], group_keys=False).apply(_redis_group)
    df = df.drop(columns=["freed_share"])
    return df

def compute_player_volume(team_vol: pd.DataFrame,
                          roles: pd.DataFrame,
                          player_form: pd.DataFrame,
                          injuries: pd.DataFrame,
                          season: int) -> pd.DataFrame:
    # Filter 2025 and assert inputs
    if "season" in player_form.columns:
        player_form = player_form[player_form["season"] == season]
    roles = roles.copy()
    assert_cols(roles, REQ_ROLES_COLS, "roles")
    missing_opt = [c for c in OPT_PLAYER_FORM_COLS if c not in player_form.columns]
    if missing_opt:
        fail(f"player_form is missing required share/efficiency priors columns: {missing_opt}")

    # Merge baseline player shares/priors onto roles
    pf = player_form[OPT_PLAYER_FORM_COLS].copy()
    df = roles.merge(pf, on=["player","team"], how="inner")
    if df.empty:
        fail("No overlap between roles and player_form for the season—cannot compute opportunities.")

    # Attach team volume
    df = df.merge(team_vol, on=["team","event_id"], how="inner")
    if df.empty:
        fail("No overlap between team volume and player rows—check event_id/team normalization.")

    # Injury redistribution for target_share
    df = apply_injury_redistribution(df, injuries)

    # Opportunities
    for c in ["target_share","route_rate","rush_share"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    assert_not_null(df, ["pass_att_team_est","rush_att_team_est"], "compute_player_volume: team volume")
    if df["target_share"].isna().any() or df["rush_share"].isna().any():
        fail("Missing target_share/rush_share after merges; cannot fill blank rows by spec.")

    df["targets_est"] = df["pass_att_team_est"] * df["target_share"].clip(lower=0.0)
    df["routes_est"]  = df["pass_att_team_est"] * df["route_rate"].clip(lower=0.0)
    df["rushes_est"]  = df["rush_att_team_est"] * df["rush_share"].clip(lower=0.0)

    # Priors presence check
    for c in ["ypt_prior","yprr_prior","ypc_prior","catch_rate_prior","ypa_prior"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    assert_not_null(df, ["ypt_prior","yprr_prior","ypc_prior","catch_rate_prior","ypa_prior"], "player priors")

    out_cols = [
        "event_id","team","player","role","season",
        "plays_est","pass_rate_est","rush_rate_est",
        "pass_att_team_est","rush_att_team_est",
        "targets_est","routes_est","rushes_est",
        "ypt_prior","yprr_prior","ypc_prior","catch_rate_prior","ypa_prior"
    ]
    return df[out_cols].copy()

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--season", type=int, default=2025)
    p.add_argument("--team_form", default="data/team_form.csv")
    p.add_argument("--game_lines", default="outputs/game_lines.csv")
    p.add_argument("--roles", default="data/roles.csv")
    p.add_argument("--player_form", default="data/player_form.csv")
    p.add_argument("--injuries", default="data/injuries.csv")
    p.add_argument("--out_team_volume", default="metrics/team_volume.csv")
    p.add_argument("--out_player_volume", default="metrics/volume_features.csv")
    args = p.parse_args()

    tf = _read_csv(args.team_form)
    gl = _read_csv(args.game_lines)
    roles = _read_csv(args.roles)
    pf = _read_csv(args.player_form)
    inj = _read_csv(args.injuries)

    if tf.empty:  fail("data/team_form.csv is empty.")
    if gl.empty:  fail("outputs/game_lines.csv is empty (H2H lines step must run first).")
    if roles.empty: fail("data/roles.csv is empty.")
    if pf.empty:   fail("data/player_form.csv is empty.")

    # Basic required columns present?
    assert_cols(tf, ["team","season","event_id","pace","proe","def_pass_epa_z","def_rush_epa_z"], "team_form")
    assert_cols(gl, REQ_LINES_COLS, "game_lines")

    team_vol = compute_team_volume(tf, gl, args.season)
    team_vol.to_csv(args.out_team_volume, index=False)

    player_vol = compute_player_volume(team_vol, roles, pf, inj, args.season)
    player_vol.to_csv(args.out_player_volume, index=False)

    print(f"[volume] ✅ wrote {args.out_team_volume} and {args.out_player_volume} for season {args.season}")

if __name__ == "__main__":
    main()
