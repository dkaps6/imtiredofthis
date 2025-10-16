# scripts/make_metrics.py
"""
Build a single, pricing-ready table that joins props with all context metrics.

Inputs (best-effort; missing inputs won’t crash):
- outputs/props_raw.csv                  # props fetched from Odds API
- data/team_form.csv                     # built by make_team_form.py
- data/player_form.csv                   # built/enriched by make_player_form.py + enrich_player_form.py
- data/coverage.csv                      # tags: top_shadow, heavy_man, heavy_zone ...
- data/cb_assignments.csv                # per-matchup CB penalties
- data/injuries.csv                      # Out / Doubtful / Limited / Probable
- data/weather.csv                       # wind_mph, temp_f, precip
- outputs/game_lines.csv                 # home_wp/away_wp from H2H (preferred)
- outputs/odds_game.csv                  # fallback if game_lines missing

Output:
- data/metrics_ready.csv                 # consumed by pricing.py

This script aims to be *non-fatal* and *schema-stable*:
- If an input is missing, we write NaNs for the corresponding features.
- Column names align with pricing expectations.
"""

from __future__ import annotations
import argparse, os, sys, warnings
import pandas as pd
import numpy as np

DATA_DIR = "data"
OUTPATH  = os.path.join(DATA_DIR, "metrics_ready.csv")

# ----------------------------
# Utilities
# ----------------------------

def _read_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    try:
        df = pd.read_csv(path)
        df.columns = [c.lower() for c in df.columns]
        return df
    except Exception:
        return pd.DataFrame()

def _safe_mkdir(p: str):
    os.makedirs(p, exist_ok=True)

def safe_div(n, d):
    n = n.astype(float); d = d.astype(float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(d == 0, np.nan, n / d)

def _normalize_team_names(s: pd.Series) -> pd.Series:
    """
    Best-effort normalization; leave values intact if unknown.
    """
    if s is None:
        return s
    norm = s.astype(str).str.upper().str.strip()
    # map some common variants if needed (extend as you meet issues)
    aliases = {
        "WSH": "WAS",
        "LA": "LAR",  # some books use LA for Rams, LAC for Chargers
    }
    return norm.replace(aliases)

# ----------------------------
# Core loaders
# ----------------------------

def load_props() -> pd.DataFrame:
    df = _read_csv(os.path.join("outputs", "props_raw.csv"))
    if df.empty:
        return df

    # Normalize common fields
    # Attempt to get consistent columns: event_id, team, opponent, player, market, line, over_odds, under_odds
    rename_map = {
        "eventid": "event_id",
        "event_id": "event_id",
        "player_name": "player",
        "name": "player",
        "market_key": "market",
        "key": "market",
        "line": "line",
        "odds_over": "over_odds",
        "odds_under": "under_odds",
        "team": "team",
        "participant": "team",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Ensure minimal schema
    for c in ["event_id","player","team","market","line","over_odds","under_odds"]:
        if c not in df.columns:
            df[c] = np.nan

    # Keep unique rows to avoid explosion at later joins
    cols_keep = [c for c in ["event_id","player","team","market","line","over_odds","under_odds"] if c in df.columns]
    if cols_keep:
        df = df[cols_keep + [c for c in df.columns if c not in cols_keep]].drop_duplicates()

    # Normalize team strings
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])

    # Some props don’t carry team; we’ll backfill via schedules join later if possible.
    return df


def load_team_form() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "team_form.csv"))
    if df.empty:
        return df
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    return df


def load_player_form() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "player_form.csv"))
    if df.empty:
        return df
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    # Player col normalization
    for pcol in ["player","player_name","name"]:
        if pcol in df.columns:
            df = df.rename(columns={pcol:"player"})
            break
    if "player" not in df.columns:
        df["player"] = np.nan
    return df


def load_coverage() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "coverage.csv"))
    # expected: defense_team, tag (top_shadow, heavy_man, heavy_zone, etc.)
    if df.empty:
        return df
    # pivot tags to boolean columns
    if {"defense_team","tag"}.issubset(df.columns):
        df["defense_team"] = _normalize_team_names(df["defense_team"])
        pivot = pd.crosstab(df["defense_team"], df["tag"]).reset_index()
        pivot.columns = [str(c).lower() for c in pivot.columns]
        # rename some common tags to pricing-friendly names
        pivot = pivot.rename(columns={
            "top_shadow": "coverage_top_shadow",
            "heavy_man":  "coverage_heavy_man",
            "heavy_zone": "coverage_heavy_zone",
        })
        # Convert counts to 0/1 indicators
        for c in pivot.columns:
            if c == "defense_team": continue
            pivot[c] = (pivot[c] > 0).astype(int)
        return pivot
    return pd.DataFrame()


def load_cb_assignments() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "cb_assignments.csv"))
    # expected columns: defense_team, receiver (player), cb, quality or penalty
    if df.empty:
        return df
    for c in ["defense_team","receiver"]:
        if c in df.columns and df[c].dtype == object:
            if c == "defense_team":
                df[c] = _normalize_team_names(df[c])
            else:
                df[c] = df[c].astype(str)
    # ensure a numeric penalty
    if "penalty" not in df.columns:
        # crude mapping from quality → penalty if present
        if "quality" in df.columns:
            m = {"elite": 0.12, "good": 0.08, "avg": 0.04, "below_avg": 0.02}
            df["penalty"] = df["quality"].map(m).fillna(0.04)
        else:
            df["penalty"] = 0.06
    return df[["defense_team","receiver","penalty"]].rename(columns={"receiver":"player","defense_team":"opp_def_team"})


def load_injuries() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "injuries.csv"))
    # expected: player, team, status
    if df.empty:
        return df
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.title()
    return df[["player","team","status"]].drop_duplicates()


def load_weather() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "weather.csv"))
    # expected: event_id, wind_mph, temp_f, precip
    if df.empty:
        return df
    return df


def load_game_lines() -> pd.DataFrame:
    """Return frame with event_id, home_team, away_team, home_wp, away_wp (best effort)."""
    gl = _read_csv(os.path.join("outputs","game_lines.csv"))
    if not gl.empty:
        # normalize
        for tcol in ["home_team","away_team"]:
            if tcol in gl.columns:
                gl[tcol] = _normalize_team_names(gl[tcol])
        # must have *_wp columns; if missing, try odds_game
        if "home_wp" in gl.columns and "away_wp" in gl.columns:
            return gl[["event_id","home_team","away_team","home_wp","away_wp"]]

    og = _read_csv(os.path.join("outputs","odds_game.csv"))
    if og.empty:
        return pd.DataFrame(columns=["event_id","home_team","away_team","home_wp","away_wp"])

    # Attempt to compute win probabilities from prices if present
    # Expecting moneyline odds or implied probs in og
    # Minimal: if 'home_implied' exists, use it; else NaN.
    cols = [c for c in og.columns if c.startswith("home_") or c.startswith("away_")]
    for tcol in ["home_team","away_team"]:
        if tcol in og.columns:
            og[tcol] = _normalize_team_names(og[tcol])

    if "home_wp" not in og.columns or "away_wp" not in og.columns:
        # If implied probs exist:
        if "home_implied" in og.columns and "away_implied" in og.columns:
            og["home_wp"] = og["home_implied"].astype(float)
            og["away_wp"] = og["away_implied"].astype(float)
        else:
            og["home_wp"] = np.nan
            og["away_wp"] = np.nan

    return og[["event_id","home_team","away_team","home_wp","away_wp"]].drop_duplicates()


# ----------------------------
# Assembler
# ----------------------------

def build_metrics(season: int) -> pd.DataFrame:
    props = load_props()
    if props.empty:
        # Return schema-like empty DF
        return pd.DataFrame(columns=[
            "event_id","player","team","market","line","over_odds","under_odds",
            "opponent",
            # player form
            "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","rz_tgt_share","rz_carry_share","position","role",
            # team form (our team & opponent)
            "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "light_box_rate_opp","heavy_box_rate_opp",
            # coverage
            "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
            # cb assignment
            "cb_penalty",
            # injuries
            "injury_status",
            # weather
            "wind_mph","temp_f","precip",
            # win probabilities for script
            "team_wp"
        ])

    # Merge player & team form
    pf = load_player_form()
    tf = load_team_form()

    # Coverage & CB
    cov = load_coverage()
    cba = load_cb_assignments()

    # Injuries
    inj = load_injuries()

    # Weather
    wx = load_weather()

    # Game lines (for team_wp)
    gl = load_game_lines()

    # Backfill team from player_form if empty in props
    if "team" in props.columns and props["team"].isna().all() and not pf.empty:
        props = props.merge(pf[["player","team"]].drop_duplicates(), on="player", how="left", suffixes=("","_pf"))
        props["team"] = props["team"].combine_first(props.get("team_pf"))
        if "team_pf" in props.columns:
            props = props.drop(columns=["team_pf"])

    # Ensure event_id exists for weather & game_lines join
    if "event_id" not in props.columns:
        props["event_id"] = np.nan

    # Base join: props + player_form (by player,team)
    base = props.copy()
    if not pf.empty:
        keep_pf = ["player","team","target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","rz_tgt_share","rz_carry_share","position","role","season"]
        keep_pf = [c for c in keep_pf if c in pf.columns]
        base = base.merge(pf[keep_pf].drop_duplicates(), on=["player","team"], how="left")

    # Attach our team’s environment (non-opponent)
    if not tf.empty:
        keep_tf = ["team","def_pass_epa","def_rush_epa","def_sack_rate","pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
                   "light_box_rate","heavy_box_rate","season"]
        keep_tf = [c for c in keep_tf if c in tf.columns]
        # we'll rename opponent columns later; for now merge our team as 'team_*' if needed
        base = base.merge(tf[keep_tf].drop_duplicates(), on=["team"], how="left", suffixes=("",""))

    # Determine opponent using game_lines
    # Strategy: if base has team + event_id, and gl has home/away with event_id, infer opponent.
    opponent = pd.Series(np.nan, index=base.index, dtype=object)
    if "event_id" in base.columns and not gl.empty:
        tmp = base.merge(gl, on="event_id", how="left")
        # If our team equals home_team → opponent is away_team; vice versa.
        opp = np.where(tmp.get("team").eq(tmp.get("home_team")), tmp.get("away_team"),
               np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("home_team"), np.nan))
        opponent = pd.Series(opp, index=base.index)

        # Add team win probability (team_wp)
        team_wp = np.where(tmp.get("team").eq(tmp.get("home_team")), tmp.get("home_wp"),
                  np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("away_wp"), np.nan))
        base["team_wp"] = team_wp
    else:
        base["team_wp"] = np.nan

    base["opponent"] = _normalize_team_names(opponent.fillna(np.nan))

    # Join opponent team_form metrics and coverage flags
    if not tf.empty and "opponent" in base.columns:
        opp_tf = tf.rename(columns={
            "team":"opponent",
            "def_pass_epa":"def_pass_epa_opp",
            "def_rush_epa":"def_rush_epa_opp",
            "def_sack_rate":"def_sack_rate_opp",
            "light_box_rate":"light_box_rate_opp",
            "heavy_box_rate":"heavy_box_rate_opp"
        })
        keep_opp = ["opponent","def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp"]
        base = base.merge(opp_tf[keep_opp].drop_duplicates(), on="opponent", how="left")

    # Coverage tags attach via opponent defense team
    if not cov.empty and "opponent" in base.columns and "defense_team" in cov.columns:
        cov2 = cov.rename(columns={"defense_team":"opponent"})
        for c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            if c not in cov2.columns:
                cov2[c] = 0
        base = base.merge(cov2[["opponent","coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]].drop_duplicates(),
                          on="opponent", how="left")

        # suffix them with _opp for clarity
        for c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            base[c + "_opp"] = base[c]
            if c in base.columns:
                base.drop(columns=[c], inplace=True)

    # CB assignment (opponent-specific, per player)
    if not cba.empty and "opponent" in base.columns:
        # cba has columns: opp_def_team, player, penalty
        # join on player + opponent=opp_def_team
        cba2 = cba.rename(columns={"opp_def_team":"opponent"})
        base = base.merge(cba2, on=["player","opponent"], how="left")
        base["cb_penalty"] = base["penalty"]
        if "penalty" in base.columns:
            base.drop(columns=["penalty"], inplace=True)
    else:
        base["cb_penalty"] = np.nan

    # Injuries
    if not inj.empty:
        base = base.merge(inj.rename(columns={"status":"injury_status"}), on=["player","team"], how="left")
    else:
        base["injury_status"] = np.nan

    # Weather via event_id
    if not wx.empty and "event_id" in base.columns and "event_id" in wx.columns:
        keep_wx = ["event_id","wind_mph","temp_f","precip"]
        keep_wx = [c for c in keep_wx if c in wx.columns]
        base = base.merge(wx[keep_wx].drop_duplicates(), on="event_id", how="left")
    else:
        for c in ["wind_mph","temp_f","precip"]:
            base[c] = np.nan

    # Final tidy and column order
    want = [
        "event_id","player","team","opponent","market","line","over_odds","under_odds",
        # player form
        "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","rz_tgt_share","rz_carry_share","position","role",
        # our team context (kept generic: pace/proe/rz/12p/slot/ay)
        "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
        # opponent defense context
        "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
        # coverage (opponent)
        "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
        # CB assignment penalty (player-specific)
        "cb_penalty",
        # injuries
        "injury_status",
        # weather
        "wind_mph","temp_f","precip",
        # script
        "team_wp",
        # season (if present)
        "season"
    ]
    for c in want:
        if c not in base.columns:
            base[c] = np.nan

    base = base[want].drop_duplicates().reset_index(drop=True)
    return base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--season", type=int, default=2025)
    args = parser.parse_args()
    _safe_mkdir(DATA_DIR)

    try:
        df = build_metrics(args.season)
    except Exception as e:
        print(f"[make_metrics] ERROR: {e}", file=sys.stderr)
        df = pd.DataFrame(columns=[
            "event_id","player","team","opponent","market","line","over_odds","under_odds",
            "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","rz_tgt_share","rz_carry_share","position","role",
            "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
            "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
            "cb_penalty","injury_status","wind_mph","temp_f","precip","team_wp","season"
        ])

    df.to_csv(OUTPATH, index=False)
    print(f"[make_metrics] Wrote {len(df)} rows → {OUTPATH}")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
