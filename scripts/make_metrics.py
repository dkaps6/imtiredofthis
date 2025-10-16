# scripts/make_metrics.py
"""
Build a single, pricing-ready table that joins props with all context metrics.

Inputs (best-effort; missing inputs won’t crash):
- outputs/props_raw.csv
- data/team_form.csv
- data/player_form.csv
- data/coverage.csv
- data/cb_assignments.csv
- data/injuries.csv
- data/weather.csv
- outputs/game_lines.csv (preferred) or outputs/odds_game.csv (fallback)

Output:
- data/metrics_ready.csv
"""

from __future__ import annotations
import argparse, os, sys, warnings, re
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
    """Map common sportsbook aliases to nflverse team codes (best effort)."""
    if s is None:
        return s
    norm = s.astype(str).str.upper().str.strip()
    aliases = {
        # books ↔ nflverse
        "WSH": "WAS", "WDC": "WAS",
        "JAX": "JAC",
        "ARZ": "ARI", "AZ": "ARI",
        "LA":  "LAR", "STL": "LAR",
        "LVR": "LV",  "OAK": "LV",
        "SFO": "SF",
        "TAM": "TB",
        "GBP": "GB",
        "KAN": "KC",
        "NOS": "NO", "N.O.": "NO",
        "SD":  "LAC",
        # occasional typos seen in feeds
        "CLV": "CLE",
    }
    return norm.replace(aliases)

_SUFFIX_RE = re.compile(r"\s+(JR|SR|II|III|IV|V)\.?$", flags=re.IGNORECASE)

def _normalize_player_name(x: pd.Series | str) -> pd.Series | str:
    """Remove punctuation/suffixes, collapse spaces."""
    def _one(name: str) -> str:
        if not isinstance(name, str):
            return name
        n = name.strip()
        n = n.replace(".", "")
        n = _SUFFIX_RE.sub("", n)
        n = re.sub(r"\s+", " ", n)
        return n
    if isinstance(x, pd.Series):
        return x.map(_one)
    return _one(x)

# ----------------------------
# Core loaders
# ----------------------------

def load_props() -> pd.DataFrame:
    df = _read_csv(os.path.join("outputs", "props_raw.csv"))
    if df.empty:
        return df

    rename_map = {
        "eventid": "event_id",
        "player_name": "player",
        "name": "player",
        "market_key": "market",
        "key": "market",
        "odds_over": "over_odds",
        "odds_under": "under_odds",
        "participant": "team",
    }
    for k, v in rename_map.items():
        if k in df.columns:
            df = df.rename(columns={k: v})

    # Ensure minimal schema
    for c in ["event_id","player","team","market","line","over_odds","under_odds"]:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize strings
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    if "player" in df.columns:
        df["player"] = _normalize_player_name(df["player"])

    # Deduplicate
    keep = ["event_id","player","team","market","line","over_odds","under_odds"]
    df = df[keep + [c for c in df.columns if c not in keep]].drop_duplicates()
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
    # normalize player column
    for pcol in ["player","player_name","name"]:
        if pcol in df.columns:
            df = df.rename(columns={pcol:"player"})
            break
    if "player" not in df.columns:
        df["player"] = np.nan
    df["player"] = _normalize_player_name(df["player"])
    return df

def load_coverage() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "coverage.csv"))
    if df.empty:
        return df
    if {"defense_team","tag"}.issubset(df.columns):
        df["defense_team"] = _normalize_team_names(df["defense_team"])
        pivot = pd.crosstab(df["defense_team"], df["tag"]).reset_index()
        pivot.columns = [str(c).lower() for c in pivot.columns]
        pivot = pivot.rename(columns={
            "top_shadow": "coverage_top_shadow",
            "heavy_man":  "coverage_heavy_man",
            "heavy_zone": "coverage_heavy_zone",
        })
        for c in pivot.columns:
            if c == "defense_team": continue
            pivot[c] = (pivot[c] > 0).astype(int)
        return pivot
    return pd.DataFrame()

def load_cb_assignments() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "cb_assignments.csv"))
    if df.empty:
        return df
    for c in ["defense_team","receiver"]:
        if c in df.columns and df[c].dtype == object:
            if c == "defense_team":
                df[c] = _normalize_team_names(df[c])
            else:
                df[c] = _normalize_player_name(df[c].astype(str))
    if "penalty" not in df.columns:
        if "quality" in df.columns:
            m = {"elite": 0.12, "good": 0.08, "avg": 0.04, "below_avg": 0.02}
            df["penalty"] = df["quality"].map(m).fillna(0.04)
        else:
            df["penalty"] = 0.06
    return df[["defense_team","receiver","penalty"]].rename(
        columns={"receiver":"player","defense_team":"opp_def_team"})

def load_injuries() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "injuries.csv"))
    if df.empty:
        return df
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    if "status" in df.columns:
        df["status"] = df["status"].astype(str).str.title()
    df["player"] = _normalize_player_name(df.get("player", pd.Series([], dtype=object)))
    return df[["player","team","status"]].drop_duplicates()

def load_weather() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "weather.csv"))
    return df

def load_game_lines() -> pd.DataFrame:
    gl = _read_csv(os.path.join("outputs","game_lines.csv"))
    if not gl.empty:
        for tcol in ["home_team","away_team"]:
            if tcol in gl.columns:
                gl[tcol] = _normalize_team_names(gl[tcol])
        if "home_wp" in gl.columns and "away_wp" in gl.columns:
            return gl[["event_id","home_team","away_team","home_wp","away_wp"]]

    og = _read_csv(os.path.join("outputs","odds_game.csv"))
    if og.empty:
        return pd.DataFrame(columns=["event_id","home_team","away_team","home_wp","away_wp"])

    for tcol in ["home_team","away_team"]:
        if tcol in og.columns:
            og[tcol] = _normalize_team_names(og[tcol])
    if "home_wp" not in og.columns or "away_wp" not in og.columns:
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
        return pd.DataFrame(columns=[
            "event_id","player","team","opponent","market","line","over_odds","under_odds",
            "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","rz_tgt_share","rz_carry_share","position","role",
            "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
            "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
            "cb_penalty","injury_status","wind_mph","temp_f","precip","team_wp","season"
        ])

    pf = load_player_form()
    tf = load_team_form()
    cov = load_coverage()
    cba = load_cb_assignments()
    inj = load_injuries()
    wx = load_weather()
    gl = load_game_lines()

    # if props.team missing entirely, try to backfill from player_form first
    if "team" in props.columns and props["team"].isna().all() and not pf.empty:
        props = props.merge(pf[["player","team"]].drop_duplicates(), on="player", how="left", suffixes=("","_pf"))
        props["team"] = props["team"].combine_first(props.get("team_pf"))
        if "team_pf" in props.columns:
            props = props.drop(columns=["team_pf"])

    # a few sources don't carry event_id; keep NaN and we still keep the rows
    if "event_id" not in props.columns:
        props["event_id"] = np.nan

    # Base: props + player_form
    base = props.copy()
    if not pf.empty:
        keep_pf = ["player","team","target_share","rush_share","route_rate","yprr_proxy","ypt","ypc",
                   "rz_tgt_share","rz_carry_share","position","role","season"]
        keep_pf = [c for c in keep_pf if c in pf.columns]

        # 1) player-only merge to capture players when team codes differ on props
        base = base.merge(pf[keep_pf].drop_duplicates(), on=["player"], how="left", suffixes=("","_pf"))
        # backfill props.team if empty
        if "team" in base.columns and "team_pf" in base.columns:
            base["team"] = base["team"].combine_first(base["team_pf"])
            base.drop(columns=[c for c in ["team_pf"] if c in base.columns], inplace=True)

    # Attach our team env (pace/proe/rz/12p/slot/ay) — left join, never drop rows
    if not tf.empty:
        keep_tf = ["team","def_pass_epa","def_rush_epa","def_sack_rate",
                   "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
                   "light_box_rate","heavy_box_rate","season"]
        keep_tf = [c for c in keep_tf if c in tf.columns]
        base = base.merge(tf[keep_tf].drop_duplicates(), on=["team"], how="left")

    # Infer opponent using game lines if possible (keep NaN otherwise)
    base["opponent"] = np.nan
    base["team_wp"] = np.nan
    if "event_id" in base.columns and not gl.empty:
        tmp = base.merge(gl, on="event_id", how="left")
        opp = np.where(tmp.get("team").eq(tmp.get("home_team")), tmp.get("away_team"),
               np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("home_team"), np.nan))
        base["opponent"] = _normalize_team_names(pd.Series(opp, index=base.index))
        base["team_wp"] = np.where(tmp.get("team").eq(tmp.get("home_team")), tmp.get("home_wp"),
                            np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("away_wp"), np.nan))

    # Opponent defense & coverage (left joins, keep rows)
    if not tf.empty:
        opp_tf = tf.rename(columns={
            "team":"opponent",
            "def_pass_epa":"def_pass_epa_opp",
            "def_rush_epa":"def_rush_epa_opp",
            "def_sack_rate":"def_sack_rate_opp",
            "light_box_rate":"light_box_rate_opp",
            "heavy_box_rate":"heavy_box_rate_opp",
        })
    keep_opp = ["opponent","def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp"]
    base = base.merge(opp_tf[keep_opp].drop_duplicates(), on="opponent", how="left")

    if not cov.empty and "defense_team" in cov.columns:
        cov2 = cov.rename(columns={"defense_team":"opponent"})
        for c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            if c not in cov2.columns:
                cov2[c] = 0
        base = base.merge(
            cov2[["opponent","coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]].drop_duplicates(),
            on="opponent", how="left"
        )
        for c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            base[c + "_opp"] = base[c]
            if c in base.columns:
                base.drop(columns=[c], inplace=True)

    # CB assignment (player + opponent)
    if not cba.empty:
        cba2 = cba.rename(columns={"opp_def_team":"opponent"})
        base = base.merge(cba2, on=["player","opponent"], how="left")
        base["cb_penalty"] = base.get("penalty")
        if "penalty" in base.columns:
            base.drop(columns=["penalty"], inplace=True)
    else:
        base["cb_penalty"] = np.nan

    # Injuries (player, team)
    if not inj.empty:
        base = base.merge(inj.rename(columns={"status":"injury_status"}), on=["player","team"], how="left")
    else:
        base["injury_status"] = np.nan

    # Weather via event_id (keep rows without event_id)
    wx_keep = ["event_id","wind_mph","temp_f","precip"]
    if not _read_csv(os.path.join(DATA_DIR,"weather.csv")).empty:
        w = load_weather()
        base = base.merge(w[[c for c in wx_keep if c in w.columns]].drop_duplicates(), on="event_id", how="left")
    else:
        for c in ["wind_mph","temp_f","precip"]:
            base[c] = np.nan

    # Final tidy and order
    want = [
        "event_id","player","team","opponent","market","line","over_odds","under_odds",
        "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","rz_tgt_share","rz_carry_share","position","role",
        "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
        "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
        "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
        "cb_penalty","injury_status","wind_mph","temp_f","precip","team_wp","season"
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
