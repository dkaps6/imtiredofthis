#!/usr/bin/env python3
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
- outputs/odds_game.csv (preferred) or outputs/game_lines.csv (fallback)
- (optional) data/team_form_weekly.csv (for opponent week-specific env)

Output:
- data/metrics_ready.csv
"""

from __future__ import annotations
import argparse, os, sys, warnings, re, traceback
import pandas as pd
import numpy as np



def _inj_load_lines_preferring_odds():
    for p in ["outputs/odds_game.csv", "outputs/game_lines.csv"]:
        if os.path.exists(p):
            try:
                gl = pd.read_csv(p)
                gl.columns = [c.lower() for c in gl.columns]
                for c in ["home_team","away_team"]:
                    if c in gl.columns:
                        gl[c] = _inj_normalize_team(gl[c])
                return gl
            except Exception:
                pass
    return pd.DataFrame()
# --- injected helper utilities (surgical, idempotent) ---
import re as _re_inj

def _inj_normalize_player(s):
    suf = _re_inj.compile(r"\s+(JR|SR|II|III|IV|V)\.?$", flags=_re_inj.I)
    return (s.astype(str)
              .str.replace(r"\.", "", regex=True)
              .str.replace(suf, "", regex=True)
              .str.replace(r"\s+", " ", regex=True)
              .str.strip())

def _inj_normalize_team(s):
    aliases = {
        "WSH":"WAS","WDC":"WAS","JAC":"JAX","ARZ":"ARI","AZ":"ARI","LA":"LAR",
        "LVR":"LV","OAK":"LV","SFO":"SF","TAM":"TB","GBP":"GB","KAN":"KC",
        "NOS":"NO","SD":"LAC","CLV":"CLE"
    }
    return s.astype(str).str.upper().str.strip().replace(aliases)

def _inj_player_key(s):
    return s.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]","",regex=True)
DATA_DIR = "data"
OUTPATH  = os.path.join(DATA_DIR, "metrics_ready.csv")

PLAYER_FORM_CONSENSUS_OPPONENT = "ALL"


# --- helpers for initials+last fallback matching ---
import re as _re_nm2
def _nm_initial_last(name: str):
    if not isinstance(name, str):
        return ("","")
    n = name.replace(".", " ").strip()
    if n and " " not in n:
        caps = [i for i,ch in enumerate(n) if ch.isupper()]
        if len(caps) >= 2:
            return (n[caps[0]].upper(), n[caps[1]:].upper())
    toks = _re_nm2.split(r"\s+", n)
    if not toks or not toks[0]:
        return ("","")
    return (toks[0][0].upper(), toks[-1].upper())

def _nm_add_fallback_keys_df(df, player_col="player"):
    if player_col not in df.columns:
        df["first_initial"] = ""; df["last_name_u"] = ""
        return df
    fi, ln = [], []
    for nm in df[player_col].astype(str):
        f,l = _nm_initial_last(nm); fi.append(f); ln.append(l)
    df["first_initial"] = fi; df["last_name_u"] = ln
    return df
def _tm_norm(s):
    return _normalize_team_names(s.astype(str).str.upper().str.strip())

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
    original_null_mask = s.isna()
    norm = s.astype(str).str.upper().str.strip()
    aliases = {
        # books ↔ nflverse
        "WSH": "WAS", "WDC": "WAS",
        "JAX": "JAX", "JAC": "JAX",
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
    norm = norm.replace(aliases)

    # add full-name -> code mapping (fix odds_game full names)
    full_to_code = {
        "ARIZONA CARDINALS":"ARI","ATLANTA FALCONS":"ATL","BALTIMORE RAVENS":"BAL","BUFFALO BILLS":"BUF",
        "CAROLINA PANTHERS":"CAR","CHICAGO BEARS":"CHI","CINCINNATI BENGALS":"CIN","CLEVELAND BROWNS":"CLE",
        "DALLAS COWBOYS":"DAL","DENVER BRONCOS":"DEN","DETROIT LIONS":"DET","GREEN BAY PACKERS":"GB",
        "HOUSTON TEXANS":"HOU","INDIANAPOLIS COLTS":"IND","JACKSONVILLE JAGUARS":"JAX","KANSAS CITY CHIEFS":"KC",
        "LAS VEGAS RAIDERS":"LV","LOS ANGELES CHARGERS":"LAC","LOS ANGELES RAMS":"LAR","MIAMI DOLPHINS":"MIA",
        "MINNESOTA VIKINGS":"MIN","NEW ENGLAND PATRIOTS":"NE","NEW ORLEANS SAINTS":"NO","NEW YORK GIANTS":"NYG",
        "NEW YORK JETS":"NYJ","PHILADELPHIA EAGLES":"PHI","PITTSBURGH STEELERS":"PIT","SAN FRANCISCO 49ERS":"SF",
        "SEATTLE SEAHAWKS":"SEA","TAMPA BAY BUCCANEERS":"TB","TENNESSEE TITANS":"TEN","WASHINGTON COMMANDERS":"WAS"
    }
    norm = norm.replace(full_to_code)

    empty_sentinels = {"", "NAN", "NONE", "NULL", "NA", "N/A"}
    norm = norm.mask(norm.isin(empty_sentinels))
    norm = norm.mask(original_null_mask)
    return norm

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

def _player_key_series(s: pd.Series) -> pd.Series:
    """Stable key for joins; lowercase, alnum only."""
    return s.fillna("").astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)


def _player_form_consensus_mask(series: pd.Series) -> pd.Series:
    if series is None:
        return pd.Series(dtype=bool)
    norm = series.fillna("").astype(str).str.upper().str.strip()
    return norm.eq("") | norm.eq(PLAYER_FORM_CONSENSUS_OPPONENT)

# ----------------------------
def load_roles() -> pd.DataFrame:
    """
    Load roles from roles_ourlads.csv or roles.csv to backfill team/role/position.
    """
    paths = [os.path.join(DATA_DIR, 'roles_ourlads.csv'), os.path.join(DATA_DIR, 'roles.csv')]
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df.columns = [c.lower() for c in df.columns]
                if df.empty:
                    continue
                if 'team' in df.columns:
                    df['team'] = _normalize_team_names(df['team'])
                for pcol in ['player','player_name','name']:
                    if pcol in df.columns:
                        df = df.rename(columns={pcol:'player'})
                        break
                if 'player' not in df.columns:
                    continue
                df['player'] = _normalize_player_name(df['player'])
                df['player_key'] = _player_key_series(df['player'])
                keep = [c for c in ['player_key','player','team','role','position'] if c in df.columns]
                return df[keep].drop_duplicates()
            except Exception:
                continue
    return pd.DataFrame()
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

    # Pivot side rows into over/under odds if present
    if "side" in df.columns and "price_american" in df.columns:
        keycols = [c for c in ["event_id","player","team","market","line"] if c in df.columns]
        if keycols:
            tmp = df[keycols + ["side","price_american"]].copy()
            tmp["side"] = tmp["side"].astype(str).str.upper().str.strip()
            pvt = tmp.pivot_table(index=keycols, columns="side", values="price_american", aggfunc="first").reset_index()
            pvt.columns = [("over_odds" if c=="OVER" else "under_odds" if c=="UNDER" else c) for c in pvt.columns]
            for c in ["over_odds","under_odds"]:
                if c not in pvt.columns:
                    pvt[c] = np.nan
            df = df.merge(pvt, on=keycols, how="left", suffixes=("","_pvt"))
            for c in ["over_odds","under_odds"]:
                c_p = c + "_pvt"
                if c_p in df.columns:
                    df[c] = df[c].combine_first(df[c_p])
                    df.drop(columns=[c_p], inplace=True)

    return df

def load_team_form() -> pd.DataFrame:
    df = _read_csv(os.path.join(DATA_DIR, "team_form.csv"))
    if df.empty:
        return df
    if "team" in df.columns:
        df["team"] = _normalize_team_names(df["team"])
    return df

def load_player_form() -> pd.DataFrame:
    # prefer consensus if present
    consensus_path = os.path.join(DATA_DIR, "player_form_consensus.csv")
    base_path = os.path.join(DATA_DIR, "player_form.csv")
    df = pd.DataFrame()
    if os.path.exists(consensus_path):
        try:
            df = pd.read_csv(consensus_path)
            df.columns = [c.lower() for c in df.columns]
        except pd.errors.EmptyDataError:
            df = pd.DataFrame()
        except Exception:
            df = pd.DataFrame()
    if df.empty:
        # Fallback to the base player_form extract when consensus has no rows
        df = _read_csv(base_path)
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

    # --- alias column names for compatibility with make_player_form ---
    if "target_share" not in df.columns and "tgt_share" in df.columns:
        df["target_share"] = df["tgt_share"]
    if "yprr_proxy" not in df.columns and "yprr" in df.columns:
        df["yprr_proxy"] = df["yprr"]
    if "rz_carry_share" not in df.columns and "rz_rush_share" in df.columns:
        df["rz_carry_share"] = df["rz_rush_share"]

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

# Preferred game-lines: Odds API file, then fallback to schedule
def _read_csv_any(paths):
    for p in paths:
        if os.path.exists(p):
            try:
                df = pd.read_csv(p)
                df.columns = [c.lower() for c in df.columns]
                if not df.empty:
                    return df.assign(_source=p)
            except Exception:
                pass
    return pd.DataFrame()

def load_game_lines_preferring_odds() -> pd.DataFrame:
    og = _read_csv_any(["outputs/odds_game.csv"])
    if not og.empty and {"home_team","away_team"}.issubset(og.columns):
        gl = og.copy()
    else:
        gl = _read_csv_any(["outputs/game_lines.csv"])

    if gl.empty:
        return gl

    # Normalize teams
    for col in ["home_team","away_team"]:
        if col in gl.columns:
            gl[col] = _normalize_team_names(gl[col])

    # Ensure event_id
    if "event_id" not in gl.columns:
        gl["event_id"] = (gl.get("season", np.nan).astype(str) + "_" +
                          gl.get("week", np.nan).astype(str) + "_" +
                          gl.get("home_team", "").astype(str) + "_" +
                          gl.get("away_team", "").astype(str))

    # Derive week from commence_time if missing
    if ("week" not in gl.columns or gl["week"].isna().all()) and "commence_time" in gl.columns:
        gl["week"] = pd.to_datetime(gl["commence_time"], errors="coerce", utc=True).dt.isocalendar().week.astype("Int64")

    keep = [c for c in ["event_id","home_team","away_team","week","season","commence_time","home_wp","away_wp"] if c in gl.columns]
    return gl[keep].drop_duplicates()

# ----------------------------
# NEW: schedule fallback (surgical)
# ----------------------------

def _load_schedule_long(season: int) -> pd.DataFrame:
    """
    Returns (team, opponent, week) long form for the given season.
    Only used as a fallback when event_id-based opponent inference is missing.
    """
    try:
        try:
            import nflreadpy as nflv  # preferred
            sch = nflv.load_schedules(seasons=[season])
        except Exception:
            import nfl_data_py as nflv  # fallback
            sch = nflv.import_schedules([season])  # type: ignore
    except Exception:
        return pd.DataFrame(columns=["team","opponent","week"])

    df = pd.DataFrame(sch)
    if df.empty:
        return pd.DataFrame(columns=["team","opponent","week"])

    df.columns = [c.lower() for c in df.columns]
    # normalize team codes
    for col in ["home_team","away_team"]:
        if col in df.columns:
            df[col] = _normalize_team_names(df[col])

    if {"home_team","away_team","week"}.issubset(df.columns):
        h = df[["home_team","away_team","week"]].rename(columns={"home_team":"team","away_team":"opponent"})
        a = df[["away_team","home_team","week"]].rename(columns={"away_team":"team","home_team":"opponent"})
        long = pd.concat([h, a], ignore_index=True).dropna()
        long["team"] = _normalize_team_names(long["team"])
        long["opponent"] = _normalize_team_names(long["opponent"])
        # dedupe in case feeds have duplicates
        return long.drop_duplicates()
    return pd.DataFrame(columns=["team","opponent","week"])

def _correct_wr_roles(df: pd.DataFrame) -> pd.DataFrame:
    """
    Re-rank WRs within each team (and week if present) by route_rate (primary) then target_share.
    Assign WR1/WR2/WR3 accordingly. Only overwrite if role is missing or disagrees with rank.
    """
    if df.empty:
        return df
    keys = []
    if 'team' in df.columns: keys.append('team')
    if 'week' in df.columns: keys.append('week')
    if not keys:
        return df
    poscol = 'position' if 'position' in df.columns else None
    if poscol is None:
        return df
    wr_mask = df[poscol].fillna('').astype(str).str.upper().str.startswith('WR')
    if not wr_mask.any():
        return df
    rr = df.get('route_rate')
    ts = df.get('target_share')
    if rr is None:
        df['route_rate'] = 0.0
        rr = df['route_rate']
    if ts is None:
        df['target_share'] = 0.0
        ts = df['target_share']
    df['_wr_sort_rr'] = rr.fillna(0).astype(float)
    df['_wr_sort_ts'] = ts.fillna(0).astype(float)
    def assign_role(g):
        g = g.sort_values(by=['_wr_sort_rr','_wr_sort_ts'], ascending=[False, False]).copy()
        new_roles = []
        for i, idx in enumerate(g.index, start=1):
            tag = f'WR{i}' if i <= 5 else 'WR5+'
            new_roles.append(tag)
        g['_wr_new_role'] = new_roles
        return g
    df_wr = df[wr_mask].copy()
    df_wr = df_wr.groupby(keys, dropna=False, group_keys=False).apply(assign_role)
    df = df.merge(df_wr[['_wr_new_role']], left_index=True, right_index=True, how='left')
    if 'role' not in df.columns:
        df['role'] = np.nan
    need_fix = df['_wr_new_role'].notna() & (df['role'].isna() | (~df['role'].astype(str).str.upper().eq(df['_wr_new_role'])))
    df.loc[need_fix, 'role'] = df.loc[need_fix, '_wr_new_role']
    df.drop(columns=['_wr_new_role','_wr_sort_rr','_wr_sort_ts'], inplace=True, errors='ignore')
    return df
# ----------------------------
# Assembler
# ----------------------------

def build_metrics(season: int) -> pd.DataFrame:
    props = load_props()
    if props.empty:
        return pd.DataFrame(columns=[
            "event_id","player","team","opponent","market","line","over_odds","under_odds",
            "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","ypa_prior","rz_tgt_share","rz_carry_share","position","role",
            "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
            "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
            "cb_penalty","injury_status","wind_mph","temp_f","precip","team_wp","season",
            "week","opp_plays_wk","opp_pace_wk","opp_proe_wk"
        ])

    pf  = load_player_form()
    tf  = load_team_form()
    cov = load_coverage()
    cba = load_cb_assignments()
    inj = load_injuries()
    wx  = load_weather()
    gl  = load_game_lines_preferring_odds()

    # Week inference for props (if missing)
    if "week" not in props.columns or props["week"].isna().all():
        if "commence_time" in props.columns:
            props["week"] = pd.to_datetime(props["commence_time"], errors="coerce", utc=True)\
                                .dt.isocalendar().week.astype("Int64")
        elif not gl.empty and {"event_id","commence_time"}.issubset(gl.columns) and "event_id" in props.columns:
            wk_src = gl[["event_id","commence_time"]].dropna().copy()
            wk_src["week"] = pd.to_datetime(wk_src["commence_time"], errors="coerce", utc=True)\
                                .dt.isocalendar().week.astype("Int64")
            props = props.merge(wk_src[["event_id","week"]], on="event_id", how="left", suffixes=("", "_gl"))
            if "week_gl" in props.columns:
                props["week"] = props["week"].combine_first(props["week_gl"])
                props.drop(columns=["week_gl"], inplace=True, errors="ignore")
    if "week" not in props.columns:
        props["week"] = pd.Series(pd.array([], dtype="Int64"))

    # Stable player keys for joins
    props["player_key"] = _player_key_series(props.get("player", pd.Series(dtype=object)))
    pf_consensus = pd.DataFrame()
    pf_by_opponent = pd.DataFrame()
    if not pf.empty:
        pf["player_key"] = _player_key_series(pf.get("player", pd.Series(dtype=object)))
        if "opponent" in pf.columns:
            consensus_mask = _player_form_consensus_mask(pf["opponent"])
        else:
            consensus_mask = pd.Series(True, index=pf.index)
        pf_consensus = pf.loc[consensus_mask].copy()
        pf_by_opponent = pf.loc[~consensus_mask].copy()
        if pf_consensus.empty:
            pf_consensus = pf.copy()
    else:
        pf_consensus = pf.copy()
        pf_by_opponent = pd.DataFrame(columns=pf.columns)

    # Backfill team from player_form if all missing
    team_fill_source = pf_consensus if not pf_consensus.empty else pf
    if (
        "team" in props.columns
        and props["team"].isna().all()
        and not team_fill_source.empty
        and {"player", "team"}.issubset(team_fill_source.columns)
    ):
        props = props.merge(team_fill_source[["player","team"]].drop_duplicates(), on="player", how="left", suffixes=("","_pf"))
        props["team"] = props["team"].combine_first(props.get("team_pf"))
        if "team_pf" in props.columns:
            props = props.drop(columns=["team_pf"])

    # a few sources don't carry event_id; keep NaN and we still keep the rows
    if "event_id" not in props.columns:
        props["event_id"] = np.nan

    # Backfill team/role/position from roles files if still missing
    try:
        roles_df = load_roles()
        if not roles_df.empty:
            props = props.merge(roles_df, on='player_key', how='left', suffixes=('', '_roles'))
            if 'team' in props.columns and 'team_roles' in props.columns:
                props['team'] = props['team'].combine_first(props['team_roles'])
                props.drop(columns=['team_roles'], inplace=True)
            if 'role' in props.columns and 'role_roles' in props.columns:
                props['role'] = props['role'].combine_first(props['role_roles'])
                props.drop(columns=['role_roles'], inplace=True)
            elif 'role_roles' in props.columns:
                props.rename(columns={'role_roles':'role'}, inplace=True)
            if 'position' in props.columns and 'position_roles' in props.columns:
                props['position'] = props['position'].combine_first(props['position_roles'])
                props.drop(columns=['position_roles'], inplace=True)
            elif 'position_roles' in props.columns:
                props.rename(columns={'position_roles':'position'}, inplace=True)
    except Exception as _e:
        print('[make_metrics] roles fill skipped:', _e)

    # Fallback fill: match roles by (first_initial, last_name_u) when player_key failed
    try:
        props = _nm_add_fallback_keys_df(props, "player")
        try:
            roles_df = roles_df.copy()
        except NameError:
            roles_df = load_roles()
        if not roles_df.empty:
            roles_df = _nm_add_fallback_keys_df(roles_df, "player")
            cols_key = ["first_initial","last_name_u"]
            fill_cols = [c for c in ["team","position","role"] if c in roles_df.columns]
            if fill_cols:
                fb = roles_df[cols_key + fill_cols].drop_duplicates()
                before_team_missing = props["team"].isna().sum() if "team" in props.columns else None
                props = props.merge(fb, on=cols_key, how="left", suffixes=("","_fbroles"))
                if "team" in props.columns and "team_fbroles" in props.columns:
                    props["team"] = props["team"].combine_first(props["team_fbroles"])
                if "position" in props.columns and "position_fbroles" in props.columns:
                    props["position"] = props["position"].combine_first(props["position_fbroles"])
                if "role" in props.columns and "role_fbroles" in props.columns:
                    props["role"] = props["role"].combine_first(props["role_fbroles"])
                props.drop(columns=[c for c in props.columns if c.endswith("_fbroles")], inplace=True, errors="ignore")
    except Exception as _e:
        print("[make_metrics] roles initials/last fallback skipped:", _e)
    # ---- Opponent mapping
    if not gl.empty:
        if "event_id" in props.columns and "event_id" in gl.columns:
            cols_have = [c for c in ["event_id","home_team","away_team","week","home_wp","away_wp"] if c in gl.columns]
            tmp = props.merge(gl[cols_have], on="event_id", how="left")
            opp = np.where(tmp.get("team").eq(tmp.get("home_team")), tmp.get("away_team"),
                   np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("home_team"), np.nan))
            props["opponent"] = _normalize_team_names(pd.Series(opp, index=props.index))
            # carry week from gl if needed
            if "week" not in props.columns or props["week"].isna().all():
                props["week"] = tmp.get("week")
            # team win prob
            if {"home_wp","away_wp"}.issubset(tmp.columns):
                props["team_wp"] = np.where(
                    tmp.get("team").eq(tmp.get("home_team")),
                    tmp.get("home_wp"),
                    np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("away_wp"), np.nan),
                )
            else:
                if "team_wp" not in props.columns:
                    props["team_wp"] = np.nan
        elif {"team","week"}.issubset(props.columns) and {"home_team","away_team","week"}.issubset(gl.columns):
            cols_have_w = [c for c in ["home_team","away_team","week","home_wp","away_wp"] if c in gl.columns]
            left = props.merge(gl[cols_have_w], on="week", how="left")
            opp = np.where(left.get("team").eq(left.get("home_team")), left.get("away_team"),
                   np.where(left.get("team").eq(left.get("away_team")), left.get("home_team"), np.nan))
            props["opponent"] = _normalize_team_names(pd.Series(opp, index=props.index))
            if {"home_wp","away_wp"}.issubset(left.columns):
                props["team_wp"] = np.where(
                    left.get("team").eq(left.get("home_team")),
                    left.get("home_wp"),
                    np.where(left.get("team").eq(left.get("away_team")), left.get("away_wp"), np.nan),
                )

    # Base: props + player_form (non-destructive)
    base = props.copy()
    if not pf_consensus.empty:
        keep_pf = ["player","team","target_share","rush_share","route_rate","yprr_proxy","ypt","ypc",
                   "rz_tgt_share","rz_carry_share","position","role","season","player_key","week","opponent","ypa_prior"]
        keep_pf = [c for c in keep_pf if c in pf_consensus.columns]
        if keep_pf:
            base = base.merge(pf_consensus[keep_pf].drop_duplicates(), on=["player_key"], how="left", suffixes=("","_pf"))
        # backfill props.team from pf if missing
        if "team" in base.columns and "team_pf" in base.columns:
            base["team"] = base["team"].combine_first(base["team_pf"])
            base.drop(columns=[c for c in ["team_pf"] if c in base.columns], inplace=True)
    # --- fallback usage merge on (team, position, first_initial, last_name_u) ---
    try:
        # ensure initials/last exist
        base = _nm_add_fallback_keys_df(base, "player")
        pf_fb = pf_consensus.copy()
        pf_fb = _nm_add_fallback_keys_df(pf_fb, "player")
        cols_key = [c for c in ["team","position","first_initial","last_name_u"] if c in base.columns and c in pf_fb.columns]
        fb_cols = [c for c in ["target_share","route_rate","rush_share","yprr_proxy","ypt","ypc","ypa_prior","rz_tgt_share","rz_carry_share","role","position"] if c in pf_fb.columns]
        if fb_cols and len(cols_key) >= 3:
            # choose best candidate per key (highest route_rate then target_share)
            sort_cols = [c for c in ["route_rate","target_share"] if c in pf_fb.columns]
            if sort_cols:
                pf_fb = pf_fb.sort_values(cols_key + sort_cols, ascending=[True,True,True,True] + [False]*len(sort_cols))
            pf_fb = pf_fb.drop_duplicates(subset=cols_key, keep="first")
            miss_mask = base.get("target_share").isna() if "target_share" in base.columns else base.get("route_rate").isna()
            if miss_mask is None:
                miss_mask = base.index == -1  # no-op
            if miss_mask.any():
                fill = base.loc[miss_mask, cols_key].merge(pf_fb[cols_key + fb_cols], on=cols_key, how="left")
                for c in fb_cols:
                    if c in base.columns and c in fill.columns:
                        base.loc[miss_mask, c] = base.loc[miss_mask, c].combine_first(fill[c])
    except Exception as _e:
        print("[make_metrics] fallback usage merge skipped:", _e)


    # carry week explicitly
    if "week" not in base.columns and "week" in props.columns:
        base["week"] = props["week"]

    # Fix WR roles by usage (route_rate then target_share)
    base = _correct_wr_roles(base)
    # Attach team env (pace/proe/rz/12p/slot/ay + boxes)
    if not tf.empty:
        keep_tf = ["team","def_pass_epa","def_rush_epa","def_sack_rate",
                   "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
                   "light_box_rate","heavy_box_rate","season"]
        keep_tf = [c for c in keep_tf if c in tf.columns]
        base = base.merge(tf[keep_tf].drop_duplicates(), on=["team"], how="left")

    # Attach opponent defenses (EPA/sacks/boxes)
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
    else:
        for c in ["def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp"]:
            if c not in base.columns:
                base[c] = np.nan

    # Coverage tags (opponent defense)
    cov = load_coverage()
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
    w = load_weather()
    if not w.empty and "event_id" in base.columns:
        wx_keep = [c for c in ["event_id","wind_mph","temp_f","precip"] if c in w.columns]
        base = base.merge(w[wx_keep].drop_duplicates(), on="event_id", how="left")
    else:
        for c in ["wind_mph","temp_f","precip"]:
            if c not in base.columns:
                base[c] = np.nan

    # stamp season explicitly
    base["season"] = season

    # Default QB YPA prior if missing
    if "ypa_prior" in base.columns:
        pos_col = base.get("position") if "position" in base.columns else None
        if pos_col is not None:
            qb_mask = pos_col.astype(str).str.upper().str.startswith("QB")
            base.loc[qb_mask & base["ypa_prior"].isna(), "ypa_prior"] = 6.8

    # Final tidy and order
    # --- injected: attach coverage (opponent-level flags) and CB assignments (player-level) ---
    try:
        _cov_df = load_coverage()
    except Exception:
        _cov_df = pd.DataFrame()
    if not _cov_df.empty:
        _cov2 = _cov_df.rename(columns={"defense_team":"opponent"})
        for _c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            if _c not in _cov2.columns:
                _cov2[_c] = 0
        base = base.merge(
            _cov2[["opponent","coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]].drop_duplicates(),
            on="opponent", how="left"
        )
        for _c in ["coverage_top_shadow","coverage_heavy_man","coverage_heavy_zone"]:
            if _c + "_opp" not in base.columns:
                base[_c + "_opp"] = base[_c]
        if "coverage_top_shadow_opp" in base.columns:
            base["has_cb_shadow"] = (base["coverage_top_shadow_opp"].fillna(0).astype(int) == 1).astype(int)
        else:
            base["has_cb_shadow"] = 0

    try:
        _cba_df = load_cb_assignments()
    except Exception:
        _cba_df = pd.DataFrame()
    if not _cba_df.empty:
        _cba2 = _cba_df.rename(columns={"opp_def_team":"opponent"})
        base = base.merge(_cba2, on=["player","opponent"], how="left", suffixes=("","_cba"))
        need_fb = base["penalty"].isna().mean() > 0.80 if "penalty" in base.columns else True
        if need_fb:
            if "first_initial" not in base.columns or "last_name_u" not in base.columns:
                fi, ln = _player_init_last(base["player"])
                base["first_initial"] = base.get("first_initial", fi)
                base["last_name_u"] = base.get("last_name_u", ln)
            _cba2["fi"], _cba2["ln"] = _player_init_last(_cba2["player"] if "player" in _cba2.columns else pd.Series([], dtype=object))
            join_left = ["first_initial","last_name_u","opponent"]
            join_right= ["fi","ln","opponent"]
            fb_cols = ["penalty"]
            _t = base[join_left].merge(_cba2[join_right+fb_cols].drop_duplicates(), left_on=join_left, right_on=join_right, how="left")
            if "penalty" in _t.columns:
                if "penalty" not in base.columns:
                    base["penalty"] = np.nan
                base["penalty"] = base["penalty"].combine_first(_t["penalty"])
        base["cb_penalty"] = base.get("penalty")
        if "penalty" in base.columns:
            base.drop(columns=["penalty"], inplace=True, errors="ignore")
    else:
        if "cb_penalty" not in base.columns:
            base["cb_penalty"] = np.nan


    want = [
        "event_id","player","team","opponent","market","line","over_odds","under_odds",
        "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","ypa_prior","rz_tgt_share","rz_carry_share","position","role",
        "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
        "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
        "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
        "cb_penalty","injury_status","wind_mph","temp_f","precip","team_wp","season",
        "week","opp_plays_wk","opp_pace_wk","opp_proe_wk"
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
        # ## -- injected: final odds rejoin (idempotent) --
        try:
            props_raw = pd.read_csv(os.path.join("outputs","props_raw.csv"))
            props_raw.columns = [c.lower() for c in props_raw.columns]
            if {"side","price_american"}.issubset(props_raw.columns):
                k = [c for c in ["event_id","player","market","line"] if c in props_raw.columns]
                if k:
                    tmp = props_raw[k + ["side","price_american"]].copy()
                    tmp["side"] = tmp["side"].astype(str).str.upper().str.strip()
                    pvt = tmp.pivot_table(index=k, columns="side", values="price_american", aggfunc="first").reset_index()
                    pvt = pvt.rename(columns={"OVER":"over_odds","UNDER":"under_odds"})
                    join_key = [c for c in ["event_id","player","market","line"] if c in df.columns and c in pvt.columns]
                    if join_key:
                        df = df.merge(pvt[join_key+["over_odds","under_odds"]], on=join_key, how="left", suffixes=("","_fromprops"))
                        for c in ["over_odds","under_odds"]:
                            alt = f"{c}_fromprops"
                            if alt in df.columns:
                                df[c] = df[c].combine_first(df[alt])
                                df.drop(columns=[alt], inplace=True)
        except Exception as _e:
            print("[make_metrics] final odds rejoin skipped:", _e)

        # ## -- injected: final player_form_consensus backfill (idempotent) --
        try:
            pfc = pd.read_csv(os.path.join("data","player_form_consensus.csv"))
            pfc.columns = [c.lower() for c in pfc.columns]
            # normalize expected columns
            rename_map = {
                "tgt_share": "target_share",
                "yprr": "yprr_proxy",
                "rz_rush_share": "rz_carry_share",
            }
            for k,v in rename_map.items():
                if k in pfc.columns and v not in pfc.columns:
                    pfc[v] = pfc[k]

            # initials+last on both sides
            def _mk_init_last(series):
                import re as _re
                fi, ln = [], []
                for nm in series.astype(str):
                    n = nm.replace("."," ").strip()
                    if n and " " not in n:
                        caps = [i for i,ch in enumerate(n) if ch.isupper()]
                        if len(caps) >= 2:
                            fi.append(n[caps[0]].upper()); ln.append(n[caps[1]:].upper()); continue
                    toks = _re.split(r"\\s+", n)
                    if toks and toks[0]:
                        fi.append(toks[0][0].upper()); ln.append(toks[-1].upper())
                    else:
                        fi.append(""); ln.append("")
                return fi, ln

            if "first_initial" not in df.columns or "last_name_u" not in df.columns:
                fi, ln = _mk_init_last(df["player"] if "player" in df.columns else pd.Series([]))
                df["first_initial"] = df.get("first_initial", pd.Series(fi))
                df["last_name_u"]   = df.get("last_name_u",  pd.Series(ln))

            if "first_initial" not in pfc.columns or "last_name_u" not in pfc.columns:
                fi2, ln2 = _mk_init_last(pfc["player"] if "player" in pfc.columns else pd.Series([]))
                pfc["first_initial"] = fi2
                pfc["last_name_u"] = ln2

            # choose keys: prefer (team, position, initial, last); otherwise (team, initial, last)
            keys_full = [k for k in ["team","position","first_initial","last_name_u"] if k in df.columns and k in pfc.columns]
            keys_lo   = [k for k in ["team","first_initial","last_name_u"] if k in df.columns and k in pfc.columns]
            use_keys = keys_full if len(keys_full)==4 else (keys_lo if len(keys_lo)==3 else None)

            if use_keys:
                use_cols = [c for c in [
                    "target_share","route_rate","rush_share","yprr_proxy","ypt","ypc","ypa","ypa_prior",
                    "rz_share","rz_tgt_share","rz_carry_share","role","position"
                ] if c in pfc.columns]

                if use_cols:
                    sort_cols = [c for c in ["route_rate","target_share"] if c in pfc.columns]
                    pf_fb = pfc.sort_values(use_keys + sort_cols, ascending=[True]*len(use_keys) + [False]*len(sort_cols))\
                               .drop_duplicates(subset=use_keys, keep="first")

                    # only fill rows that lack usage currently
                    if set(["target_share","route_rate","rush_share"]).issubset(df.columns):
                        miss_mask = (~df[["target_share","route_rate","rush_share"]].notna().any(axis=1))
                    else:
                        miss_mask = pd.Series([True]*len(df))

                    fill = df.loc[miss_mask, use_keys].merge(pf_fb[use_keys + use_cols], on=use_keys, how="left")
                    for c in use_cols:
                        if c not in df.columns:
                            df[c] = np.nan
                        if c in fill.columns:
                            df.loc[miss_mask, c] = df.loc[miss_mask, c].combine_first(fill[c])

        except Exception as _e:
            print("[make_metrics] final pf_consensus backfill skipped:", _e)
    
        # ## -- injected: final odds rejoin --
        try:
            props_raw = pd.read_csv(os.path.join("outputs","props_raw.csv"))
            props_raw.columns = [c.lower() for c in props_raw.columns]
            if {"side","price_american"}.issubset(props_raw.columns):
                key = [c for c in ["event_id","player","market","line"] if c in props_raw.columns]
                if key:
                    tmp = props_raw[key + ["side","price_american"]].copy()
                    tmp["side"] = tmp["side"].astype(str).str.upper().str.strip()
                    pvt = tmp.pivot_table(index=key, columns="side", values="price_american", aggfunc="first").reset_index()
                    pvt = pvt.rename(columns={"OVER":"over_odds","UNDER":"under_odds"})
                    # attach where missing
                    join_key = [c for c in ["event_id","player","market","line"] if c in df.columns and c in pvt.columns]
                    if join_key:
                        df = df.merge(pvt[join_key+["over_odds","under_odds"]], on=join_key, how="left", suffixes=("","_fromprops"))
                        for c in ["over_odds","under_odds"]:
                            alt = f"{c}_fromprops"
                            if alt in df.columns:
                                df[c] = df[c].combine_first(df[alt])
                                df.drop(columns=[alt], inplace=True)
        except Exception as _e:
            print("[make_metrics] final odds rejoin skipped:", _e)

    except Exception as e:
        print(f"[make_metrics] ERROR: {e}", file=sys.stderr)
        traceback.print_exc()
        # emit empty but schema-correct file so pipeline continues
        df = pd.DataFrame(columns=[
            "event_id","player","team","opponent","market","line","over_odds","under_odds",
            "target_share","rush_share","route_rate","yprr_proxy","ypt","ypc","ypa_prior","rz_tgt_share","rz_carry_share","position","role",
            "pace","proe","rz_rate","12p_rate","slot_rate","ay_per_att",
            "def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp","light_box_rate_opp","heavy_box_rate_opp",
            "coverage_top_shadow_opp","coverage_heavy_man_opp","coverage_heavy_zone_opp",
            "cb_penalty","injury_status","wind_mph","temp_f","precip","team_wp","season",
            "week","opp_plays_wk","opp_pace_wk","opp_proe_wk"
        ])

    # Optional: add opponent weekly env (pace/proe/plays_est) if present
    try:
        tfw = _read_csv(os.path.join(DATA_DIR, "team_form_weekly.csv"))
        if (
            not df.empty
            and not tfw.empty
            and {"opponent","week"}.issubset(df.columns)
            and {"team","week"}.issubset(tfw.columns)
        ):
            tfw = tfw.rename(columns={"team":"opponent"})
            cols = [c for c in ["opponent","week","plays_est","pace","proe"] if c in tfw.columns]
            tfw = tfw[cols].drop_duplicates().rename(columns={
                "plays_est":"opp_plays_wk","pace":"opp_pace_wk","proe":"opp_proe_wk"
            })
            df = df.merge(tfw, on=["opponent","week"], how="left")
            for c in ["opp_plays_wk","opp_pace_wk","opp_proe_wk"]:
                if c not in df.columns:
                    df[c] = np.nan
    except Exception:
        pass

    df.replace({'NAN': np.nan, 'nan': np.nan, 'NaN': np.nan, 'None': np.nan, '': np.nan}, inplace=True)
    # FINAL_OPPONENT_SANITY
    if 'opponent' in df.columns:
        df['opponent'] = df['opponent'].replace({'NAN': np.nan, 'nan': np.nan, 'NaN': np.nan, '': np.nan})
    df.to_csv(OUTPATH, index=False)
    print(f"[make_metrics] Wrote {len(df)} rows → {OUTPATH}")

if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        main()
