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

DATA_DIR = "data"
OUTPATH  = os.path.join(DATA_DIR, "metrics_ready.csv")

PLAYER_FORM_CONSENSUS_OPPONENT = "ALL"

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
    # ---- Opponent mapping
    if not gl.empty:
        if "event_id" in props.columns and "event_id" in gl.columns:
            tmp = props.merge(gl[["event_id","home_team","away_team","week","home_wp","away_wp"]], on="event_id", how="left")
            opp = np.where(tmp.get("team").eq(tmp.get("home_team")), tmp.get("away_team"),
                   np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("home_team"), np.nan))
            props["opponent"] = _normalize_team_names(pd.Series(opp, index=props.index))
            # carry week from gl if needed
            if "week" not in props.columns or props["week"].isna().all():
                props["week"] = tmp.get("week")
            # team win prob
            props["team_wp"] = np.where(
                tmp.get("team").eq(tmp.get("home_team")),
                tmp.get("home_wp"),
                np.where(tmp.get("team").eq(tmp.get("away_team")), tmp.get("away_wp"), np.nan),
            )
        elif {"team","week"}.issubset(props.columns) and {"home_team","away_team","week"}.issubset(gl.columns):
            left = props.merge(gl[["home_team","away_team","week","home_wp","away_wp"]], on="week", how="left")
            opp = np.where(left.get("team").eq(left.get("home_team")), left.get("away_team"),
                   np.where(left.get("team").eq(left.get("away_team")), left.get("home_team"), np.nan))
            props["opponent"] = _normalize_team_names(pd.Series(opp, index=props.index))
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
