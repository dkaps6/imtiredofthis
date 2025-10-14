# scripts/make_metrics.py
# Builds a single, model-ready table with guaranteed columns for pricing.
# Inputs (best-effort, all optional and safely defaulted):
#   data/team_form.csv              # team-level: EPA splits, sack rate, pace, PROE, box rates, etc.
#   data/player_form.csv            # player-level: target_share, rush_share, yprr_proxy, ypc, qb_ypa, route_profile_*
#   data/injuries.csv               # statuses: Out/Doubtful/Questionable/Limited/Probable
#   data/coverage.csv               # defense tags/rates: heavy_man, heavy_zone, press_rate, man_rate, zone_rate
#   data/cb_assignments.csv         # matchups: defense_team, receiver, cb, cb_penalty (0..0.25)
#   data/weather.csv                # event_id, wind_mph, temp_f, precip
#   outputs/game_lines.csv OR outputs/odds_game.csv
#
# Output:
#   data/metrics_ready.csv
#   outputs/metrics/metrics_ready.csv (debug mirror)

from __future__ import annotations
import pandas as pd
import numpy as np
from pathlib import Path

REQ_COLS = {
    # join keys / id
    "event_id", "player", "team", "opponent", "position",

    # player volumes/priors
    "target_share", "rush_share", "route_rate",
    "yprr_proxy", "ypc", "qb_ypa",

    # environment / opponent
    "def_pass_epa_opp", "def_rush_epa_opp", "def_sack_rate_opp",
    "light_box_rate_opp", "heavy_box_rate_opp",
    "pace_opp", "proe_opp",

    # script/lines
    "team_wp",

    # injuries + coverage + cb
    "status",
    "coverage_tags",
    "man_rate_opp", "zone_rate_opp", "press_rate_opp",
    "cb_penalty",

    # weather
    "wind_mph", "temp_f", "precip",

    # route profiles (share of time)
    "route_profile_press", "route_profile_man", "route_profile_zone",
}

def _read_csv(path: str, usecols=None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

def _safe_cols(df: pd.DataFrame, cols: set[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out

def _status_simplify(s: str) -> str:
    s = str(s or "").strip().lower()
    if s in {"out", "doubtful"}:
        return "Out"
    if s in {"questionable", "q"}:
        return "Questionable"
    if s in {"limited", "ltd", "lp"}:
        return "Limited"
    if s in {"probable", "p"}:
        return "Probable"
    return "Healthy"

def _cb_penalty_merge(players: pd.DataFrame, cbs: pd.DataFrame) -> pd.Series:
    if players.empty or cbs.empty:
        return pd.Series(0.0, index=players.index)
    cols_needed = {"defense_team","receiver","cb_penalty"}
    missing = cols_needed - set(cbs.columns)
    if missing:
        cbs = cbs.copy()
        if "team" in cbs.columns and "defense_team" not in cbs.columns:
            cbs = cbs.rename(columns={"team":"defense_team"})
        if "penalty" in cbs.columns and "cb_penalty" not in cbs.columns:
            cbs = cbs.rename(columns={"penalty":"cb_penalty"})
        if "cb_penalty" not in cbs.columns and "quality" in cbs.columns:
            def _pen_from_quality(q):
                q = str(q or "").strip().lower()
                return {"elite":0.20, "good":0.12, "avg":0.05}.get(q, 0.10)
            cbs["cb_penalty"] = cbs["quality"].map(_pen_from_quality)
    tmp = players[["opponent", "player"]].rename(columns={"opponent":"defense_team", "player":"receiver"})
    tmp = tmp.merge(cbs[["defense_team","receiver","cb","cb_penalty"]].drop_duplicates(),
                    on=["defense_team","receiver"], how="left")
    return tmp["cb_penalty"].fillna(0.0)

def _coverage_rates(opponent: str, coverage: pd.DataFrame) -> tuple[float,float,float]:
    if coverage.empty:
        return (0.33, 0.5, 0.5)
    cov = coverage.copy()
    if "team" not in cov.columns and "defense_team" in cov.columns:
        cov = cov.rename(columns={"defense_team":"team"})
    if "tags" not in cov.columns and "tag" in cov.columns:
        tags_by_team = (
            cov.groupby("team")["tag"]
            .apply(lambda s: ",".join(sorted(set(str(x).strip().lower() for x in s if pd.notna(x)))))
            .rename("tags")
            .reset_index()
        )
        cov = cov.drop(columns=["tag"]).drop_duplicates()
        cov = cov.merge(tags_by_team, on="team", how="left")
    row = cov.loc[cov.get("team","").astype(str).str.upper() == str(opponent).upper()]
    if row.empty:
        return (0.33, 0.5, 0.5)
    pr = float(row.iloc[0].get("press_rate", np.nan))
    mr = float(row.iloc[0].get("man_rate", np.nan))
    zr = float(row.iloc[0].get("zone_rate", np.nan))
    pr = 0.33 if np.isnan(pr) else max(0.0, min(1.0, pr))
    mr = 0.5  if np.isnan(mr) else max(0.0, min(1.0, mr))
    zr = 0.5  if np.isnan(zr) else max(0.0, min(1.0, zr))
    if mr + zr == 0:
        mr, zr = 0.5, 0.5
    return (pr, mr, zr)

def main(season: int | None = None) -> None:
    # Load all sources (best-effort)
    team_form   = _read_csv("data/team_form.csv")
    player_form = _read_csv("data/player_form.csv")
    injuries    = _read_csv("data/injuries.csv")
    coverage    = _read_csv("data/coverage.csv")
    cb_asgn     = _read_csv("data/cb_assignments.csv")
    weather     = _read_csv("data/weather.csv")

    # --- NEW: accept either game_lines.csv or odds_game.csv
    glines = _read_csv("outputs/game_lines.csv")
    if glines.empty:
        glines = _read_csv("outputs/odds_game.csv")

    # Normalize key columns
    for df in (team_form, player_form, injuries, coverage, cb_asgn, weather, glines):
        if not df.empty:
            for c in ("team","opponent","home_team","away_team"):
                if c in df.columns:
                    df[c] = df[c].astype(str)

    # --- z->raw passthrough if only *_z exist
    if not team_form.empty:
        z_to_raw = {
            "def_pass_epa_z": "def_pass_epa",
            "def_rush_epa_z": "def_rush_epa",
            "def_sack_rate_z": "def_sack_rate",
            "pace_z": "pace",
            "proe_z": "proe",
            "light_box_rate_z": "light_box_rate",
            "heavy_box_rate_z": "heavy_box_rate",
        }
        for zcol, raw in z_to_raw.items():
            if raw not in team_form.columns and zcol in team_form.columns:
                team_form[raw] = team_form[zcol]

    # --- coverage schema normalization + default rates
    if not coverage.empty:
        if "team" not in coverage.columns and "defense_team" in coverage.columns:
            coverage = coverage.rename(columns={"defense_team": "team"})
        if "tags" not in coverage.columns and "tag" in coverage.columns:
            tags_by_team = (
                coverage.groupby("team")["tag"]
                .apply(lambda s: ",".join(sorted(set(str(x).strip().lower() for x in s if pd.notna(x)))))
                .rename("tags")
                .reset_index()
            )
            coverage = coverage.drop(columns=["tag"]).drop_duplicates()
            coverage = coverage.merge(tags_by_team, on="team", how="left")
        for c, default in [("press_rate", 0.33), ("man_rate", 0.50), ("zone_rate", 0.50)]:
            if c not in coverage.columns:
                coverage[c] = default

    # Base player frame
    if player_form.empty:
        player_form = pd.DataFrame(columns=["player","team","position"])
    player_form["position"] = player_form.get("position","").fillna("")

    # Ensure join keys
    if "event_id" not in player_form.columns:
        player_form["event_id"] = np.nan
    if "opponent" not in player_form.columns:
        player_form["opponent"] = np.nan

    # Injuries
    if not injuries.empty:
        injuries = injuries.rename(columns={"status":"status_raw"})
        injuries["status"] = injuries["status_raw"].apply(_status_simplify)
        injuries = injuries[["player","status"]]
    else:
        injuries = pd.DataFrame(columns=["player","status"])
    players = player_form.merge(injuries, on="player", how="left")
    players["status"] = players["status"].fillna("Healthy")

    # ---- carry team enrich columns when present
    team_enrich_candidates = {
        "pass_rush_win_rate","pressure_rate","run_stop_rate",
        "man_coverage_rate","zone_coverage_rate",
        "light_box_rate_obs","heavy_box_rate_obs"
    }

    if not team_form.empty:
        tf = team_form.copy()
        cols_keep = [
            "event_id","team","opponent","def_pass_epa","def_rush_epa","def_sack_rate",
            "light_box_rate","heavy_box_rate","pace","proe"
        ]
        # include any enrich cols that exist
        cols_keep += [c for c in team_enrich_candidates if c in tf.columns]
        for c in cols_keep:
            if c not in tf.columns:
                tf[c] = np.nan
        tf = tf[cols_keep]
        players = players.merge(tf, on=["event_id","team"], how="left", suffixes=("","_team"))
        players["def_pass_epa_opp"]   = players.get("def_pass_epa",   np.nan)
        players["def_rush_epa_opp"]   = players.get("def_rush_epa",   np.nan)
        players["def_sack_rate_opp"]  = players.get("def_sack_rate",  np.nan)
        players["light_box_rate_opp"] = players.get("light_box_rate", np.nan)
        players["heavy_box_rate_opp"] = players.get("heavy_box_rate", np.nan)
        players["pace_opp"]           = players.get("pace",           np.nan)
        players["proe_opp"]           = players.get("proe",           np.nan)
    else:
        for c in ["def_pass_epa_opp","def_rush_epa_opp","def_sack_rate_opp",
                  "light_box_rate_opp","heavy_box_rate_opp","pace_opp","proe_opp"]:
            players[c] = np.nan

    # Route profile defaults if missing
    for c in ["route_profile_press","route_profile_man","route_profile_zone"]:
        if c not in players.columns:
            players[c] = np.nan
    players[["route_profile_press","route_profile_man","route_profile_zone"]] = \
        players[["route_profile_press","route_profile_man","route_profile_zone"]].fillna(0.33)

    # CB penalty per receiver
    players["cb_penalty"] = _cb_penalty_merge(players, cb_asgn)

    # Weather merge
    if "wind_mph" not in players.columns: players["wind_mph"] = np.nan
    if "temp_f"   not in players.columns: players["temp_f"]   = np.nan
    if "precip"   not in players.columns: players["precip"]   = np.nan
    if not weather.empty:
        for c in ("event_id","wind_mph","temp_f","precip"):
            if c not in weather.columns: weather[c] = np.nan
        players = players.merge(weather[["event_id","wind_mph","temp_f","precip"]],
                                on="event_id", how="left", suffixes=("","_w"))
        players["wind_mph"] = players["wind_mph"].fillna(players.get("wind_mph_w"))
        players["temp_f"]   = players["temp_f"].fillna(players.get("temp_f_w"))
        players["precip"]   = players["precip"].fillna(players.get("precip_w"))
        for c in ("wind_mph_w","temp_f_w","precip_w"):
            if c in players.columns: players.drop(columns=[c], inplace=True)

    # team_wp from lines
    players["team_wp"] = np.nan
    if not glines.empty:
        if {"home_wp","away_wp","event_id","home_team","away_team"}.issubset(glines.columns):
            players = players.merge(glines[["event_id","home_team","away_team","home_wp","away_wp"]],
                                    on="event_id", how="left")
            players["team_wp"] = np.where(players["team"]==players["home_team"], players["home_wp"],
                                   np.where(players["team"]==players["away_team"], players["away_wp"], np.nan))
            players.drop(columns=[c for c in ["home_team","away_team","home_wp","away_wp"] if c in players.columns], inplace=True)
        else:
            has_h2h = ("market" in glines.columns) and (glines["market"].astype(str).str.lower()=="h2h").any()
            if has_h2h and "price_american" in glines.columns and "outcome" in glines.columns:
                g = glines.copy()
                h2h = g[g["market"].astype(str).str.lower()=="h2h"].copy()
                def _p_from_american(o):
                    try:
                        o = float(o)
                        return (100.0/(o+100.0)) if o>0 else ((-o)/((-o)+100.0))
                    except Exception:
                        return np.nan
                h2h["p"] = h2h["price_american"].apply(_p_from_american)
                grouped = []
                for (eid, bk), grp in h2h.groupby(["event_id","book"], dropna=True):
                    s = grp["p"].sum()
                    rows = grp.copy()
                    rows["pfair"] = rows["p"]/s if s and s>0 else rows["p"]
                    grouped.append(rows)
                if grouped:
                    h2h = pd.concat(grouped, ignore_index=True)
                    teams = g[["event_id","home_team","away_team"]].dropna().drop_duplicates()
                    h2h = h2h.merge(teams, on="event_id", how="left")
                    h2h["is_home"] = h2h.apply(lambda r: str(r.get("outcome","")).upper()==str(r.get("home_team","")).upper(), axis=1)
                    home = h2h[h2h["is_home"]].groupby("event_id")["pfair"].mean().rename("home_wp")
                    away = h2h[~h2h["is_home"]].groupby("event_id")["pfair"].mean().rename("away_wp")
                    wp = pd.concat([home, away], axis=1).reset_index()
                    players = players.merge(wp, on="event_id", how="left")
                    players = players.merge(teams, on="event_id", how="left")
                    players["team_wp"] = np.where(players["team"]==players["home_team"], players["home_wp"],
                                           np.where(players["team"]==players["away_team"], players["away_wp"], np.nan))
                    players.drop(columns=[c for c in ["home_team","away_team","home_wp","away_wp"] if c in players.columns], inplace=True)

    players = _safe_cols(players, REQ_COLS)
    players["target_share"] = players["target_share"].fillna(0.18)
    players["rush_share"]   = players["rush_share"].fillna(0.30)
    players["route_rate"]   = players["route_rate"].fillna(0.75)
    players["yprr_proxy"]   = players["yprr_proxy"].fillna(1.7)
    players["ypc"]          = players["ypc"].fillna(4.2)
    players["qb_ypa"]       = players["qb_ypa"].fillna(6.9)
    players["team_wp"]      = players["team_wp"].fillna(0.5)

    Path("outputs/metrics").mkdir(parents=True, exist_ok=True)
    players.to_csv("data/metrics_ready.csv", index=False)
    players.to_csv("outputs/metrics/metrics_ready.csv", index=False)
    print(f"[metrics] rows={len(players)} → data/metrics_ready.csv")

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=None)
    args = ap.parse_args()
    main(args.season)
