"""M95A: RB role x defensive rushing vulnerability truth test.

Research-only diagnostic. This migration tests the user's central football hypothesis:
pregame workhorse evidence plus a demonstrably weak run defense should predict
more RB opportunity and production. All matchup features are leakage-safe: each
target row uses only games completed before that season/week. No sportsbook
inputs are used and no production projection code is changed.

The diagnostic:
1. builds a rich pregame RB role/workload profile from historical player logs;
2. builds a rich opponent run-defense profile from nflverse play-by-play plus
   RB outcomes previously allowed by that defense;
3. ranks defenses weekly from strong to weak using only entering-week data;
4. shows actual RB outcome distributions by role x defense quartile;
5. runs prospective 2023->2024 and 2023-24->2025 role-only vs
   role+defense vs role+defense+interaction tests;
6. exports bottom-8/top-8 matchup ledgers and feature availability/provenance.

Unavailable premium charting metrics (e.g. true yards-before-contact, missed
tackles forced/allowed, run-block win rate) are explicitly reported rather than
fabricated.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

TARGET_SEASONS = (2023, 2024, 2025)
PBP_SEASONS = (2022, 2023, 2024, 2025)
WINDOWS = (1, 3, 5)
TEAM_KEYS = ["season", "week", "team"]
PLAYER_KEYS = ["season", "week", "team", "player_clean_key"]

TEAM_MAP = {
    "ARZ": "ARI", "JAC": "JAX", "LA": "LAR", "STL": "LAR",
    "OAK": "LV", "SD": "LAC", "WSH": "WAS",
}

ROLE_FEATURES = [
    "rb_games_before", "rb_carries_avg1", "rb_carries_avg3", "rb_carries_avg5",
    "rb_rush_yards_avg3", "rb_rush_yards_avg5",
    "rb_rb_share_avg1", "rb_rb_share_avg3", "rb_rb_share_avg5",
    "rb_targets_avg3", "rb_targets_avg5", "rb_rec_yards_avg3", "rb_rec_yards_avg5",
    "rb_15plus_rate5", "rb_20plus_rate5",
    "team_rb_pool_avg3", "team_rb_pool_avg5",
    "team_total_rush_avg3", "team_total_rush_avg5",
    "team_top1_share_avg3", "team_top1_share_avg5",
    "team_rb_used_avg3", "team_rb_used_avg5",
    "team_qb_rush_share_avg3", "team_qb_rush_share_avg5",
    "team_rush_ypc_avg3", "team_rush_ypc_avg5",
    "home",
]

CLEAR_DEF_BASES = [
    "rush_ypa_allowed", "rush_epa_allowed", "rush_success_allowed",
    "explosive10_rate_allowed", "explosive15_rate_allowed", "explosive20_rate_allowed",
    "stuff_rate_allowed", "rush_first_down_rate_allowed",
    "short_success_allowed", "redzone_success_allowed",
    "inside10_ypa_allowed", "inside5_td_rate_allowed",
    "left_ypa_allowed", "middle_ypa_allowed", "right_ypa_allowed",
    "shotgun_ypa_allowed", "non_scramble_ypa_allowed",
    "rb_carries_allowed", "rb_rush_yards_allowed", "rb_ypc_allowed",
    "top_rb_carries_allowed", "top_rb_rush_yards_allowed",
    "rb_75plus_rate_allowed", "rb_100plus_rate_allowed",
    "rb_15plus_carry_rate_allowed", "rb_20plus_carry_rate_allowed",
    "rb_over_prior5_rush_yards_allowed",
]

COMPOSITE_FEATURES = [
    "def_rb_rush_yards_allowed_avg5",
    "def_rb_ypc_allowed_avg5",
    "def_top_rb_rush_yards_allowed_avg5",
    "def_rb_over_prior5_rush_yards_allowed_avg5",
    "def_rush_epa_allowed_avg5",
    "def_rush_success_allowed_avg5",
    "def_explosive10_rate_allowed_avg5",
    "def_explosive20_rate_allowed_avg5",
    "def_stuff_rate_allowed_avg5",
]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _team(v: object) -> str:
    x = str(v).upper().strip()
    return TEAM_MAP.get(x, x)


def _clean_name(v: object) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v).lower())


def _num(s: pd.Series, default: float = np.nan) -> pd.Series:
    z = pd.to_numeric(s, errors="coerce")
    return z.fillna(default) if np.isfinite(default) else z


def _alias(df: pd.DataFrame, names: Iterable[str], required: bool = False) -> str | None:
    for n in names:
        if n in df.columns:
            return n
    if required:
        raise RuntimeError(f"none of required aliases present: {list(names)}")
    return None


def _find_all(root: Path, name: str) -> list[Path]:
    return sorted({p.resolve() for p in root.rglob(name) if p.is_file() and p.stat().st_size > 0})


def _read_concat(root: Path, name: str) -> pd.DataFrame:
    paths = _find_all(root, name)
    if not paths:
        raise RuntimeError(f"missing {name} under {root}")
    frames = []
    for p in paths:
        x = _lower(pd.read_csv(p, low_memory=False))
        x["_source_path"] = str(p)
        frames.append(x)
    return pd.concat(frames, ignore_index=True, sort=False)


def _prepare_logs(root: Path) -> pd.DataFrame:
    x = _read_concat(root, "player_game_logs_history.csv")
    season = _alias(x, ["season"], True)
    week = _alias(x, ["week"], True)
    team = _alias(x, ["team", "recent_team"], True)
    player = _alias(x, ["player", "player_name", "name"], True)
    pos = _alias(x, ["position", "pos"], True)
    rushes = _alias(x, ["rushes", "carries", "rushing_attempts", "rush_attempts"], True)
    rush_yards = _alias(x, ["rush_yards", "rushing_yards", "yards_rushing"], True)
    targets = _alias(x, ["targets"], False)
    receptions = _alias(x, ["receptions", "rec"], False)
    rec_yards = _alias(x, ["rec_yards", "receiving_yards", "yards_receiving"], False)

    out = pd.DataFrame({
        "season": _num(x[season]),
        "week": _num(x[week]),
        "team": x[team].map(_team),
        "player": x[player].astype(str),
        "position": x[pos].astype(str).str.upper().str.strip(),
        "rushes": _num(x[rushes], 0.0),
        "rush_yards": _num(x[rush_yards], 0.0),
        "targets": _num(x[targets], 0.0) if targets else 0.0,
        "receptions": _num(x[receptions], 0.0) if receptions else 0.0,
        "rec_yards": _num(x[rec_yards], 0.0) if rec_yards else 0.0,
    })
    out["player_clean_key"] = out["player"].map(_clean_name)
    out = out.loc[out["season"].between(2022, 2025) & out["week"].between(1, 22)].copy()
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    out = out.sort_values(["season", "week", "team", "player_clean_key"]).drop_duplicates(
        ["season", "week", "team", "player_clean_key"], keep="last"
    )
    return out.reset_index(drop=True)


def _prepare_team_history(root: Path) -> pd.DataFrame:
    try:
        x = _read_concat(root, "team_weekly_history.csv")
    except RuntimeError:
        return pd.DataFrame()
    if not {"season", "week", "team"}.issubset(x.columns):
        return pd.DataFrame()
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    x["team"] = x["team"].map(_team)
    x = x.loc[x["season"].between(2022, 2025) & x["week"].between(1, 22)].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x = x.sort_values(["season", "week", "team"]).drop_duplicates(
        ["season", "week", "team"], keep="last"
    )
    return x.reset_index(drop=True)


def _read_pbp(root: Path) -> pd.DataFrame:
    frames = []
    for season in PBP_SEASONS:
        p = root / f"play_by_play_{season}.parquet"
        if not p.exists() or p.stat().st_size == 0:
            raise RuntimeError(f"missing nflverse PBP: {p}")
        cols = pd.read_parquet(p, engine="pyarrow").columns.tolist()
        wanted = [
            "season", "week", "posteam", "defteam", "home_team", "away_team",
            "rush_attempt", "rushing_yards", "epa", "success", "down", "ydstogo",
            "yardline_100", "first_down_rush", "touchdown", "run_location", "run_gap",
            "shotgun", "no_huddle", "qb_kneel", "qb_scramble",
        ]
        use = [c for c in wanted if c in cols]
        z = pd.read_parquet(p, columns=use, engine="pyarrow")
        z = _lower(z)
        frames.append(z)
    x = pd.concat(frames, ignore_index=True, sort=False)
    for c in ["posteam", "defteam", "home_team", "away_team"]:
        if c in x.columns:
            x[c] = x[c].map(_team)
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    return x.loc[x["season"].between(min(PBP_SEASONS), max(PBP_SEASONS))].copy()


def _schedule_from_pbp(pbp: pd.DataFrame) -> pd.DataFrame:
    g = pbp.loc[pbp["posteam"].notna() & pbp["defteam"].notna()].copy()
    rows = []
    for (s, w, team, opp), q in g.groupby(["season", "week", "posteam", "defteam"], dropna=False):
        if int(s) not in TARGET_SEASONS:
            continue
        home_team = q["home_team"].dropna().iloc[0] if "home_team" in q.columns and q["home_team"].notna().any() else None
        rows.append({
            "season": int(s), "week": int(w), "team": _team(team), "opponent": _team(opp),
            "home": int(home_team is not None and _team(team) == _team(home_team)),
        })
    return pd.DataFrame(rows).drop_duplicates(TEAM_KEYS)


def _team_game_from_logs(logs: pd.DataFrame) -> pd.DataFrame:
    z = logs.copy()
    z["is_rb"] = z["position"].eq("RB")
    z["is_qb"] = z["position"].eq("QB")
    rows = []
    for (s, w, team), g in z.groupby(TEAM_KEYS, dropna=False):
        rb = g.loc[g["is_rb"]]
        total_rush = float(g["rushes"].sum())
        rb_pool = float(rb["rushes"].sum())
        shares = rb["rushes"] / rb_pool if rb_pool > 0 and len(rb) else pd.Series(dtype=float)
        rows.append({
            "season": int(s), "week": int(w), "team": team,
            "team_total_rush": total_rush,
            "team_rush_yards": float(g["rush_yards"].sum()),
            "team_rush_ypc": float(g["rush_yards"].sum() / total_rush) if total_rush > 0 else np.nan,
            "team_rb_pool": rb_pool,
            "team_top1_share": float(shares.max()) if len(shares) else np.nan,
            "team_rb_used": int((rb["rushes"] > 0).sum()),
            "team_qb_rush_share": float(g.loc[g["is_qb"], "rushes"].sum() / total_rush) if total_rush > 0 else 0.0,
        })
    return pd.DataFrame(rows)


def _player_prior_features(logs: pd.DataFrame, team_games: pd.DataFrame) -> pd.DataFrame:
    rb = logs.loc[logs["position"].eq("RB") & logs["season"].isin(TARGET_SEASONS)].copy()
    team_rb = logs.loc[logs["position"].eq("RB")].copy()
    pool = team_rb.groupby(TEAM_KEYS)["rushes"].sum().rename("rb_pool").reset_index()
    team_rb = team_rb.merge(pool, on=TEAM_KEYS, how="left")
    team_rb["rb_share"] = np.where(team_rb["rb_pool"].gt(0), team_rb["rushes"] / team_rb["rb_pool"], 0.0)
    keyhist = {
        key: g.sort_values(["season", "week"]).reset_index(drop=True)
        for key, g in team_rb.groupby(["team", "player_clean_key"], dropna=False)
    }
    teamhist = {
        team: g.sort_values(["season", "week"]).reset_index(drop=True)
        for team, g in team_games.groupby("team", dropna=False)
    }

    rows = []
    for r in rb.itertuples(index=False):
        hist = keyhist.get((r.team, r.player_clean_key), pd.DataFrame())
        if len(hist):
            mask = (hist["season"] < r.season) | ((hist["season"] == r.season) & (hist["week"] < r.week))
            ph = hist.loc[mask]
        else:
            ph = pd.DataFrame()
        th = teamhist.get(r.team, pd.DataFrame())
        if len(th):
            maskt = (th["season"] < r.season) | ((th["season"] == r.season) & (th["week"] < r.week))
            th = th.loc[maskt]
        rec = {
            "season": r.season, "week": r.week, "team": r.team,
            "player": r.player, "player_clean_key": r.player_clean_key,
            "actual_carries": float(r.rushes), "actual_rush_yards": float(r.rush_yards),
            "actual_targets": float(r.targets), "actual_rec_yards": float(r.rec_yards),
            "actual_rush_rec_yards": float(r.rush_yards + r.rec_yards),
            "rb_games_before": int(len(ph)),
        }
        for n in WINDOWS:
            q = ph.tail(n)
            rec[f"rb_carries_avg{n}"] = float(q["rushes"].mean()) if len(q) else np.nan
            rec[f"rb_rush_yards_avg{n}"] = float(q["rush_yards"].mean()) if len(q) else np.nan
            rec[f"rb_rb_share_avg{n}"] = float(q["rb_share"].mean()) if len(q) else np.nan
            rec[f"rb_targets_avg{n}"] = float(q["targets"].mean()) if len(q) else np.nan
            rec[f"rb_rec_yards_avg{n}"] = float(q["rec_yards"].mean()) if len(q) else np.nan
            rec[f"rb_15plus_rate{n}"] = float(q["rushes"].ge(15).mean()) if len(q) else np.nan
            rec[f"rb_20plus_rate{n}"] = float(q["rushes"].ge(20).mean()) if len(q) else np.nan
            tq = th.tail(n) if len(th) else th
            for c in ["team_rb_pool", "team_total_rush", "team_top1_share", "team_rb_used", "team_qb_rush_share", "team_rush_ypc"]:
                rec[f"{c}_avg{n}"] = float(tq[c].mean()) if len(tq) and c in tq else np.nan

        c5 = rec.get("rb_carries_avg5", np.nan)
        s5 = rec.get("rb_rb_share_avg5", np.nan)
        c3 = rec.get("rb_carries_avg3", np.nan)
        s3 = rec.get("rb_rb_share_avg3", np.nan)
        if (pd.notna(s5) and pd.notna(c5) and s5 >= 0.60 and c5 >= 14) or (
            pd.notna(s3) and pd.notna(c3) and s3 >= 0.65 and c3 >= 14
        ):
            role = "workhorse"
        elif pd.notna(s5) and pd.notna(c5) and s5 >= 0.45 and c5 >= 9:
            role = "strong_starter"
        elif pd.notna(s5) and pd.notna(c5) and s5 >= 0.25 and c5 >= 5:
            role = "committee"
        elif len(ph):
            role = "light"
        else:
            role = "unknown"
        rec["pregame_role"] = role
        rec["pregame_prior5_ry"] = rec.get("rb_rush_yards_avg5", np.nan)
        rec["actual_ry_over_prior5"] = (
            rec["actual_rush_yards"] - rec["pregame_prior5_ry"]
            if pd.notna(rec["pregame_prior5_ry"]) else np.nan
        )
        rows.append(rec)
    return pd.DataFrame(rows)


def _pbp_defense_games(pbp: pd.DataFrame) -> pd.DataFrame:
    x = pbp.copy()
    rush = _num(x.get("rush_attempt", pd.Series(index=x.index, dtype=float)), 0.0).eq(1)
    kneel = _num(x.get("qb_kneel", pd.Series(index=x.index, dtype=float)), 0.0).eq(1)
    x = x.loc[rush & ~kneel & x["defteam"].notna()].copy()
    x["rushing_yards"] = _num(x.get("rushing_yards", pd.Series(index=x.index, dtype=float)))
    x["epa"] = _num(x.get("epa", pd.Series(index=x.index, dtype=float)))
    x["success"] = _num(x.get("success", pd.Series(index=x.index, dtype=float)))
    x["first_down_rush"] = _num(x.get("first_down_rush", pd.Series(index=x.index, dtype=float)), 0.0)
    x["touchdown"] = _num(x.get("touchdown", pd.Series(index=x.index, dtype=float)), 0.0)
    x["down"] = _num(x.get("down", pd.Series(index=x.index, dtype=float)))
    x["ydstogo"] = _num(x.get("ydstogo", pd.Series(index=x.index, dtype=float)))
    x["yardline_100"] = _num(x.get("yardline_100", pd.Series(index=x.index, dtype=float)))
    x["shotgun"] = _num(x.get("shotgun", pd.Series(index=x.index, dtype=float)), 0.0)
    x["qb_scramble"] = _num(x.get("qb_scramble", pd.Series(index=x.index, dtype=float)), 0.0)
    x["run_location"] = x.get("run_location", pd.Series(index=x.index, dtype=object)).astype(str).str.lower()

    rows = []
    for (s, w, d), g in x.groupby(["season", "week", "defteam"], dropna=False):
        y = g["rushing_yards"]
        short = g["down"].isin([3, 4]) & g["ydstogo"].le(2)
        rz = g["yardline_100"].le(20)
        in10 = g["yardline_100"].le(10)
        in5 = g["yardline_100"].le(5)
        shot = g["shotgun"].eq(1)
        nonscr = g["qb_scramble"].ne(1)
        rec = {
            "season": int(s), "week": int(w), "defense": _team(d),
            "rush_att_allowed": int(len(g)),
            "rush_ypa_allowed": float(y.mean()) if y.notna().any() else np.nan,
            "rush_epa_allowed": float(g["epa"].mean()) if g["epa"].notna().any() else np.nan,
            "rush_success_allowed": float(g["success"].mean()) if g["success"].notna().any() else np.nan,
            "explosive10_rate_allowed": float(y.ge(10).mean()),
            "explosive15_rate_allowed": float(y.ge(15).mean()),
            "explosive20_rate_allowed": float(y.ge(20).mean()),
            "stuff_rate_allowed": float(y.le(0).mean()),
            "rush_first_down_rate_allowed": float(g["first_down_rush"].mean()),
            "short_success_allowed": float(g.loc[short, "success"].mean()) if short.any() else np.nan,
            "redzone_success_allowed": float(g.loc[rz, "success"].mean()) if rz.any() else np.nan,
            "inside10_ypa_allowed": float(g.loc[in10, "rushing_yards"].mean()) if in10.any() else np.nan,
            "inside5_td_rate_allowed": float(g.loc[in5, "touchdown"].mean()) if in5.any() else np.nan,
            "shotgun_ypa_allowed": float(g.loc[shot, "rushing_yards"].mean()) if shot.any() else np.nan,
            "non_scramble_ypa_allowed": float(g.loc[nonscr, "rushing_yards"].mean()) if nonscr.any() else np.nan,
        }
        for loc in ["left", "middle", "right"]:
            q = g["run_location"].eq(loc)
            rec[f"{loc}_ypa_allowed"] = float(g.loc[q, "rushing_yards"].mean()) if q.any() else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _rb_allowed_games(rb_games: pd.DataFrame, schedule: pd.DataFrame) -> pd.DataFrame:
    x = rb_games.merge(schedule, on=TEAM_KEYS, how="left", validate="many_to_one")
    x = x.loc[x["opponent"].notna()].copy()
    rows = []
    for (s, w, defense), g in x.groupby(["season", "week", "opponent"], dropna=False):
        carries = g["actual_carries"]
        yards = g["actual_rush_yards"]
        total_c = float(carries.sum())
        rows.append({
            "season": int(s), "week": int(w), "defense": _team(defense),
            "rb_carries_allowed": total_c,
            "rb_rush_yards_allowed": float(yards.sum()),
            "rb_ypc_allowed": float(yards.sum() / total_c) if total_c > 0 else np.nan,
            "top_rb_carries_allowed": float(carries.max()) if len(g) else np.nan,
            "top_rb_rush_yards_allowed": float(yards.max()) if len(g) else np.nan,
            "rb_75plus_rate_allowed": float(yards.ge(75).mean()) if len(g) else np.nan,
            "rb_100plus_rate_allowed": float(yards.ge(100).mean()) if len(g) else np.nan,
            "rb_15plus_carry_rate_allowed": float(carries.ge(15).mean()) if len(g) else np.nan,
            "rb_20plus_carry_rate_allowed": float(carries.ge(20).mean()) if len(g) else np.nan,
            "rb_over_prior5_rush_yards_allowed": float(g["actual_ry_over_prior5"].mean())
                if g["actual_ry_over_prior5"].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)


def _team_context_defense_games(team_hist: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if team_hist.empty:
        return pd.DataFrame(), []
    requested = [
        "def_rush_epa", "success_rate_def", "explosive_play_rate_allowed",
        "avg_defenders_in_box", "light_box_rate", "heavy_box_rate", "middle_open_rate",
        "rz_rate", "ay_per_att", "pressure_rate_generated",
    ]
    available = [c for c in requested if c in team_hist.columns and _num(team_hist[c]).notna().any()]
    if not available:
        return pd.DataFrame(), []
    out = team_hist[["season", "week", "team"] + available].copy().rename(columns={"team": "defense"})
    for c in available:
        out[c] = _num(out[c])
    return out, available


def _rolling_defense_profiles(
    defense_games: pd.DataFrame,
    schedule: pd.DataFrame,
    metric_cols: list[str],
) -> pd.DataFrame:
    hist = {
        d: g.sort_values(["season", "week"]).reset_index(drop=True)
        for d, g in defense_games.groupby("defense", dropna=False)
    }
    target_def = schedule[["season", "week", "opponent"]].drop_duplicates().rename(columns={"opponent": "defense"})
    rows = []
    for r in target_def.itertuples(index=False):
        g = hist.get(r.defense, pd.DataFrame())
        if len(g):
            prior = g.loc[(g["season"] < r.season) | ((g["season"] == r.season) & (g["week"] < r.week))]
        else:
            prior = pd.DataFrame()
        rec = {"season": r.season, "week": r.week, "defense": r.defense, "def_games_before": int(len(prior))}
        for c in metric_cols:
            if c not in defense_games.columns:
                continue
            vals = _num(prior[c]) if len(prior) else pd.Series(dtype=float)
            for n in (3, 5):
                q = vals.tail(n).dropna()
                rec[f"def_{c}_avg{n}"] = float(q.mean()) if len(q) else np.nan
            season_q = prior.loc[prior["season"].eq(r.season), c] if len(prior) else pd.Series(dtype=float)
            season_q = _num(season_q).dropna()
            rec[f"def_{c}_std"] = float(season_q.mean()) if len(season_q) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def _add_defense_composite(profiles: pd.DataFrame) -> pd.DataFrame:
    x = profiles.copy()
    rank_cols = []
    for c in COMPOSITE_FEATURES:
        if c not in x.columns or _num(x[c]).notna().sum() < 50:
            continue
        source = -_num(x[c]) if "stuff_rate" in c else _num(x[c])
        rc = f"{c}_weekly_bad_pct"
        x[rc] = source.groupby([x["season"], x["week"]]).rank(pct=True, method="average")
        rank_cols.append(rc)
    if rank_cols:
        x["def_vulnerability_score"] = x[rank_cols].mean(axis=1, skipna=True)
        x["def_vulnerability_metrics_n"] = x[rank_cols].notna().sum(axis=1)
        x.loc[x["def_vulnerability_metrics_n"].lt(3), "def_vulnerability_score"] = np.nan
    else:
        x["def_vulnerability_score"] = np.nan
        x["def_vulnerability_metrics_n"] = 0
    x["def_weekly_rank_bad"] = x.groupby(["season", "week"])["def_vulnerability_score"].rank(
        ascending=False, method="min"
    )
    x["def_weekly_rank_good"] = x.groupby(["season", "week"])["def_vulnerability_score"].rank(
        ascending=True, method="min"
    )
    x["def_bucket"] = np.select(
        [x["def_weekly_rank_bad"].le(8), x["def_weekly_rank_good"].le(8)],
        ["bottom8_weak", "top8_strong"], default="middle16",
    )
    return x


def _merge_external_context(
    profiles: pd.DataFrame,
    team_ctx_games: pd.DataFrame,
    team_ctx_cols: list[str],
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    if team_ctx_games.empty or not team_ctx_cols:
        return profiles
    rolled = _rolling_defense_profiles(team_ctx_games, schedule, team_ctx_cols)
    keep = ["season", "week", "defense"] + [
        c for c in rolled.columns if c.startswith("def_") and c not in {"def_games_before"}
    ]
    return profiles.merge(rolled[keep], on=["season", "week", "defense"], how="left", validate="one_to_one")


def _truth_trace(rb_games: pd.DataFrame, schedule: pd.DataFrame, defense_profiles: pd.DataFrame) -> pd.DataFrame:
    x = rb_games.merge(schedule, on=TEAM_KEYS, how="left", validate="many_to_one")
    x = x.merge(
        defense_profiles,
        left_on=["season", "week", "opponent"],
        right_on=["season", "week", "defense"],
        how="left",
        validate="many_to_one",
    )
    x["actual_ypc"] = np.where(x["actual_carries"].gt(0), x["actual_rush_yards"] / x["actual_carries"], np.nan)
    x["actual_15plus_carries"] = x["actual_carries"].ge(15).astype(int)
    x["actual_20plus_carries"] = x["actual_carries"].ge(20).astype(int)
    x["actual_25plus_carries"] = x["actual_carries"].ge(25).astype(int)
    x["actual_75plus_rush_yards"] = x["actual_rush_yards"].ge(75).astype(int)
    x["actual_100plus_rush_yards"] = x["actual_rush_yards"].ge(100).astype(int)
    x["role_is_workhorse"] = x["pregame_role"].eq("workhorse").astype(int)
    x["role_is_starter_plus"] = x["pregame_role"].isin(["workhorse", "strong_starter"]).astype(int)

    dv = _num(x["def_vulnerability_score"])
    x["workhorse_x_def_vulnerability"] = x["role_is_workhorse"] * dv
    x["starter_x_def_vulnerability"] = x["role_is_starter_plus"] * dv
    for c in ["rb_rb_share_avg5", "rb_carries_avg5", "team_total_rush_avg5", "team_rush_ypc_avg5"]:
        if c in x.columns:
            x[f"{c}_x_def_vulnerability"] = _num(x[c]) * dv
    return x


def _bucket_truth(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for scope, g0 in [("combined", trace)] + [(str(s), trace.loc[trace["season"].eq(s)]) for s in TARGET_SEASONS]:
        for role in ["workhorse", "strong_starter", "committee", "light"]:
            for db in ["bottom8_weak", "middle16", "top8_strong"]:
                g = g0.loc[g0["pregame_role"].eq(role) & g0["def_bucket"].eq(db)]
                if not len(g):
                    continue
                rows.append({
                    "season_scope": scope, "pregame_role": role, "def_bucket": db, "n": len(g),
                    "actual_carries_mean": g["actual_carries"].mean(),
                    "actual_rush_yards_mean": g["actual_rush_yards"].mean(),
                    "actual_ypc_mean": g["actual_ypc"].mean(),
                    "actual_rush_rec_yards_mean": g["actual_rush_rec_yards"].mean(),
                    "actual_ry_over_prior5_mean": g["actual_ry_over_prior5"].mean(),
                    "carry_15plus_rate": g["actual_15plus_carries"].mean(),
                    "carry_20plus_rate": g["actual_20plus_carries"].mean(),
                    "carry_25plus_rate": g["actual_25plus_carries"].mean(),
                    "rush_75plus_rate": g["actual_75plus_rush_yards"].mean(),
                    "rush_100plus_rate": g["actual_100plus_rush_yards"].mean(),
                })
    return pd.DataFrame(rows)


def _weak_strong(bucket: pd.DataFrame) -> pd.DataFrame:
    rows = []
    metrics = [
        "actual_carries_mean", "actual_rush_yards_mean", "actual_ypc_mean",
        "actual_rush_rec_yards_mean", "actual_ry_over_prior5_mean",
        "carry_15plus_rate", "carry_20plus_rate", "carry_25plus_rate",
        "rush_75plus_rate", "rush_100plus_rate",
    ]
    for (scope, role), g in bucket.groupby(["season_scope", "pregame_role"]):
        w = g.loc[g["def_bucket"].eq("bottom8_weak")]
        s = g.loc[g["def_bucket"].eq("top8_strong")]
        if w.empty or s.empty:
            continue
        rec = {
            "season_scope": scope, "pregame_role": role,
            "weak_n": int(w["n"].iloc[0]), "strong_n": int(s["n"].iloc[0]),
        }
        for m in metrics:
            rec[f"weak_{m}"] = float(w[m].iloc[0])
            rec[f"strong_{m}"] = float(s[m].iloc[0])
            rec[f"weak_minus_strong_{m}"] = float(w[m].iloc[0] - s[m].iloc[0])
        rows.append(rec)
    return pd.DataFrame(rows)


def _numeric_feature_cols(trace: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    role = [c for c in ROLE_FEATURES if c in trace.columns and _num(trace[c]).notna().any()]
    blocked_terms = ("actual_", "weekly_bad_pct", "rank_bad", "rank_good")
    defense = []
    for c in trace.columns:
        if not c.startswith("def_"):
            continue
        if c in {"defense", "def_bucket", "def_vulnerability_metrics_n"}:
            continue
        if any(t in c for t in blocked_terms):
            continue
        if _num(trace[c]).notna().sum() >= 100:
            defense.append(c)
    interactions = [
        c for c in trace.columns
        if c.endswith("_x_def_vulnerability") and _num(trace[c]).notna().sum() >= 100
    ]
    defense = sorted(set(defense))
    return role, defense, sorted(interactions)


def _ridge() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", Ridge(alpha=20.0)),
    ])


def _logit() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.20, max_iter=2000)),
    ])


def _reg_metrics(actual: pd.Series, pred: np.ndarray) -> dict[str, float | int]:
    z = pd.DataFrame({"a": _num(actual), "p": pd.Series(pred, index=actual.index)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan}
    e = z["p"] - z["a"]
    corr = z["a"].corr(z["p"]) if z["a"].nunique() > 1 and z["p"].nunique() > 1 else np.nan
    return {
        "n": len(z), "mae": e.abs().mean(), "rmse": math.sqrt(float(np.square(e).mean())),
        "bias": e.mean(), "corr": corr,
    }


def _prospective_models(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    role, defense, interactions = _numeric_feature_cols(trace)
    families = {
        "role_only": role,
        "role_plus_defense": role + defense,
        "role_defense_interactions": role + defense + interactions,
    }
    splits = [
        ("train_2023_test_2024", [2023], 2024),
        ("train_2023_24_test_2025", [2023, 2024], 2025),
    ]
    targets = [
        ("carries", "actual_carries"),
        ("rush_yards", "actual_rush_yards"),
        ("rush_rec_yards", "actual_rush_rec_yards"),
    ]
    rows, coef_rows = [], []
    for split_name, train_seasons, test_season in splits:
        tr = trace.loc[trace["season"].isin(train_seasons) & trace["pregame_role"].ne("unknown")].copy()
        te = trace.loc[trace["season"].eq(test_season) & trace["pregame_role"].ne("unknown")].copy()
        for family, feats in families.items():
            feats = [c for c in feats if c in tr.columns and _num(tr[c]).notna().any()]
            if not feats:
                continue
            Xtr = tr[feats].apply(pd.to_numeric, errors="coerce")
            Xte = te[feats].apply(pd.to_numeric, errors="coerce")
            for target_name, target_col in targets:
                m = _ridge()
                ytr = _num(tr[target_col])
                valid = ytr.notna()
                m.fit(Xtr.loc[valid], ytr.loc[valid])
                pred = m.predict(Xte)
                met = _reg_metrics(te[target_col], pred)
                rows.append({
                    "split": split_name, "train_seasons": ",".join(map(str, train_seasons)),
                    "test_season": test_season, "family": family, "target": target_name,
                    "feature_count": len(feats), **met,
                })
                if target_name == "rush_yards":
                    coefs = np.ravel(m.named_steps["model"].coef_)
                    for f, v in zip(feats, coefs):
                        coef_rows.append({
                            "split": split_name, "family": family, "target": target_name,
                            "feature": f, "standardized_coefficient": float(v),
                            "abs_coefficient": abs(float(v)),
                        })

            ytrc = tr["actual_100plus_rush_yards"].astype(int)
            ytec = te["actual_100plus_rush_yards"].astype(int)
            if ytrc.nunique() > 1 and ytec.nunique() > 1:
                lm = _logit()
                lm.fit(Xtr, ytrc)
                prob = lm.predict_proba(Xte)[:, 1]
                rows.append({
                    "split": split_name, "train_seasons": ",".join(map(str, train_seasons)),
                    "test_season": test_season, "family": family, "target": "rush_100plus_auc",
                    "feature_count": len(feats), "n": len(te), "mae": np.nan, "rmse": np.nan,
                    "bias": np.nan, "corr": float(roc_auc_score(ytec, prob)),
                })
    return pd.DataFrame(rows), pd.DataFrame(coef_rows)


def _incremental(prospective: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (split, target), g in prospective.groupby(["split", "target"]):
        b = g.loc[g["family"].eq("role_only")]
        if b.empty:
            continue
        for fam in ["role_plus_defense", "role_defense_interactions"]:
            c = g.loc[g["family"].eq(fam)]
            if c.empty:
                continue
            if target == "rush_100plus_auc":
                gain = float(c["corr"].iloc[0] - b["corr"].iloc[0])
                metric = "auc_gain"
            else:
                gain = float(b["mae"].iloc[0] - c["mae"].iloc[0])
                metric = "mae_gain"
            rows.append({
                "split": split, "target": target, "candidate_family": fam,
                "comparison_metric": metric, "gain_vs_role_only": gain,
                "role_only_mae": float(b["mae"].iloc[0]) if pd.notna(b["mae"].iloc[0]) else np.nan,
                "candidate_mae": float(c["mae"].iloc[0]) if pd.notna(c["mae"].iloc[0]) else np.nan,
                "role_only_corr_or_auc": float(b["corr"].iloc[0]) if pd.notna(b["corr"].iloc[0]) else np.nan,
                "candidate_corr_or_auc": float(c["corr"].iloc[0]) if pd.notna(c["corr"].iloc[0]) else np.nan,
            })
    return pd.DataFrame(rows)


def _availability(
    logs: pd.DataFrame,
    pbp: pd.DataFrame,
    team_ctx_cols: list[str],
    trace: pd.DataFrame,
) -> pd.DataFrame:
    requested = [
        ("RB recent carry/share/target role", "M91 player_game_logs_history", True, "Pregame rolling 1/3/5."),
        ("RB rush yards / receiving yards", "M91 player_game_logs_history", True, "Actual target and prior form."),
        ("Opponent RB yards/carries/100+ games allowed", "M91 player logs + opponent mapping", True, "Direct RB-results-against-defense history."),
        ("Rush EPA allowed", "nflverse PBP", "def_rush_epa_allowed_avg5" in trace, "Pregame rolling defense."),
        ("Rush success allowed", "nflverse PBP", "def_rush_success_allowed_avg5" in trace, "Pregame rolling defense."),
        ("10+/15+/20+ explosive rush rate allowed", "nflverse PBP", "def_explosive20_rate_allowed_avg5" in trace, "Pregame rolling defense."),
        ("Stuff rate", "nflverse PBP", "def_stuff_rate_allowed_avg5" in trace, "Rush yards <= 0 proxy."),
        ("First-down rushing rate", "nflverse PBP", "def_rush_first_down_rate_allowed_avg5" in trace, "Pregame rolling defense."),
        ("Short-yardage defense", "nflverse PBP", "def_short_success_allowed_avg5" in trace, "3rd/4th down, <=2 yards."),
        ("Red-zone rushing success", "nflverse PBP", "def_redzone_success_allowed_avg5" in trace, "Inside 20."),
        ("Inside-10 / inside-5", "nflverse PBP", "def_inside5_td_rate_allowed_avg5" in trace, "Goal-line vulnerability."),
        ("Run direction left/middle/right", "nflverse PBP", "def_left_ypa_allowed_avg5" in trace, "Direction-specific YPA."),
        ("Shotgun rushing", "nflverse PBP", "def_shotgun_ypa_allowed_avg5" in trace, "Shotgun rushing YPA allowed."),
        ("QB scramble separation", "nflverse PBP", "def_non_scramble_ypa_allowed_avg5" in trace, "Prevents scramble-heavy games from fully defining RB matchup."),
        ("Defensive rush EPA legacy context", "M91 team_weekly_history", "def_rush_epa" in team_ctx_cols, "Research only; legacy/provider-audit provenance."),
        ("Box rates", "M91 team_weekly_history", any(c in team_ctx_cols for c in ["avg_defenders_in_box","light_box_rate","heavy_box_rate"]), "Research only; provider provenance may require audit."),
        ("Explosive play rate allowed context", "M91 team_weekly_history", "explosive_play_rate_allowed" in team_ctx_cols, "Available context may be all-play rather than rush-only; PBP rush explosives also used."),
        ("True yards before contact", "premium charting / tracking", False, "Not present in frozen sources; not fabricated."),
        ("True yards after contact", "premium charting / tracking", False, "Not present in frozen sources; not fabricated."),
        ("Missed tackles forced/allowed", "premium charting", False, "Not present in frozen sources; not fabricated."),
        ("Run-block win rate", "ESPN/PFF style charting", False, "Not present in frozen sources; not fabricated."),
        ("Adjusted line yards / second-level yards", "external OL analytics", False, "Not present in frozen sources; candidate for later OL migration."),
        ("Run concept (inside zone/outside zone/duo/power/counter)", "charting source", False, "Standard PBP run_gap/location is not enough to truthfully label full concepts."),
        ("DL/LB individual injury strength", "historical injury + defensive personnel model", False, "Not yet joined in M95A; should be tested in later personnel migration."),
    ]
    return pd.DataFrame([
        {"metric_family": a, "source": b, "available_in_m95a": int(bool(c)), "notes": d}
        for a, b, c, d in requested
    ])


def _bottom_top_ledgers(trace: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    cols = [
        "season", "week", "team", "opponent", "player", "pregame_role",
        "rb_carries_avg5", "rb_rb_share_avg5", "def_vulnerability_score",
        "def_weekly_rank_bad", "def_weekly_rank_good",
        "actual_carries", "actual_rush_yards", "actual_ypc", "actual_rush_rec_yards",
        "actual_ry_over_prior5", "actual_20plus_carries", "actual_25plus_carries",
        "actual_75plus_rush_yards", "actual_100plus_rush_yards",
    ]
    cols = [c for c in cols if c in trace.columns]
    starter = trace["pregame_role"].isin(["workhorse", "strong_starter"])
    weak = trace.loc[starter & trace["def_bucket"].eq("bottom8_weak"), cols].copy()
    strong = trace.loc[starter & trace["def_bucket"].eq("top8_strong"), cols].copy()
    return weak.sort_values(["season", "week", "def_weekly_rank_bad"]), strong.sort_values(["season", "week", "def_weekly_rank_good"])


def _disposition(weak_strong: pd.DataFrame, incremental: pd.DataFrame) -> pd.DataFrame:
    ws = weak_strong.loc[
        weak_strong["pregame_role"].isin(["workhorse", "strong_starter"])
        & weak_strong["season_scope"].isin(["2024", "2025"])
    ]
    weak_ry_positive = bool(len(ws) >= 2 and (ws["weak_minus_strong_actual_rush_yards_mean"] > 0).mean() >= 0.75)
    inc = incremental.loc[
        incremental["candidate_family"].eq("role_defense_interactions")
        & incremental["target"].isin(["carries", "rush_yards"])
    ]
    prospective_positive = bool(len(inc) >= 4 and (inc["gain_vs_role_only"] > 0).sum() >= 3)
    advance = weak_ry_positive and prospective_positive
    return pd.DataFrame([{
        "weak_vs_strong_direction_consistent": int(weak_ry_positive),
        "prospective_role_plus_matchup_signal": int(prospective_positive),
        "advance_matchup_family_to_candidate_testing": int(advance),
        "disposition": "ADVANCE_M95_MATCHUP_FAMILY" if advance else "RETAIN_DIAGNOSTIC_ONLY",
        "production_change": 0,
        "note": "M95A is a truth test only; no production coefficient/model is changed.",
    }])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--pbp-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m95a"))
    args = p.parse_args()

    logs = _prepare_logs(args.m91_root)
    team_hist = _prepare_team_history(args.m91_root)
    pbp = _read_pbp(args.pbp_root)
    schedule = _schedule_from_pbp(pbp)
    team_games = _team_game_from_logs(logs)
    rb_games = _player_prior_features(logs, team_games)

    rb_allowed = _rb_allowed_games(rb_games, schedule)
    pbp_def = _pbp_defense_games(pbp)
    defense_games = pbp_def.merge(rb_allowed, on=["season", "week", "defense"], how="outer", validate="one_to_one")
    metric_cols = [c for c in defense_games.columns if c not in {"season", "week", "defense"}]
    profiles = _rolling_defense_profiles(defense_games, schedule, metric_cols)
    team_ctx_games, team_ctx_cols = _team_context_defense_games(team_hist)
    profiles = _merge_external_context(profiles, team_ctx_games, team_ctx_cols, schedule)
    profiles = _add_defense_composite(profiles)

    trace = _truth_trace(rb_games, schedule, profiles)
    truth = trace.loc[
        trace["season"].isin(TARGET_SEASONS)
        & trace["opponent"].notna()
        & trace["def_vulnerability_score"].notna()
    ].copy()

    bucket = _bucket_truth(truth)
    weak_strong = _weak_strong(bucket)
    prospective, coefs = _prospective_models(truth)
    incremental = _incremental(prospective)
    weak_ledger, strong_ledger = _bottom_top_ledgers(truth)
    availability = _availability(logs, pbp, team_ctx_cols, truth)
    disposition = _disposition(weak_strong, incremental)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    profiles.to_csv(args.out_dir / "m95a_defense_week_profiles.csv", index=False)
    trace.to_csv(args.out_dir / "m95a_rb_game_trace.csv", index=False)
    bucket.to_csv(args.out_dir / "m95a_role_x_defense_bucket_truth.csv", index=False)
    weak_strong.to_csv(args.out_dir / "m95a_weak_vs_strong_summary.csv", index=False)
    prospective.to_csv(args.out_dir / "m95a_prospective_model_comparison.csv", index=False)
    incremental.to_csv(args.out_dir / "m95a_incremental_matchup_gain.csv", index=False)
    coefs.sort_values("abs_coefficient", ascending=False).to_csv(
        args.out_dir / "m95a_rush_yards_standardized_coefficients.csv", index=False
    )
    weak_ledger.to_csv(args.out_dir / "m95a_bottom8_run_defense_starter_games.csv", index=False)
    strong_ledger.to_csv(args.out_dir / "m95a_top8_run_defense_starter_games.csv", index=False)
    availability.to_csv(args.out_dir / "m95a_feature_availability.csv", index=False)
    disposition.to_csv(args.out_dir / "m95a_disposition.csv", index=False)

    print("[m95a] feature availability")
    print(availability.to_string(index=False))
    print("\n[m95a] workhorse/strong-starter weak-vs-strong truth")
    print(weak_strong.loc[
        weak_strong["pregame_role"].isin(["workhorse", "strong_starter"])
    ].to_string(index=False))
    print("\n[m95a] prospective incremental matchup gain")
    print(incremental.to_string(index=False))
    print("\n[m95a] disposition")
    print(disposition.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
