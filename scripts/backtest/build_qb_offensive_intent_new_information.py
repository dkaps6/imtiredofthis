#!/usr/bin/env python3
"""Migration 67: recover genuinely new pregame QB information.

This builder is deliberately separate from the M66 feature universe.  It creates
pregame team-week features from three source families:

1. nflverse PBP: granular offensive intent (early-down/score-bin/two-minute,
   no-huddle, shotgun) using only games strictly before the target week.
2. nflverse participation: formation/personnel and lineup-continuity diagnostics.
   For 2023+ this feed is released after the postseason, so these fields are
   explicitly marked historical-only and cannot silently become live-2026 inputs.
3. nflverse injuries: current-week official injury-report burden by offensive
   position group. These are pregame report fields and are treated as live-capable.

2024/2025 target outcomes are never used to create their own features.
"""
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

HISTORY_GAMES = 8
RECENT_GAMES = 3

INTENT_BASE = [
    "intent_early_down_neutral_dbr",
    "intent_q1_dbr",
    "intent_first_half_dbr",
    "intent_two_minute_dbr",
    "intent_third_long_dbr",
    "intent_trailing_1_7_dbr",
    "intent_trailing_8_14_dbr",
    "intent_trailing_15plus_dbr",
    "intent_leading_1_7_dbr",
    "intent_leading_8_14_dbr",
    "intent_leading_15plus_dbr",
    "intent_no_huddle_rate",
    "intent_shotgun_rate",
]

PERSONNEL_BASE = [
    "personnel_formation_shotgun_rate",
    "personnel_formation_pistol_rate",
    "personnel_formation_singleback_rate",
    "personnel_formation_i_form_rate",
    "personnel_formation_empty_rate",
    "personnel_11_rate",
    "personnel_12_rate",
    "personnel_21_rate",
    "personnel_22_rate",
    "personnel_10_rate",
    "personnel_shotgun_dbr",
    "personnel_under_center_dbr",
    "personnel_11_dbr",
    "personnel_12_dbr",
    "personnel_formation_entropy",
    "personnel_group_entropy",
]

CONTINUITY_BASE = [
    "continuity_ol_top5_overlap_prev",
    "continuity_skill_top6_overlap_prev",
    "continuity_ol_snap_concentration",
]

AVAILABILITY_FEATURES = [
    "availability_report_rows",
    "availability_out_doubtful_total",
    "availability_questionable_total",
    "availability_dnp_total",
    "availability_limited_total",
    "availability_ol_out_doubtful",
    "availability_ol_questionable",
    "availability_rb_out_doubtful",
    "availability_wrte_out_doubtful",
    "availability_skill_out_doubtful",
]


def to_pd(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def num(v):
    return pd.to_numeric(v, errors="coerce")


def text(v) -> pd.Series:
    return v.astype("string").str.strip().fillna("")


def join_keys(part: pd.DataFrame, pbp: pd.DataFrame) -> list[str]:
    for keys in (["nflverse_game_id", "play_id"], ["old_game_id", "play_id"], ["game_id", "play_id"]):
        if all(k in part.columns and k in pbp.columns for k in keys):
            return list(keys)
    raise RuntimeError("M67 participation/PBP have no shared game+play key")


def entropy(series: pd.Series) -> float:
    q = text(series)
    q = q[q.ne("")]
    if q.empty:
        return np.nan
    p = q.value_counts(normalize=True).to_numpy(dtype=float)
    return float(-(p * np.log(p)).sum())


def rate(g: pd.DataFrame, mask: pd.Series, col: str = "_dbr") -> float:
    q = g.loc[mask & num(g[col]).notna(), col]
    return float(num(q).mean()) if len(q) else np.nan


def split_list(v) -> list[str]:
    s = str(v or "").strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return []
    return [x.strip() for x in re.split(r"[;,]", s) if x.strip()]


def is_ol_position(pos: str) -> bool:
    p = str(pos or "").strip().upper()
    return p in {"C", "G", "T", "OL", "OG", "OT", "LT", "RT", "LG", "RG"}


def is_skill_position(pos: str) -> bool:
    p = str(pos or "").strip().upper()
    return p in {"RB", "FB", "WR", "TE"}


def top_ids(g: pd.DataFrame, predicate, n: int) -> tuple[str, float]:
    counts: dict[str, int] = {}
    total_slots = 0
    if "offense_players" not in g.columns or "offense_positions" not in g.columns:
        return "", np.nan
    for _, r in g[["offense_players", "offense_positions"]].iterrows():
        ids = split_list(r["offense_players"])
        pos = split_list(r["offense_positions"])
        if len(ids) != len(pos):
            continue
        for pid, pp in zip(ids, pos):
            if predicate(pp):
                counts[pid] = counts.get(pid, 0) + 1
                total_slots += 1
    if not counts:
        return "", np.nan
    ordered = sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))
    ids = [k for k, _ in ordered[:n]]
    concentration = sum(v for _, v in ordered[:n]) / total_slots if total_slots else np.nan
    return ";".join(ids), float(concentration)


def personnel_code(v) -> str:
    s = str(v or "").upper()
    vals = {}
    for label in ("RB", "TE", "WR"):
        m = re.search(rf"(\d+)\s*{label}", s)
        vals[label] = int(m.group(1)) if m else 0
    return f"{vals['RB']}{vals['TE']}" if any(vals.values()) else ""


def load_sources(seasons: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict]]:
    import nflreadpy as nfl

    pbp = lower(to_pd(nfl.load_pbp(seasons=seasons)))
    part = lower(to_pd(nfl.load_participation(seasons=seasons)))
    injury_ok = True
    try:
        inj = lower(to_pd(nfl.load_injuries(seasons=seasons)))
    except Exception as exc:
        print(f"[M67] injury provider unavailable: {exc}")
        inj = pd.DataFrame()
        injury_ok = False

    manifest = [
        {
            "family": "pbp_intent",
            "source": "nflverse_pbp",
            "status": "recovered" if len(pbp) else "unavailable",
            "live_2026_capability": "live_in_season",
            "notes": "rolling prior-game intent; no target-game plays used",
        },
        {
            "family": "formation_personnel_continuity",
            "source": "nflverse_participation",
            "status": "recovered" if len(part) else "unavailable",
            "live_2026_capability": "historical_only_postseason_release_2023plus",
            "notes": "diagnostic until equivalent live source is acquired",
        },
        {
            "family": "injury_availability",
            "source": "nflverse_injuries",
            "status": "recovered" if injury_ok else "unavailable",
            "live_2026_capability": "live_pregame",
            "notes": "official report status/practice fields by offensive position",
        },
        {
            "family": "actual_playcaller",
            "source": "not_in_current_nflverse_stack",
            "status": "not_recovered_m67",
            "live_2026_capability": "requires_new_source",
            "notes": "preserved as next new-information family",
        },
        {
            "family": "true_playoff_leverage",
            "source": "not_built_m67",
            "status": "not_recovered_m67",
            "live_2026_capability": "requires_new_derived_source",
            "notes": "preserved after offensive-intent audit",
        },
    ]
    return pbp, part, inj, manifest


def build_team_games(pbp: pd.DataFrame, part: pd.DataFrame) -> pd.DataFrame:
    keys = join_keys(part, pbp)
    needed = [
        *keys, "season", "week", "season_type", "posteam", "defteam", "qb_dropback",
        "pass_attempt", "rush_attempt", "down", "ydstogo", "score_differential",
        "qtr", "quarter_seconds_remaining", "shotgun", "no_huddle",
    ]
    right = pbp[[c for c in needed if c in pbp.columns]].drop_duplicates(keys)
    x = part.merge(right, on=keys, how="inner", suffixes=("", "_pbp"), validate="one_to_one")
    if x.empty:
        raise RuntimeError("M67 participation/PBP join produced zero rows")
    if "season_type" in x.columns:
        reg = x[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            x = reg
    team_col = "posteam" if "posteam" in x.columns else "possession_team"
    x["team"] = x[team_col].map(canon_team)
    x = x[x.team.ne("")].copy()
    x["_dbr"] = num(x.get("qb_dropback", 0)).fillna(0).clip(0, 1)
    rush = num(x.get("rush_attempt", 0)).fillna(0).eq(1)
    pas = num(x.get("pass_attempt", 0)).fillna(0).eq(1)
    x["_scrimmage"] = (x["_dbr"].eq(1) | rush | pas).astype(int)
    x = x[x._scrimmage.eq(1)].copy()
    x["_down"] = num(x.get("down", np.nan))
    x["_ydstogo"] = num(x.get("ydstogo", np.nan))
    x["_score"] = num(x.get("score_differential", np.nan))
    x["_qtr"] = num(x.get("qtr", np.nan))
    x["_qsec"] = num(x.get("quarter_seconds_remaining", np.nan))
    x["_no_huddle"] = num(x.get("no_huddle", np.nan))
    x["_shotgun_pbp"] = num(x.get("shotgun", np.nan))

    rows = []
    for (season, week, team), g in x.groupby(["season", "week", "team"], sort=True):
        score = g._score
        down = g._down
        qtr = g._qtr
        rec: dict = {"season": int(season), "week": int(week), "team": canon_team(team)}
        rec["intent_early_down_neutral_dbr"] = rate(g, down.le(2) & score.abs().le(7))
        rec["intent_q1_dbr"] = rate(g, qtr.eq(1))
        rec["intent_first_half_dbr"] = rate(g, qtr.le(2))
        rec["intent_two_minute_dbr"] = rate(g, qtr.isin([2, 4]) & g._qsec.le(120))
        rec["intent_third_long_dbr"] = rate(g, down.eq(3) & g._ydstogo.ge(7))
        rec["intent_trailing_1_7_dbr"] = rate(g, score.between(-7, -1))
        rec["intent_trailing_8_14_dbr"] = rate(g, score.between(-14, -8))
        rec["intent_trailing_15plus_dbr"] = rate(g, score.le(-15))
        rec["intent_leading_1_7_dbr"] = rate(g, score.between(1, 7))
        rec["intent_leading_8_14_dbr"] = rate(g, score.between(8, 14))
        rec["intent_leading_15plus_dbr"] = rate(g, score.ge(15))
        rec["intent_no_huddle_rate"] = float(g._no_huddle.mean()) if g._no_huddle.notna().any() else np.nan
        rec["intent_shotgun_rate"] = float(g._shotgun_pbp.mean()) if g._shotgun_pbp.notna().any() else np.nan

        form = text(g.get("offense_formation", pd.Series("", index=g.index))).str.upper()
        pers = text(g.get("offense_personnel", pd.Series("", index=g.index))).str.upper()
        form_valid = form.ne("")
        pers_valid = pers.ne("")
        denom_f = int(form_valid.sum())
        denom_p = int(pers_valid.sum())
        def fr(pattern: str) -> float:
            return float(form[form_valid].str.contains(pattern, regex=True).mean()) if denom_f else np.nan
        rec["personnel_formation_shotgun_rate"] = fr("SHOTGUN")
        rec["personnel_formation_pistol_rate"] = fr("PISTOL")
        rec["personnel_formation_singleback_rate"] = fr("SINGLEBACK|SINGLE BACK")
        rec["personnel_formation_i_form_rate"] = fr("I_FORM|I FORM")
        rec["personnel_formation_empty_rate"] = fr("EMPTY")
        codes = pers.map(personnel_code)
        for code in ("11", "12", "21", "22", "10"):
            rec[f"personnel_{code}_rate"] = float(codes[pers_valid].eq(code).mean()) if denom_p else np.nan
        shotgun = form.str.contains("SHOTGUN", na=False)
        rec["personnel_shotgun_dbr"] = rate(g, shotgun)
        rec["personnel_under_center_dbr"] = rate(g, form_valid & ~shotgun)
        rec["personnel_11_dbr"] = rate(g, codes.eq("11"))
        rec["personnel_12_dbr"] = rate(g, codes.eq("12"))
        rec["personnel_formation_entropy"] = entropy(form[form_valid])
        rec["personnel_group_entropy"] = entropy(codes[pers_valid])
        ol_ids, ol_conc = top_ids(g, is_ol_position, 5)
        skill_ids, _ = top_ids(g, is_skill_position, 6)
        rec["_ol_top5_ids"] = ol_ids
        rec["_skill_top6_ids"] = skill_ids
        rec["continuity_ol_snap_concentration"] = ol_conc
        rows.append(rec)

    out = pd.DataFrame(rows).sort_values(["team", "season", "week"]).reset_index(drop=True)
    if out.empty:
        raise RuntimeError("M67 team-game feature table is empty")
    out["continuity_ol_top5_overlap_prev"] = np.nan
    out["continuity_skill_top6_overlap_prev"] = np.nan
    for team, idx in out.groupby("team", sort=False).groups.items():
        order = list(out.loc[idx].sort_values(["season", "week"]).index)
        prev_ol: set[str] | None = None
        prev_skill: set[str] | None = None
        for i in order:
            cur_ol = set(split_list(out.at[i, "_ol_top5_ids"]))
            cur_skill = set(split_list(out.at[i, "_skill_top6_ids"]))
            if prev_ol is not None and cur_ol:
                out.at[i, "continuity_ol_top5_overlap_prev"] = len(cur_ol & prev_ol) / max(1, min(5, len(cur_ol)))
            if prev_skill is not None and cur_skill:
                out.at[i, "continuity_skill_top6_overlap_prev"] = len(cur_skill & prev_skill) / max(1, min(6, len(cur_skill)))
            if cur_ol:
                prev_ol = cur_ol
            if cur_skill:
                prev_skill = cur_skill
    return out


def injury_team_week(inj: pd.DataFrame) -> pd.DataFrame:
    cols = ["season", "week", "team"] + AVAILABILITY_FEATURES
    if inj.empty:
        return pd.DataFrame(columns=cols)
    x = inj.copy()
    if "season_type" in x.columns:
        reg = x[x.season_type.astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            x = reg
    if "team" not in x or "week" not in x or "season" not in x:
        return pd.DataFrame(columns=cols)
    x["team"] = x.team.map(canon_team)
    pos = text(x.get("position", pd.Series("", index=x.index))).str.upper()
    status = text(x.get("report_status", x.get("status", pd.Series("", index=x.index)))).str.upper()
    practice = text(x.get("practice_status", pd.Series("", index=x.index))).str.upper()
    x["_outd"] = status.str.contains("OUT|DOUBTFUL", regex=True)
    x["_q"] = status.str.contains("QUESTIONABLE", regex=True)
    x["_dnp"] = practice.str.contains("DNP|DID NOT", regex=True)
    x["_limited"] = practice.str.contains("LIMITED", regex=True)
    x["_ol"] = pos.isin(["C", "G", "T", "OL", "OG", "OT", "LT", "RT", "LG", "RG"])
    x["_rb"] = pos.isin(["RB", "FB"])
    x["_wrte"] = pos.isin(["WR", "TE"])
    x["_skill"] = x._rb | x._wrte
    rows = []
    for (season, week, team), g in x.groupby(["season", "week", "team"], sort=True):
        rows.append({
            "season": int(season), "week": int(week), "team": canon_team(team),
            "availability_report_rows": int(len(g)),
            "availability_out_doubtful_total": int(g._outd.sum()),
            "availability_questionable_total": int(g._q.sum()),
            "availability_dnp_total": int(g._dnp.sum()),
            "availability_limited_total": int(g._limited.sum()),
            "availability_ol_out_doubtful": int((g._outd & g._ol).sum()),
            "availability_ol_questionable": int((g._q & g._ol).sum()),
            "availability_rb_out_doubtful": int((g._outd & g._rb).sum()),
            "availability_wrte_out_doubtful": int((g._outd & g._wrte).sum()),
            "availability_skill_out_doubtful": int((g._outd & g._skill).sum()),
        })
    return pd.DataFrame(rows)


def rolling_features(master: pd.DataFrame, team_games: pd.DataFrame, injuries: pd.DataFrame) -> pd.DataFrame:
    numeric_history = INTENT_BASE + PERSONNEL_BASE + CONTINUITY_BASE
    out = []
    master = master[["season", "week", "team"]].drop_duplicates().copy()
    for _, r in master.iterrows():
        season, week, team = int(r.season), int(r.week), canon_team(r.team)
        h = team_games[(team_games.team.eq(team)) & ((num(team_games.season) < season) | ((num(team_games.season) == season) & (num(team_games.week) < week)))].copy()
        h = h[num(h.season).ge(season - 1)].sort_values(["season", "week"]).tail(HISTORY_GAMES)
        rec = {"season": season, "week": week, "team": team, "history_games": int(len(h))}
        for c in numeric_history:
            s = num(h[c]).dropna() if c in h else pd.Series(dtype=float)
            rec[f"{c}_mean8"] = float(s.tail(HISTORY_GAMES).mean()) if len(s) else np.nan
            rec[f"{c}_mean3"] = float(s.tail(RECENT_GAMES).mean()) if len(s) else np.nan
        out.append(rec)
    feat = pd.DataFrame(out)
    if not injuries.empty:
        feat = feat.merge(injuries, on=["season", "week", "team"], how="left", validate="one_to_one")
    for c in AVAILABILITY_FEATURES:
        if c not in feat:
            feat[c] = np.nan
    return feat


def coverage_table(features: pd.DataFrame) -> pd.DataFrame:
    families = {
        "pbp_intent": [c for c in features if c.startswith("intent_")],
        "formation_personnel": [c for c in features if c.startswith("personnel_")],
        "lineup_continuity": [c for c in features if c.startswith("continuity_")],
        "injury_availability": [c for c in features if c.startswith("availability_")],
    }
    rows = []
    n = len(features)
    for family, cols in families.items():
        for c in cols:
            nn = int(num(features[c]).notna().sum())
            rows.append({"family": family, "feature": c, "non_null": nn, "n": n, "coverage": nn / n if n else np.nan})
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--master-game-level", type=Path, required=True)
    p.add_argument("--seasons", default="2023,2024,2025")
    p.add_argument("--out-dir", type=Path, required=True)
    args = p.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    seasons = [int(x.strip()) for x in args.seasons.split(",") if x.strip()]
    master = lower(pd.read_csv(args.master_game_level))
    master["team"] = master.team.map(canon_team)
    pbp, part, inj, manifest = load_sources(seasons)
    games = build_team_games(pbp, part)
    injuries = injury_team_week(inj)
    features = rolling_features(master, games, injuries)
    if len(features) != master[["season", "week", "team"]].drop_duplicates().shape[0]:
        raise RuntimeError("M67 target feature row count changed unexpectedly")
    games.to_csv(args.out_dir / "m67_historical_team_game_new_information.csv", index=False)
    features.to_csv(args.out_dir / "m67_pregame_new_information_features.csv", index=False)
    pd.DataFrame(manifest).to_csv(args.out_dir / "m67_source_manifest.csv", index=False)
    cov = coverage_table(features)
    cov.to_csv(args.out_dir / "m67_feature_coverage.csv", index=False)
    print("=== M67 SOURCE MANIFEST ===")
    print(pd.DataFrame(manifest).to_string(index=False))
    print("=== M67 COVERAGE SUMMARY ===")
    print(cov.groupby("family").agg(features=("feature", "nunique"), median_coverage=("coverage", "median"), min_coverage=("coverage", "min")).reset_index().to_string(index=False))
    print(f"[M67] target_team_weeks={len(features)} historical_team_games={len(games)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
