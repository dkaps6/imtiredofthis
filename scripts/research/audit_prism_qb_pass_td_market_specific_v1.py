#!/usr/bin/env python3
"""Source-only audit for PRISM QB passing-TD market-specific V1.

This script does not score a TD projection. It verifies that historical
sportsbook-independent nflverse play-by-play preserves stable named-QB passing
TD and scoring-area pass topology with enough chronology for a later,
separately frozen predictive mechanism.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

VERSION = "PRISM_QB_PASS_TD_MARKET_SPECIFIC_V1_SOURCE_AUDIT"
FREEZE_COMMIT = "af709a937f61c39240b73a1d6960ab47e60eb152"
SEASONS = (2022, 2023, 2024, 2025)


def to_pd(x) -> pd.DataFrame:
    return x.to_pandas() if hasattr(x, "to_pandas") else pd.DataFrame(x)


def lower(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    y.columns = [str(c).strip().lower() for c in y.columns]
    return y


def pick(columns: Iterable[str], *names: str) -> str | None:
    cols = set(columns)
    return next((n for n in names if n in cols), None)


def num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_pbp(seasons: list[int]) -> pd.DataFrame:
    import nflreadpy as nfl
    frames = []
    for season in seasons:
        q = lower(to_pd(nfl.load_pbp(seasons=[int(season)])))
        if q.empty:
            raise RuntimeError(f"nflreadpy returned zero PBP rows for {season}")
        if "season_type" in q.columns:
            st = q["season_type"].astype("string").str.upper()
            q = q.loc[st.eq("REG")].copy()
        if q.empty:
            raise RuntimeError(f"zero REG PBP rows for {season}")
        frames.append(q)
    return pd.concat(frames, ignore_index=True, sort=False)


def resolve_schema(x: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "season": ("season",),
        "week": ("week",),
        "team": ("posteam", "offense_team", "team"),
        "pass_attempt": ("pass_attempt",),
        "pass_td": ("pass_touchdown", "passing_touchdown", "pass_td"),
        "passer_id": ("passer_player_id", "passer_id"),
        "passer_name": ("passer_player_name", "passer", "passer_name"),
        "yardline": ("yardline_100",),
        "game_id": ("game_id",),
    }
    out: dict[str, str] = {}
    for key, names in aliases.items():
        col = pick(x.columns, *names)
        if key == "passer_name" or key == "game_id":
            if col:
                out[key] = col
            continue
        if not col:
            raise RuntimeError(f"required PBP semantic {key} missing; tried {names}")
        out[key] = col
    return out


def aggregate(x: pd.DataFrame, schema: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    season = num(x[schema["season"]]).astype("Int64")
    week = num(x[schema["week"]]).astype("Int64")
    pass_attempt = num(x[schema["pass_attempt"]])
    pass_td = num(x[schema["pass_td"]])
    yardline = num(x[schema["yardline"]])
    team = x[schema["team"]].astype("string").fillna("").str.strip()
    passer_id = x[schema["passer_id"]].astype("string").fillna("").str.strip()

    attempts = x.loc[pass_attempt.eq(1)].copy()
    attempts["_season"] = season.loc[attempts.index]
    attempts["_week"] = week.loc[attempts.index]
    attempts["_team"] = team.loc[attempts.index]
    attempts["_passer_id"] = passer_id.loc[attempts.index]
    attempts["_pass_td"] = pass_td.loc[attempts.index]
    attempts["_yardline"] = yardline.loc[attempts.index]

    if "passer_name" in schema:
        attempts["_passer_name"] = (
            attempts[schema["passer_name"]].astype("string").fillna("").str.strip()
        )
    else:
        attempts["_passer_name"] = ""
    if "game_id" in schema:
        attempts["_game_id"] = (
            attempts[schema["game_id"]].astype("string").fillna("").str.strip()
        )
    else:
        attempts["_game_id"] = ""

    coverage = {
        "pass_attempt_rows": int(len(attempts)),
        "passer_id_coverage": float(attempts["_passer_id"].ne("").mean()) if len(attempts) else 0.0,
        "pass_td_coverage": float(attempts["_pass_td"].notna().mean()) if len(attempts) else 0.0,
        "yardline_coverage": float(attempts["_yardline"].notna().mean()) if len(attempts) else 0.0,
    }

    valid = attempts.loc[
        attempts["_passer_id"].ne("")
        & attempts["_team"].ne("")
        & attempts["_season"].notna()
        & attempts["_week"].notna()
    ].copy()
    valid["_pass_td"] = valid["_pass_td"].fillna(0.0)
    valid["_inside20"] = valid["_yardline"].le(20)
    valid["_inside10"] = valid["_yardline"].le(10)
    valid["_inside5"] = valid["_yardline"].le(5)
    for band in ("20", "10", "5"):
        valid[f"_inside{band}_td"] = valid[f"_inside{band}"] & valid["_pass_td"].eq(1)

    group = ["_season", "_week", "_team", "_passer_id"]
    rows = (
        valid.groupby(group, dropna=False)
        .agg(
            passer_name=("_passer_name", lambda s: next((str(v) for v in s if str(v).strip()), "")),
            game_id=("_game_id", lambda s: next((str(v) for v in s if str(v).strip()), "")),
            pass_attempts=("_pass_td", "size"),
            pass_tds=("_pass_td", "sum"),
            inside20_pass_attempts=("_inside20", "sum"),
            inside10_pass_attempts=("_inside10", "sum"),
            inside5_pass_attempts=("_inside5", "sum"),
            inside20_pass_tds=("_inside20_td", "sum"),
            inside10_pass_tds=("_inside10_td", "sum"),
            inside5_pass_tds=("_inside5_td", "sum"),
        )
        .reset_index()
        .rename(columns={
            "_season":"season", "_week":"week", "_team":"team", "_passer_id":"passer_id"
        })
    )
    for c in [
        "pass_attempts","pass_tds","inside20_pass_attempts","inside10_pass_attempts",
        "inside5_pass_attempts","inside20_pass_tds","inside10_pass_tds","inside5_pass_tds"
    ]:
        rows[c] = num(rows[c]).fillna(0).astype(int)
    rows["pass_td_rate"] = rows["pass_tds"] / rows["pass_attempts"].replace(0, np.nan)

    team_rows = (
        rows.groupby(["season","week","team"], as_index=False)
        .agg(
            team_pass_attempts=("pass_attempts","sum"),
            team_pass_tds=("pass_tds","sum"),
            passers_with_attempts=("passer_id","nunique"),
            top_passer_attempts=("pass_attempts","max"),
        )
    )
    team_rows["top_passer_attempt_share"] = (
        team_rows["top_passer_attempts"] / team_rows["team_pass_attempts"].replace(0, np.nan)
    )
    rows = rows.merge(
        team_rows,
        on=["season","week","team"],
        how="left",
        validate="many_to_one",
    )

    rows = rows.sort_values(["season","week","team","passer_id"]).reset_index(drop=True)
    return rows, team_rows, coverage


def add_prior_counts(rows: pd.DataFrame) -> pd.DataFrame:
    x = rows.sort_values(["season","week","passer_id"]).copy()
    x["_order"] = np.arange(len(x))
    prior_counts = pd.Series(index=x.index, dtype=int)
    for _, g in x.groupby("passer_id", sort=False):
        g = g.sort_values(["season","week","_order"])
        prior_counts.loc[g.index] = np.arange(len(g), dtype=int)
    x["strict_prior_qb_games"] = prior_counts.astype(int)
    return x.drop(columns=["_order"]).sort_values(["season","week","team","passer_id"]).reset_index(drop=True)


def audit_season(raw: pd.DataFrame, rows: pd.DataFrame, season: int, schema: dict[str,str]) -> dict:
    rs = raw.loc[num(raw[schema["season"]]).eq(season)].copy()
    if "season_type" in rs.columns:
        rs = rs.loc[rs["season_type"].astype("string").str.upper().eq("REG")].copy()
    attempt = num(rs[schema["pass_attempt"]]).eq(1)
    ra = rs.loc[attempt].copy()
    pass_td = num(ra[schema["pass_td"]])
    yard = num(ra[schema["yardline"]])
    pid = ra[schema["passer_id"]].astype("string").fillna("").str.strip()
    team = ra[schema["team"]].astype("string").fillna("").str.strip()
    weeks = sorted(int(w) for w in num(ra[schema["week"]]).dropna().unique())
    srows = rows.loc[rows["season"].eq(season)].copy()

    week_checks = []
    for w in weeks:
        q = ra.loc[num(ra[schema["week"]]).eq(w)]
        qtd = num(q[schema["pass_td"]]).fillna(0)
        qyl = num(q[schema["yardline"]])
        week_checks.append({
            "week": w,
            "pass_attempts": int(len(q)),
            "pass_tds": int(qtd.eq(1).sum()),
            "inside20_pass_attempts": int(qyl.le(20).sum()),
        })

    impossible = int((srows["pass_tds"] > srows["pass_attempts"]).sum())
    count_mono = int((
        (srows["inside5_pass_attempts"] > srows["inside10_pass_attempts"])
        | (srows["inside10_pass_attempts"] > srows["inside20_pass_attempts"])
        | (srows["inside20_pass_attempts"] > srows["pass_attempts"])
    ).sum())
    td_mono = int((
        (srows["inside5_pass_tds"] > srows["inside10_pass_tds"])
        | (srows["inside10_pass_tds"] > srows["inside20_pass_tds"])
        | (srows["inside20_pass_tds"] > srows["pass_tds"])
    ).sum())
    duplicates = int(srows.duplicated(["season","week","team","passer_id"]).sum())

    gates = {
        "reg_pbp_nonempty": len(rs) > 0,
        "teams_ge_30": int(team[team.ne("")].nunique()) >= 30,
        "passers_ge_40": int(pid[pid.ne("")].nunique()) >= 40,
        "passer_id_coverage_ge_99pct": float(pid.ne("").mean()) >= 0.99 if len(ra) else False,
        "pass_td_coverage_ge_99_9pct": float(pass_td.notna().mean()) >= 0.999 if len(ra) else False,
        "yardline_coverage_ge_99pct": float(yard.notna().mean()) >= 0.99 if len(ra) else False,
        "all_weeks_have_pass_attempts": bool(week_checks) and all(z["pass_attempts"] > 0 for z in week_checks),
        "all_weeks_have_pass_tds": bool(week_checks) and all(z["pass_tds"] > 0 for z in week_checks),
        "all_weeks_have_inside20_pass_attempts": bool(week_checks) and all(z["inside20_pass_attempts"] > 0 for z in week_checks),
        "duplicates_zero": duplicates == 0,
        "impossible_td_counts_zero": impossible == 0,
        "scoring_area_count_monotonicity": count_mono == 0,
        "scoring_area_td_monotonicity": td_mono == 0,
    }
    return {
        "season": season,
        "regular_weeks": weeks,
        "pbp_rows": int(len(rs)),
        "pass_attempt_rows": int(len(ra)),
        "teams": int(team[team.ne("")].nunique()),
        "distinct_passers": int(pid[pid.ne("")].nunique()),
        "passer_id_coverage": float(pid.ne("").mean()) if len(ra) else 0.0,
        "pass_td_coverage": float(pass_td.notna().mean()) if len(ra) else 0.0,
        "yardline_coverage": float(yard.notna().mean()) if len(ra) else 0.0,
        "aggregated_qb_games": int(len(srows)),
        "duplicates": duplicates,
        "impossible_td_count_rows": impossible,
        "scoring_area_count_monotonicity_violations": count_mono,
        "scoring_area_td_monotonicity_violations": td_mono,
        "week_checks": week_checks,
        "source_gates": gates,
        "source_ready": bool(all(gates.values())),
    }


def history_gate(rows: pd.DataFrame, season: int) -> dict:
    g = rows.loc[rows["season"].eq(season)].copy()
    supported = g.loc[g["strict_prior_qb_games"].ge(3)].copy()
    gates = {
        "supported_rows_ge_300": int(len(supported)) >= 300,
        "supported_qbs_ge_35": int(supported["passer_id"].nunique()) >= 35,
    }
    return {
        "season": season,
        "target_rows": int(len(g)),
        "target_qbs": int(g["passer_id"].nunique()),
        "rows_prior_ge_3": int(g["strict_prior_qb_games"].ge(3).sum()),
        "rows_prior_ge_5": int(g["strict_prior_qb_games"].ge(5).sum()),
        "rows_prior_ge_8": int(g["strict_prior_qb_games"].ge(8).sum()),
        "qbs_prior_ge_3": int(supported["passer_id"].nunique()),
        "history_gates": gates,
        "history_ready": bool(all(gates.values())),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    result = {
        "version": VERSION,
        "freeze_commit": FREEZE_COMMIT,
        "seasons": list(SEASONS),
        "predictive_relationships_scored": 0,
        "models_fit": 0,
        "threshold_searches": 0,
        "qb_pass_td_projections_created": 0,
        "outcomes_2026_used": 0,
        "week6_week7_outcomes_used": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "production_changed": False,
    }
    try:
        raw = load_pbp(list(SEASONS))
        schema = resolve_schema(raw)
        rows, team_rows, coverage = aggregate(raw, schema)
        rows = add_prior_counts(rows)

        summaries = {str(s): audit_season(raw, rows, s, schema) for s in SEASONS}
        history = {str(s): history_gate(rows, s) for s in (2024, 2025)}
        source_ready = all(v["source_ready"] for v in summaries.values())
        history_ready = all(v["history_ready"] for v in history.values())

        rows.to_csv(args.out_dir / "qb_pass_td_source_rows.csv", index=False)
        team_rows.to_csv(args.out_dir / "qb_pass_td_team_game_source.csv", index=False)
        pd.DataFrame([
            {
                "season": s,
                **{k:v for k,v in summaries[str(s)].items() if k not in {"week_checks","source_gates"}},
                **{f"gate_{k}":v for k,v in summaries[str(s)]["source_gates"].items()},
            }
            for s in SEASONS
        ]).to_csv(args.out_dir / "qb_pass_td_source_support.csv", index=False)

        result.update({
            "resolved_schema": schema,
            "aggregate_coverage": coverage,
            "season_summaries": summaries,
            "history_feasibility": history,
            "disposition": (
                "PRISM_QB_PASS_TD_MARKET_SPECIFIC_SOURCE_READY"
                if source_ready and history_ready
                else "PRISM_QB_PASS_TD_MARKET_SPECIFIC_SOURCE_BLOCKED"
            ),
        })
    except Exception as exc:
        result.update({
            "disposition": "PRISM_QB_PASS_TD_MARKET_SPECIFIC_SOURCE_INTEGRITY_FAILURE",
            "error": str(exc),
        })
        (args.out_dir / "qb_pass_td_source_result.json").write_text(
            json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (args.out_dir / "qb_pass_td_source_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
