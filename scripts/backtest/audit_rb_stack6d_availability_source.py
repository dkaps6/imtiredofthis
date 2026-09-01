#!/usr/bin/env python3
"""RB STACK6D / ND5: pregame backfield availability source audit.

No rushing outcome model is fit and no sportsbook data is loaded.
Target-game participation is used only as delayed semantic benchmark truth.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

TEAM_GAME_COVERAGE_GATE = 0.95
IDENTITY_GATE = 0.95
PARTICIPATION_GAME_GATE = 0.90
INJURY_TIMESTAMP_PARSE_GATE = 0.95
INA_ABSENCE_GATE = 0.98
OUT_PREDEADLINE_RATE_GATE = 0.95
OUT_ABSENCE_GATE = 0.98
P3_AVAILABILITY_COVERAGE_GATE = 0.50


def to_pd(x):
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def lower(x):
    z = x.copy()
    z.columns = [str(c).strip().lower() for c in z.columns]
    return z


def num(x):
    return pd.to_numeric(x, errors="coerce")


def clean_id(v):
    if pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def clean_status(v):
    return "" if pd.isna(v) else str(v).strip().upper()


def split_list(v):
    s = "" if pd.isna(v) else str(v).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return []
    return [q.strip() for q in re.split(r"[;,]", s) if q.strip()]


def one(root: Path, name: str):
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name} under {root}; found {len(hits)}")
    return lower(pd.read_csv(hits[0], low_memory=False))


def load_sources(seasons):
    import nflreadpy as nfl

    rosters = lower(to_pd(nfl.load_rosters_weekly(seasons=seasons)))
    injuries = lower(to_pd(nfl.load_injuries(seasons=seasons)))
    schedules = lower(to_pd(nfl.load_schedules(seasons=seasons)))
    participation = lower(to_pd(nfl.load_participation(seasons=seasons)))
    return rosters, injuries, schedules, participation


def choose(z, candidates):
    return next((c for c in candidates if c in z.columns), None)


def regular_only(z):
    if "season_type" not in z.columns:
        return z.copy()
    return z.loc[z.season_type.astype(str).str.upper().eq("REG")].copy()


def prepare_schedule(x):
    z = regular_only(x.copy())
    required = ["season", "week", "home_team", "away_team"]
    missing = [c for c in required if c not in z.columns]
    if missing:
        raise RuntimeError(f"STACK6D schedule missing {missing}")
    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    z["home_team"] = z.home_team.map(canon_team)
    z["away_team"] = z.away_team.map(canon_team)
    game_id_col = choose(z, ["game_id", "nflverse_game_id", "old_game_id"])
    if game_id_col:
        z["game_key"] = z[game_id_col].astype(str)
    else:
        z["game_key"] = (
            z.season.astype(str) + "_" + z.week.astype(str) + "_" + z.away_team + "_" + z.home_team
        )

    # nflverse schedules expose gameday + gametime. NFL schedule clock times are
    # interpreted in America/New_York for this audit; convert to UTC before
    # comparing with explicitly-zoned injury timestamps.
    if "gameday" in z.columns and "gametime" in z.columns:
        raw = z.gameday.astype(str).str.strip() + " " + z.gametime.astype(str).str.strip()
        naive = pd.to_datetime(raw, errors="coerce")
        try:
            local = naive.dt.tz_localize("America/New_York", ambiguous="NaT", nonexistent="NaT")
            z["kickoff_utc"] = local.dt.tz_convert("UTC")
        except Exception:
            z["kickoff_utc"] = pd.NaT
    else:
        dt_col = choose(z, ["game_datetime", "start_time", "start_time_utc", "kickoff"])
        z["kickoff_utc"] = pd.to_datetime(z[dt_col], utc=True, errors="coerce") if dt_col else pd.NaT

    z["decision_deadline_utc"] = z.kickoff_utc - pd.Timedelta(minutes=90)
    home = z[["season", "week", "game_key", "home_team", "kickoff_utc", "decision_deadline_utc"]].rename(columns={"home_team": "team"})
    away = z[["season", "week", "game_key", "away_team", "kickoff_utc", "decision_deadline_utc"]].rename(columns={"away_team": "team"})
    tg = pd.concat([home, away], ignore_index=True)
    return z, tg


def prepare_rosters(x, team_games):
    z = regular_only(x.copy())
    required = ["season", "week", "team", "position", "status"]
    missing = [c for c in required if c not in z.columns]
    if missing:
        raise RuntimeError(f"STACK6D weekly rosters missing {missing}")
    id_col = choose(z, ["gsis_id", "player_id"])
    if not id_col:
        raise RuntimeError("STACK6D weekly rosters missing GSIS/player identity")
    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(canon_team)
    z["position"] = z.position.astype(str).str.upper().str.strip()
    z["status"] = z.status.map(clean_status)
    z["player_id"] = z[id_col].map(clean_id)
    z = z.loc[z.position.isin(["RB", "FB"]) & z.player_id.ne("")].copy()
    z = z.merge(team_games, on=["season", "week", "team"], how="left", validate="many_to_one")

    # Discover candidate historical timing/provenance fields conservatively.
    timing_cols = []
    for c in x.columns:
        lc = str(c).lower()
        if any(tok in lc for tok in ["modified", "updated", "timestamp", "transaction_date", "status_date"]):
            timing_cols.append(c)
    return z, timing_cols


def explicit_timezone_mask(s):
    text = s.astype(str).str.strip()
    return text.str.contains(r"(?:Z|UTC|[+-]\d{2}:?\d{2})$", case=False, regex=True, na=False)


def prepare_injuries(x, team_games):
    z = regular_only(x.copy())
    required = ["season", "week", "team", "position", "report_status", "date_modified"]
    missing = [c for c in required if c not in z.columns]
    if missing:
        raise RuntimeError(f"STACK6D injuries missing {missing}")
    id_col = choose(z, ["gsis_id", "player_id"])
    if not id_col:
        raise RuntimeError("STACK6D injuries missing GSIS/player identity")
    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    z["team"] = z.team.map(canon_team)
    z["position"] = z.position.astype(str).str.upper().str.strip()
    z["player_id"] = z[id_col].map(clean_id)
    z["report_status_norm"] = z.report_status.map(clean_status)
    z = z.loc[z.position.isin(["RB", "FB"]) & z.player_id.ne("")].copy()
    z = z.merge(team_games, on=["season", "week", "team"], how="left", validate="many_to_one")

    raw = z.date_modified.astype(str)
    z["date_modified_has_clock"] = raw.str.contains(r"\d{1,2}:\d{2}", regex=True, na=False)
    z["date_modified_explicit_tz"] = explicit_timezone_mask(raw)
    z["date_modified_utc"] = pd.to_datetime(z.date_modified, utc=True, errors="coerce")
    z["date_modified_parseable"] = z.date_modified_utc.notna()
    # A timing comparison is considered provable only when the raw historical
    # timestamp includes both a clock time and explicit timezone/offset.
    z["timestamp_provenance_usable"] = (
        z.date_modified_parseable & z.date_modified_has_clock & z.date_modified_explicit_tz
    )
    z["modified_by_deadline"] = (
        z.timestamp_provenance_usable
        & z.decision_deadline_utc.notna()
        & z.date_modified_utc.le(z.decision_deadline_utc)
    )
    return z


def participation_benchmark(x, team_games):
    z = regular_only(x.copy())
    # Participation rows carry game/week plus offensive player and position arrays.
    if "season" not in z.columns or "week" not in z.columns:
        gid = choose(z, ["nflverse_game_id", "game_id", "old_game_id"])
        if gid:
            parts = z[gid].astype(str).str.split("_", expand=True)
            if "season" not in z.columns:
                z["season"] = num(parts[0])
            if "week" not in z.columns and parts.shape[1] > 1:
                z["week"] = num(parts[1])
    z["season"] = num(z.season).astype(int)
    z["week"] = num(z.week).astype(int)
    team_col = choose(z, ["posteam", "possession_team", "team"])
    if not team_col:
        raise RuntimeError("STACK6D participation missing possession team")
    z["team"] = z[team_col].map(canon_team)
    player_col = choose(z, ["offense_players"])
    pos_col = choose(z, ["offense_positions"])
    if not player_col or not pos_col:
        raise RuntimeError("STACK6D participation missing offense arrays")

    present = set()
    game_rows = set()
    aligned = 0
    eligible = 0
    for _, r in z.iterrows():
        if not r.team:
            continue
        game_rows.add((int(r.season), int(r.week), str(r.team)))
        ids = split_list(r[player_col])
        pos = split_list(r[pos_col])
        if ids or pos:
            eligible += 1
        if not ids or len(ids) != len(pos):
            continue
        aligned += 1
        for pid, pp in zip(ids, pos):
            if str(pp).upper() in {"RB", "FB"} and clean_id(pid):
                present.add((int(r.season), int(r.week), str(r.team), clean_id(pid)))
    return present, game_rows, aligned / max(eligible, 1)


def enrich_participation_flag(z, present, game_rows):
    q = z.copy()
    q["participation_game_available"] = [
        int((int(s), int(w), str(t)) in game_rows)
        for s, w, t in zip(q.season, q.week, q.team)
    ]
    q["offensive_participation_present"] = [
        int((int(s), int(w), str(t), str(pid)) in present)
        for s, w, t, pid in zip(q.season, q.week, q.team, q.player_id)
    ]
    return q


def p3_team_population(root):
    x = one(root, "stack6_2025_casebook.csv")
    required = ["season", "week", "team", "depth_rank", "stack6_risk"]
    missing = [c for c in required if c not in x.columns]
    if missing:
        raise RuntimeError(f"STACK6D P3 population casebook missing {missing}")
    x["season"] = num(x.season).astype(int)
    x["week"] = num(x.week).astype(int)
    x["team"] = x.team.map(canon_team)
    risk = x.stack6_risk.astype(str).str.lower().isin(["true", "1", "yes"])
    mask = num(x.week).ge(6) & (~risk) & num(x.depth_rank).ge(2)
    return x.loc[mask, ["season", "week", "team"]].drop_duplicates()


def safe_rate(mask_or_series):
    s = pd.Series(mask_or_series)
    return float(s.mean()) if len(s) else np.nan


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seasons", default="2024,2025")
    ap.add_argument("--stack6-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)
    seasons = [int(v) for v in str(a.seasons).split(",")]

    rosters_raw, injuries_raw, schedules_raw, participation_raw = load_sources(seasons)
    schedules, team_games = prepare_schedule(schedules_raw)
    rosters, roster_timing_cols = prepare_rosters(rosters_raw, team_games)
    injuries = prepare_injuries(injuries_raw, team_games)
    present, participation_games, array_alignment = participation_benchmark(participation_raw, team_games)
    rosters = enrich_participation_flag(rosters, present, participation_games)
    injuries = enrich_participation_flag(injuries, present, participation_games)

    scheduled_tg = team_games[["season", "week", "team"]].drop_duplicates()
    roster_tg = rosters[["season", "week", "team"]].drop_duplicates()
    team_game_coverage = len(scheduled_tg.merge(roster_tg, on=["season", "week", "team"], how="inner")) / max(len(scheduled_tg), 1)
    roster_identity_rate = safe_rate(rosters.player_id.ne(""))
    part_game_rate = safe_rate(rosters.participation_game_available.eq(1))

    injury_parse_rate = safe_rate(injuries.date_modified_parseable)
    injury_usable_tz_rate = safe_rate(injuries.timestamp_provenance_usable)
    injury_by_deadline_rate = safe_rate(injuries.loc[injuries.timestamp_provenance_usable, "modified_by_deadline"])

    out = injuries.report_status_norm.eq("OUT")
    out_usable = out & injuries.timestamp_provenance_usable
    out_by_deadline_rate = safe_rate(injuries.loc[out_usable, "modified_by_deadline"])
    predeadline_out = out & injuries.modified_by_deadline
    predeadline_q = injuries.report_status_norm.eq("QUESTIONABLE") & injuries.modified_by_deadline
    predeadline_d = injuries.report_status_norm.eq("DOUBTFUL") & injuries.modified_by_deadline

    ina = rosters.status.eq("INA") & rosters.participation_game_available.eq(1)
    act = rosters.status.eq("ACT") & rosters.participation_game_available.eq(1)
    ina_absence_rate = safe_rate(rosters.loc[ina, "offensive_participation_present"].eq(0))
    act_presence_rate = safe_rate(rosters.loc[act, "offensive_participation_present"].eq(1))
    out_absence_rate = safe_rate(injuries.loc[predeadline_out & injuries.participation_game_available.eq(1), "offensive_participation_present"].eq(0))
    q_presence_rate = safe_rate(injuries.loc[predeadline_q & injuries.participation_game_available.eq(1), "offensive_participation_present"].eq(1))
    d_presence_rate = safe_rate(injuries.loc[predeadline_d & injuries.participation_game_available.eq(1), "offensive_participation_present"].eq(1))

    roster_injury = rosters.merge(
        injuries[["season", "week", "team", "player_id", "report_status_norm", "modified_by_deadline"]],
        on=["season", "week", "team", "player_id"],
        how="left",
    )
    both = roster_injury.report_status_norm.notna() & roster_injury.modified_by_deadline.fillna(False)
    ina_vs_out_agreement = safe_rate(
        roster_injury.loc[both, "status"].eq("INA")
        == roster_injury.loc[both, "report_status_norm"].eq("OUT")
    )
    injury_coverage_roster = safe_rate(roster_injury.report_status_norm.notna())

    # Weekly roster data has no accepted exact-game inactive timing unless a
    # qualifying provenance field exists. Generic roster dictionary status INA
    # is not, by itself, an official game-day inactive declaration contract.
    roster_has_timing_provenance = int(len(roster_timing_cols) > 0)
    roster_exact_inactive_contract = 0
    exact_inactive_timing_valid = int(roster_has_timing_provenance and roster_exact_inactive_contract)

    p3teams = p3_team_population(a.stack6_root)
    pre_out_tg = injuries.loc[predeadline_out, ["season", "week", "team"]].drop_duplicates()
    p3_out_info_coverage = len(p3teams.merge(pre_out_tg, on=["season", "week", "team"], how="inner")) / max(len(p3teams), 1)

    infrastructure = {
        "gate_team_game_coverage": int(team_game_coverage >= TEAM_GAME_COVERAGE_GATE),
        "gate_identity": int(roster_identity_rate >= IDENTITY_GATE),
        "gate_participation_game": int(part_game_rate >= PARTICIPATION_GAME_GATE),
        "gate_injury_timestamp_parse": int(injury_parse_rate >= INJURY_TIMESTAMP_PARSE_GATE),
    }
    exact_gates = {
        "gate_exact_inactive_timing_provenance": int(exact_inactive_timing_valid == 1),
        "gate_ina_absence_ge_098": int(pd.notna(ina_absence_rate) and ina_absence_rate >= INA_ABSENCE_GATE),
    }
    out_gates = {
        "gate_out_predeadline_ge_095": int(pd.notna(out_by_deadline_rate) and out_by_deadline_rate >= OUT_PREDEADLINE_RATE_GATE),
        "gate_out_absence_ge_098": int(pd.notna(out_absence_rate) and out_absence_rate >= OUT_ABSENCE_GATE),
        "gate_p3_out_info_coverage_ge_050": int(p3_out_info_coverage >= P3_AVAILABILITY_COVERAGE_GATE),
    }

    infra_pass = int(all(infrastructure.values()))
    exact_pass = int(infra_pass and all(exact_gates.values()))
    out_pass = int(infra_pass and all(out_gates.values()))
    if exact_pass:
        disposition = "GO_STACK6D_EXACT_INACTIVE_COMPETITOR_STATE"
    elif out_pass:
        disposition = "GO_STACK6D_DEFINITE_OUT_COMPETITOR_STATE_ONLY"
    else:
        disposition = "STACK6D_AVAILABILITY_SOURCE_NOT_TIMESTAMP_SAFE"

    source = pd.DataFrame(
        [
            {
                "weekly_roster_rows_all_positions": int(len(rosters_raw)),
                "injury_rows_all_positions": int(len(injuries_raw)),
                "schedule_games": int(len(schedules)),
                "rbfb_weekly_roster_player_games": int(len(rosters)),
                "rbfb_injury_rows": int(len(injuries)),
                "scheduled_team_games": int(len(scheduled_tg)),
                "rbfb_roster_team_game_coverage": team_game_coverage,
                "rbfb_roster_identity_rate": roster_identity_rate,
                "participation_benchmark_team_game_rate": part_game_rate,
                "participation_array_alignment": array_alignment,
                "schedule_kickoff_parse_rate": safe_rate(team_games.kickoff_utc.notna()),
                "injury_date_modified_parse_rate": injury_parse_rate,
                "injury_explicit_timezone_clock_rate": injury_usable_tz_rate,
                "injury_modified_by_deadline_rate_when_timing_usable": injury_by_deadline_rate,
                "out_rows": int(out.sum()),
                "out_rows_with_usable_timestamp": int(out_usable.sum()),
                "out_modified_by_deadline_rate_when_timing_usable": out_by_deadline_rate,
                "roster_ina_rows_benchmarked": int(ina.sum()),
                "roster_ina_offensive_participation_absence_rate": ina_absence_rate,
                "roster_act_rows_benchmarked": int(act.sum()),
                "roster_act_offensive_participation_presence_rate": act_presence_rate,
                "predeadline_out_rows_benchmarked": int((predeadline_out & injuries.participation_game_available.eq(1)).sum()),
                "predeadline_out_offensive_participation_absence_rate": out_absence_rate,
                "predeadline_questionable_participation_presence_rate": q_presence_rate,
                "predeadline_doubtful_participation_presence_rate": d_presence_rate,
                "roster_injury_report_row_coverage": injury_coverage_roster,
                "roster_ina_vs_predeadline_report_out_agreement": ina_vs_out_agreement,
                "weekly_roster_timing_candidate_column_count": len(roster_timing_cols),
                "weekly_roster_exact_gameday_inactive_contract": roster_exact_inactive_contract,
                "p3_eligible_team_games": int(len(p3teams)),
                "p3_team_games_with_predeadline_out_rbfb": int(len(p3teams.merge(pre_out_tg, on=["season", "week", "team"], how="inner"))),
                "p3_predeadline_out_information_coverage": p3_out_info_coverage,
                "outcome_model_fit": 0,
                "sportsbook_used": 0,
                "target_game_participation_feature_used": 0,
            }
        ]
    )

    gates = pd.DataFrame(
        [
            {
                **infrastructure,
                **exact_gates,
                **out_gates,
                "infrastructure_pass": infra_pass,
                "exact_inactive_source_pass": exact_pass,
                "definite_out_source_pass": out_pass,
                "source_gate_pass": int(exact_pass or out_pass),
                "disposition": disposition,
                "production_change": 0,
            }
        ]
    )

    schema_rows = []
    for source_name, frame in [
        ("weekly_rosters", rosters_raw),
        ("injuries", injuries_raw),
        ("schedules", schedules_raw),
        ("participation", participation_raw),
    ]:
        for c in frame.columns:
            schema_rows.append(
                {
                    "source": source_name,
                    "column": c,
                    "dtype": str(frame[c].dtype),
                    "nonnull_rate": float(frame[c].notna().mean()),
                    "sample_values": " | ".join(frame[c].dropna().astype(str).head(3).tolist()),
                }
            )
    schema = pd.DataFrame(schema_rows)
    roster_timing = pd.DataFrame(
        [{"candidate_timing_column": c} for c in roster_timing_cols]
        or [{"candidate_timing_column": "NONE"}]
    )

    roster_status = (
        rosters.groupby("status", dropna=False)
        .agg(
            n=("player_id", "size"),
            benchmark_n=("participation_game_available", "sum"),
            offensive_presence_rate=("offensive_participation_present", "mean"),
        )
        .reset_index()
    )
    injury_status = (
        injuries.groupby("report_status_norm", dropna=False)
        .agg(
            n=("player_id", "size"),
            timestamp_parse_rate=("date_modified_parseable", "mean"),
            usable_explicit_timing_rate=("timestamp_provenance_usable", "mean"),
            predeadline_rate=("modified_by_deadline", "mean"),
            offensive_presence_rate=("offensive_participation_present", "mean"),
        )
        .reset_index()
    )

    manifest = pd.DataFrame(
        [
            {
                "family": "weekly_roster_status",
                "source": "nflverse load_rosters_weekly",
                "candidate_field": "status",
                "exact_target_game_inactive_qualified": exact_pass,
                "allowed_role": "exact inactive only if timing/provenance and semantic gates pass",
            },
            {
                "family": "official_injury_game_status",
                "source": "nflverse load_injuries",
                "candidate_field": "report_status + date_modified",
                "exact_target_game_inactive_qualified": 0,
                "allowed_role": "definitely-out signal only if predeadline timestamp and absence gates pass",
            },
            {
                "family": "historical_participation",
                "source": "nflverse participation",
                "candidate_field": "offense_players/offense_positions",
                "exact_target_game_inactive_qualified": 0,
                "allowed_role": "delayed benchmark truth only; never fitted target-game pregame feature",
            },
        ]
    )

    outputs = {
        "stack6d_availability_source_audit.csv": source,
        "stack6d_availability_source_gates.csv": gates,
        "stack6d_availability_source_manifest.csv": manifest,
        "stack6d_availability_schema.csv": schema,
        "stack6d_weekly_roster_timing_columns.csv": roster_timing,
        "stack6d_roster_status_semantics.csv": roster_status,
        "stack6d_injury_status_timing_semantics.csv": injury_status,
        "stack6d_rbfb_weekly_roster_casebook.csv": rosters,
        "stack6d_rbfb_injury_casebook.csv": injuries,
    }
    for name, df in outputs.items():
        df.to_csv(a.out_dir / name, index=False)

    print("=== STACK6D availability source audit ===")
    print(source.to_string(index=False))
    print("=== STACK6D frozen source gates ===")
    print(gates.to_string(index=False))
    print("=== Weekly roster timing candidates ===")
    print(roster_timing.to_string(index=False))
    print("=== RB/FB roster status semantics ===")
    print(roster_status.to_string(index=False))
    print("=== RB/FB injury status timing semantics ===")
    print(injury_status.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
