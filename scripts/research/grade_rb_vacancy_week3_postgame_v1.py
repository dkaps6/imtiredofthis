#!/usr/bin/env python3
"""Grade the frozen 2026 Week-3 RB Vacancy Opportunity V1 lock.

This is a postgame attachment/grade only. The candidate projections, transfer
weights, YPC, ensemble weights, and cohort are immutable inputs from the
pregame artifact. No sportsbook data are used.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.player_stats_loader_v2 import load_weekly_player_stats
from scripts.utils.canonical_names import canonicalize_player_name_safe

SEASON = 2026
WEEK = 3


def _to_pandas(obj):
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def _pick_col(df: pd.DataFrame, names: tuple[str, ...]) -> str:
    for c in names:
        if c in df.columns:
            return c
    raise RuntimeError(f"none of required columns present: {names}; have={sorted(df.columns)}")


def _load_schedule() -> pd.DataFrame:
    import nflreadpy as nfl

    x = _to_pandas(nfl.load_schedules(seasons=[SEASON])).copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "season" in x.columns:
        x = x.loc[pd.to_numeric(x["season"], errors="coerce").eq(SEASON)]
    if "game_type" in x.columns:
        x = x.loc[x["game_type"].astype(str).str.upper().eq("REG")]
    x = x.loc[pd.to_numeric(x["week"], errors="coerce").eq(WEEK)].copy()
    if x.empty:
        raise RuntimeError("Week-3 schedule is empty")
    for c in ("home_team", "away_team", "home_score", "away_score"):
        if c not in x.columns:
            raise RuntimeError(f"schedule missing {c}")
    x["home_team"] = x["home_team"].map(canon_team)
    x["away_team"] = x["away_team"].map(canon_team)
    return x


def _assert_team_game_final(schedule: pd.DataFrame, team: str, opponent: str) -> None:
    t, o = canon_team(team), canon_team(opponent)
    q = schedule.loc[
        (
            schedule["home_team"].eq(t) & schedule["away_team"].eq(o)
        ) | (
            schedule["home_team"].eq(o) & schedule["away_team"].eq(t)
        )
    ].copy()
    if len(q) != 1:
        raise RuntimeError(f"expected exactly one Week-3 game {t}-{o}, found={len(q)}")
    r = q.iloc[0]
    if pd.isna(pd.to_numeric(pd.Series([r["home_score"]]), errors="coerce").iloc[0]) or pd.isna(
        pd.to_numeric(pd.Series([r["away_score"]]), errors="coerce").iloc[0]
    ):
        raise RuntimeError(f"game not final for {t}-{o}")


def _load_actuals() -> tuple[pd.DataFrame, pd.DataFrame]:
    stats = load_weekly_player_stats(SEASON).copy()
    stats.columns = [str(c).strip().lower() for c in stats.columns]
    stats = stats.loc[pd.to_numeric(stats["week"], errors="coerce").eq(WEEK)].copy()

    team_col = _pick_col(stats, ("recent_team", "team", "team_abbr", "club"))
    name_col = _pick_col(stats, ("player_display_name", "player_name", "player"))
    rush_att_col = _pick_col(stats, ("carries", "rushing_attempts", "rush_attempts"))
    rush_yd_col = _pick_col(stats, ("rushing_yards", "rush_yards"))

    stats["team"] = stats[team_col].astype("string").fillna("").str.strip().map(canon_team)
    canon = stats[name_col].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    stats["player"] = canon.map(lambda t: t[0])
    stats["player_clean_key"] = canon.map(lambda t: t[1])
    stats["actual_rush_att"] = pd.to_numeric(stats[rush_att_col], errors="coerce").fillna(0.0)
    stats["actual_rush_yards"] = pd.to_numeric(stats[rush_yd_col], errors="coerce").fillna(0.0)

    actual = (
        stats[["team", "player", "player_clean_key", "actual_rush_att", "actual_rush_yards"]]
        .drop_duplicates(["team", "player_clean_key"], keep="last")
        .reset_index(drop=True)
    )

    import nflreadpy as nfl

    roster = _to_pandas(nfl.load_rosters_weekly(SEASON)).copy()
    roster.columns = [str(c).strip().lower() for c in roster.columns]
    roster = roster.loc[pd.to_numeric(roster["week"], errors="coerce").eq(WEEK)].copy()
    rteam = _pick_col(roster, ("team", "team_abbr", "club_code"))
    rname = _pick_col(roster, ("full_name", "football_name", "player_name", "player"))
    roster["team"] = roster[rteam].astype("string").fillna("").str.strip().map(canon_team)
    rcanon = roster[rname].astype("string").fillna("").str.strip().map(canonicalize_player_name_safe)
    roster["player_clean_key"] = rcanon.map(lambda t: t[1])
    roster = roster[["team", "player_clean_key"]].drop_duplicates()
    return actual, roster


def _attach_actuals(lock: pd.DataFrame) -> pd.DataFrame:
    schedule = _load_schedule()
    actual, roster = _load_actuals()

    out = lock.copy()
    for c in ("team", "opponent"):
        out[c] = out[c].map(canon_team)

    unique_players = out[
        ["event_id", "team", "opponent", "player", "player_clean_key", "position", "direct_transfer_recipient"]
    ].drop_duplicates().copy()

    rows = []
    for r in unique_players.itertuples(index=False):
        _assert_team_game_final(schedule, r.team, r.opponent)
        q = actual.loc[
            actual["team"].eq(r.team)
            & actual["player_clean_key"].astype(str).eq(str(r.player_clean_key))
        ].copy()
        if len(q) > 1:
            raise RuntimeError(
                f"ambiguous weekly-stat identity {r.team} {r.player} key={r.player_clean_key} rows={len(q)}"
            )
        source = "player_game_logs_weekly_stats"
        if len(q) == 1:
            rush_att = float(q.iloc[0]["actual_rush_att"])
            rush_yd = float(q.iloc[0]["actual_rush_yards"])
        else:
            rq = roster.loc[
                roster["team"].eq(r.team)
                & roster["player_clean_key"].astype(str).eq(str(r.player_clean_key))
            ]
            if len(rq) != 1:
                raise RuntimeError(
                    f"missing player-game row lacks exact unambiguous roster verification: "
                    f"{r.team} {r.player} key={r.player_clean_key} roster_rows={len(rq)}"
                )
            rush_att = 0.0
            rush_yd = 0.0
            source = "final_game_exact_roster_verified_zero"

        rows.append(
            {
                "event_id": r.event_id,
                "team": r.team,
                "opponent": r.opponent,
                "player_clean_key": r.player_clean_key,
                "actual_rush_att": rush_att,
                "actual_rush_yards": rush_yd,
                "actual_source": source,
            }
        )

    actual_rows = pd.DataFrame(rows)
    out = out.merge(
        actual_rows,
        on=["event_id", "team", "opponent", "player_clean_key"],
        how="left",
        validate="many_to_one",
    )
    if out[["actual_rush_att", "actual_rush_yards"]].isna().any().any():
        raise RuntimeError("actual attachment left missing values")
    out["actual"] = np.where(
        out["market"].astype(str).eq("rush_att"),
        out["actual_rush_att"],
        out["actual_rush_yards"],
    )
    return out


def _metric(q: pd.DataFrame) -> dict:
    if q.empty:
        return {
            "rows": 0,
            "baseline_mae": np.nan,
            "candidate_mae": np.nan,
            "candidate_minus_baseline_mae": np.nan,
            "baseline_bias_actual_minus_projection": np.nan,
            "candidate_bias_actual_minus_projection": np.nan,
            "absolute_bias_delta_candidate_minus_baseline": np.nan,
            "candidate_closer": 0,
            "baseline_closer": 0,
            "tie": 0,
        }

    actual = pd.to_numeric(q["actual"], errors="raise")
    b = pd.to_numeric(q["baseline_ensemble_proj"], errors="raise")
    c = pd.to_numeric(q["candidate_ensemble_proj"], errors="raise")
    be = (actual - b)
    ce = (actual - c)
    ba = be.abs()
    ca = ce.abs()
    tol = 1e-12
    return {
        "rows": int(len(q)),
        "baseline_mae": float(ba.mean()),
        "candidate_mae": float(ca.mean()),
        "candidate_minus_baseline_mae": float(ca.mean() - ba.mean()),
        "baseline_bias_actual_minus_projection": float(be.mean()),
        "candidate_bias_actual_minus_projection": float(ce.mean()),
        "absolute_bias_delta_candidate_minus_baseline": float(abs(ce.mean()) - abs(be.mean())),
        "candidate_closer": int((ca < ba - tol).sum()),
        "baseline_closer": int((ba < ca - tol).sum()),
        "tie": int((ca - ba).abs().le(tol).sum()),
    }


def _summaries(detail: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []

    def add(scope: str, value: str, q: pd.DataFrame) -> None:
        for market in ("rush_att", "rush_yards"):
            m = q.loc[q["market"].astype(str).eq(market)]
            rows.append({"scope": scope, "scope_value": value, "market": market, **_metric(m)})

    add("ALL_LOCKED", "ALL", detail)
    add("DIRECT_RECIPIENTS", "ALL", detail.loc[pd.to_numeric(detail["direct_transfer_recipient"], errors="coerce").eq(1)])
    for team, q in detail.groupby("team", dropna=False):
        add("TEAM", str(team), q)

    player_actual = detail[
        ["event_id", "team", "player_clean_key", "actual_rush_att"]
    ].drop_duplicates()
    hi20 = set(
        tuple(x)
        for x in player_actual.loc[player_actual["actual_rush_att"].ge(20), ["event_id", "team", "player_clean_key"]]
        .itertuples(index=False, name=None)
    )
    hi25 = set(
        tuple(x)
        for x in player_actual.loc[player_actual["actual_rush_att"].ge(25), ["event_id", "team", "player_clean_key"]]
        .itertuples(index=False, name=None)
    )
    keys = list(zip(detail["event_id"], detail["team"], detail["player_clean_key"]))
    add("HIGH_VOLUME", "actual_rush_att>=20", detail.loc[[k in hi20 for k in keys]])
    add("HIGH_VOLUME", "actual_rush_att>=25", detail.loc[[k in hi25 for k in keys]])
    return pd.DataFrame(rows)


def _team_room(detail: pd.DataFrame) -> pd.DataFrame:
    players = detail.loc[detail["market"].astype(str).eq("rush_att")].copy()
    rows = []
    for (event_id, team), q in players.groupby(["event_id", "team"], dropna=False):
        rows.append(
            {
                "event_id": event_id,
                "team": team,
                "actual_rbfb_carries": float(q["actual_rush_att"].sum()),
                "actual_rbfb_rush_yards": float(
                    detail.loc[
                        detail["event_id"].eq(event_id)
                        & detail["team"].eq(team)
                        & detail["market"].astype(str).eq("rush_yards"),
                        ["player_clean_key", "actual_rush_yards"],
                    ].drop_duplicates("player_clean_key")["actual_rush_yards"].sum()
                ),
                "baseline_projected_active_room_carries": float(q["baseline_ensemble_proj"].sum()),
                "candidate_projected_active_room_carries": float(q["candidate_ensemble_proj"].sum()),
                "baseline_realized_mc_active_room_carries": float(q["baseline_realized_multinomial_mean_carries"].sum()),
                "candidate_realized_mc_active_room_carries": float(q["candidate_realized_multinomial_mean_carries"].sum()),
                "baseline_final_probability_sum": float(q["baseline_final_player_probability"].sum()),
                "candidate_final_probability_sum": float(q["candidate_final_player_probability"].sum()),
                "baseline_residual_probability": float(q["baseline_residual_probability"].iloc[0]),
                "candidate_residual_probability": float(q["candidate_residual_probability"].iloc[0]),
            }
        )
    return pd.DataFrame(rows)


def _classify(summary: pd.DataFrame) -> str:
    q = summary.loc[
        summary["scope"].eq("ALL_LOCKED")
        & summary["scope_value"].eq("ALL")
        & summary["market"].isin(["rush_att", "rush_yards"])
    ].set_index("market")
    if set(q.index) != {"rush_att", "rush_yards"}:
        return "WEEK3_EVALUATION_INVALID"

    deltas = q["candidate_minus_baseline_mae"]
    bias_deltas = q["absolute_bias_delta_candidate_minus_baseline"]

    if (deltas < 0).all() and (bias_deltas <= 0).all():
        return "WEEK3_OBSERVATIONAL_DIRECTIONALLY_SUPPORTIVE"
    if (deltas > 0).all():
        return "WEEK3_OBSERVATIONAL_ADVERSE"
    return "WEEK3_OBSERVATIONAL_MIXED"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lock", type=Path, required=True)
    ap.add_argument("--team-audit", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    lock = pd.read_csv(args.lock)
    team_audit = pd.read_csv(args.team_audit)

    if len(lock) != 10:
        raise RuntimeError(f"frozen projection lock expected 10 rows, found={len(lock)}")
    if set(lock["team"].astype(str)) != {"DEN", "PIT"}:
        raise RuntimeError(f"unexpected frozen teams: {sorted(set(lock['team'].astype(str)))}")
    if int(pd.to_numeric(lock["direct_transfer_recipient"], errors="coerce").eq(1).groupby(lock["market"]).sum().min()) != 4:
        raise RuntimeError("frozen lock no longer has four direct recipients per market")
    if int(lock["simulation_iterations"].drop_duplicates().iloc[0]) != 25000:
        raise RuntimeError("simulation iteration contract drift")
    if int(lock["simulation_seed"].drop_duplicates().iloc[0]) != 42:
        raise RuntimeError("simulation seed contract drift")

    # Mechanical invariants preserved from the lock.
    for _, q in lock.loc[lock["market"].eq("rush_att")].groupby(["event_id", "team"]):
        if not np.isclose(float(q["baseline_residual_probability"].iloc[0]), 0.05, atol=1e-12):
            raise RuntimeError("baseline residual probability drift")
        if not np.isclose(float(q["candidate_residual_probability"].iloc[0]), 0.05, atol=1e-12):
            raise RuntimeError("candidate residual probability drift")
    if len(team_audit) != 2:
        raise RuntimeError("team allocation audit expected two rows")

    detail = _attach_actuals(lock)
    summary = _summaries(detail)
    rooms = _team_room(detail)
    disposition = _classify(summary)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    detail.to_csv(args.out_dir / "rb_vacancy_week3_postgame_detail.csv", index=False)
    summary.to_csv(args.out_dir / "rb_vacancy_week3_postgame_summary.csv", index=False)
    rooms.to_csv(args.out_dir / "rb_vacancy_week3_team_room_actuals.csv", index=False)

    result = {
        "status": "RB_VACANCY_WEEK3_POSTGAME_GRADED",
        "disposition": disposition,
        "season": SEASON,
        "week": WEEK,
        "locked_players": int(detail[["team", "player_clean_key"]].drop_duplicates().shape[0]),
        "projection_rows": int(len(detail)),
        "direct_recipients": int(
            detail.loc[pd.to_numeric(detail["direct_transfer_recipient"], errors="coerce").eq(1),
                       ["team", "player_clean_key"]].drop_duplicates().shape[0]
        ),
        "sportsbook_inputs_used": 0,
        "candidate_variants_scored": 1,
        "production_changed": False,
        "summary": summary.to_dict(orient="records"),
        "team_room": rooms.to_dict(orient="records"),
    }
    (args.out_dir / "rb_vacancy_week3_postgame_result.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"DISPOSITION={disposition}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
