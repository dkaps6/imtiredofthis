#!/usr/bin/env python3
"""Player Active-Role Participation V1 — frozen source audit.

Builds a historical pregame skill-player universe from weekly roster authority,
removes exact official game-day inactives, and audits whether PFR offensive snap
data can support a later player-specific participation-probability experiment.

No predictive model is fit and no target-game result is used as a predictor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.backtest import audit_qb_40s_information_frontier as m76
from scripts.backtest import audit_qb_official_inactive_availability as m78
from scripts.backtest import test_qb_exact_personnel_discontinuity as m77

VERSION = "PLAYER_ACTIVE_ROLE_PARTICIPATION_V1_SOURCE_AUDIT"
M78_SHA = "d39aaf0feea101f3e0d2721ebd4118ef33fb1a4d3c76670e2a4f17734e37b609"
SEASONS = (2024, 2025)
FAMILY_POS = {
    "QB": {"QB"},
    "RB": {"RB", "FB", "HB"},
    "WR": {"WR", "LWR", "RWR", "SWR"},
    "TE": {"TE"},
}
SUPPORT = {
    "QB": {"rows": 450, "players": 45, "prior4": 30},
    "RB": {"rows": 1500, "players": 100, "prior4": 80},
    "WR": {"rows": 2000, "players": 180, "prior4": 140},
    "TE": {"rows": 1000, "players": 100, "prior4": 75},
}
READY = {p: f"{p}_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY" for p in FAMILY_POS}
BLOCKED = {p: f"{p}_ACTIVE_ROLE_PARTICIPATION_SOURCE_BLOCKED" for p in FAMILY_POS}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def family(pos: object) -> str:
    p = str(pos or "").upper().strip().replace("-", "").replace(" ", "")
    for fam, vals in FAMILY_POS.items():
        if p in vals:
            return fam
    return ""


def parse_inactives(path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    if sha256(path) != M78_SHA:
        raise RuntimeError("frozen M78 inactive snapshot SHA drift")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    need = {"season", "week", "team", "inactive_tokens", "inactive_count"}
    if need - set(x.columns):
        raise RuntimeError(f"M78 snapshot missing {sorted(need-set(x.columns))}")
    if len(x) != 1088:
        raise RuntimeError(f"M78 team-week row drift {len(x)} != 1088")
    if x.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("M78 duplicate team-week")
    x["season"] = num(x["season"]).astype(int)
    x["week"] = num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)

    rows = []
    for r in x.itertuples(index=False):
        for tok in str(r.inactive_tokens or "").split("|"):
            if not tok.strip():
                continue
            parts = tok.rsplit(":", 2)
            if len(parts) != 3:
                continue
            name_key, pos, matched = parts
            rows.append({
                "season": int(r.season),
                "week": int(r.week),
                "team": str(r.team),
                "inactive_name_key": str(name_key),
                "listed_position": str(pos).upper(),
                "family": family(pos),
                "roster_identity_match": int(str(matched).strip() == "1"),
            })
    d = pd.DataFrame(rows)
    if d.empty:
        raise RuntimeError("M78 inactive token parse empty")
    return x, d


def normalize_roster(raw: pd.DataFrame, season: int) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    wc = first_col(x, ["week"])
    tc = first_col(x, ["team", "club_code"])
    ic = first_col(x, ["gsis_id", "player_id"])
    pc = first_col(x, ["position", "depth_chart_position", "ngs_position", "pos"])
    sc = first_col(x, ["status", "status_description_abbr"])
    nc = first_col(x, ["full_name", "player_name", "display_name", "football_name"])
    if not all([wc, tc, pc, nc]):
        raise RuntimeError(f"weekly roster schema incomplete for {season}")
    if ic is None:
        x["_player_id"] = ""
        ic = "_player_id"
    if sc is None:
        x["_status"] = ""
        sc = "_status"

    out = pd.DataFrame({
        "season": int(season),
        "week": num(x[wc]),
        "team": x[tc].map(canon_team),
        "player_id": x[ic].astype("string").fillna("").str.strip(),
        "position": x[pc].astype("string").fillna("").str.upper().str.strip(),
        "status": x[sc].astype("string").fillna("").str.upper().str.strip(),
        "player_name_key": x[nc].astype("string").fillna("").str.strip().map(m78.norm_name),
    })
    out["family"] = out["position"].map(family)
    out = out.loc[
        out["week"].notna()
        & out["team"].astype(str).ne("")
        & out["player_name_key"].ne("")
        & out["family"].isin(FAMILY_POS)
    ].copy()
    out["week"] = out["week"].astype(int)

    # The frozen contract requires ACT where a usable status field exists.
    nonblank = out["status"].ne("")
    if nonblank.any():
        out = out.loc[out["status"].eq("ACT")].copy()

    return out.drop_duplicates(
        ["season", "week", "team", "player_name_key"], keep="first"
    ).reset_index(drop=True)


def load_rosters(source_meta: list[dict]) -> pd.DataFrame:
    parts = []
    for season in SEASONS:
        raw = m78.load_weekly_rosters(season, source_meta)
        q = normalize_roster(raw, season)
        if q.empty:
            raise RuntimeError(f"normalized weekly roster empty for {season}")
        parts.append(q)
    return pd.concat(parts, ignore_index=True)


def load_snaps(source_meta: list[dict]) -> pd.DataFrame:
    as_of = datetime.now(timezone.utc).isoformat()
    parts = []
    for season in SEASONS:
        raw = m76.download_table(
            "pfr_snap_counts",
            season,
            m76.release_urls("snaps", season),
            source_meta,
            as_of,
        )
        if raw.empty:
            raise RuntimeError(f"PFR snap source empty for {season}")
        parts.append(raw)
    raw = pd.concat(parts, ignore_index=True)
    raw, _, _ = m76.build_id_bridge(raw, pd.DataFrame(), source_meta, as_of)
    q = m77.prepare_snap_history(raw)
    if q.empty:
        raise RuntimeError("prepared PFR snap history empty")
    q["player_id"] = q["player_id"].astype("string").fillna("").str.strip()
    q["off_pct"] = num(q["off_pct"])
    q = q.loc[q["season"].isin(SEASONS) & q["week"].between(1, 18)].copy()
    return q.drop_duplicates(["season", "week", "player_id"], keep="first").reset_index(drop=True)


def inactive_name_map(inactive_rows: pd.DataFrame) -> dict[tuple[int,int,str], set[str]]:
    return {
        (int(s), int(w), str(t)): set(g["inactive_name_key"].astype(str))
        for (s, w, t), g in inactive_rows.groupby(["season", "week", "team"], sort=False)
    }


def build_universe(
    roster: pd.DataFrame,
    inactive_rows: pd.DataFrame,
    teamweeks: pd.DataFrame,
) -> pd.DataFrame:
    # Weekly roster files include bye/non-game weeks.  The player-game universe
    # must be restricted to the frozen official M78 team-game authority before
    # any snap coverage or strict-prior game counts are computed.
    game_keys = teamweeks[["season", "week", "team"]].drop_duplicates()
    x = roster.merge(
        game_keys,
        on=["season", "week", "team"],
        how="inner",
        validate="many_to_one",
    )
    imap = inactive_name_map(inactive_rows)
    x["known_official_inactive"] = [
        int(str(r.player_name_key) in imap.get((int(r.season), int(r.week), str(r.team)), set()))
        for r in x.itertuples(index=False)
    ]
    x["stable_roster_identity"] = x["player_id"].astype(str).str.strip().ne("").astype(int)
    return x.loc[x["known_official_inactive"].eq(0)].copy().reset_index(drop=True)


def attach_snap_source(universe: pd.DataFrame, snaps: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    positive = snaps.loc[num(snaps["off_pct"]).fillna(0).gt(0)].copy()
    pos = positive[["season", "week", "player_id", "off_pct"]].copy()

    # Map positive snap rows through the weekly-roster universe. Team is obtained
    # from the exact weekly roster, not guessed from the snap table.
    roster_key = universe[
        ["season", "week", "team", "player_id", "family"]
    ].loc[universe["player_id"].ne("")].drop_duplicates(
        ["season", "week", "player_id"], keep="first"
    )
    mapped_positive = positive.merge(
        roster_key,
        on=["season", "week", "player_id"],
        how="left",
        validate="one_to_one",
    )

    valid_teamweeks = set(
        tuple(r)
        for r in mapped_positive.loc[
            mapped_positive["team"].notna(),
            ["season", "week", "team"],
        ].drop_duplicates().itertuples(index=False, name=None)
    )

    x = universe.merge(
        pos,
        on=["season", "week", "player_id"],
        how="left",
        validate="one_to_one",
    )
    x["teamweek_snap_source_valid"] = [
        int((int(r.season), int(r.week), str(r.team)) in valid_teamweeks)
        for r in x.itertuples(index=False)
    ]
    x["off_pct"] = num(x["off_pct"])
    x["positive_offensive_participation"] = (
        x["teamweek_snap_source_valid"].eq(1)
        & x["off_pct"].fillna(0).gt(0)
    ).astype(int)
    x["valid_zero_snap"] = (
        x["teamweek_snap_source_valid"].eq(1)
        & x["off_pct"].isna()
    ).astype(int)

    return x, mapped_positive


def attach_prior_universe_support(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["ordinal"] = out["season"].astype(int) * 100 + out["week"].astype(int)
    groups = {
        (str(pid), str(team)): g.sort_values("ordinal")
        for (pid, team), g in out.loc[out["player_id"].ne("")].groupby(
            ["player_id", "team"], sort=False
        )
    }
    prior_count = []
    chronology = []
    for r in out.itertuples(index=False):
        g = groups.get((str(r.player_id), str(r.team)))
        if g is None:
            prior_count.append(0)
            chronology.append(0)
            continue
        q = g.loc[g["ordinal"].lt(int(r.ordinal))]
        prior_count.append(int(len(q)))
        chronology.append(int((q["ordinal"] >= int(r.ordinal)).sum()))
    out["strict_prior_same_team_universe_games"] = prior_count
    out["chronology_violations"] = chronology
    return out


def metrics(
    teamweeks: pd.DataFrame,
    inactive_rows: pd.DataFrame,
    universe: pd.DataFrame,
    mapped_positive: pd.DataFrame,
    season: int,
    fam: str,
) -> dict:
    tw = teamweeks.loc[teamweeks["season"].eq(season)]
    uq = universe.loc[
        universe["season"].eq(season) & universe["family"].eq(fam)
    ].copy()
    iq = inactive_rows.loc[
        inactive_rows["season"].eq(season) & inactive_rows["family"].eq(fam)
    ].copy()
    mp = mapped_positive.loc[
        mapped_positive["season"].eq(season)
        & mapped_positive["family"].eq(fam)
    ].copy()

    roster_teamweeks = int(
        uq[["season", "week", "team"]].drop_duplicates().shape[0]
    )
    target_teamweeks = int(
        tw[["season", "week", "team"]].drop_duplicates().shape[0]
    )
    snap_teamweeks = int(
        uq.loc[uq["teamweek_snap_source_valid"].eq(1), ["season", "week", "team"]]
        .drop_duplicates().shape[0]
    )

    prior4_players = int(
        uq.loc[
            uq["strict_prior_same_team_universe_games"].ge(4)
            & uq["stable_roster_identity"].eq(1),
            "player_id",
        ].nunique()
    )
    prior8_players = int(
        uq.loc[
            uq["strict_prior_same_team_universe_games"].ge(8)
            & uq["stable_roster_identity"].eq(1),
            "player_id",
        ].nunique()
    )

    return {
        "season": season,
        "family": fam,
        "official_teamweeks": target_teamweeks,
        "roster_teamweeks": roster_teamweeks,
        "roster_teamweek_coverage": roster_teamweeks / max(target_teamweeks, 1),
        "pregame_universe_rows": int(len(uq)),
        "distinct_players": int(uq.loc[uq["player_id"].ne(""), "player_id"].nunique()),
        "stable_roster_identity_rate": float(uq["stable_roster_identity"].mean()) if len(uq) else 0.0,
        "official_inactive_rows": int(len(iq)),
        "inactive_roster_identity_bridge": float(iq["roster_identity_match"].mean()) if len(iq) else 1.0,
        "snap_teamweeks": snap_teamweeks,
        "snap_teamweek_coverage": snap_teamweeks / max(roster_teamweeks, 1),
        "positive_snap_rows": int(uq["positive_offensive_participation"].sum()),
        "valid_zero_snap_rows": int(uq["valid_zero_snap"].sum()),
        "positive_participation_rate": float(uq["positive_offensive_participation"].mean()) if len(uq) else 0.0,
        "positive_snap_roster_bridge": float(mp["team"].notna().mean()) if len(mp) else 0.0,
        "players_with_prior4_same_team_universe_games": prior4_players,
        "players_with_prior8_same_team_universe_games": prior8_players,
        "chronology_violations": int(uq["chronology_violations"].sum()),
    }


def gates(m: dict) -> dict:
    req = SUPPORT[m["family"]]
    return {
        "official_teamweeks_exact_544": m["official_teamweeks"] == 544,
        "roster_teamweek_coverage_ge_99": m["roster_teamweek_coverage"] >= 0.99,
        "stable_roster_identity_ge_99": m["stable_roster_identity_rate"] >= 0.99,
        "inactive_roster_identity_bridge_ge_95": m["inactive_roster_identity_bridge"] >= 0.95,
        "snap_teamweek_coverage_ge_99": m["snap_teamweek_coverage"] >= 0.99,
        "positive_snap_roster_bridge_ge_98": m["positive_snap_roster_bridge"] >= 0.98,
        "pregame_rows_support": m["pregame_universe_rows"] >= req["rows"],
        "distinct_players_support": m["distinct_players"] >= req["players"],
        "prior4_players_support": m["players_with_prior4_same_team_universe_games"] >= req["prior4"],
        "chronology_violations_zero": m["chronology_violations"] == 0,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inactives", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    try:
        teamweeks, inactive_rows = parse_inactives(args.inactives)
        source_meta: list[dict] = []
        roster = load_rosters(source_meta)
        universe = build_universe(roster, inactive_rows, teamweeks)
        snaps = load_snaps(source_meta)
        universe, mapped_positive = attach_snap_source(universe, snaps)
        universe = attach_prior_universe_support(universe)

        rows = []
        pos_disp = {}
        for fam in FAMILY_POS:
            fam_pass = True
            for season in SEASONS:
                m = metrics(
                    teamweeks, inactive_rows, universe, mapped_positive, season, fam
                )
                g = gates(m)
                rows.append({**m, **{f"gate_{k}": bool(v) for k, v in g.items()}})
                fam_pass = fam_pass and all(g.values())
            pos_disp[fam] = READY[fam] if fam_pass else BLOCKED[fam]

        nready = sum(v.endswith("_SOURCE_READY") for v in pos_disp.values())
        if nready == 4:
            overall = "PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_ALL"
        elif nready:
            overall = "PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_READY_PARTIAL"
        else:
            overall = "PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_BLOCKED"

        result = {
            "version": VERSION,
            "disposition": overall,
            "position_dispositions": pos_disp,
            "m78_sha256": M78_SHA,
            "m78_teamweek_rows": int(len(teamweeks)),
            "metric_rows": rows,
            "target_game_snap_result_used_as_predictor": 0,
            "predictive_models_fit": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "projection_values_changed": 0,
            "production_changed": False,
        }

        pd.DataFrame(source_meta).to_csv(
            args.out_dir / "player_active_role_source_snapshots.csv", index=False
        )
        inactive_rows.to_csv(
            args.out_dir / "player_active_role_inactive_rows.csv", index=False
        )
        universe.to_csv(
            args.out_dir / "player_active_role_pregame_universe.csv", index=False
        )
        pd.DataFrame(rows).to_csv(
            args.out_dir / "player_active_role_source_metrics.csv", index=False
        )
    except Exception as exc:
        result = {
            "version": VERSION,
            "disposition": "PLAYER_ACTIVE_ROLE_PARTICIPATION_SOURCE_INTEGRITY_FAILURE",
            "error": str(exc),
            "target_game_snap_result_used_as_predictor": 0,
            "predictive_models_fit": 0,
            "sportsbook_inputs_used": 0,
            "oddsapi_calls": 0,
            "outcomes_2026_read": 0,
            "projection_values_changed": 0,
            "production_changed": False,
        }
        (
            args.out_dir / "player_active_role_participation_source_result.json"
        ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
        print(json.dumps(result, indent=2, sort_keys=True))
        raise

    (
        args.out_dir / "player_active_role_participation_source_result.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
