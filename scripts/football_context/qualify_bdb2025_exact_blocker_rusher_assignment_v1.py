#!/usr/bin/env python3
"""Outcome-free qualification for BDB2025 exact blocker-rusher assignment data.

Purpose: decide whether the public competition slice is rich enough for a separately
frozen, explicitly nondeployable value-of-information experiment. This audit never
reads pressure outcome values; it reads assignment identities only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ASSIGNMENT_COLS = [
    "gameId",
    "playId",
    "nflId",
    "blockedPlayerNFLId1",
    "blockedPlayerNFLId2",
    "blockedPlayerNFLId3",
]
OUTCOME_SCHEMA_COLS = [
    "pressureAllowedAsBlocker",
    "timeToPressureAllowedAsBlocker",
]
MIN_STABLE_ID_COVERAGE = 0.99
MIN_ASSIGNMENT_EDGES = 1000
MIN_WEEKS = 8
MIN_PRIOR10_EDGES = 500


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _norm_id(value: object) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>"}:
        return ""
    try:
        v = float(text)
        if np.isfinite(v) and v.is_integer():
            return str(int(v))
    except (TypeError, ValueError):
        pass
    return text


def _find_col(df: pd.DataFrame, names: list[str]) -> str | None:
    lower = {str(c).lower(): c for c in df.columns}
    for name in names:
        if name.lower() in lower:
            return lower[name.lower()]
    return None


def build_identity_bridge(ids: set[str], players: pd.DataFrame) -> tuple[dict[str, str], dict[str, object]]:
    p = players.copy()
    nfl_col = _find_col(p, ["nfl_id", "nfl_player_id", "nflid"])
    gsis_col = _find_col(p, ["gsis_id", "player_id", "player_gsis_id"])
    if not nfl_col or not gsis_col:
        raise RuntimeError(f"nflverse player crosswalk missing direct nfl/gsis IDs: {sorted(p.columns)}")

    q = p[[nfl_col, gsis_col]].copy()
    q["nfl_id_norm"] = q[nfl_col].map(_norm_id)
    q["gsis_norm"] = q[gsis_col].map(_norm_id)
    q = q.loc[q["nfl_id_norm"].ne("") & q["gsis_norm"].ne("")].drop_duplicates()

    amb = q.groupby("nfl_id_norm")["gsis_norm"].nunique()
    ambiguous = set(amb[amb.gt(1)].index.astype(str))
    safe = q.loc[~q["nfl_id_norm"].isin(ambiguous)].drop_duplicates("nfl_id_norm")
    mapping = dict(zip(safe["nfl_id_norm"].astype(str), safe["gsis_norm"].astype(str)))

    mapped = sum(i in mapping for i in ids)
    coverage = mapped / len(ids) if ids else 0.0
    return mapping, {
        "unique_assignment_ids": len(ids),
        "mapped_assignment_ids": mapped,
        "stable_id_coverage": coverage,
        "ambiguous_nfl_id_count": len(ambiguous & ids),
        "name_fallback_used": False,
    }


def explode_edges(player_play: pd.DataFrame, games: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, object]]:
    g = games[["gameId", "season", "week"]].drop_duplicates()
    if g["gameId"].duplicated().any():
        raise RuntimeError("games.csv contains duplicate gameId")
    pp = player_play.merge(g, on="gameId", how="left", validate="many_to_one")
    missing_game = int(pp["week"].isna().sum())
    rows: list[dict[str, object]] = []
    for col in ["blockedPlayerNFLId1", "blockedPlayerNFLId2", "blockedPlayerNFLId3"]:
        z = pp[["season", "week", "gameId", "playId", "nflId", col]].copy()
        z["blocker_nfl_id"] = z["nflId"].map(_norm_id)
        z["rusher_nfl_id"] = z[col].map(_norm_id)
        z = z.loc[z["blocker_nfl_id"].ne("") & z["rusher_nfl_id"].ne("")].copy()
        z["assignment_slot"] = col[-1]
        rows.extend(
            z[
                ["season", "week", "gameId", "playId", "blocker_nfl_id", "rusher_nfl_id", "assignment_slot"]
            ].to_dict("records")
        )
    edges = pd.DataFrame(rows)
    if edges.empty:
        return edges, {"missing_game_week_rows": missing_game, "duplicate_exact_edges": 0}
    duplicate_exact = int(
        edges.duplicated(
            ["gameId", "playId", "blocker_nfl_id", "rusher_nfl_id"], keep=False
        ).sum()
    )
    edges = edges.drop_duplicates(
        ["gameId", "playId", "blocker_nfl_id", "rusher_nfl_id"]
    ).reset_index(drop=True)
    return edges, {
        "missing_game_week_rows": missing_game,
        "duplicate_exact_edges": duplicate_exact,
    }


def add_strict_prior_support(edges: pd.DataFrame) -> pd.DataFrame:
    if edges.empty:
        return edges.copy()
    out: list[pd.DataFrame] = []
    prior = edges.iloc[0:0].copy()
    for (season, week), current in edges.groupby(["season", "week"], sort=True):
        cur = current.copy()
        hist = prior.loc[
            (pd.to_numeric(prior["season"], errors="coerce") < int(season))
            | (
                pd.to_numeric(prior["season"], errors="coerce").eq(int(season))
                & pd.to_numeric(prior["week"], errors="coerce").lt(int(week))
            )
        ]
        blocker_counts = hist.groupby("blocker_nfl_id").size()
        rusher_counts = hist.groupby("rusher_nfl_id").size()
        pair_counts = hist.groupby(["blocker_nfl_id", "rusher_nfl_id"]).size()
        cur["prior_blocker_assignment_edges"] = cur["blocker_nfl_id"].map(blocker_counts).fillna(0).astype(int)
        cur["prior_rusher_assignment_edges"] = cur["rusher_nfl_id"].map(rusher_counts).fillna(0).astype(int)
        cur["prior_pair_assignment_edges"] = [
            int(pair_counts.get((b, r), 0))
            for b, r in zip(cur["blocker_nfl_id"], cur["rusher_nfl_id"])
        ]
        out.append(cur)
        prior = pd.concat([prior, current], ignore_index=True)
    return pd.concat(out, ignore_index=True)


def run(corpus_dir: Path, players_path: Path, out_dir: Path) -> dict[str, object]:
    games_path = next(corpus_dir.rglob("games.csv"))
    player_play_path = next(corpus_dir.rglob("player_play.csv"))
    header = pd.read_csv(player_play_path, nrows=0)
    missing_assign = [c for c in ASSIGNMENT_COLS if c not in header.columns]
    if missing_assign:
        raise RuntimeError(f"BDB2025 player_play missing assignment columns: {missing_assign}")

    outcomes_present = {c: c in header.columns for c in OUTCOME_SCHEMA_COLS}
    # Deliberately do not load outcome columns.
    pp = pd.read_csv(player_play_path, usecols=ASSIGNMENT_COLS, low_memory=False)
    games = pd.read_csv(games_path, usecols=["gameId", "season", "week"])
    players = pd.read_csv(players_path, low_memory=False)

    edges, integrity = explode_edges(pp, games)
    supported = add_strict_prior_support(edges)

    ids = set(supported.get("blocker_nfl_id", pd.Series(dtype=str)).astype(str))
    ids |= set(supported.get("rusher_nfl_id", pd.Series(dtype=str)).astype(str))
    ids.discard("")
    _, identity = build_identity_bridge(ids, players)

    weeks = sorted(
        int(x) for x in pd.to_numeric(supported.get("week"), errors="coerce").dropna().unique()
    ) if len(supported) else []
    seasons = sorted(
        int(x) for x in pd.to_numeric(supported.get("season"), errors="coerce").dropna().unique()
    ) if len(supported) else []

    if len(supported):
        target = supported.loc[pd.to_numeric(supported["week"], errors="coerce").ge(5)].copy()
        prior10 = (
            target["prior_blocker_assignment_edges"].ge(10)
            & target["prior_rusher_assignment_edges"].ge(10)
        )
        pair_prior = target["prior_pair_assignment_edges"].ge(1)
        week_rows = (
            supported.groupby(["season", "week"])
            .agg(
                assignment_edges=("rusher_nfl_id", "size"),
                blockers=("blocker_nfl_id", "nunique"),
                rushers=("rusher_nfl_id", "nunique"),
                pairs=("rusher_nfl_id", lambda s: 0),
            )
            .reset_index()
        )
        # overwrite pair counts deterministically
        pair_counts = (
            supported.groupby(["season", "week"])
            .apply(lambda g: g[["blocker_nfl_id", "rusher_nfl_id"]].drop_duplicates().shape[0])
            .rename("pairs")
            .reset_index()
        )
        week_rows = week_rows.drop(columns=["pairs"]).merge(pair_counts, on=["season", "week"], how="left")
    else:
        target = supported.copy()
        prior10 = pd.Series(dtype=bool)
        pair_prior = pd.Series(dtype=bool)
        week_rows = pd.DataFrame()

    ready = (
        len(supported) >= MIN_ASSIGNMENT_EDGES
        and len(weeks) >= MIN_WEEKS
        and identity["stable_id_coverage"] >= MIN_STABLE_ID_COVERAGE
        and integrity["missing_game_week_rows"] == 0
        and int(prior10.sum()) >= MIN_PRIOR10_EDGES
    )
    disposition = (
        "VALUE_OF_INFORMATION_LAB_READY_SOURCE_SLICE"
        if ready
        else "SOURCE_SLICE_INSUFFICIENT_FOR_FROZEN_VOI_LAB"
    )

    report = {
        "audit": "BDB2025_EXACT_BLOCKER_RUSHER_ASSIGNMENT_QUALIFICATION_V1",
        "source_file_sha256": {
            "games.csv": sha256(games_path),
            "player_play.csv": sha256(player_play_path),
        },
        "seasons": seasons,
        "weeks": weeks,
        "assignment_edges": int(len(supported)),
        "unique_blockers": int(supported["blocker_nfl_id"].nunique()) if len(supported) else 0,
        "unique_rushers": int(supported["rusher_nfl_id"].nunique()) if len(supported) else 0,
        "unique_pairs": int(
            supported[["blocker_nfl_id", "rusher_nfl_id"]].drop_duplicates().shape[0]
        ) if len(supported) else 0,
        "week5plus_edges": int(len(target)),
        "week5plus_edges_with_blocker_and_rusher_prior10": int(prior10.sum()) if len(target) else 0,
        "week5plus_blocker_and_rusher_prior10_coverage": float(prior10.mean()) if len(target) else 0.0,
        "week5plus_edges_with_prior_same_pair": int(pair_prior.sum()) if len(target) else 0,
        "week5plus_prior_same_pair_coverage": float(pair_prior.mean()) if len(target) else 0.0,
        "identity": identity,
        "integrity": integrity,
        "pressure_outcome_columns_present_in_schema": outcomes_present,
        "pressure_outcome_values_read": False,
        "target_game_assignment_is_realized_not_pregame": True,
        "deployable_pregame_source_contract": False,
        "sportsbook_read": False,
        "production_change_authorized": False,
        "issue_535_touched": False,
        "disposition": disposition,
        "next_if_ready": (
            "Freeze a nondeployable hindsight value-of-information experiment asking whether "
            "realized exact assignment exposure plus strictly-prior blocker/rusher quality "
            "materially improves target-game pressure prediction versus aggregate quality alone."
        ),
        "rb_research_note": (
            "RB remains unresolved: Week-1 specialist authorities exist, Weeks 2-18 rushing "
            "and receiving-mean gaps remain, and the separately frozen PD2 yard-difficulty "
            "MC-width lane remains an open RB research priority."
        ),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    week_rows.to_csv(out_dir / "bdb2025_blocker_rusher_assignment_support_by_week_v1.csv", index=False)
    (out_dir / "bdb2025_blocker_rusher_assignment_qualification_v1.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n"
    )
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus-dir", required=True, type=Path)
    ap.add_argument("--players", required=True, type=Path)
    ap.add_argument("--out-dir", required=True, type=Path)
    args = ap.parse_args()
    print(json.dumps(run(args.corpus_dir, args.players, args.out_dir), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
