#!/usr/bin/env python3
"""Generalize the free Action Network archive across every graded market.

Migration 60B (``prepare_free_qb_prop_archive.py``) proved this free source
and its reconciliation fix out for QB passing yards only. This module reuses
every low-level piece of that script unchanged (download, identity
reconciliation, the GSIS-player-id-over-stale-team-label fix) and generalizes
only the bet_type -> market normalization so rushing yards, receiving yards,
receptions, and rush+rec yards can be graded the same way, for free, across
the seasons the archive actually covers (2024-2025 at present).
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from scripts.backtest.benchmark_identity_v1 import assert_benchmark_identity
from scripts.backtest.prepare_free_qb_prop_archive import (
    BOOKS,
    FULL_GAME_PERIODS,
    SOURCE_TEMPLATE,
    attach_projection_games,
    attach_projection_player_ids,
    clean_id,
    clean_key,
    clean_team,
    first_existing,
    num,
    text,
    to_american,
)

MARKET_BET_TYPES: dict[str, set[str]] = {
    "pass_yards": {
        "passing_yards", "player_pass_yds", "player_passing_yards", "pass_yards",
        "core_bet_type_9_passing_yards",
    },
    "rush_yards": {"rushing_yards", "player_rush_yds", "rush_yards"},
    "rec_yards": {"receiving_yards", "player_reception_yds", "player_receiving_yards", "rec_yards"},
    "receptions": {"receptions", "player_receptions"},
    "rush_rec_yards": {"rushing_receiving_yards", "player_rush_reception_yds", "rush_rec_yards"},
}


def download_parquet(url: str, timeout: int = 90) -> tuple[pd.DataFrame, int]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "imtiredofthis-market-archive/1.0"})
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content)), len(r.content)


def normalize_source_all_markets(raw: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict]:
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    audit: dict[str, object] = {"season": int(season), "raw_rows": len(x)}

    required = {"bet_type", "book_id", "side", "value", "odds", "week"}
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"free prop archive {season} missing required columns: {missing}")

    if "season" in x.columns:
        x = x.loc[num(x.season).eq(int(season))].copy()
    x["week"] = num(x.week)
    x = x.loc[x.week.between(1, 18, inclusive="both")].copy()
    audit["regular_season_rows"] = len(x)

    bet = text(x.bet_type)
    bet_type_to_market = {bt: market for market, bts in MARKET_BET_TYPES.items() for bt in bts}
    x["market"] = bet.map(bet_type_to_market)
    x = x.loc[x.market.notna()].copy()
    audit["gradeable_market_rows"] = len(x)
    if x.empty:
        return pd.DataFrame(), audit

    if "period" not in x.columns:
        audit["period_status"] = "missing_period_column"
        return pd.DataFrame(), audit
    pnorm = text(x.period).str.replace(r"\s+", " ", regex=True)
    full = pnorm.isin(FULL_GAME_PERIODS)
    audit["full_game_rows"] = int(full.sum())
    if not full.any():
        audit["period_status"] = "ambiguous_no_known_full_game_period"
        return pd.DataFrame(), audit
    audit["period_status"] = "known_full_game_period"
    x = x.loc[full].copy()

    x["book_id"] = num(x.book_id)
    x = x.loc[x.book_id.isin(BOOKS)].copy()
    x["book"] = x.book_id.map(BOOKS)
    audit["dk_fd_rows"] = len(x)

    side = text(x.side)
    x["side_norm"] = np.select(
        [side.isin(["over", "o"]), side.isin(["under", "u"])],
        ["OVER", "UNDER"], default="",
    )
    x = x.loc[x.side_norm.ne("")].copy()
    x["line"] = num(x.value)
    x["price"] = x.odds.map(to_american)
    x = x.loc[x.line.notna() & x.line.gt(0)].copy()
    audit["valid_side_line_rows"] = len(x)

    name_col = first_existing(x, ["join_name", "player_name", "player", "name", "full_name"])
    if name_col is None:
        raise RuntimeError(f"free prop archive {season} has no usable player-name column")
    x["player"] = x[name_col].astype("string").fillna("").str.strip()
    x["source_name_key"] = x.player.map(clean_key)

    id_col = first_existing(x, ["player_id", "gsis_id", "player_gsis_id"])
    x["source_player_id"] = x[id_col].map(clean_id) if id_col else ""
    audit["rows_with_source_player_id"] = int(x.source_player_id.ne("").sum())

    x["identity_key"] = np.where(
        x.source_player_id.ne(""),
        "id:" + x.source_player_id,
        "name:" + x.source_name_key,
    )
    x = x.loc[~x.identity_key.eq("name:")].copy()

    team_col = first_existing(x, ["team", "team_abbr", "team_abbreviation"])
    x["source_team"] = x[team_col].map(clean_team) if team_col else ""
    x["season"] = int(season)
    x["week"] = num(x.week).astype(int)

    rows = []
    conflicts = 0
    keys = ["season", "week", "market", "identity_key", "book"]
    for key, g in x.groupby(keys, dropna=False):
        lines = sorted(set(num(g.line).dropna().round(6)))
        if len(lines) != 1:
            conflicts += 1
            continue
        line = float(lines[0])
        over = g.loc[g.side_norm.eq("OVER")]
        under = g.loc[g.side_norm.eq("UNDER")]
        rows.append({
            "season": int(key[0]), "week": int(key[1]), "market": str(key[2]),
            "identity_key": str(key[3]), "book": str(key[4]), "line": line,
            "over_odds": float(over.price.dropna().iloc[-1]) if over.price.notna().any() else np.nan,
            "under_odds": float(under.price.dropna().iloc[-1]) if under.price.notna().any() else np.nan,
            "player": str(g.player.dropna().iloc[-1]) if g.player.notna().any() else "",
            "source_name_key": str(g.source_name_key.dropna().iloc[-1]) if g.source_name_key.notna().any() else "",
            "source_player_id": str(g.source_player_id.dropna().iloc[-1]) if g.source_player_id.notna().any() else "",
            "source_team": str(g.source_team.dropna().iloc[-1]) if g.source_team.notna().any() else "",
            "source_line_definition": "archived_latest_per_book",
            "source_dataset": "gcampb41/nfl_data- Action Network-derived archive",
        })
    out = pd.DataFrame(rows)
    audit["consolidated_book_player_market_rows"] = len(out)
    audit["conflicting_line_groups_dropped"] = int(conflicts)
    return out, audit


def attach_projection_games_per_market(props: pd.DataFrame, projections: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """attach_projection_games joins on (season, week, player_id); grouping by
    market first keeps that contract intact while covering every market."""
    if props.empty:
        return props.copy(), {}
    parts = []
    stats: dict[str, dict] = {}
    for market, g in props.groupby("market"):
        matched, s = attach_projection_games(g.drop(columns=["market"]), projections)
        if not matched.empty:
            matched = matched.copy()
            matched["market"] = market
        parts.append(matched)
        stats[market] = s
    out = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    return out, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-file", action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-template", default=SOURCE_TEMPLATE)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    raw_projections = pd.concat([pd.read_csv(Path(p)) for p in args.projection_file], ignore_index=True)
    raw_projections.columns = [str(c).strip().lower() for c in raw_projections.columns]
    assert_benchmark_identity(
        raw_projections,
        label="projection input to free market archive",
        require_team=True,
        require_opponent=("opponent" in raw_projections.columns),
    )

    # The projection trace has one row per (season, week, team, player_clean_key,
    # market) -- market/mc_proj/actual are irrelevant to identity reconciliation
    # and are reattached later when grading. Only a single identity row per
    # player-game is needed here.
    identity_cols = [c for c in ["season", "week", "team", "player_clean_key", "player_id", "game_id"] if c in raw_projections.columns]
    projections = raw_projections[identity_cols].drop_duplicates()
    projections, projection_id_stats = attach_projection_player_ids(projections)
    seasons = sorted(set(num(projections.season).dropna().astype(int)))

    normalized, audits = [], []
    for season in seasons:
        url = args.source_template.format(season=int(season))
        raw, size = download_parquet(url)
        n, audit = normalize_source_all_markets(raw, int(season))
        audit["source_url"] = url
        audit["download_bytes"] = int(size)
        normalized.append(n)
        audits.append(audit)

    props = pd.concat([x for x in normalized if not x.empty], ignore_index=True) if any(not x.empty for x in normalized) else pd.DataFrame()
    matched = pd.DataFrame()
    match_stats: dict = {}
    if not props.empty:
        matched, match_stats = attach_projection_games_per_market(props, projections)
        if not matched.empty:
            assert_benchmark_identity(
                matched,
                label="matched historical market props",
                require_team=False,
                require_opponent=False,
            )

    pd.DataFrame(audits).to_csv(args.out_dir / "market_archive_source_audit.csv", index=False)
    out_cols = [
        "game_id", "player_clean_key", "book", "market", "line", "over_odds", "under_odds",
        "player", "season", "week", "source_team", "source_player_id", "join_method",
        "team_mismatch", "source_line_definition", "source_dataset",
    ]
    if matched.empty:
        pd.DataFrame(columns=out_cols).to_csv(args.out_dir / "historical_market_props.csv", index=False)
    else:
        matched[[c for c in out_cols if c in matched.columns]].to_csv(
            args.out_dir / "historical_market_props.csv", index=False
        )

    print("=== MARKET ARCHIVE SOURCE AUDIT ===")
    print(pd.DataFrame(audits).to_string(index=False))
    if not matched.empty:
        print("\n=== MATCHED ROWS BY MARKET ===")
        print(matched.groupby(["season", "market"]).size().rename("rows").reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())