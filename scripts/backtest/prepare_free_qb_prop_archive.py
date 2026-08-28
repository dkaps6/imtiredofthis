#!/usr/bin/env python3
"""Migration 60B: prepare free archived NFL QB passing-yard props.

Downloads the public 2024/2025 player-prop parquet files from
``gcampb41/nfl_data-`` (derived from Action Network), audits their schema, and
normalizes DraftKings/FanDuel full-game passing-yard rows into the schema used by
Migration 60's market grader.

Important scientific constraint: this source does not preserve a trustworthy
30-minutes-before-kickoff timestamp. These rows are therefore labeled as
``archived_latest_per_book`` / closing-like and must never be described as the
fixed M60 30-minute snapshot.

Migration 60B joins sportsbook rows to frozen football projections primarily by
the nflverse/GSIS ``player_id`` carried in both datasets. Name matching is only
a fallback for archive rows where that stable identifier is absent.
"""
from __future__ import annotations

import argparse
import io
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

SOURCE_TEMPLATE = (
    "https://raw.githubusercontent.com/gcampb41/nfl_data-/main/"
    "data/processed/football/nfl/player_props/{season}.parquet"
)
BOOKS = {68: "draftkings", 69: "fanduel"}
FULL_GAME_PERIODS = {
    "0", "0.0", "game", "full", "fullgame", "full_game", "full game", "event", "match", "all"
}
PASSING_YARD_TYPES = {
    "passing_yards", "player_pass_yds", "player_passing_yards", "pass_yards",
    "core_bet_type_9_passing_yards",
}


def num(v):
    return pd.to_numeric(v, errors="coerce")


def clean_key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(value or "") if ch.isalnum())


def clean_id(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return ""
    return s


def clean_team(value) -> str:
    if pd.isna(value):
        return ""
    s = str(value).strip()
    if not s or s.lower() in {"nan", "none", "<na>"}:
        return ""
    return canon_team(s)


def text(v) -> pd.Series:
    return v.astype("string").fillna("").str.strip().str.lower()


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return str(lookup[n.lower()])
    return None


def to_american(v):
    """Normalize obvious American or decimal prices to American odds."""
    try:
        x = float(v)
    except Exception:
        return np.nan
    if not np.isfinite(x) or x == 0:
        return np.nan
    if x <= -100 or x >= 100:
        return x
    if 1.0 < x < 20.0:
        return (x - 1.0) * 100.0 if x >= 2.0 else -100.0 / (x - 1.0)
    return np.nan


def download_parquet(url: str, timeout: int = 90) -> tuple[pd.DataFrame, int]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "imtiredofthis-m60b/1.1"})
    r.raise_for_status()
    return pd.read_parquet(io.BytesIO(r.content)), len(r.content)


def normalize_source(raw: pd.DataFrame, season: int) -> tuple[pd.DataFrame, dict, pd.DataFrame, pd.DataFrame]:
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    audit: dict[str, object] = {"season": int(season), "raw_rows": len(x), "raw_columns": len(x.columns)}

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
    x = x.loc[bet.isin(PASSING_YARD_TYPES)].copy()
    audit["passing_yard_rows"] = len(x)

    period_dist = pd.DataFrame(columns=["season", "period", "rows"])
    if "period" not in x.columns:
        audit["period_status"] = "missing_period_column"
        return pd.DataFrame(), audit, period_dist, pd.DataFrame()
    pnorm = text(x.period).str.replace(r"\s+", " ", regex=True)
    period_dist = pnorm.value_counts(dropna=False).rename_axis("period").rename("rows").reset_index()
    period_dist.insert(0, "season", int(season))
    full = pnorm.isin(FULL_GAME_PERIODS)
    audit["period_values"] = ";".join(sorted(set(pnorm.dropna().astype(str))))
    audit["full_game_rows"] = int(full.sum())
    if not full.any():
        audit["period_status"] = "ambiguous_no_known_full_game_period"
        return pd.DataFrame(), audit, period_dist, pd.DataFrame()
    audit["period_status"] = "known_full_game_period"
    x = x.loc[full].copy()

    x["book_id"] = num(x.book_id)
    book_dist = x.book_id.value_counts(dropna=False).rename_axis("book_id").rename("rows").reset_index()
    book_dist.insert(0, "season", int(season))
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
    audit["source_player_id_column"] = id_col or ""

    # Stable ID is authoritative. Name is retained only as a fallback identity
    # for rows where the archive has no mapped GSIS ID.
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
    keys = ["season", "week", "identity_key", "book"]
    for key, g in x.groupby(keys, dropna=False):
        lines = sorted(set(num(g.line).dropna().round(6)))
        if len(lines) != 1:
            conflicts += 1
            continue
        line = float(lines[0])
        over = g.loc[g.side_norm.eq("OVER")]
        under = g.loc[g.side_norm.eq("UNDER")]
        rows.append({
            "season": int(key[0]), "week": int(key[1]), "identity_key": str(key[2]),
            "book": str(key[3]), "line": line,
            "over_odds": float(over.price.dropna().iloc[-1]) if over.price.notna().any() else np.nan,
            "under_odds": float(under.price.dropna().iloc[-1]) if under.price.notna().any() else np.nan,
            "player": str(g.player.dropna().iloc[-1]) if g.player.notna().any() else "",
            "source_name_key": str(g.source_name_key.dropna().iloc[-1]) if g.source_name_key.notna().any() else "",
            "source_player_id": str(g.source_player_id.dropna().iloc[-1]) if g.source_player_id.notna().any() else "",
            "source_team": str(g.source_team.dropna().iloc[-1]) if g.source_team.notna().any() else "",
            "source_event_id": str(g.event_id.dropna().iloc[-1]) if "event_id" in g and g.event_id.notna().any() else "",
            "source_line_definition": "archived_latest_per_book",
            "source_dataset": "gcampb41/nfl_data- Action Network-derived archive",
        })
    out = pd.DataFrame(rows)
    audit["consolidated_book_player_rows"] = len(out)
    audit["conflicting_line_groups_dropped"] = int(conflicts)
    return out, audit, period_dist, book_dist


def attach_projection_games(props: pd.DataFrame, projections: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    p = projections.copy()
    p.columns = [str(c).strip().lower() for c in p.columns]
    for c in ("season", "week"):
        p[c] = num(p[c]).astype("Int64")
    required = {"player_clean_key", "player_id", "game_id", "team"}
    if not required.issubset(p.columns):
        raise RuntimeError(f"projection files missing GSIS join columns: {sorted(required-set(p.columns))}")
    p["player_clean_key"] = p.player_clean_key.astype("string").fillna("").str.strip()
    p["player_id"] = p.player_id.map(clean_id)
    p["proj_team"] = p.team.map(clean_team)

    stats: dict[str, int] = {
        "normalized_prop_rows": len(props),
        "source_rows_with_player_id": int(props.source_player_id.astype(str).ne("").sum()),
    }

    # Primary GSIS join.
    id_key = ["season", "week", "player_id"]
    id_map = p.loc[p.player_id.ne(""), id_key + ["game_id", "proj_team", "player_clean_key"]].drop_duplicates()
    amb_id = id_map.groupby(id_key).size().gt(1)
    if amb_id.any():
        sample = amb_id[amb_id].reset_index().head(10).to_dict(orient="records")
        raise RuntimeError(f"ambiguous projection GSIS keys: {sample}")
    id_map = id_map.drop_duplicates(id_key)

    with_id = props.loc[props.source_player_id.astype(str).ne("")].copy()
    with_id["player_id"] = with_id.source_player_id.map(clean_id)
    with_id = with_id.merge(id_map, on=id_key, how="left", validate="many_to_one")
    with_id["join_method"] = np.where(with_id.game_id.notna(), "gsis_player_id", "unmatched_gsis_player_id")
    stats["gsis_matched_rows_before_team_check"] = int(with_id.game_id.notna().sum())

    # Fallback is deliberately restricted to archive rows that do not have a
    # GSIS ID. Never override a conflicting stable ID with a fuzzy/name match.
    no_id = props.loc[props.source_player_id.astype(str).eq("")].copy()
    if not no_id.empty:
        name_key = ["season", "week", "player_clean_key"]
        name_map = p[name_key + ["game_id", "proj_team"]].drop_duplicates()
        counts = name_map.groupby(name_key).size().rename("n").reset_index()
        name_map = name_map.merge(counts, on=name_key, how="left")
        name_map = name_map.loc[name_map.n.eq(1)].drop(columns="n")
        no_id["player_clean_key"] = no_id.source_name_key.astype(str)
        no_id = no_id.merge(name_map, on=name_key, how="left", validate="many_to_one")
        no_id["join_method"] = np.where(no_id.game_id.notna(), "exact_name_fallback", "unmatched_name_fallback")
    stats["name_fallback_matched_rows_before_team_check"] = int(no_id.game_id.notna().sum()) if not no_id.empty else 0

    z = pd.concat([with_id, no_id], ignore_index=True, sort=False)
    z["matched_projection"] = z.game_id.notna()
    src = z.source_team.fillna("").astype(str).str.strip()
    proj = z.proj_team.fillna("").astype(str).str.strip()
    z["team_mismatch"] = z.matched_projection & src.ne("") & proj.ne("") & src.ne(proj)
    stats["matched_projection_rows_before_team_check"] = int(z.matched_projection.sum())
    stats["team_mismatch_rows_dropped"] = int(z.team_mismatch.sum())

    z = z.loc[z.matched_projection & ~z.team_mismatch].copy()
    stats["matched_projection_rows_after_team_check"] = len(z)

    keep = [
        "game_id", "player_clean_key", "book", "line", "over_odds", "under_odds", "player",
        "season", "week", "source_team", "source_player_id", "source_event_id", "join_method",
        "source_line_definition", "source_dataset",
    ]
    return z[[c for c in keep if c in z.columns]].copy(), stats


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--projection-file", action="append", required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--source-template", default=SOURCE_TEMPLATE)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    projections = pd.concat([pd.read_csv(Path(p)) for p in args.projection_file], ignore_index=True)
    seasons = sorted(set(num(projections.season).dropna().astype(int)))

    normalized, audits, period_parts, book_parts = [], [], [], []
    status = "complete"
    reason = "free archive normalized successfully"
    for season in seasons:
        url = args.source_template.format(season=int(season))
        try:
            raw, size = download_parquet(url)
            n, audit, periods, books = normalize_source(raw, int(season))
            audit["source_url"] = url
            audit["download_bytes"] = int(size)
            normalized.append(n)
            audits.append(audit)
            period_parts.append(periods)
            book_parts.append(books)
            if n.empty:
                status = "source_validation_blocked"
                reason = f"season {season} produced no validated full-game DK/FD passing-yard rows"
        except Exception as exc:
            audits.append({"season": int(season), "source_url": url, "error": str(exc)})
            status = "source_validation_blocked"
            reason = f"season {season} source failure: {exc}"

    props = pd.concat([x for x in normalized if not x.empty], ignore_index=True) if any(not x.empty for x in normalized) else pd.DataFrame()
    match_stats = {}
    if status == "complete" and not props.empty:
        props, match_stats = attach_projection_games(props, projections)
        if props.empty:
            status = "source_validation_blocked"
            reason = "validated archive rows did not match any stable-QB projection games"

    audit_df = pd.DataFrame(audits)
    for k, v in match_stats.items():
        audit_df[k] = v
    audit_df.to_csv(args.out_dir / "m60b_free_source_audit.csv", index=False)
    if period_parts:
        pd.concat(period_parts, ignore_index=True).to_csv(args.out_dir / "m60b_period_distribution.csv", index=False)
    if book_parts:
        pd.concat(book_parts, ignore_index=True).to_csv(args.out_dir / "m60b_book_distribution.csv", index=False)

    out_cols = [
        "game_id", "player_clean_key", "book", "line", "over_odds", "under_odds", "player",
        "season", "week", "source_team", "source_player_id", "source_event_id", "join_method",
        "source_line_definition", "source_dataset",
    ]
    if props.empty:
        pd.DataFrame(columns=out_cols).to_csv(args.out_dir / "m60b_historical_qb_pass_props.csv", index=False)
    else:
        props.to_csv(args.out_dir / "m60b_historical_qb_pass_props.csv", index=False)

    status_df = pd.DataFrame([{
        "status": status,
        "reason": reason,
        "projection_rows": len(projections),
        "normalized_book_player_rows": len(props),
        "source_line_definition": "archived_latest_per_book",
        "source_is_exact_30min_snapshot": False,
        "book_priority": "draftkings,fanduel",
        "seasons": ",".join(map(str, seasons)),
        **match_stats,
    }])
    status_df.to_csv(args.out_dir / "m60b_free_source_status.csv", index=False)

    print("=== M60B FREE SOURCE STATUS ===")
    print(status_df.to_string(index=False))
    print("\n=== M60B FREE SOURCE AUDIT ===")
    print(audit_df.to_string(index=False))
    if not props.empty:
        print("\n=== M60B NORMALIZED BOOK COVERAGE ===")
        print(props.groupby(["season", "book", "join_method"]).size().rename("rows").reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
