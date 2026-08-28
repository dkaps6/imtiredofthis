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


def text(v) -> pd.Series:
    return v.astype(str).str.strip().str.lower()


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
    # Action Network normally stores American prices. Keep those directly.
    if x <= -100 or x >= 100:
        return x
    # Defensive support for decimal prices if the archive format changes.
    if 1.0 < x < 20.0:
        return (x - 1.0) * 100.0 if x >= 2.0 else -100.0 / (x - 1.0)
    return np.nan


def download_parquet(url: str, timeout: int = 90) -> tuple[pd.DataFrame, int]:
    r = requests.get(url, timeout=timeout, headers={"User-Agent": "imtiredofthis-m60b/1.0"})
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
    period_dist = (
        pnorm.value_counts(dropna=False).rename_axis("period").rename("rows").reset_index()
    )
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
    book_dist = (
        x.book_id.value_counts(dropna=False).rename_axis("book_id").rename("rows").reset_index()
    )
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
    x["player"] = x[name_col].astype(str).str.strip()
    x["player_clean_key"] = x.player.map(clean_key)
    x = x.loc[x.player_clean_key.ne("")].copy()

    team_col = first_existing(x, ["team", "team_abbr", "team_abbreviation"])
    if team_col:
        x["source_team"] = x[team_col].astype(str).map(canon_team)
    else:
        x["source_team"] = ""

    x["season"] = int(season)
    x["week"] = num(x.week).astype(int)

    rows = []
    conflicts = 0
    keys = ["season", "week", "player_clean_key", "book"]
    for key, g in x.groupby(keys, dropna=False):
        lines = sorted(set(num(g.line).dropna().round(6)))
        if len(lines) != 1:
            conflicts += 1
            continue
        line = float(lines[0])
        over = g.loc[g.side_norm.eq("OVER")]
        under = g.loc[g.side_norm.eq("UNDER")]
        # The processed source is intended to retain the latest row per side/book.
        # If duplicates survive, keep the last archive row without inventing a timestamp.
        rows.append({
            "season": int(key[0]), "week": int(key[1]), "player_clean_key": str(key[2]),
            "book": str(key[3]), "line": line,
            "over_odds": float(over.price.dropna().iloc[-1]) if over.price.notna().any() else np.nan,
            "under_odds": float(under.price.dropna().iloc[-1]) if under.price.notna().any() else np.nan,
            "player": str(g.player.dropna().iloc[-1]) if g.player.notna().any() else "",
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
        p[c] = num(p[c])
    if "player_clean_key" not in p.columns:
        raise RuntimeError("projection files missing player_clean_key")
    if "game_id" not in p.columns:
        raise RuntimeError("projection files missing game_id")
    p["player_clean_key"] = p.player_clean_key.astype(str)
    p["proj_team"] = p.get("team", "").astype(str).map(canon_team)

    key = ["season", "week", "player_clean_key"]
    counts = p.groupby(key).size().rename("projection_key_count").reset_index()
    unique = p.merge(counts, on=key, how="left")
    unique = unique.loc[unique.projection_key_count.eq(1), key + ["game_id", "proj_team"]].drop_duplicates(key)

    z = props.merge(unique, on=key, how="left", validate="many_to_one")
    z["matched_projection"] = z.game_id.notna()
    src = z.source_team.astype(str).str.strip()
    proj = z.proj_team.astype(str).str.strip()
    z["team_mismatch"] = z.matched_projection & src.ne("") & proj.ne("") & src.ne(proj)
    stats = {
        "normalized_prop_rows": len(z),
        "matched_projection_rows_before_team_check": int(z.matched_projection.sum()),
        "team_mismatch_rows_dropped": int(z.team_mismatch.sum()),
    }
    z = z.loc[z.matched_projection & ~z.team_mismatch].copy()
    stats["matched_projection_rows_after_team_check"] = len(z)
    keep = [
        "game_id", "player_clean_key", "book", "line", "over_odds", "under_odds", "player",
        "season", "week", "source_team", "source_event_id", "source_line_definition", "source_dataset",
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
        "season", "week", "source_team", "source_event_id", "source_line_definition", "source_dataset",
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
    }])
    status_df.to_csv(args.out_dir / "m60b_free_source_status.csv", index=False)

    print("=== M60B FREE SOURCE STATUS ===")
    print(status_df.to_string(index=False))
    print("\n=== M60B FREE SOURCE AUDIT ===")
    print(audit_df.to_string(index=False))
    if not props.empty:
        print("\n=== M60B NORMALIZED BOOK COVERAGE ===")
        print(props.groupby(["season", "book"]).size().rename("rows").reset_index().to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
