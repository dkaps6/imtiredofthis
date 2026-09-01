#!/usr/bin/env python3
"""Downstream RB market benchmark: frozen M94C vs real archived DK/FD rush-yard lines.

Research audit only. No sportsbook input is fed upstream into football projections.
The source is the same Action Network-derived free archive audited in M60B.
"""
from __future__ import annotations

import argparse
import io
import math
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import requests

from scripts.utils.canonical_names import canonicalize_player_name_safe

SOURCE_URL = (
    "https://raw.githubusercontent.com/gcampb41/nfl_data-/main/"
    "data/processed/football/nfl/player_props/2025.parquet"
)
BOOKS = {68: "draftkings", 69: "fanduel"}
FULL_GAME_PERIODS = {
    "0", "0.0", "game", "full", "fullgame", "full_game", "full game", "event", "match", "all"
}
TEAM_MAP = {"OAK": "LV", "SD": "LAC", "STL": "LA", "LAR": "LA", "JAX": "JAC", "ARZ": "ARI", "WSH": "WAS"}


def num(s):
    return pd.to_numeric(s, errors="coerce")


def text(s):
    return s.astype("string").fillna("").str.strip().str.lower()


def first_existing(df: pd.DataFrame, names: Iterable[str]) -> str | None:
    lookup = {str(c).lower(): c for c in df.columns}
    for n in names:
        if n.lower() in lookup:
            return str(lookup[n.lower()])
    return None


def clean_team(v) -> str:
    if pd.isna(v):
        return ""
    s = str(v).strip().upper()
    if not s or s in {"NAN", "NONE", "<NA>"}:
        return ""
    return TEAM_MAP.get(s, s)


def clean_name(v) -> str:
    try:
        _, key = canonicalize_player_name_safe(v)
        if key:
            return str(key)
    except Exception:
        pass
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def point_metrics(actual, pred) -> dict:
    q = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if q.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan,
                "actual_mean": np.nan, "pred_mean": np.nan}
    err = q.pred - q.actual
    corr = q.actual.corr(q.pred) if len(q) >= 3 and q.actual.nunique() > 1 and q.pred.nunique() > 1 else np.nan
    return {
        "n": int(len(q)),
        "mae": float(err.abs().mean()),
        "rmse": float(math.sqrt(np.square(err).mean())),
        "bias": float(err.mean()),
        "corr": float(corr) if pd.notna(corr) else np.nan,
        "actual_mean": float(q.actual.mean()),
        "pred_mean": float(q.pred.mean()),
    }


def load_market() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    r = requests.get(SOURCE_URL, timeout=120, headers={"User-Agent": "imtiredofthis-rb-market-benchmark/1.0"})
    r.raise_for_status()
    raw = pd.read_parquet(io.BytesIO(r.content))
    raw.columns = [str(c).strip().lower() for c in raw.columns]

    bet_col = first_existing(raw, ["bet_type", "market", "market_key"])
    if not bet_col:
        raise RuntimeError("player-prop archive missing bet_type/market column")
    btxt = text(raw[bet_col])
    bet_audit = btxt.value_counts(dropna=False).rename_axis("bet_type").rename("rows").reset_index()
    bet_audit["rush_yard_candidate"] = (
        bet_audit.bet_type.astype(str).str.contains("rush", case=False, regex=False)
        & bet_audit.bet_type.astype(str).str.contains("yard", case=False, regex=False)
    ).astype(int)

    x = raw.copy()
    if "season" in x.columns:
        x = x.loc[num(x.season).eq(2025)].copy()
    if "week" not in x.columns:
        raise RuntimeError("archive missing week")
    x["week"] = num(x.week)
    x = x.loc[x.week.between(1, 18, inclusive="both")].copy()

    market = text(x[bet_col])
    # General, source-schema-tolerant definition. Attempts cannot pass because 'yard' is required.
    x = x.loc[market.str.contains("rush", regex=False) & market.str.contains("yard", regex=False)].copy()
    if x.empty:
        candidates = bet_audit.loc[bet_audit.rush_yard_candidate.eq(1)].head(30).to_dict(orient="records")
        raise RuntimeError(f"no rushing-yard market rows after filter; candidates={candidates}")

    period_status = "missing"
    if "period" in x.columns:
        p = text(x.period).str.replace(r"\s+", " ", regex=True)
        full = p.isin(FULL_GAME_PERIODS)
        period_status = "known_full_game_period" if full.any() else "no_known_full_game_period"
        if not full.any():
            raise RuntimeError(f"rushing-yard rows have no recognized full-game period; values={sorted(set(p))[:30]}")
        x = x.loc[full].copy()

    if "book_id" not in x.columns:
        raise RuntimeError("archive missing book_id")
    x["book_id"] = num(x.book_id)
    x = x.loc[x.book_id.isin(BOOKS)].copy()
    x["book"] = x.book_id.map(BOOKS)

    line_col = first_existing(x, ["value", "line", "point"])
    side_col = first_existing(x, ["side", "label", "outcome"])
    name_col = first_existing(x, ["join_name", "player_name", "player", "name", "full_name"])
    if not all([line_col, side_col, name_col]):
        raise RuntimeError(f"archive missing line/side/name columns: line={line_col} side={side_col} name={name_col}")

    x["line"] = num(x[line_col])
    side = text(x[side_col])
    x["side_norm"] = np.select(
        [side.isin(["over", "o"]), side.isin(["under", "u"])],
        ["OVER", "UNDER"], default="",
    )
    x = x.loc[x.line.notna() & x.line.gt(0) & x.side_norm.ne("")].copy()
    x["player"] = x[name_col].astype("string").fillna("").str.strip()
    x["name_key"] = x.player.map(clean_name)
    x = x.loc[x.name_key.ne("")].copy()

    team_col = first_existing(x, ["team", "team_abbr", "team_abbreviation"])
    x["source_team"] = x[team_col].map(clean_team) if team_col else ""
    x["week"] = x.week.astype(int)

    rows = []
    conflicts = 0
    one_sided = 0
    for (week, name_key, book), g in x.groupby(["week", "name_key", "book"], dropna=False):
        lines = sorted(set(num(g.line).dropna().round(6)))
        if len(lines) != 1:
            conflicts += 1
            continue
        sides = set(g.side_norm)
        if not {"OVER", "UNDER"}.issubset(sides):
            one_sided += 1
            continue
        teams = [clean_team(v) for v in g.source_team if clean_team(v)]
        team = teams[-1] if teams else ""
        rows.append({
            "season": 2025,
            "week": int(week),
            "name_key": str(name_key),
            "book": str(book),
            "line": float(lines[0]),
            "source_team": team,
            "source_player": str(g.player.dropna().iloc[-1]) if g.player.notna().any() else "",
            "source_line_definition": "archived_latest_per_book",
        })
    market_rows = pd.DataFrame(rows)
    source_audit = pd.DataFrame([{
        "source_url": SOURCE_URL,
        "download_bytes": int(len(r.content)),
        "raw_rows": int(len(raw)),
        "regular_season_rush_yard_rows_before_book_filter": int(len(x)),
        "eligible_book_player_rows": int(len(market_rows)),
        "conflicting_player_week_book_groups_dropped": int(conflicts),
        "one_sided_player_week_book_groups_dropped": int(one_sided),
        "period_status": period_status,
        "line_definition": "archived_latest_per_book_closing_like_not_fixed_timestamp",
    }])
    return market_rows, source_audit, bet_audit


def load_m94c(root: Path) -> pd.DataFrame:
    hits = list(root.rglob("m94c_2025_rb_trace.csv"))
    if len(hits) != 1:
        raise RuntimeError(f"expected one m94c_2025_rb_trace.csv under {root}; found {len(hits)}")
    x = pd.read_csv(hits[0], low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    x = x.loc[num(x.season).eq(2025)].copy()
    if "position" in x.columns:
        x = x.loc[x.position.astype(str).str.upper().isin(["RB", "FB"])].copy()
    x["week"] = num(x.week).astype(int)
    x["team"] = x.team.map(clean_team)
    x["name_key"] = x.player.map(clean_name)
    for c in ["candidate_rush_yards", "actual_rush_yards"]:
        x[c] = num(x[c])
    x = x.loc[x.name_key.ne("") & x.candidate_rush_yards.notna() & x.actual_rush_yards.notna()].copy()
    if x.duplicated(["week", "name_key"]).any():
        d = x.loc[x.duplicated(["week", "name_key"], keep=False), ["week", "name_key", "player", "team"]]
        raise RuntimeError(f"ambiguous M94C player-week name keys: {d.head(20).to_dict(orient='records')}")
    return x[["season", "week", "player", "name_key", "team", "opponent", "candidate_rush_yards", "actual_rush_yards", "actual_rush_att"]].copy()


def join_market(m94: pd.DataFrame, market: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    if market.empty:
        raise RuntimeError("no eligible market rows")
    j = market.merge(m94, on=["season", "week", "name_key"], how="left", validate="many_to_one")
    raw_matched = j.player.notna()
    team_mismatch = raw_matched & j.source_team.ne("") & j.team.ne("") & j.source_team.ne(j.team)
    mismatches = j.loc[team_mismatch, ["week", "source_player", "source_team", "player", "team", "book", "line"]].copy()
    j = j.loc[raw_matched & ~team_mismatch].copy()

    piv = j.pivot_table(index=["season", "week", "name_key", "player", "team", "opponent", "candidate_rush_yards", "actual_rush_yards", "actual_rush_att"],
                        columns="book", values="line", aggfunc="first").reset_index()
    for c in ["draftkings", "fanduel"]:
        if c not in piv.columns:
            piv[c] = np.nan
    piv["consensus_line"] = piv[["draftkings", "fanduel"]].median(axis=1, skipna=True)
    both = piv.draftkings.notna() & piv.fanduel.notna()
    piv["two_book_consensus"] = np.where(both, (piv.draftkings + piv.fanduel) / 2.0, np.nan)
    piv["market_books"] = piv[["draftkings", "fanduel"]].notna().sum(axis=1)
    piv["model_minus_market"] = piv.candidate_rush_yards - piv.consensus_line
    piv["abs_disagreement"] = piv.model_minus_market.abs()
    piv["model_abs_error"] = (piv.candidate_rush_yards - piv.actual_rush_yards).abs()
    piv["market_abs_error"] = (piv.consensus_line - piv.actual_rush_yards).abs()
    piv["winner"] = np.select(
        [piv.model_abs_error < piv.market_abs_error, piv.market_abs_error < piv.model_abs_error],
        ["MODEL", "MARKET"], default="TIE",
    )
    audit = pd.DataFrame([{
        "m94c_rows": int(len(m94)),
        "eligible_market_book_rows": int(len(market)),
        "name_week_matched_book_rows": int(raw_matched.sum()),
        "team_mismatch_book_rows_dropped": int(team_mismatch.sum()),
        "final_consensus_player_games": int(piv.consensus_line.notna().sum()),
        "two_book_player_games": int(piv.two_book_consensus.notna().sum()),
        "draftkings_player_games": int(piv.draftkings.notna().sum()),
        "fanduel_player_games": int(piv.fanduel.notna().sum()),
    }])
    return piv, pd.concat([audit, pd.DataFrame([{"mismatch_examples": mismatches.head(20).to_json(orient="records")}])], axis=1)


def summaries(z: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows = []
    for arm, col in [
        ("M94C_MODEL", "candidate_rush_yards"),
        ("VEGAS_CONSENSUS", "consensus_line"),
        ("DRAFTKINGS", "draftkings"),
        ("FANDUEL", "fanduel"),
        ("TWO_BOOK_CONSENSUS", "two_book_consensus"),
    ]:
        rows.append({"arm": arm, **point_metrics(z.actual_rush_yards, z[col])})
    summary = pd.DataFrame(rows)

    common = z.loc[z.consensus_line.notna()].copy()
    wins = common.winner.value_counts().to_dict()
    h2h = pd.DataFrame([{
        "n": int(len(common)),
        "model_closer": int(wins.get("MODEL", 0)),
        "market_closer": int(wins.get("MARKET", 0)),
        "ties": int(wins.get("TIE", 0)),
        "model_closer_rate_ex_ties": float(wins.get("MODEL", 0) / max(wins.get("MODEL", 0) + wins.get("MARKET", 0), 1)),
        "mean_market_abs_error_minus_model_abs_error": float((common.market_abs_error - common.model_abs_error).mean()),
        "median_market_abs_error_minus_model_abs_error": float((common.market_abs_error - common.model_abs_error).median()),
    }])

    bins = [-np.inf, 5, 10, 15, np.inf]
    labels = ["lt5", "5_to_lt10", "10_to_lt15", "15_plus"]
    common["disagreement_bucket"] = pd.cut(common.abs_disagreement, bins=bins, labels=labels, right=False)
    drows = []
    for bucket, q in common.groupby("disagreement_bucket", observed=False):
        if q.empty:
            continue
        w = q.winner.value_counts().to_dict()
        drows.append({
            "bucket": str(bucket), "n": int(len(q)),
            "mean_abs_disagreement": float(q.abs_disagreement.mean()),
            "model_mae": float(q.model_abs_error.mean()), "market_mae": float(q.market_abs_error.mean()),
            "model_mae_gain_vs_market": float(q.market_abs_error.mean() - q.model_abs_error.mean()),
            "model_closer": int(w.get("MODEL", 0)), "market_closer": int(w.get("MARKET", 0)), "ties": int(w.get("TIE", 0)),
            "model_closer_rate_ex_ties": float(w.get("MODEL", 0) / max(w.get("MODEL", 0) + w.get("MARKET", 0), 1)),
        })
    disagreement = pd.DataFrame(drows)

    srows = []
    for threshold in [0, 5, 10, 15]:
        for direction in ["MODEL_OVER_MARKET", "MODEL_UNDER_MARKET"]:
            if direction == "MODEL_OVER_MARKET":
                q = common.loc[common.model_minus_market >= threshold].copy() if threshold else common.loc[common.model_minus_market > 0].copy()
                correct = q.actual_rush_yards > q.consensus_line
            else:
                q = common.loc[common.model_minus_market <= -threshold].copy() if threshold else common.loc[common.model_minus_market < 0].copy()
                correct = q.actual_rush_yards < q.consensus_line
            pushes = q.actual_rush_yards.eq(q.consensus_line)
            denom = int((~pushes).sum())
            w = q.winner.value_counts().to_dict()
            srows.append({
                "threshold": threshold, "direction": direction, "n": int(len(q)), "pushes": int(pushes.sum()),
                "directional_market_side_accuracy_ex_pushes": float(correct.loc[~pushes].mean()) if denom else np.nan,
                "model_closer_rate_ex_ties": float(w.get("MODEL", 0) / max(w.get("MODEL", 0) + w.get("MARKET", 0), 1)),
                "model_mae": float(q.model_abs_error.mean()) if len(q) else np.nan,
                "market_mae": float(q.market_abs_error.mean()) if len(q) else np.nan,
            })
    signal = pd.DataFrame(srows)
    return summary, h2h, disagreement, signal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--m94c-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    market, source_audit, bet_audit = load_market()
    m94 = load_m94c(args.m94c_root)
    joined, join_audit = join_market(m94, market)
    summary, h2h, disagreement, signal = summaries(joined)

    market.to_csv(args.out_dir / "rb_market_normalized_book_lines.csv", index=False)
    source_audit.to_csv(args.out_dir / "rb_market_source_audit.csv", index=False)
    bet_audit.to_csv(args.out_dir / "rb_market_bet_type_audit.csv", index=False)
    join_audit.to_csv(args.out_dir / "rb_market_join_audit.csv", index=False)
    joined.sort_values(["week", "player"]).to_csv(args.out_dir / "rb_market_casebook.csv", index=False)
    summary.to_csv(args.out_dir / "rb_market_summary.csv", index=False)
    h2h.to_csv(args.out_dir / "rb_market_head_to_head.csv", index=False)
    disagreement.to_csv(args.out_dir / "rb_market_disagreement_buckets.csv", index=False)
    signal.to_csv(args.out_dir / "rb_market_directional_signal.csv", index=False)

    print("=== source audit ==="); print(source_audit.to_string(index=False))
    print("=== rush-yard bet types ==="); print(bet_audit.loc[bet_audit.rush_yard_candidate.eq(1)].head(40).to_string(index=False))
    print("=== join audit ==="); print(join_audit.to_string(index=False))
    print("=== summary ==="); print(summary.to_string(index=False))
    print("=== head-to-head ==="); print(h2h.to_string(index=False))
    print("=== disagreement ==="); print(disagreement.to_string(index=False))
    print("=== directional signal ==="); print(signal.to_string(index=False))


if __name__ == "__main__":
    main()
