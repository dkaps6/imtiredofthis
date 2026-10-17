#!/usr/bin/env python3
"""PRISM 2026 Week-5 strict-prior named-player replay (forensic QA only).

Uses the already-preserved Weeks 1-4 player-output decomposition as the only
history source and the preserved Week-5 full football board for scoring.

This is NOT a prospective lock and cannot promote a production rule. Its purpose
is to answer a narrow player-first QA question: does the same named player's
same-market projection difficulty over Weeks 1-4 carry into Week 5?
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

VERSION = "PRISM_2026_WEEK5_STRICT_PRIOR_PLAYER_REPLAY_V1"
SEASON = 2026
PRIOR_WEEKS = [1, 2, 3, 4]
TARGET_WEEK = 5
ALLOWED_MARKETS = {"pass_yards", "rush_yards", "rec_yards", "receptions"}


def _read(path: Path) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def build(history: pd.DataFrame, target: pd.DataFrame):
    history_required = {
        "season", "week", "player_clean_key", "market", "baseline_error",
        "grading_identity_valid", "component_decomposition_eligible",
    }
    target_required = {
        "season", "week", "event_id", "player_clean_key", "player", "team",
        "opponent", "position", "market", "model_proj", "actual",
        "football_settlement_status",
    }
    if history_required - set(history.columns):
        raise RuntimeError(
            f"history missing {sorted(history_required-set(history.columns))}"
        )
    if target_required - set(target.columns):
        raise RuntimeError(
            f"target missing {sorted(target_required-set(target.columns))}"
        )

    h = history.copy()
    t = target.copy()
    h["season"] = pd.to_numeric(h["season"], errors="coerce")
    h["week"] = pd.to_numeric(h["week"], errors="coerce")
    t["season"] = pd.to_numeric(t["season"], errors="coerce")
    t["week"] = pd.to_numeric(t["week"], errors="coerce")

    h = h.loc[h["season"].eq(SEASON)].copy()
    t = t.loc[t["season"].eq(SEASON) & t["week"].eq(TARGET_WEEK)].copy()

    bad = sorted(
        h.loc[~h["week"].isin(PRIOR_WEEKS), "week"]
        .dropna().astype(int).unique().tolist()
    )
    if bad:
        raise RuntimeError(f"history contains non-prior weeks: {bad}")

    h = h.loc[
        h["grading_identity_valid"].eq(True)
        & h["component_decomposition_eligible"].eq(True)
        & h["market"].isin(ALLOWED_MARKETS)
    ].copy()
    t = t.loc[
        t["football_settlement_status"].eq("SETTLED")
        & t["market"].isin(ALLOWED_MARKETS)
    ].copy()

    h["abs_error"] = pd.to_numeric(h["baseline_error"], errors="coerce").abs()
    t["w5_abs_error"] = (
        pd.to_numeric(t["model_proj"], errors="coerce")
        - pd.to_numeric(t["actual"], errors="coerce")
    ).abs()
    if h["abs_error"].isna().any() or t["w5_abs_error"].isna().any():
        raise RuntimeError("nonfinite error")

    agg = (
        h.groupby(["player_clean_key", "market"], as_index=False)
        .agg(
            prior_games=("week", "nunique"),
            prior_min_week=("week", "min"),
            prior_max_week=("week", "max"),
            prior_mae=("abs_error", "mean"),
        )
    )
    agg = agg.loc[
        agg["prior_games"].eq(4)
        & agg["prior_min_week"].eq(1)
        & agg["prior_max_week"].eq(4)
    ].copy()

    out = t.merge(
        agg,
        on=["player_clean_key", "market"],
        how="inner",
        validate="many_to_one",
    )
    if out.empty:
        raise RuntimeError("zero eligible Week-5 rows")

    out["prior_fragility_pct"] = out.groupby("market")["prior_mae"].rank(
        method="average", pct=True
    )
    out["w5_error_pct"] = out.groupby("market")["w5_abs_error"].rank(
        method="average", pct=True
    )
    out["trust_bucket"] = "MID"
    out.loc[out["prior_fragility_pct"].le(0.25), "trust_bucket"] = "HIGH_TRUST"
    out.loc[out["prior_fragility_pct"].ge(0.75), "trust_bucket"] = "LOW_TRUST"

    rows = []
    for market, g in out.groupby("market", sort=True):
        high = g.loc[g["trust_bucket"].eq("HIGH_TRUST")]
        low = g.loc[g["trust_bucket"].eq("LOW_TRUST")]
        rows.append(
            {
                "market": market,
                "rows": len(g),
                "players": g["player_clean_key"].nunique(),
                "spearman_prior_mae_vs_w5_abs_error": g["prior_mae"].corr(
                    g["w5_abs_error"], method="spearman"
                ),
                "high_trust_n": len(high),
                "high_trust_w5_mae": high["w5_abs_error"].mean(),
                "low_trust_n": len(low),
                "low_trust_w5_mae": low["w5_abs_error"].mean(),
            }
        )
    market_summary = pd.DataFrame(rows)

    result = {
        "version": VERSION,
        "status": "FORENSIC_STRICT_PRIOR_REPLAY_COMPLETE__NO_PROMOTION",
        "season": SEASON,
        "prior_weeks": PRIOR_WEEKS,
        "target_week": TARGET_WEEK,
        "eligible_rows": int(len(out)),
        "unique_players": int(out["player_clean_key"].nunique()),
        "distinct_games": int(out["event_id"].nunique()),
        "pooled_within_market_rank_spearman": float(
            out["prior_fragility_pct"].corr(out["w5_error_pct"], method="spearman")
        ),
        "high_trust_rows": int(out["trust_bucket"].eq("HIGH_TRUST").sum()),
        "low_trust_rows": int(out["trust_bucket"].eq("LOW_TRUST").sum()),
        "high_trust_mean_w5_error_pct": float(
            out.loc[out["trust_bucket"].eq("HIGH_TRUST"), "w5_error_pct"].mean()
        ),
        "low_trust_mean_w5_error_pct": float(
            out.loc[out["trust_bucket"].eq("LOW_TRUST"), "w5_error_pct"].mean()
        ),
        "sportsbook_used_as_predictor": False,
        "models_fit": 0,
        "thresholds_fit": 0,
        "production_changed": False,
        "interpretation_boundary": (
            "retrospective strict-prior QA only; Week-5 outcomes were already "
            "known when this replay was designed, so no new production/research "
            "promotion may be inferred from this result alone"
        ),
    }

    payload = out.sort_values(["market", "player_clean_key"]).to_csv(
        index=False, float_format="%.12f", lineterminator="\n"
    ).encode()
    result["rows_sha256"] = "sha256:" + hashlib.sha256(payload).hexdigest()
    return out, market_summary, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--history", type=Path, required=True)
    ap.add_argument("--target", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, summary, result = build(_read(args.history), _read(args.target))
    rows.to_csv(
        args.out_dir / "prism_week5_strict_prior_player_replay_rows.csv",
        index=False,
        float_format="%.12f",
    )
    summary.to_csv(
        args.out_dir / "prism_week5_strict_prior_player_replay_market_summary.csv",
        index=False,
        float_format="%.12f",
    )
    (
        args.out_dir / "prism_week5_strict_prior_player_replay_result.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
