#!/usr/bin/env python3
"""WR-R2 frozen player-level NGS tracking residual diagnostic.

Exact M38 receiving-yard predictions come from the successful WR-R1 artifact.
All NGS features use only strictly prior player games. No model is fit and no
sportsbook data is read.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

EXPECTED_M38_WR_ROWS = 12396
MIN_ELIGIBLE = 3000
MIN_COVERAGE = 0.60
MIN_SPEARMAN = 0.08
MIN_GAP = 4.0
MIN_ENRICH = 1.25
TARGET_SEASONS = list(range(2020, 2026))
SOURCE_SEASONS = list(range(2019, 2026))
HIST = 8

SIGNALS = {
    "NGS_ADOT_PRIOR8": ["avg_intended_air_yards", "avg_air_distance"],
    "NGS_YACOE_PRIOR8": ["avg_yac_above_expectation"],
    "NGS_SEPARATION_PRIOR8": ["avg_separation"],
    "NGS_IAY_SHARE_PRIOR8": ["percent_share_of_intended_air_yards"],
}


def _read(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError(f"empty input {path}")
    return x


def _one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}, found {len(hits)}")
    return hits[0]


def _key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _first(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def build_casebook(paired: pd.DataFrame) -> pd.DataFrame:
    q = paired.loc[
        paired["market"].astype(str).str.lower().eq("rec_yards")
        & paired["position"].astype(str).str.upper().eq("WR")
    ].copy()
    if len(q) != EXPECTED_M38_WR_ROWS:
        raise RuntimeError(f"WR-R1 M38 row parity drift expected={EXPECTED_M38_WR_ROWS} got={len(q)}")
    q["season"] = _num(q["season"])
    q["week"] = _num(q["week"])
    q["player_key"] = q["player_clean_key"].map(_key)
    q["actual"] = _num(q["actual_m38"])
    q["m38_proj"] = _num(q["mc_proj_m38"])
    if q[["season", "week", "actual", "m38_proj"]].isna().any().any():
        raise RuntimeError("canonical M38 casebook contains non-finite core values")
    if sorted(q["season"].astype(int).unique().tolist()) != TARGET_SEASONS:
        raise RuntimeError(f"unexpected target seasons: {sorted(q['season'].unique())}")
    q["residual_actual_minus_m38"] = q["actual"] - q["m38_proj"]
    q["error_m38_minus_actual"] = -q["residual_actual_minus_m38"]
    q["under25"] = q["residual_actual_minus_m38"].ge(25).astype(int)
    q["under50"] = q["residual_actual_minus_m38"].ge(50).astype(int)
    q["actual100"] = q["actual"].ge(100).astype(int)
    return q


def individual_profiles(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (player_key, player), g in q.groupby(["player_key", "player"], sort=True):
        e = g["error_m38_minus_actual"].astype(float)
        rows.append({
            "player_key": player_key,
            "player": player,
            "games": int(len(g)),
            "m38_mae": float(e.abs().mean()),
            "m38_bias_projection_minus_actual": float(e.mean()),
            "m38_median_abs_error": float(e.abs().median()),
            "m38_p90_abs_error": float(np.quantile(e.abs(), 0.90)),
            "m38_miss20_rate": float(e.abs().ge(20).mean()),
            "m38_miss30_rate": float(e.abs().ge(30).mean()),
            "m38_miss40_rate": float(e.abs().ge(40).mean()),
        })
    return pd.DataFrame(rows)


def load_ngs() -> tuple[pd.DataFrame, dict]:
    import nflreadpy as nfl

    raw = nfl.load_nextgen_stats(seasons=SOURCE_SEASONS, stat_type="receiving")
    if hasattr(raw, "to_pandas"):
        raw = raw.to_pandas()
    x = pd.DataFrame(raw)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if x.empty:
        raise RuntimeError("NGS receiving source returned zero rows")
    if "season_type" in x.columns:
        x = x.loc[x["season_type"].fillna("").astype(str).str.upper().isin(["REG", "REGULAR", "RS", ""])].copy()
    name_col = _first(x, ["player_display_name", "player_name", "receiver_player_name", "player"])
    if not name_col or "season" not in x.columns or "week" not in x.columns:
        raise RuntimeError(f"NGS source missing player/season/week columns: {list(x.columns)}")
    resolved = {}
    for sig, candidates in SIGNALS.items():
        col = _first(x, candidates)
        if not col:
            raise RuntimeError(f"NGS source missing frozen signal {sig}; candidates={candidates}; columns={list(x.columns)}")
        resolved[sig] = col
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    x = x.loc[x["season"].isin(SOURCE_SEASONS) & x["week"].gt(0)].copy()
    x["player_key"] = x[name_col].map(_key)
    for sig, col in resolved.items():
        x[sig] = _num(x[col])
    x = x.loc[x["player_key"].ne("")].sort_values(["player_key", "season", "week"])
    meta = {
        "raw_rows": int(len(x)),
        "name_column": name_col,
        "resolved_columns": resolved,
        "source_seasons_present": sorted(_num(x["season"]).dropna().astype(int).unique().tolist()),
        "source_players": int(x["player_key"].nunique()),
    }
    return x, meta


def attach_prior8(casebook: pd.DataFrame, ngs: pd.DataFrame) -> pd.DataFrame:
    by_player = {k: g.copy() for k, g in ngs.groupby("player_key", sort=False)}
    rows = []
    for r in casebook.itertuples(index=False):
        rec = {
            "season": int(r.season),
            "week": int(r.week),
            "team": r.team,
            "player": r.player,
            "player_key": r.player_key,
            "actual": float(r.actual),
            "m38_proj": float(r.m38_proj),
            "residual_actual_minus_m38": float(r.residual_actual_minus_m38),
            "under25": int(r.under25),
            "under50": int(r.under50),
            "actual100": int(r.actual100),
        }
        h = by_player.get(r.player_key)
        if h is None:
            prior = pd.DataFrame()
        else:
            prior = h.loc[
                h["season"].lt(r.season)
                | (h["season"].eq(r.season) & h["week"].lt(r.week))
            ].sort_values(["season", "week"]).tail(HIST)
        rec["ngs_prior_games"] = int(len(prior))
        if len(prior):
            last = prior.iloc[-1]
            rec["ngs_last_prior_season"] = int(last["season"])
            rec["ngs_last_prior_week"] = int(last["week"])
        else:
            rec["ngs_last_prior_season"] = np.nan
            rec["ngs_last_prior_week"] = np.nan
        for sig in SIGNALS:
            rec[sig] = float(_num(prior[sig]).dropna().mean()) if len(prior) and _num(prior[sig]).notna().any() else np.nan
        rows.append(rec)
    out = pd.DataFrame(rows)
    leak = out.loc[
        out["ngs_last_prior_season"].notna()
        & (
            (out["ngs_last_prior_season"] > out["season"])
            | ((out["ngs_last_prior_season"] == out["season"]) & (out["ngs_last_prior_week"] >= out["week"]))
        )
    ]
    if len(leak):
        raise RuntimeError(f"target/future NGS leakage detected rows={len(leak)}")
    return out


def enrichment(g: pd.DataFrame, event: str, high_mask: pd.Series) -> float:
    overall = float(g[event].mean())
    high = float(g.loc[high_mask, event].mean()) if int(high_mask.sum()) else np.nan
    return high / overall if np.isfinite(overall) and overall > 0 and np.isfinite(high) else np.nan


def score_signal(x: pd.DataFrame, sig: str) -> tuple[dict, list[dict]]:
    g = x.loc[_num(x[sig]).notna()].copy()
    eligible = int(len(g))
    coverage = eligible / EXPECTED_M38_WR_ROWS
    if eligible < MIN_ELIGIBLE or coverage < MIN_COVERAGE:
        return {
            "signal": sig, "eligible_rows": eligible, "coverage": coverage,
            "scoreable": False, "passes": False, "reason": "coverage_or_row_gate",
        }, []

    vals = _num(g[sig])
    residual = _num(g["residual_actual_minus_m38"])
    spearman = float(vals.corr(residual, method="spearman"))
    q25, q75 = float(vals.quantile(0.25)), float(vals.quantile(0.75))
    low = vals.le(q25)
    high = vals.ge(q75)
    gap = float(residual.loc[high].mean() - residual.loc[low].mean())
    u25 = enrichment(g, "under25", high)
    u50 = enrichment(g, "under50", high)
    a100 = enrichment(g, "actual100", high)

    season_rows = []
    positive = 0
    evaluable = 0
    gaps = {}
    for season in TARGET_SEASONS:
        s = g.loc[g["season"].eq(season)].copy()
        if len(s) < 40 or _num(s[sig]).nunique() < 4:
            season_rows.append({"signal": sig, "season": season, "n": int(len(s)), "residual_gap": np.nan})
            continue
        sv = _num(s[sig]); sr = _num(s["residual_actual_minus_m38"])
        s25, s75 = float(sv.quantile(.25)), float(sv.quantile(.75))
        sgap = float(sr.loc[sv.ge(s75)].mean() - sr.loc[sv.le(s25)].mean())
        season_rows.append({"signal": sig, "season": season, "n": int(len(s)), "residual_gap": sgap})
        gaps[season] = sgap
        evaluable += 1
        positive += int(sgap > 0)

    passes = bool(
        eligible >= MIN_ELIGIBLE
        and coverage >= MIN_COVERAGE
        and np.isfinite(spearman) and spearman >= MIN_SPEARMAN
        and np.isfinite(gap) and gap >= MIN_GAP
        and np.isfinite(u25) and u25 >= MIN_ENRICH
        and ((np.isfinite(u50) and u50 >= MIN_ENRICH) or (np.isfinite(a100) and a100 >= MIN_ENRICH))
        and positive >= 4
        and gaps.get(2024, -np.inf) > 0
        and gaps.get(2025, -np.inf) > 0
    )
    row = {
        "signal": sig,
        "eligible_rows": eligible,
        "coverage": coverage,
        "scoreable": True,
        "spearman": spearman,
        "q25": q25,
        "q75": q75,
        "high_low_residual_gap": gap,
        "under25_enrichment": u25,
        "under50_enrichment": u50,
        "actual100_enrichment": a100,
        "evaluable_seasons": evaluable,
        "positive_seasons": positive,
        "gap_2024": gaps.get(2024, np.nan),
        "gap_2025": gaps.get(2025, np.nan),
        "passes": passes,
        "reason": "pass" if passes else "frozen_gate_failure",
    }
    return row, season_rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--wr-r1-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, default=Path("data/backtests/wr_r2_player_tracking_residuals"))
    a = ap.parse_args()

    paired = _read(_one(a.wr_r1_root, "wr_r1_paired_wr_casebook.csv"))
    casebook = build_casebook(paired)
    profiles = individual_profiles(casebook)
    ngs, source = load_ngs()
    joined = attach_prior8(casebook, ngs)

    score_rows = []
    season_rows = []
    for sig in SIGNALS:
        row, seasons = score_signal(joined, sig)
        score_rows.append(row)
        season_rows.extend(seasons)
    scores = pd.DataFrame(score_rows)
    season_scores = pd.DataFrame(season_rows)

    scoreable = bool(scores.get("scoreable", pd.Series(dtype=bool)).any())
    if not scoreable:
        disposition = "PLAYER_TRACKING_SOURCE_OR_INTEGRITY_FAILURE"
        winners = []
    else:
        winners = scores.loc[scores["passes"].fillna(False), "signal"].tolist()
        if len(winners) == 0:
            disposition = "NO_ACTIONABLE_PLAYER_TRACKING_SIGNAL"
        elif len(winners) == 1:
            disposition = f"{winners[0]}_TRACKING_SIGNAL"
        else:
            disposition = "MULTIPLE_PLAYER_TRACKING_SIGNALS"

    summary = {
        "migration": "WR_R2_PLAYER_TRACKING_RESIDUALS",
        "canonical_m38_rows": int(len(casebook)),
        "canonical_seasons": TARGET_SEASONS,
        "individual_players": int(len(profiles)),
        "source": source,
        "target_or_future_ngs_violations": 0,
        "model_fitting_used": False,
        "sportsbook_inputs_used": False,
        "production_changed": False,
        "winners": winners,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    profiles.sort_values(["games", "m38_mae"], ascending=[False, False]).to_csv(a.out_dir / "wr_r2_individual_m38_error_profiles.csv", index=False)
    joined.to_csv(a.out_dir / "wr_r2_tracking_casebook.csv", index=False)
    scores.to_csv(a.out_dir / "wr_r2_signal_metrics.csv", index=False)
    season_scores.to_csv(a.out_dir / "wr_r2_signal_season_metrics.csv", index=False)
    (a.out_dir / "wr_r2_result.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("=== WR-R2 SOURCE ===")
    print(json.dumps(source, indent=2, sort_keys=True))
    print("=== WR-R2 SIGNAL METRICS ===")
    print(scores.to_string(index=False))
    print("=== WR-R2 RESULT ===")
    print(json.dumps(summary, indent=2, sort_keys=True))

    if disposition == "PLAYER_TRACKING_SOURCE_OR_INTEGRITY_FAILURE":
        raise RuntimeError(disposition)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
