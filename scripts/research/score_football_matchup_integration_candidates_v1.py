#!/usr/bin/env python3
"""Football Matchup Transmission V1 — frozen integration-candidate score.

Contract:
docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_INTEGRATION_CANDIDATE_CONTRACT.md

Three candidates only:
- FMT-RB1: RB rush yards <- lower opponent pass_rate_faced
- FMT-WR1: WR receiving yards <- higher offense true_proe
- FMT-TE1: TE receiving yards <- higher opponent def_pass_success_allowed

Coefficient training is 2022 W2-18 only. Primary confirmation is 2023 W2-18.
2024/2025 are fixed-coefficient consistency replays only. No sportsbook data.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

VERSION = "FOOTBALL_MATCHUP_TRANSMISSION_INTEGRATION_CANDIDATES_V1"
TRAIN_SEASON = 2022
PRIMARY_SEASON = 2023
SECONDARY_SEASONS = (2024, 2025)
TARGET_SEASONS = (2022, 2023, 2024, 2025)
TARGET_WEEKS = tuple(range(2, 19))
HISTORY_GAMES = 8
MIN_ROWS = 200
MIN_GAMES = 50
BOOT_REPS = 5000
BOOT_SEED = 20261006
TOL = 1e-10
RB_POS = {"RB", "FB", "HB"}

FORBIDDEN_TOKENS = (
    "sportsbook", "bookmaker", "prop_line", "market_line", "over_odds",
    "under_odds", "spread_line", "total_line", "moneyline", "closing_line",
    "no_vig", "implied_prob",
)


@dataclass(frozen=True)
class Candidate:
    candidate_id: str
    cohort: str
    market: str
    position: str
    feature: str
    sign: int


CANDIDATES = (
    Candidate("FMT-RB1", "RB_RUSH", "rush_yards", "RB", "def_pass_rate_faced", -1),
    Candidate("FMT-WR1", "WR_REC", "rec_yards", "WR", "off_true_proe", +1),
    Candidate("FMT-TE1", "TE_REC", "rec_yards", "TE", "def_pass_success_allowed", +1),
)


def _read(path: Path, label: str, usecols=None) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False, usecols=usecols)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def _key(v) -> str:
    return "".join(ch.lower() for ch in str(v or "") if ch.isalnum())


def _validate_no_forbidden(label: str, frame: pd.DataFrame) -> None:
    bad = [
        c for c in frame.columns
        if any(t in str(c).lower() for t in FORBIDDEN_TOKENS)
    ]
    if bad:
        raise RuntimeError(f"forbidden sportsbook/odds fields in {label}: {bad}")


def _mean_col(x: pd.DataFrame, names: list[str]) -> float:
    for name in names:
        if name in x.columns:
            s = _num(x[name])
            if s.notna().any():
                return float(s.mean())
    return np.nan


def _prior_team_rows(
    team_weekly: pd.DataFrame,
    season: int,
    week: int,
    team: str,
) -> pd.DataFrame:
    q = team_weekly.loc[
        team_weekly["team"].eq(team)
        & (
            team_weekly["season"].lt(season)
            | (team_weekly["season"].eq(season) & team_weekly["week"].lt(week))
        )
    ].sort_values(["season", "week"])
    return q.tail(HISTORY_GAMES)


def build_team_features(
    team_weekly: pd.DataFrame,
    schedule: pd.DataFrame,
) -> pd.DataFrame:
    tw = team_weekly.copy()
    tw["season"] = _num(tw["season"]).astype("Int64")
    tw["week"] = _num(tw["week"]).astype("Int64")
    tw["team"] = tw["team"].map(canon_team)

    s = schedule.copy()
    s["season"] = _num(s["season"]).astype("Int64")
    s["week"] = _num(s["week"]).astype("Int64")
    s["team"] = s["team"].map(canon_team)
    s["opponent"] = s["opponent"].map(canon_team)
    s = s.loc[
        s["season"].isin(TARGET_SEASONS)
        & s["week"].isin(TARGET_WEEKS),
        ["season", "week", "team", "opponent"],
    ].drop_duplicates()
    if s.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate target schedule team-week")

    rows = []
    for r in s.itertuples(index=False):
        season, week = int(r.season), int(r.week)
        oh = _prior_team_rows(tw, season, week, str(r.team))
        dh = _prior_team_rows(tw, season, week, str(r.opponent))
        rows.append(
            {
                "season": season,
                "week": week,
                "team": str(r.team),
                "opponent": str(r.opponent),
                "off_true_proe": _mean_col(oh, ["true_proe", "proe"]),
                "def_pass_rate_faced": _mean_col(dh, ["pass_rate_faced"]),
                "def_pass_success_allowed": _mean_col(
                    dh, ["def_pass_success_allowed", "success_rate_def"]
                ),
            }
        )
    out = pd.DataFrame(rows)
    for c in sorted({x.feature for x in CANDIDATES}):
        raw = _num(out[c])
        z = pd.Series(np.nan, index=out.index, dtype=float)
        for _, idx in out.groupby(["season", "week"]).groups.items():
            vals = raw.loc[idx]
            m = float(vals.mean()) if vals.notna().any() else np.nan
            sd = float(vals.std(ddof=0)) if vals.notna().sum() >= 2 else np.nan
            if np.isfinite(sd) and sd > 0:
                z.loc[idx] = (vals - m) / sd
        out[f"{c}__z"] = z
    return out


def attach_position(
    frame: pd.DataFrame,
    logs: pd.DataFrame,
    *,
    projection_col: str,
) -> pd.DataFrame:
    x = frame.copy()
    for c in ("season", "week"):
        x[c] = _num(x[c]).astype(int)
    x = x.loc[
        x["season"].isin(TARGET_SEASONS)
        & x["week"].isin(TARGET_WEEKS)
    ].copy()
    x["team"] = x["team"].map(canon_team)
    x["opponent"] = x["opponent"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].map(_key)
    x["market"] = x["market"].astype(str).str.lower()
    x["actual"] = _num(x["actual"])
    x["baseline_projection"] = _num(x[projection_col])
    x = x.loc[x["actual"].notna() & x["baseline_projection"].notna()].copy()

    meta = logs.copy()
    meta["season"] = _num(meta["season"]).astype("Int64")
    meta["week"] = _num(meta["week"]).astype("Int64")
    meta["team"] = meta["team"].map(canon_team)
    meta["player_clean_key"] = meta["player_clean_key"].map(_key)
    if "player_identity_key" not in meta.columns:
        meta["player_identity_key"] = meta["player_clean_key"]
    meta["position"] = meta["position"].astype(str).str.upper().str.strip()
    meta = meta[
        [
            "season", "week", "team", "player_clean_key",
            "player_identity_key", "position",
        ]
    ].drop_duplicates()

    # The rebuilt full-stack trace may already carry position/identity columns.
    # Canonicalize those fields from historical player logs rather than allowing
    # pandas to silently suffix them to position_x/position_y.
    x = x.drop(
        columns=["position", "player_identity_key"],
        errors="ignore",
    )
    x = x.merge(
        meta,
        on=["season", "week", "team", "player_clean_key"],
        how="left",
        validate="many_to_one",
    )
    miss = x["position"].isna()
    if miss.mean() > 0.02:
        raise RuntimeError(
            f"position identity coverage too low: {int(miss.sum())}/{len(x)}"
        )
    x["player_identity_key"] = x["player_identity_key"].fillna(x["player_clean_key"]).astype(str)
    return x


def candidate_cohort(
    projection: pd.DataFrame,
    team_features: pd.DataFrame,
    cand: Candidate,
) -> pd.DataFrame:
    if cand.position == "RB":
        pos = projection["position"].isin(RB_POS)
    else:
        pos = projection["position"].eq(cand.position)
    q = projection.loc[
        projection["market"].eq(cand.market) & pos
    ].copy()
    q = q.merge(
        team_features[
            [
                "season", "week", "team", "opponent",
                cand.feature, f"{cand.feature}__z",
            ]
        ],
        on=["season", "week", "team", "opponent"],
        how="left",
        validate="many_to_one",
    )
    q["weakness_z"] = cand.sign * _num(q[f"{cand.feature}__z"])
    q = q.replace([np.inf, -np.inf], np.nan)
    return q


def fit_beta(q: pd.DataFrame) -> dict:
    z = q.loc[q["season"].eq(TRAIN_SEASON)].dropna(
        subset=["actual", "baseline_projection", "weakness_z", "game_id"]
    ).copy()
    rows = int(len(z))
    games = int(z["game_id"].nunique())
    support = rows >= MIN_ROWS and games >= MIN_GAMES
    beta = np.nan
    if support:
        x = z["weakness_z"].to_numpy(float)
        residual = (z["actual"] - z["baseline_projection"]).to_numpy(float)
        den = float(np.dot(x, x))
        if den > 0:
            beta = float(np.dot(x, residual) / den)
    return {
        "rows": rows,
        "games": games,
        "support": bool(support),
        "beta_train": beta,
        "positive_beta": bool(np.isfinite(beta) and beta > 0),
    }


def metric(actual: pd.Series, pred: pd.Series) -> dict:
    z = pd.DataFrame({"actual": _num(actual), "pred": _num(pred)}).dropna()
    if z.empty:
        return {
            "n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan,
            "median_ae": np.nan, "correlation": np.nan,
            "tail75": 0, "tail100": 0,
        }
    err = z["pred"] - z["actual"]
    corr = (
        float(z["pred"].corr(z["actual"]))
        if len(z) > 1 and z["pred"].nunique() > 1 and z["actual"].nunique() > 1
        else np.nan
    )
    return {
        "n": int(len(z)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "median_ae": float(err.abs().median()),
        "correlation": corr,
        "tail75": int(err.abs().ge(75).sum()),
        "tail100": int(err.abs().ge(100).sum()),
    }


def paired_game_bootstrap(q: pd.DataFrame, *, seed: int) -> dict:
    z = q.dropna(
        subset=["actual", "baseline_projection", "candidate_projection", "game_id"]
    ).copy()
    if z.empty or z["game_id"].nunique() < 2:
        return {"p_improve": np.nan, "ci_low": np.nan, "ci_high": np.nan, "valid_reps": 0}
    z["base_abs"] = (z["baseline_projection"] - z["actual"]).abs()
    z["cand_abs"] = (z["candidate_projection"] - z["actual"]).abs()
    g = z.groupby("game_id", as_index=False).agg(
        n=("actual", "size"),
        base_sum=("base_abs", "sum"),
        cand_sum=("cand_abs", "sum"),
    )
    a = g[["n", "base_sum", "cand_sum"]].to_numpy(float)
    rng = np.random.default_rng(seed)
    probs = np.full(len(g), 1.0 / len(g))
    vals = []
    done = 0
    while done < BOOT_REPS:
        k = min(250, BOOT_REPS - done)
        counts = rng.multinomial(len(g), probs, size=k).astype(float)
        s = counts @ a
        valid = s[:, 0] > 0
        diff = (s[valid, 1] - s[valid, 2]) / s[valid, 0]
        vals.append(diff)
        done += k
    v = np.concatenate(vals)
    v = v[np.isfinite(v)]
    return {
        "p_improve": float((v > 0).mean()) if len(v) else np.nan,
        "ci_low": float(np.quantile(v, 0.025)) if len(v) else np.nan,
        "ci_high": float(np.quantile(v, 0.975)) if len(v) else np.nan,
        "valid_reps": int(len(v)),
    }


def spearman_residual(q: pd.DataFrame, pred_col: str) -> float:
    z = q[["actual", pred_col, "weakness_z"]].dropna().copy()
    if len(z) < 2 or z["weakness_z"].nunique() < 2:
        return np.nan
    residual = z["actual"] - z[pred_col]
    return float(residual.corr(z["weakness_z"], method="spearman"))


def score_candidate(q: pd.DataFrame, cand: Candidate, seed_offset: int) -> tuple[dict, list[dict]]:
    train = fit_beta(q)
    beta = train["beta_train"]
    q = q.copy()
    q["candidate_projection"] = (
        q["baseline_projection"] + beta * q["weakness_z"]
        if np.isfinite(beta)
        else np.nan
    )
    q["abs_adjustment"] = (q["candidate_projection"] - q["baseline_projection"]).abs()

    rows = []
    season_metrics = {}
    for season in (TRAIN_SEASON, PRIMARY_SEASON, *SECONDARY_SEASONS):
        s = q.loc[q["season"].eq(season)].copy()
        base = metric(s["actual"], s["baseline_projection"])
        candidate = metric(s["actual"], s["candidate_projection"])
        boot = paired_game_bootstrap(s, seed=BOOT_SEED + seed_offset + season)
        pre_rho = spearman_residual(s, "baseline_projection")
        post_rho = spearman_residual(s, "candidate_projection")
        adj = _num(s["abs_adjustment"]).dropna()
        rec = {
            "candidate_id": cand.candidate_id,
            "season": season,
            "rows": int(len(s.dropna(subset=["actual", "baseline_projection", "weakness_z"]))),
            "games": int(s.dropna(subset=["game_id"])["game_id"].nunique()),
            "players": int(s.dropna(subset=["player_identity_key"])["player_identity_key"].nunique()),
            "beta_train": beta,
            "baseline_mae": base["mae"],
            "candidate_mae": candidate["mae"],
            "mae_improvement": base["mae"] - candidate["mae"],
            "baseline_rmse": base["rmse"],
            "candidate_rmse": candidate["rmse"],
            "baseline_bias": base["bias"],
            "candidate_bias": candidate["bias"],
            "baseline_median_ae": base["median_ae"],
            "candidate_median_ae": candidate["median_ae"],
            "baseline_correlation": base["correlation"],
            "candidate_correlation": candidate["correlation"],
            "baseline_tail75": base["tail75"],
            "candidate_tail75": candidate["tail75"],
            "baseline_tail100": base["tail100"],
            "candidate_tail100": candidate["tail100"],
            "bootstrap_p_improve": boot["p_improve"],
            "bootstrap_ci_low": boot["ci_low"],
            "bootstrap_ci_high": boot["ci_high"],
            "bootstrap_valid_reps": boot["valid_reps"],
            "pre_residual_spearman": pre_rho,
            "post_residual_spearman": post_rho,
            "mean_abs_adjustment": float(adj.mean()) if len(adj) else np.nan,
            "p95_abs_adjustment": float(adj.quantile(0.95)) if len(adj) else np.nan,
            "max_abs_adjustment": float(adj.max()) if len(adj) else np.nan,
        }
        rows.append(rec)
        season_metrics[season] = rec

    pooled = q.loc[q["season"].isin(SECONDARY_SEASONS)].copy()
    base_p = metric(pooled["actual"], pooled["baseline_projection"])
    cand_p = metric(pooled["actual"], pooled["candidate_projection"])

    p = season_metrics[PRIMARY_SEASON]
    primary_gates = {
        "mae_improves": bool(np.isfinite(p["mae_improvement"]) and p["mae_improvement"] > 0),
        "rmse_nonworse": bool(
            np.isfinite(p["candidate_rmse"]) and np.isfinite(p["baseline_rmse"])
            and p["candidate_rmse"] <= p["baseline_rmse"] + TOL
        ),
        "bootstrap_p_ge_080": bool(
            np.isfinite(p["bootstrap_p_improve"]) and p["bootstrap_p_improve"] >= 0.80
        ),
        "tail75_nonincrease": bool(p["candidate_tail75"] <= p["baseline_tail75"]),
        "tail100_nonincrease": bool(p["candidate_tail100"] <= p["baseline_tail100"]),
        "residual_signal_reduced": bool(
            np.isfinite(p["pre_residual_spearman"])
            and np.isfinite(p["post_residual_spearman"])
            and abs(p["post_residual_spearman"]) < abs(p["pre_residual_spearman"])
        ),
    }
    secondary_gates = {
        "mae_nonworse_2024": bool(
            season_metrics[2024]["candidate_mae"] <= season_metrics[2024]["baseline_mae"] + TOL
        ),
        "mae_nonworse_2025": bool(
            season_metrics[2025]["candidate_mae"] <= season_metrics[2025]["baseline_mae"] + TOL
        ),
        "pooled_mae_improves": bool(cand_p["mae"] < base_p["mae"]),
        "pooled_tail75_nonincrease": bool(cand_p["tail75"] <= base_p["tail75"]),
        "pooled_tail100_nonincrease": bool(cand_p["tail100"] <= base_p["tail100"]),
    }
    all_gates = bool(
        train["support"]
        and train["positive_beta"]
        and all(primary_gates.values())
        and all(secondary_gates.values())
    )
    result = {
        "candidate_id": cand.candidate_id,
        "cohort": cand.cohort,
        "market": cand.market,
        "feature": cand.feature,
        "orientation_sign": cand.sign,
        "train": train,
        "primary_2023_gates": primary_gates,
        "secondary_2024_2025_gates": secondary_gates,
        "secondary_pooled": {
            "baseline_mae": base_p["mae"],
            "candidate_mae": cand_p["mae"],
            "mae_improvement": base_p["mae"] - cand_p["mae"],
            "baseline_tail75": base_p["tail75"],
            "candidate_tail75": cand_p["tail75"],
            "baseline_tail100": base_p["tail100"],
            "candidate_tail100": cand_p["tail100"],
        },
        "disposition": (
            "INTEGRATION_CANDIDATE_CONFIRMED"
            if all_gates
            else "INTEGRATION_CANDIDATE_CLOSED"
        ),
    }
    return result, rows


def load_parent_detail(path: Path, logs: pd.DataFrame) -> pd.DataFrame:
    allowed = {
        "season", "week", "game_id", "team", "opponent", "player_clean_key",
        "market", "actual", "final_mean",
    }
    x = _read(
        path,
        "parent right-tail detail",
        usecols=lambda c: str(c).strip().lower() in allowed,
    )
    x = x.rename(columns={"final_mean": "baseline_projection"})
    return attach_position(x, logs, projection_col="baseline_projection").drop(
        columns=["baseline_projection_y"], errors="ignore"
    )


def assert_parent_parity(
    current: pd.DataFrame,
    parent: pd.DataFrame,
) -> dict:
    keys = [
        "season", "week", "game_id", "team", "opponent",
        "player_clean_key", "market",
    ]
    checked = 0
    max_gap = 0.0
    per_candidate = {}
    for cand in CANDIDATES:
        a = candidate_cohort(current, pd.DataFrame(), cand) if False else None
        if cand.position == "RB":
            cur_pos = current["position"].isin(RB_POS)
            par_pos = parent["position"].isin(RB_POS)
        else:
            cur_pos = current["position"].eq(cand.position)
            par_pos = parent["position"].eq(cand.position)
        aa = current.loc[
            current["season"].isin(SECONDARY_SEASONS)
            & current["week"].isin(TARGET_WEEKS)
            & current["market"].eq(cand.market)
            & cur_pos,
            keys + ["baseline_projection"],
        ].copy()
        bb = parent.loc[
            parent["season"].isin(SECONDARY_SEASONS)
            & parent["week"].isin(TARGET_WEEKS)
            & parent["market"].eq(cand.market)
            & par_pos,
            keys + ["baseline_projection"],
        ].copy()
        aa = aa.sort_values(keys).reset_index(drop=True)
        bb = bb.sort_values(keys).reset_index(drop=True)
        if len(aa) != len(bb):
            raise RuntimeError(
                f"parent parity row-count mismatch {cand.candidate_id}: {len(aa)} vs {len(bb)}"
            )
        if not aa[keys].astype(str).equals(bb[keys].astype(str)):
            raise RuntimeError(f"parent parity identity mismatch {cand.candidate_id}")
        av = _num(aa["baseline_projection"]).to_numpy(float)
        bv = _num(bb["baseline_projection"]).to_numpy(float)
        if not np.array_equal(np.isnan(av), np.isnan(bv)):
            raise RuntimeError(f"parent parity missingness mismatch {cand.candidate_id}")
        mask = np.isfinite(av) & np.isfinite(bv)
        gap = float(np.max(np.abs(av[mask] - bv[mask]))) if mask.any() else 0.0
        if gap > TOL:
            raise RuntimeError(
                f"parent baseline numerical drift {cand.candidate_id}: max_gap={gap}"
            )
        per_candidate[cand.candidate_id] = {"rows": int(len(aa)), "max_gap": gap}
        checked += len(aa)
        max_gap = max(max_gap, gap)
    return {
        "status": "PASS",
        "rows": int(checked),
        "max_gap": float(max_gap),
        "per_candidate": per_candidate,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-trace", type=Path, required=True)
    ap.add_argument("--parent-right-tail-detail", type=Path, required=True)
    ap.add_argument("--team-weekly-matchup", type=Path, required=True)
    ap.add_argument("--player-logs", type=Path, required=True)
    ap.add_argument("--schedule", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    projection = _read(args.projection_trace, "candidate baseline projection trace")
    team = _read(args.team_weekly_matchup, "corrected matchup team history")
    logs = _read(args.player_logs, "historical player logs")
    schedule = _read(args.schedule, "authoritative historical schedule")
    for label, frame in (
        ("projection", projection),
        ("team", team),
        ("logs", logs),
        ("schedule", schedule),
    ):
        _validate_no_forbidden(label, frame)

    required_projection = {
        "season", "week", "game_id", "team", "opponent",
        "player_clean_key", "market", "actual", "ensemble_proj",
    }
    miss = sorted(required_projection - set(projection.columns))
    if miss:
        raise RuntimeError(f"projection trace missing columns: {miss}")

    current = attach_position(projection, logs, projection_col="ensemble_proj")
    parent = load_parent_detail(args.parent_right_tail_detail, logs)
    parity = assert_parent_parity(current, parent)

    team_features = build_team_features(team, schedule)
    _validate_no_forbidden("team_features", team_features)

    candidate_results = []
    score_rows = []
    trace_rows = []
    for i, cand in enumerate(CANDIDATES):
        q = candidate_cohort(current, team_features, cand)
        result, rows = score_candidate(q, cand, i * 1000)
        candidate_results.append(result)
        score_rows.extend(rows)
        keep = [
            "season", "week", "game_id", "team", "opponent",
            "player_clean_key", "player_identity_key", "position", "market",
            "actual", "baseline_projection", cand.feature, "weakness_z",
        ]
        beta = result["train"]["beta_train"]
        qt = q[keep].copy()
        qt["candidate_id"] = cand.candidate_id
        qt["beta_train"] = beta
        qt["candidate_projection"] = (
            qt["baseline_projection"] + beta * qt["weakness_z"]
            if np.isfinite(beta)
            else np.nan
        )
        trace_rows.append(qt)

    confirmed = [
        r["candidate_id"]
        for r in candidate_results
        if r["disposition"] == "INTEGRATION_CANDIDATE_CONFIRMED"
    ]
    payload = {
        "version": VERSION,
        "contract": (
            "docs/research/"
            "FOOTBALL_MATCHUP_TRANSMISSION_V1_INTEGRATION_CANDIDATE_CONTRACT.md"
        ),
        "train_season": TRAIN_SEASON,
        "primary_confirmation_season": PRIMARY_SEASON,
        "secondary_consistency_seasons": list(SECONDARY_SEASONS),
        "evaluation_weeks": [2, 18],
        "sportsbook_inputs_used": 0,
        "production_changed": False,
        "combined_candidate_scored": False,
        "parent_2024_2025_baseline_parity": parity,
        "candidates": candidate_results,
        "confirmed_candidates": confirmed,
        "confirmed_count": len(confirmed),
        "next_step": (
            "FREEZE_SEPARATE_PRODUCTION_ORDER_SHADOW_CONTRACT"
            if confirmed
            else "CLOSE_ALL_INTEGRATION_CANDIDATES"
        ),
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(score_rows).to_csv(
        args.out_dir / "football_matchup_integration_candidate_scorecard.csv",
        index=False,
    )
    pd.concat(trace_rows, ignore_index=True).to_csv(
        args.out_dir / "football_matchup_integration_candidate_trace.csv",
        index=False,
    )
    team_features.to_csv(
        args.out_dir / "football_matchup_integration_candidate_team_features.csv",
        index=False,
    )
    (
        args.out_dir / "football_matchup_integration_candidate_result.json"
    ).write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(pd.DataFrame(score_rows).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
