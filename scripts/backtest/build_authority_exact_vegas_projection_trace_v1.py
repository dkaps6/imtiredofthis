#!/usr/bin/env python3
"""Build an authority-exact historical Vegas projection trace.

This module refuses to reconstruct promoted football science from a generic
historical harness. It consumes the exact frozen OOS authority artifacts for
QB M89/M90, WR-R15 and TE-R5P, verifies their canonical row counts and frozen
football metrics, and only then emits rows that may be joined to sportsbook
history.

Two benchmark arms are emitted on the same authority identities:
- AUTHORITY_OOS: the exact OOS projection stored in the promoting artifact.
- CURRENT_PRODUCTION_ORDER: QB uses the exact M89 OOS mean; WR/TE use the
  production-order replay components with the repository's current ensemble
  weights already applied upstream of this script.

RB P3/R26/R22 are intentionally not fabricated retrospectively. Their
promoted scope is 2026 Week 1, so the parity audit records RB as prospective-
only for a separate forward grade.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

QB_EXPECTED_ROWS = 884
QB_EXPECTED_SEASONS = {2024: 444, 2025: 440}
QB_BASE_MAE = 57.638995160175256
QB_SYNTH_MAE = 55.06011827391715

WR_BASE_VARIANT = "M38_EXPLICIT_BASELINE"
WR_CANDIDATE_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
WR_EXPECTED_ROWS = 4193
WR_EXPECTED_SEASONS = {2023: 2076, 2024: 2117}
WR_BASE_TARGET_MAE = 2.1284991789406633
WR_CANDIDATE_TARGET_MAE = 2.0438856296229853
WR_BASE_REC_YARDS_MAE = 22.85661757162215
WR_CANDIDATE_REC_YARDS_MAE = 22.52635823935656

TE_EXPECTED_ROWS = 3214
TE_EXPECTED_SEASONS = {2023: 1019, 2024: 1082, 2025: 1113}
TE_BASE_TARGET_MAE = 1.682530106839412
TE_CANDIDATE_TARGET_MAE = 1.6183581460350753
TE_BASE_REC_YARDS_MAE = 16.26797094476944
TE_CANDIDATE_REC_YARDS_MAE = 15.850607934418186

TOL = 1e-9
KEYS = ["season", "week", "team", "player_clean_key"]


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size == 0:
        raise RuntimeError(f"missing {label}: {path}")
    out = pd.read_csv(path, low_memory=False)
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _mae(pred: pd.Series, actual: pd.Series) -> float:
    p = pd.to_numeric(pred, errors="coerce")
    a = pd.to_numeric(actual, errors="coerce")
    if p.isna().any() or a.isna().any():
        raise RuntimeError("authority metric contains non-finite projection/actual")
    return float(np.mean(np.abs(p.to_numpy(float) - a.to_numpy(float))))


def _assert_close(label: str, actual: float, expected: float, tol: float = TOL) -> None:
    if not np.isfinite(actual) or abs(float(actual) - float(expected)) > tol:
        raise RuntimeError(f"{label} drifted: actual={actual:.12f} expected={expected:.12f} tol={tol}")


def _season_counts(df: pd.DataFrame) -> dict[int, int]:
    return {int(k): int(v) for k, v in df.groupby("season").size().to_dict().items()}


def _assert_unique(df: pd.DataFrame, keys: list[str], label: str) -> None:
    dup = df.duplicated(keys, keep=False)
    if dup.any():
        sample = df.loc[dup, keys].head(20).to_dict(orient="records")
        raise RuntimeError(f"{label} duplicate identities: {sample}")


def validate_qb(df: pd.DataFrame) -> dict:
    req = {*KEYS, "opponent", "actual_pass_yards", "base_proj", "football_synthesis"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise RuntimeError(f"QB authority artifact missing columns: {miss}")
    _assert_unique(df, KEYS, "QB M89 authority")
    if len(df) != QB_EXPECTED_ROWS:
        raise RuntimeError(f"QB authority row count drifted: {len(df)} != {QB_EXPECTED_ROWS}")
    counts = _season_counts(df)
    if counts != QB_EXPECTED_SEASONS:
        raise RuntimeError(f"QB authority season counts drifted: {counts} != {QB_EXPECTED_SEASONS}")
    base_mae = _mae(df.base_proj, df.actual_pass_yards)
    synth_mae = _mae(df.football_synthesis, df.actual_pass_yards)
    _assert_close("QB base MAE", base_mae, QB_BASE_MAE)
    _assert_close("QB synthesis MAE", synth_mae, QB_SYNTH_MAE)
    return {
        "position": "QB", "authority": "QB_PASS_SYNTHESIS_V1",
        "scope": "M89 exact canonical 884 OOS 2024-2025",
        "rows": len(df), "baseline_metric": base_mae, "candidate_metric": synth_mae,
        "metric": "pass_yards_mae", "status": "PASS",
    }


def validate_wr(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    req = {"variant", *KEYS, "mc_receptions", "mc_rec_yards", "pred_targets", "actual_targets", "actual_rec_yards"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise RuntimeError(f"WR authority artifact missing columns: {miss}")
    base = df.loc[df.variant.astype(str).eq(WR_BASE_VARIANT)].copy()
    cand = df.loc[df.variant.astype(str).eq(WR_CANDIDATE_VARIANT)].copy()
    for label, part in [("WR baseline", base), ("WR candidate", cand)]:
        _assert_unique(part, KEYS, label)
        if len(part) != WR_EXPECTED_ROWS:
            raise RuntimeError(f"{label} row count drifted: {len(part)} != {WR_EXPECTED_ROWS}")
        counts = _season_counts(part)
        if counts != WR_EXPECTED_SEASONS:
            raise RuntimeError(f"{label} season counts drifted: {counts} != {WR_EXPECTED_SEASONS}")
    if set(map(tuple, base[KEYS].to_numpy())) != set(map(tuple, cand[KEYS].to_numpy())):
        raise RuntimeError("WR baseline/candidate identity sets differ")
    b_tgt = _mae(base.pred_targets, base.actual_targets)
    c_tgt = _mae(cand.pred_targets, cand.actual_targets)
    b_rec = _mae(base.mc_rec_yards, base.actual_rec_yards)
    c_rec = _mae(cand.mc_rec_yards, cand.actual_rec_yards)
    _assert_close("WR baseline target MAE", b_tgt, WR_BASE_TARGET_MAE)
    _assert_close("WR candidate target MAE", c_tgt, WR_CANDIDATE_TARGET_MAE)
    _assert_close("WR baseline rec_yards MAE", b_rec, WR_BASE_REC_YARDS_MAE)
    _assert_close("WR candidate rec_yards MAE", c_rec, WR_CANDIDATE_REC_YARDS_MAE)
    if (cand.season == 2025).any():
        raise RuntimeError("WR-R15 authority unexpectedly contains 2025 confirmation rows")
    return ({
        "position": "WR", "authority": "M38_WR1_PLUS_WR_R15_PRODUCTION_MODEL_V1",
        "scope": "WR-R15 exact OOS confirmation folds 2023-2024; 2025 forbidden",
        "rows": len(cand), "baseline_metric": b_rec, "candidate_metric": c_rec,
        "metric": "rec_yards_mae", "status": "PASS",
        "secondary_metric": "targets_mae", "secondary_baseline": b_tgt, "secondary_candidate": c_tgt,
    }, cand)


def validate_te(df: pd.DataFrame) -> dict:
    req = {*KEYS, "position", "b0_expected_targets", "candidate_targets_r5p", "targets",
           "b0_rec_yards", "candidate_rec_yards_r5p", "candidate_receptions_r5p", "receptions", "rec_yards"}
    miss = sorted(req - set(df.columns))
    if miss:
        raise RuntimeError(f"TE authority artifact missing columns: {miss}")
    _assert_unique(df, KEYS, "TE-R5P authority")
    if len(df) != TE_EXPECTED_ROWS:
        raise RuntimeError(f"TE authority row count drifted: {len(df)} != {TE_EXPECTED_ROWS}")
    counts = _season_counts(df)
    if counts != TE_EXPECTED_SEASONS:
        raise RuntimeError(f"TE authority season counts drifted: {counts} != {TE_EXPECTED_SEASONS}")
    b_tgt = _mae(df.b0_expected_targets, df.targets)
    c_tgt = _mae(df.candidate_targets_r5p, df.targets)
    b_rec = _mae(df.b0_rec_yards, df.rec_yards)
    c_rec = _mae(df.candidate_rec_yards_r5p, df.rec_yards)
    _assert_close("TE baseline target MAE", b_tgt, TE_BASE_TARGET_MAE)
    _assert_close("TE candidate target MAE", c_tgt, TE_CANDIDATE_TARGET_MAE)
    _assert_close("TE baseline rec_yards MAE", b_rec, TE_BASE_REC_YARDS_MAE)
    _assert_close("TE candidate rec_yards MAE", c_rec, TE_CANDIDATE_REC_YARDS_MAE)
    return {
        "position": "TE", "authority": "TE_R5P_PRODUCTION_MODEL_V1",
        "scope": "TE-R5P exact OOS production-contract casebook 2023-2025",
        "rows": len(df), "baseline_metric": b_rec, "candidate_metric": c_rec,
        "metric": "rec_yards_mae", "status": "PASS",
        "secondary_metric": "targets_mae", "secondary_baseline": b_tgt, "secondary_candidate": c_tgt,
    }


def validate_wrte_replay(audit: pd.DataFrame) -> dict:
    if len(audit) != 1:
        raise RuntimeError("WR/TE replay integrity audit must have exactly one row")
    r = audit.iloc[0]
    if str(r.get("status", "")).strip().upper() != "PASS":
        raise RuntimeError(f"WR/TE production-order replay integrity is not PASS: {r.to_dict()}")
    for col in ["untreated_max_abs_proj_delta", "untreated_max_abs_p_over_delta", "untreated_max_abs_model_sd_delta"]:
        if float(r.get(col, np.inf)) > 1e-9:
            raise RuntimeError(f"WR/TE replay {col} exceeded tolerance: {r.get(col)}")
    return {
        "position": "WR/TE", "authority": "WR_TE_PRODUCTION_ORDER_REPLAY_V1",
        "scope": "PR #549 exact replay integrity",
        "rows": int(r.get("authorized_rows", 0)),
        "baseline_metric": 0.0, "candidate_metric": float(r.get("untreated_max_abs_proj_delta", 0.0)),
        "metric": "untreated_projection_delta", "status": "PASS",
    }


def _schedule_from_trace(trace: pd.DataFrame) -> pd.DataFrame:
    req = {"season", "week", "team", "opponent", "game_id"}
    miss = sorted(req - set(trace.columns))
    if miss:
        raise RuntimeError(f"production-order trace missing schedule columns: {miss}")
    sched = trace[["season", "week", "team", "opponent", "game_id"]].drop_duplicates().copy()
    _assert_unique(sched, ["season", "week", "team"], "production-order schedule")
    return sched


def _attach_schedule(df: pd.DataFrame, sched: pd.DataFrame, label: str) -> pd.DataFrame:
    x = df.copy()
    had_opp = "opponent" in x.columns
    if had_opp:
        x = x.rename(columns={"opponent": "authority_opponent"})
    x = x.merge(sched, on=["season", "week", "team"], how="left", validate="many_to_one")
    if x.game_id.isna().any():
        sample = x.loc[x.game_id.isna(), KEYS].head(20).to_dict(orient="records")
        raise RuntimeError(f"{label} failed schedule attachment: {sample}")
    if had_opp:
        bad = x.authority_opponent.astype(str).str.upper().ne(x.opponent.astype(str).str.upper())
        if bad.any():
            sample = x.loc[bad, [*KEYS, "authority_opponent", "opponent"]].head(20).to_dict(orient="records")
            raise RuntimeError(f"{label} opponent mismatch: {sample}")
        x = x.drop(columns=["authority_opponent"])
    return x


def _base_output(df: pd.DataFrame, *, position: str, market: str, authority: str, scope: str,
                 benchmark_arm: str, projection: pd.Series, actual: pd.Series) -> pd.DataFrame:
    out = df[["season", "week", "team", "opponent", "game_id", "player_clean_key"]].copy()
    if "player" in df.columns:
        out["player"] = df["player"].astype(str)
    else:
        out["player"] = df["player_clean_key"].astype(str)
    out["position"] = position
    out["market"] = market
    out["ensemble_proj"] = pd.to_numeric(projection, errors="coerce")
    out["actual"] = pd.to_numeric(actual, errors="coerce")
    out["model_authority"] = authority
    out["authority_scope"] = scope
    out["benchmark_arm"] = benchmark_arm
    out["authority_parity_pass"] = 1
    if out.ensemble_proj.isna().any() or out.actual.isna().any():
        raise RuntimeError(f"{authority} {market} output contains non-finite projection/actual")
    return out


def build_authority_trace(qb: pd.DataFrame, wr_cand: pd.DataFrame, te: pd.DataFrame, sched: pd.DataFrame,
                          production_trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    q = _attach_schedule(qb, sched, "QB authority")
    rows.append(_base_output(
        q, position="QB", market="pass_yards", authority="QB_PASS_SYNTHESIS_V1",
        scope="exact canonical M89 884-game OOS authority", benchmark_arm="AUTHORITY_OOS",
        projection=q.football_synthesis, actual=q.actual_pass_yards,
    ))

    w = wr_cand.loc[wr_cand.season.eq(2024)].copy()
    w = _attach_schedule(w, sched, "WR authority")
    rows.append(_base_output(
        w, position="WR", market="rec_yards", authority="M38_WR1_PLUS_WR_R15_PRODUCTION_MODEL_V1",
        scope="exact WR-R15 2024 OOS confirmation fold", benchmark_arm="AUTHORITY_OOS",
        projection=w.mc_rec_yards, actual=w.actual_rec_yards,
    ))
    w_actual = production_trace.loc[
        production_trace.position.eq("WR") & production_trace.market.eq("receptions"),
        [*KEYS, "actual"],
    ].drop_duplicates(KEYS)
    _assert_unique(w_actual, KEYS, "WR receptions actual")
    w2 = w.merge(w_actual, on=KEYS, how="left", validate="one_to_one")
    rows.append(_base_output(
        w2, position="WR", market="receptions", authority="M38_WR1_PLUS_WR_R15_PRODUCTION_MODEL_V1",
        scope="exact WR-R15 2024 OOS fold; receptions derived downstream from promoted entitlement",
        benchmark_arm="AUTHORITY_OOS", projection=w2.mc_receptions, actual=w2.actual,
    ))

    t = te.loc[te.season.isin([2024, 2025])].copy()
    t = _attach_schedule(t, sched, "TE authority")
    rows.append(_base_output(
        t, position="TE", market="rec_yards", authority="TE_R5P_PRODUCTION_MODEL_V1",
        scope="exact TE-R5P OOS production-contract rows 2024-2025", benchmark_arm="AUTHORITY_OOS",
        projection=t.candidate_rec_yards_r5p, actual=t.rec_yards,
    ))
    rows.append(_base_output(
        t, position="TE", market="receptions", authority="TE_R5P_PRODUCTION_MODEL_V1",
        scope="exact TE-R5P OOS production-contract rows 2024-2025", benchmark_arm="AUTHORITY_OOS",
        projection=t.candidate_receptions_r5p, actual=t.receptions,
    ))
    out = pd.concat(rows, ignore_index=True)
    _assert_unique(out, ["benchmark_arm", "season", "week", "team", "player_clean_key", "market"], "authority trace")
    return out


def build_production_trace(qb: pd.DataFrame, wr_cand: pd.DataFrame, te: pd.DataFrame,
                           production_trace: pd.DataFrame) -> pd.DataFrame:
    sched = _schedule_from_trace(production_trace)
    rows = []
    q = _attach_schedule(qb, sched, "QB production mean")
    rows.append(_base_output(
        q, position="QB", market="pass_yards", authority="QB_PASS_SYNTHESIS_V1",
        scope="exact canonical M89 OOS mean; current production mean authority",
        benchmark_arm="CURRENT_PRODUCTION_ORDER", projection=q.football_synthesis, actual=q.actual_pass_yards,
    ))

    wr_ids = wr_cand.loc[wr_cand.season.eq(2024), KEYS].drop_duplicates()
    te_ids = te.loc[te.season.isin([2024, 2025]), KEYS].drop_duplicates()

    def filtered(position: str, ids: pd.DataFrame, route: str, authority: str, scope: str) -> pd.DataFrame:
        g = production_trace.loc[
            production_trace.position.eq(position)
            & production_trace.market.isin(["rec_yards", "receptions"])
            & production_trace.wrte_route.astype(str).eq(route)
            & production_trace.wrte_authorized_treatment.fillna(False).astype(bool)
        ].copy()
        g = g.merge(ids.assign(_authority_identity=1), on=KEYS, how="inner", validate="many_to_one")
        expected = len(ids) * 2
        if len(g) != expected:
            raise RuntimeError(f"{authority} production-order coverage drifted: rows={len(g)} expected={expected}")
        _assert_unique(g, [*KEYS, "market"], f"{authority} production-order rows")
        g["model_authority"] = authority
        g["authority_scope"] = scope
        g["benchmark_arm"] = "CURRENT_PRODUCTION_ORDER"
        g["authority_parity_pass"] = 1
        return g

    wrp = filtered(
        "WR", wr_ids, "WR_R15_OOS_PRODUCTION_ORDER",
        "M38_WR1_PLUS_WR_R15_PRODUCTION_MODEL_V1",
        "production-order replay on exact WR-R15 2024 OOS identities with current ensemble weights",
    )
    tep = filtered(
        "TE", te_ids, "TE_R5P_OOS_PRODUCTION_ORDER",
        "TE_R5P_PRODUCTION_MODEL_V1",
        "production-order replay on exact TE-R5P 2024-2025 OOS identities with current ensemble weights",
    )
    keep = ["season", "week", "team", "opponent", "game_id", "player", "player_clean_key", "position", "market",
            "ensemble_proj", "actual", "model_authority", "authority_scope", "benchmark_arm", "authority_parity_pass",
            "mc_proj", "ml_proj", "state_proj", "ensemble_weight_mc", "ensemble_weight_ml", "ensemble_weight_state",
            "ensemble_calibration_rows", "wrte_route", "wrte_authorized_treatment"]
    rows.extend([wrp[[c for c in keep if c in wrp.columns]], tep[[c for c in keep if c in tep.columns]]])
    out = pd.concat(rows, ignore_index=True, sort=False)
    _assert_unique(out, ["benchmark_arm", "season", "week", "team", "player_clean_key", "market"], "production trace")
    return out


def metric_summary(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for keys, g in trace.groupby(["benchmark_arm", "position", "market", "model_authority"], dropna=False):
        err = pd.to_numeric(g.ensemble_proj, errors="coerce") - pd.to_numeric(g.actual, errors="coerce")
        rows.append({
            "benchmark_arm": keys[0], "position": keys[1], "market": keys[2], "model_authority": keys[3],
            "rows": len(g), "mae": float(err.abs().mean()), "rmse": float(np.sqrt(np.mean(err**2))),
            "bias": float(err.mean()), "correlation": float(pd.to_numeric(g.ensemble_proj, errors="coerce").corr(pd.to_numeric(g.actual, errors="coerce"))),
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--qb-casebook", type=Path, required=True)
    ap.add_argument("--wr-predictions", type=Path, required=True)
    ap.add_argument("--te-casebook", type=Path, required=True)
    ap.add_argument("--wrte-replay-integrity", type=Path, required=True)
    ap.add_argument("--wrte-production-trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    qb = _read(a.qb_casebook, "QB exact authority casebook")
    wr = _read(a.wr_predictions, "WR exact authority predictions")
    te = _read(a.te_casebook, "TE exact authority casebook")
    replay = _read(a.wrte_replay_integrity, "WR/TE production-order replay integrity")
    prod = _read(a.wrte_production_trace, "WR/TE current production-order trace")

    audits = [validate_qb(qb)]
    wr_audit, wr_cand = validate_wr(wr)
    audits.append(wr_audit)
    audits.append(validate_te(te))
    audits.append(validate_wrte_replay(replay))
    audits.append({
        "position": "RB", "authority": "RB_P3_R26_R22_WEEK1_2026",
        "scope": "2026 Week-1 prospective only; no 2024-2025 retrospective authority",
        "rows": 0, "baseline_metric": np.nan, "candidate_metric": np.nan,
        "metric": "scope", "status": "PROSPECTIVE_ONLY_NOT_HISTORICALLY_GRADED",
    })

    sched = _schedule_from_trace(prod)
    authority = build_authority_trace(qb, wr_cand, te, sched, prod)
    production = build_production_trace(qb, wr_cand, te, prod)
    combined = pd.concat([authority, production], ignore_index=True, sort=False)

    a.out_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(audits).to_csv(a.out_dir / "authority_parity_audit.csv", index=False)
    authority.to_csv(a.out_dir / "authority_oos_projection_trace.csv", index=False)
    production.to_csv(a.out_dir / "current_production_order_projection_trace.csv", index=False)
    combined.to_csv(a.out_dir / "combined_authority_projection_trace.csv", index=False)
    metric_summary(combined).to_csv(a.out_dir / "football_metric_summary.csv", index=False)

    print("=== PROMOTED AUTHORITY PARITY AUDIT ===")
    print(pd.DataFrame(audits).to_string(index=False))
    print("\n=== FOOTBALL METRIC SUMMARY ===")
    print(metric_summary(combined).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
