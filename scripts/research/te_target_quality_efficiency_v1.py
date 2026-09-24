"""Blind TE target-quality residual study against frozen PR #549 football means.

Frozen plan:
docs/research/TE_TARGET_QUALITY_EFFICIENCY_V1_PREDICTIVE_PLAN.md

No sportsbook input. No 2026 outcomes. No feature/model search.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BASE_METRICS = [
    "avg_separation",
    "avg_cushion",
    "avg_intended_air_yards",
    "avg_expected_yac",
    "avg_yac_above_expectation",
    "percent_share_of_intended_air_yards",
    "ngs_catch_rate",
]
ROLLS = ["last1", "mean3", "mean8", "delta3_8"]
ALPHA = 50.0


def to_pd(x) -> pd.DataFrame:
    if isinstance(x, pd.DataFrame):
        return x.copy()
    if hasattr(x, "to_pandas"):
        return x.to_pandas()
    return pd.DataFrame(x)


def lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def num(x) -> pd.Series:
    return pd.to_numeric(x, errors="coerce")


def first_col(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def full_key(v) -> str:
    return re.sub(r"[^a-z0-9]", "", str(v or "").lower())


def base_key(v) -> str:
    s = re.sub(r"[^a-z0-9 ]", " ", str(v or "").lower())
    toks = [t for t in s.split() if t not in {"jr", "sr", "ii", "iii", "iv", "v"}]
    return "".join(toks)


def load_registry() -> pd.DataFrame:
    import nflreadpy as nfl

    p = lower(to_pd(nfl.load_players()))
    id_col = first_col(p, ["gsis_id", "player_gsis_id", "player_id"])
    name_col = first_col(p, ["display_name", "full_name", "player_name", "football_name"])
    pos_col = first_col(p, ["position", "position_group"])
    if not id_col or not name_col:
        raise RuntimeError("player registry missing id/name")
    q = pd.DataFrame({
        "player_id": p[id_col].astype("string").fillna("").str.strip(),
        "registry_name": p[name_col].astype("string").fillna("").str.strip(),
        "registry_position": p[pos_col].astype("string").fillna("").str.upper().str.strip() if pos_col else "",
    })
    q = q[q.player_id.ne("") & q.registry_name.ne("")].drop_duplicates("player_id")
    q["full_key"] = q.registry_name.map(full_key)
    q["base_key"] = q.registry_name.map(base_key)
    return q


def resolve_target_ids(target: pd.DataFrame, registry: pd.DataFrame) -> pd.DataFrame:
    t = target.copy()
    t["target_full_key"] = t["player"].map(full_key)
    t["target_base_key"] = t["player"].map(base_key)

    full_map = (
        registry.groupby("full_key")["player_id"]
        .agg(lambda s: list(pd.unique(s)))
        .to_dict()
    )
    base_map = (
        registry.groupby("base_key")["player_id"]
        .agg(lambda s: list(pd.unique(s)))
        .to_dict()
    )

    ids = []
    method = []
    for r in t.itertuples(index=False):
        cand = full_map.get(r.target_full_key, [])
        if len(cand) == 1:
            ids.append(cand[0]); method.append("FULL_NAME")
            continue
        cand = base_map.get(r.target_base_key, [])
        if len(cand) == 1:
            ids.append(cand[0]); method.append("BASE_NAME_UNIQUE")
            continue
        ids.append(""); method.append("UNRESOLVED")
    t["player_id"] = ids
    t["identity_method"] = method
    return t


def load_ngs(seasons: list[int], registry: pd.DataFrame) -> pd.DataFrame:
    import nflreadpy as nfl

    q = lower(to_pd(nfl.load_nextgen_stats(seasons=seasons, stat_type="receiving")))
    if "season_type" in q.columns:
        reg = q[q.season_type.fillna("").astype(str).str.upper().eq("REG")].copy()
        if len(reg):
            q = reg

    id_col = first_col(q, ["player_gsis_id", "gsis_id", "player_id"])
    name_col = first_col(q, ["player_display_name", "player_name", "display_name"])
    season_col = first_col(q, ["season"])
    week_col = first_col(q, ["week"])
    if not id_col or not season_col or not week_col:
        raise RuntimeError("NGS receiving missing id/season/week")

    out = pd.DataFrame({
        "player_id": q[id_col].astype("string").fillna("").str.strip(),
        "season": num(q[season_col]),
        "week": num(q[week_col]),
        "player_name": q[name_col].astype("string").fillna("").str.strip() if name_col else "",
    })

    aliases = {
        "avg_separation": ["avg_separation"],
        "avg_cushion": ["avg_cushion"],
        "avg_intended_air_yards": ["avg_intended_air_yards", "avg_air_distance"],
        "avg_expected_yac": ["avg_expected_yac"],
        "avg_yac_above_expectation": ["avg_yac_above_expectation"],
        "percent_share_of_intended_air_yards": ["percent_share_of_intended_air_yards"],
        "targets": ["targets"],
        "receptions": ["receptions"],
    }
    for label, names in aliases.items():
        c = first_col(q, names)
        out[label] = num(q[c]) if c else np.nan

    out["ngs_catch_rate"] = np.divide(
        out["receptions"],
        out["targets"],
        out=np.full(len(out), np.nan, dtype=float),
        where=out["targets"].to_numpy(float) > 0,
    )

    out = out[
        out.player_id.ne("")
        & out.season.notna()
        & out.week.notna()
    ].copy()
    out["season"] = out.season.astype(int)
    out["week"] = out.week.astype(int)
    out = out.sort_values(["player_id", "season", "week"]).drop_duplicates(
        ["player_id", "season", "week"], keep="last"
    )
    return out


def add_features(target: pd.DataFrame, ngs: pd.DataFrame) -> pd.DataFrame:
    by_player = {k: g.sort_values(["season", "week"]) for k, g in ngs.groupby("player_id")}
    rows = []
    for r in target.itertuples(index=False):
        rec = r._asdict()
        hist = by_player.get(r.player_id)
        if hist is None or not r.player_id:
            hist = ngs.iloc[0:0]
        else:
            hist = hist[
                (hist.season < int(r.season))
                | ((hist.season == int(r.season)) & (hist.week < int(r.week)))
            ].copy()
        rec["prior_ngs_games"] = int(len(hist))
        for c in BASE_METRICS:
            vals = num(hist[c]).dropna().to_numpy(float) if c in hist.columns else np.asarray([])
            rec[f"{c}_last1"] = float(vals[-1]) if len(vals) else np.nan
            rec[f"{c}_mean3"] = float(vals[-3:].mean()) if len(vals) else np.nan
            rec[f"{c}_mean8"] = float(vals[-8:].mean()) if len(vals) else np.nan
            m3 = float(vals[-3:].mean()) if len(vals) else np.nan
            m8 = float(vals[-8:].mean()) if len(vals) else np.nan
            rec[f"{c}_delta3_8"] = m3 - m8 if np.isfinite(m3) and np.isfinite(m8) else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def model() -> Pipeline:
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("ridge", Ridge(alpha=ALPHA)),
    ])


def metrics(df: pd.DataFrame, pred: np.ndarray) -> dict:
    actual = num(df.actual).to_numpy(float)
    base = num(df.ensemble_proj).to_numpy(float)
    cand = base + pred
    base_err = actual - base
    cand_err = actual - cand
    finite = np.isfinite(actual) & np.isfinite(base) & np.isfinite(cand)
    actual = actual[finite]; base = base[finite]; cand = cand[finite]
    base_err = base_err[finite]; cand_err = cand_err[finite]
    pred = pred[finite]
    realized_resid = actual - base
    corr = np.corrcoef(pred, realized_resid)[0, 1] if len(actual) > 2 and np.std(pred) > 0 and np.std(realized_resid) > 0 else np.nan
    return {
        "n": int(len(actual)),
        "baseline_mae": float(np.mean(np.abs(base_err))),
        "candidate_mae": float(np.mean(np.abs(cand_err))),
        "mae_gain": float(np.mean(np.abs(base_err)) - np.mean(np.abs(cand_err))),
        "baseline_rmse": float(np.sqrt(np.mean(base_err ** 2))),
        "candidate_rmse": float(np.sqrt(np.mean(cand_err ** 2))),
        "baseline_abs_bias": float(abs(np.mean(base_err))),
        "candidate_abs_bias": float(abs(np.mean(cand_err))),
        "baseline_p90_ae": float(np.quantile(np.abs(base_err), 0.90)),
        "candidate_p90_ae": float(np.quantile(np.abs(cand_err), 0.90)),
        "baseline_miss30": int(np.sum(np.abs(base_err) >= 30.0)),
        "candidate_miss30": int(np.sum(np.abs(cand_err) >= 30.0)),
        "correction_residual_corr": float(corr) if np.isfinite(corr) else None,
    }


def gate(m: dict) -> dict:
    checks = {
        "n_ge_250": m["n"] >= 250,
        "mae_strictly_better": m["candidate_mae"] < m["baseline_mae"],
        "rmse_nonworse": m["candidate_rmse"] <= m["baseline_rmse"] + 1e-12,
        "abs_bias_nonworse": m["candidate_abs_bias"] <= m["baseline_abs_bias"] + 1e-12,
        "p90_nonworse": m["candidate_p90_ae"] <= m["baseline_p90_ae"] + 1e-12,
        "miss30_nonincrease": m["candidate_miss30"] <= m["baseline_miss30"],
        "correction_corr_positive": (m["correction_residual_corr"] or 0.0) > 0.0,
    }
    return {**checks, "pass": all(checks.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frozen-trace", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    trace = pd.read_csv(args.frozen_trace, low_memory=False)
    target = trace[
        trace["position"].astype(str).str.upper().eq("TE")
        & trace["market"].astype(str).eq("rec_yards")
        & pd.to_numeric(trace["season"], errors="coerce").isin([2024, 2025])
    ].copy()
    target["season"] = num(target["season"]).astype(int)
    target["week"] = num(target["week"]).astype(int)
    if target.duplicated(["season", "week", "team", "player_clean_key", "market", "game_id"]).any():
        raise RuntimeError("frozen TE target identity not unique")

    registry = load_registry()
    target = resolve_target_ids(target, registry)
    unresolved = target[target.player_id.eq("")][
        ["season", "week", "team", "player", "player_clean_key", "game_id"]
    ].copy()
    unresolved.to_csv(args.out_dir / "unresolved_identity.csv", index=False)

    ngs = load_ngs(list(range(2020, 2026)), registry)
    data = add_features(target, ngs)
    feature_cols = [f"{c}_{r}" for c in BASE_METRICS for r in ROLLS] + ["prior_ngs_games"]

    scored_parts = []
    fold_results = []
    fold_gates = []
    for train_season, test_season in [(2024, 2025), (2025, 2024)]:
        tr = data[
            data.season.eq(train_season)
            & data.player_id.ne("")
            & num(data.actual).notna()
            & num(data.ensemble_proj).notna()
        ].copy()
        te = data[
            data.season.eq(test_season)
            & data.player_id.ne("")
            & num(data.actual).notna()
            & num(data.ensemble_proj).notna()
        ].copy()
        if len(tr) < 250 or len(te) < 250:
            raise RuntimeError(f"insufficient resolved fold rows train={len(tr)} test={len(te)}")

        y = num(tr.actual).to_numpy(float) - num(tr.ensemble_proj).to_numpy(float)
        pipe = model()
        pipe.fit(tr[feature_cols], y)
        pred = pipe.predict(te[feature_cols])
        m = metrics(te, pred)
        g = gate(m)
        fold_results.append({"train_season": train_season, "test_season": test_season, **m})
        fold_gates.append({"train_season": train_season, "test_season": test_season, **g})

        q = te[[
            "season", "week", "team", "opponent", "player", "player_clean_key",
            "game_id", "ensemble_proj", "actual", "prior_ngs_games"
        ]].copy()
        q["train_season"] = train_season
        q["predicted_residual"] = pred
        q["candidate_proj"] = num(q.ensemble_proj).to_numpy(float) + pred
        scored_parts.append(q)

    scored = pd.concat(scored_parts, ignore_index=True)
    fold_df = pd.DataFrame(fold_results)
    gate_df = pd.DataFrame(fold_gates)
    scored.to_csv(args.out_dir / "blind_scored_rows.csv", index=False)
    fold_df.to_csv(args.out_dir / "fold_metrics.csv", index=False)
    gate_df.to_csv(args.out_dir / "fold_gates.csv", index=False)

    pooled = metrics(scored.rename(columns={"candidate_proj": "_candidate_tmp"}), scored["predicted_residual"].to_numpy(float))
    # metrics() reconstructs candidate as ensemble + predicted residual, so the renamed candidate column is not consumed.
    pooled_gain = pooled["candidate_mae"] < pooled["baseline_mae"]
    both_pass = bool(gate_df["pass"].all())
    if both_pass and pooled_gain:
        disposition = "TE_TARGET_QUALITY_EFFICIENCY_V1_REPLICATES"
    elif bool(gate_df["pass"].any()):
        disposition = "TE_TARGET_QUALITY_EFFICIENCY_V1_NONREPLICATING_SIGNAL"
    else:
        disposition = "TE_TARGET_QUALITY_EFFICIENCY_V1_FAIL"

    payload = {
        "study": "TE_TARGET_QUALITY_EFFICIENCY_V1",
        "disposition": disposition,
        "frozen_authority": "PR549",
        "model": "StandardScaler+MedianImputer+Ridge(alpha=50)",
        "feature_count": len(feature_cols),
        "features": feature_cols,
        "unresolved_identity_rows": int(len(unresolved)),
        "folds": fold_results,
        "fold_gates": fold_gates,
        "pooled": pooled,
        "pooled_mae_improves": bool(pooled_gain),
        "sportsbook_inputs_used": 0,
        "outcomes_2026_used": 0,
        "production_changed": False,
    }
    (args.out_dir / "summary.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
