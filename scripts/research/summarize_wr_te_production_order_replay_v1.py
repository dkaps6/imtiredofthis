#!/usr/bin/env python3
"""Compare canonical empirical baseline to WR/TE production-order replay.

Research-only summarizer for Issue #535 checkpoint 18. It fails closed if row
identity or any untreated row changes, then reports the frozen specialist-
authorized cohort without threshold or cohort search.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.research.grade_empirical_fair_prob_v1 import _probability_diagnostics, _summarize
from scripts.operations.grade_market_track_record_v1 import num
from scripts.utils.canonical_names import canon_team

BASE_KEYS = ["season", "week", "team", "opponent", "player_clean_key", "market"]
RECEIVING_MARKETS = {"rec_yards", "receptions", "rush_rec_yards"}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _pos(value) -> str:
    p = "" if value is None or pd.isna(value) else str(value).upper().strip()
    if p.startswith("WR") or p in {"LWR", "RWR", "SWR"}:
        return "WR"
    if p.startswith("TE"):
        return "TE"
    return p


def _authorized(frame: pd.DataFrame) -> pd.Series:
    pos = frame["position"].map(_pos)
    market = frame["market"].astype(str).str.lower()
    season = pd.to_numeric(frame["season"], errors="raise").astype(int)
    return market.isin(RECEIVING_MARKETS) & (
        (season.eq(2024) & pos.isin(["WR", "TE"]))
        | (season.eq(2025) & pos.eq("TE"))
    )


def _canon(x: pd.DataFrame) -> pd.DataFrame:
    out = x.copy()
    out["season"] = pd.to_numeric(out["season"], errors="raise").astype(int)
    out["week"] = pd.to_numeric(out["week"], errors="raise").astype(int)
    out["team"] = out["team"].map(canon_team)
    out["opponent"] = out["opponent"].map(canon_team)
    out["player_clean_key"] = out["player_clean_key"].astype(str)
    out["market"] = out["market"].astype(str).str.lower()
    return out


def _row_identity(x: pd.DataFrame) -> pd.DataFrame:
    cols = BASE_KEYS + [c for c in ("game_id", "line") if c in x.columns]
    return x[cols].sort_values(cols).reset_index(drop=True)


def _tag(summary: pd.DataFrame, arm: str) -> pd.DataFrame:
    x = summary.copy()
    x.insert(0, "arm", arm)
    return x


def _group_diagnostics(x: pd.DataFrame, arm: str) -> pd.DataFrame:
    rows: list[dict] = []
    y = x.copy()
    y["position_family"] = y["position"].map(_pos)
    scopes: list[tuple[str, object]] = []
    for season in sorted(y["season"].unique()):
        scopes.append(("season", int(season)))
    for market in sorted(y["market"].unique()):
        scopes.append(("market", str(market)))
    for position in sorted(y["position_family"].unique()):
        scopes.append(("position", str(position)))
    scopes.append(("all", "ALL_AUTHORIZED"))
    for kind, value in scopes:
        if kind == "season":
            g = y.loc[y["season"].eq(value)]
        elif kind == "market":
            g = y.loc[y["market"].eq(value)]
        elif kind == "position":
            g = y.loc[y["position_family"].eq(value)]
        else:
            g = y
        if g.empty:
            continue
        decided = g.loc[g["actual_side"].ne("PUSH")].copy()
        p = np.clip(num(decided["p_over"]).to_numpy(float), 1e-6, 1 - 1e-6)
        actual_over = (num(decided["actual"]) > num(decided["line"])).astype(float).to_numpy()
        brier = float(np.mean((p - actual_over) ** 2)) if len(decided) else np.nan
        logloss = float(-np.mean(actual_over*np.log(p)+(1-actual_over)*np.log(1-p))) if len(decided) else np.nan
        residual = num(g["actual"]) - num(g["proj"])
        mean_sd = float(num(g["model_sd"]).mean()) if "model_sd" in g.columns else np.nan
        residual_sd = float(residual.std(ddof=1)) if len(g) > 1 else np.nan
        strong = g.loc[g["signal"].eq("STRONG_EDGE")]
        strong_decided = strong.loc[strong["bet_result"].isin(["WIN", "LOSS"])]
        rows.append({
            "arm": arm,
            "scope": kind,
            "value": value,
            "rows": int(len(g)),
            "mean_mae": float(num(g["model_error"]).abs().mean()),
            "signed_bias_actual_minus_proj": float(residual.mean()),
            "brier_over": brier,
            "log_loss_over": logloss,
            "mean_model_sd": mean_sd,
            "residual_sd": residual_sd,
            "sd_to_residual_ratio": float(mean_sd / residual_sd) if np.isfinite(mean_sd) and np.isfinite(residual_sd) and residual_sd > 0 else np.nan,
            "strong_rows": int(len(strong)),
            "strong_coverage": float(len(strong) / len(g)),
            "strong_win_rate": float(strong_decided["bet_result"].eq("WIN").mean()) if len(strong_decided) else np.nan,
            "strong_roi_per_unit": float(num(strong_decided["unit_result"]).mean()) if len(strong_decided) else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline-detail", type=Path, required=True)
    ap.add_argument("--specialist-detail", type=Path, required=True)
    ap.add_argument("--baseline-projection", type=Path, required=True)
    ap.add_argument("--specialist-projection", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    base = _canon(_read(a.baseline_detail, "baseline empirical detail"))
    spec = _canon(_read(a.specialist_detail, "specialist empirical detail"))
    _ = _canon(_read(a.baseline_projection, "baseline projection trace"))
    spec_proj = _canon(_read(a.specialist_projection, "specialist projection trace"))

    if len(base) != 17715 or len(spec) != 17715:
        raise RuntimeError(f"graded row-count drift baseline={len(base)} specialist={len(spec)}")
    if not _row_identity(base).equals(_row_identity(spec)):
        raise RuntimeError("specialist replay changed graded row identity")

    route_cols = BASE_KEYS + [
        c for c in ("position", "wrte_authorized_treatment", "wrte_route") if c in spec_proj.columns
    ]
    route = spec_proj[route_cols].drop_duplicates(BASE_KEYS)
    if route.duplicated(BASE_KEYS).any():
        raise RuntimeError("specialist projection route identity is not unique")
    for label in ("base", "spec"):
        frame = base if label == "base" else spec
        if "position" not in frame.columns or "wrte_authorized_treatment" not in frame.columns:
            merge_cols = [c for c in route.columns if c not in frame.columns or c in BASE_KEYS]
            frame = frame.merge(route[merge_cols], on=BASE_KEYS, how="left", validate="one_to_one")
            if label == "base":
                base = frame
            else:
                spec = frame

    if "position" not in spec.columns:
        raise RuntimeError("specialist detail lacks position lineage")
    auth = _authorized(spec)
    if int(auth.sum()) == 0:
        raise RuntimeError("frozen specialist-authorized cohort is empty")
    if "wrte_authorized_treatment" in spec.columns:
        routed = spec["wrte_authorized_treatment"].astype(str).str.lower().isin(["true", "1"])
        if not routed.equals(auth):
            bad = spec.loc[routed.ne(auth), BASE_KEYS + ["position", "wrte_authorized_treatment"]].head(20)
            raise RuntimeError(f"frozen cohort != persisted treatment route: {bad.to_dict('records')}")

    pos = spec["position"].map(_pos)
    wr2025 = spec["season"].eq(2025) & pos.eq("WR") & spec["market"].isin(RECEIVING_MARKETS)
    if auth.loc[wr2025].any():
        raise RuntimeError("2025 WR-R15 rows entered treatment despite frozen exclusion")

    keys = BASE_KEYS + [c for c in ("game_id", "line") if c in spec.columns and c in base.columns]
    left = base[keys + ["proj", "p_over", "model_sd"]].rename(
        columns={"proj":"proj_base", "p_over":"p_over_base", "model_sd":"model_sd_base"}
    )
    right = spec[keys + ["proj", "p_over", "model_sd", "position"]].rename(
        columns={"proj":"proj_spec", "p_over":"p_over_spec", "model_sd":"model_sd_spec"}
    )
    z = right.merge(left, on=keys, validate="one_to_one")
    z["authorized"] = _authorized(z)
    untreated = z.loc[~z["authorized"]]
    diffs = {
        "proj": float((num(untreated["proj_spec"]) - num(untreated["proj_base"])).abs().max()),
        "p_over": float((num(untreated["p_over_spec"]) - num(untreated["p_over_base"])).abs().max()),
        "model_sd": float((num(untreated["model_sd_spec"]) - num(untreated["model_sd_base"])).abs().max()),
    }
    if any((not np.isfinite(v)) or v > 1e-10 for v in diffs.values()):
        raise RuntimeError(f"untreated baseline invariance failed: {diffs}")

    b = base.loc[auth.to_numpy()].copy()
    s = spec.loc[auth.to_numpy()].copy()
    if not _row_identity(b).equals(_row_identity(s)):
        raise RuntimeError("authorized baseline/specialist rows are not paired")

    auth_summary = pd.concat([
        _tag(_summarize(b), "BASE_EMPIRICAL"),
        _tag(_summarize(s), "WR_TE_PRODUCTION_ORDER"),
    ], ignore_index=True)
    auth_prob = pd.concat([
        _tag(_probability_diagnostics(b, "BASE_EMPIRICAL"), "BASE_EMPIRICAL"),
        _tag(_probability_diagnostics(s, "WR_TE_PRODUCTION_ORDER"), "WR_TE_PRODUCTION_ORDER"),
    ], ignore_index=True)
    diagnostics = pd.concat([
        _group_diagnostics(b, "BASE_EMPIRICAL"),
        _group_diagnostics(s, "WR_TE_PRODUCTION_ORDER"),
    ], ignore_index=True)
    whole_summary = pd.concat([
        _tag(_summarize(base), "BASE_EMPIRICAL"),
        _tag(_summarize(spec), "WR_TE_PRODUCTION_ORDER"),
    ], ignore_index=True)

    paired = s[BASE_KEYS + ["position", "proj", "p_over", "model_sd", "signal", "unit_result"]].copy()
    paired = paired.rename(columns={c:f"{c}_specialist" for c in ["proj","p_over","model_sd","signal","unit_result"]})
    bp = b[BASE_KEYS + ["proj", "p_over", "model_sd", "signal", "unit_result", "actual"]].copy()
    bp = bp.rename(columns={c:f"{c}_baseline" for c in ["proj","p_over","model_sd","signal","unit_result"]})
    paired = paired.merge(bp, on=BASE_KEYS, validate="one_to_one")
    paired["projection_delta"] = num(paired["proj_specialist"]) - num(paired["proj_baseline"])
    paired["p_over_delta"] = num(paired["p_over_specialist"]) - num(paired["p_over_baseline"])
    paired["sd_delta"] = num(paired["model_sd_specialist"]) - num(paired["model_sd_baseline"])

    integrity = pd.DataFrame([{
        "graded_rows": int(len(spec)),
        "authorized_rows": int(auth.sum()),
        "untreated_rows": int((~auth).sum()),
        "wr_2025_receiving_rows": int(wr2025.sum()),
        "wr_2025_treated_rows": int((wr2025 & auth).sum()),
        "untreated_max_abs_proj_delta": diffs["proj"],
        "untreated_max_abs_p_over_delta": diffs["p_over"],
        "untreated_max_abs_model_sd_delta": diffs["model_sd"],
        "status": "PASS",
    }])

    a.out_dir.mkdir(parents=True, exist_ok=True)
    integrity.to_csv(a.out_dir / "wrte_replay_integrity_audit.csv", index=False)
    auth_summary.to_csv(a.out_dir / "wrte_authorized_summary.csv", index=False)
    auth_prob.to_csv(a.out_dir / "wrte_authorized_probability_diagnostics.csv", index=False)
    diagnostics.to_csv(a.out_dir / "wrte_authorized_scope_diagnostics.csv", index=False)
    whole_summary.to_csv(a.out_dir / "wrte_whole_cohort_summary.csv", index=False)
    paired.to_csv(a.out_dir / "wrte_authorized_paired_rows.csv", index=False)

    print("=== WR/TE AUTHORIZED STRONG SUMMARY ===")
    print(auth_summary.loc[auth_summary["tier"].eq("STRONG_ONLY_PLAY_TIER")].to_string(index=False))
    print("\n=== WR/TE AUTHORIZED DIAGNOSTICS ===")
    print(diagnostics.to_string(index=False))
    print("\n=== INTEGRITY ===")
    print(integrity.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
