#!/usr/bin/env python3
"""Frozen 2x2 interaction: heldout ensemble weights x probability translator.

A0 = current/fallback means x legacy component_sd Normal translator.
A1 = PR #545 published heldout means x legacy translator.
B0 = current/fallback means x empirical historical MC translator.
B1 = PR #545 published heldout means x empirical historical MC translator.

Important reproduction note: PR #545's published A1 result used a fixed weighted
sum with missing components contributing zero, rather than production
apply_ensemble() missing-component renormalization. The 2x2 preserves that exact
published-A1 mean construction in A1/B1 so the interaction tests the result that
was actually reported. This limitation is audited explicitly and does not alter
production.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest.grade_full_stack_vegas_benchmark_v1 import grade as legacy_grade
from scripts.research.grade_empirical_fair_prob_v1 import _probability_diagnostics, grade_empirical

TARGET_MARKETS = ("rec_yards", "receptions", "rush_rec_yards")
IDENTITY = ["season", "week", "team", "opponent", "player_clean_key", "market", "game_id"]
SUMMARY_KEYS = ["market", "tier"]
SUMMARY_NUMERIC = [
    "matched_rows", "decided_bets", "wins", "losses", "win_rate", "units",
    "roi_per_unit", "model_mae", "vegas_mae",
]


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def build_overlay_weights(current: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    cur = current.copy()
    ho = heldout.copy()
    cur["market"] = cur["market"].astype(str).str.lower().str.strip()
    ho["market"] = ho["market"].astype(str).str.lower().str.strip()
    missing = sorted(set(TARGET_MARKETS) - set(ho["market"]))
    if missing:
        raise RuntimeError(f"heldout weight artifact missing frozen markets: {missing}")
    chosen = ho.loc[ho["market"].isin(TARGET_MARKETS)].copy()
    if chosen.duplicated("market").any():
        raise RuntimeError("heldout weight artifact has duplicate target markets")
    out = pd.concat([cur.loc[~cur["market"].isin(TARGET_MARKETS)], chosen], ignore_index=True, sort=False)
    if out.duplicated("market").any():
        raise RuntimeError("overlay weights contain duplicate markets")
    return out


def build_published_a1_trace(current: pd.DataFrame, heldout_weights: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Reproduce PR #545's published A1 mean construction exactly.

    Missing component projections contribute zero and weights are not renormalized.
    This is intentionally *not* production apply_ensemble() semantics; it is the
    discovered historical application semantics required to reproduce A1 exactly.
    """
    out = current.copy()
    w = heldout_weights.copy()
    w["market"] = w["market"].astype(str).str.lower().str.strip()
    missing = sorted(set(TARGET_MARKETS) - set(w["market"]))
    if missing:
        raise RuntimeError(f"heldout weight artifact missing frozen markets: {missing}")
    w = w.set_index("market")
    audit_rows: list[dict] = []

    for market in TARGET_MARKETS:
        mask = out["market"].astype(str).str.lower().eq(market)
        row = w.loc[market]
        wm = float(row["mc_weight"])
        wl = float(row["ml_weight"])
        ws = float(row["state_weight"])
        weight_sum = wm + wl + ws
        if not np.isfinite(weight_sum) or abs(weight_sum - 1.0) > 1e-6:
            raise RuntimeError(f"heldout weights for {market} do not sum to 1: {weight_sum}")

        mc = pd.to_numeric(out.loc[mask, "mc_proj"], errors="coerce")
        ml = pd.to_numeric(out.loc[mask, "ml_proj"], errors="coerce")
        state = pd.to_numeric(out.loc[mask, "state_proj"], errors="coerce")
        if mc.isna().any() or ml.isna().any():
            raise RuntimeError(f"published-A1 reproduction found missing mc/ml projection in {market}")
        published_mean = mc * wm + ml * wl + state.fillna(0.0) * ws
        out.loc[mask, "ensemble_proj"] = published_mean.to_numpy(dtype=float)
        out.loc[mask, "ensemble_weight_mc"] = wm
        out.loc[mask, "ensemble_weight_ml"] = wl
        out.loc[mask, "ensemble_weight_state"] = ws
        if "ensemble_calibration_rows" in out.columns and "calibration_rows" in row.index:
            out.loc[mask, "ensemble_calibration_rows"] = int(row["calibration_rows"])
        if "ensemble_method" in out.columns:
            out.loc[mask, "ensemble_method"] = "PR545_PUBLISHED_FIXED_WEIGHT_NO_MISSING_RENORM"
        if "ensemble_status" in out.columns:
            out.loc[mask, "ensemble_status"] = "research_pr545_published_semantics"

        audit_rows.append({
            "market": market,
            "rows": int(mask.sum()),
            "missing_mc_rows": int(mc.isna().sum()),
            "missing_ml_rows": int(ml.isna().sum()),
            "missing_state_rows": int(state.isna().sum()),
            "state_weight": ws,
            "rows_affected_by_missing_state_semantics": int((state.isna() & (abs(ws) > 0)).sum()),
            "weight_sum": weight_sum,
            "application_semantics": "fixed_weight_missing_component_zero_no_renormalization",
        })
    return out, pd.DataFrame(audit_rows)


def _assert_trace_identity(a: pd.DataFrame, b: pd.DataFrame) -> None:
    aa = a[IDENTITY].copy().sort_values(IDENTITY).reset_index(drop=True)
    bb = b[IDENTITY].copy().sort_values(IDENTITY).reset_index(drop=True)
    if len(aa) != len(bb) or not aa.equals(bb):
        raise RuntimeError(f"projection trace identity changed under heldout means: {len(aa)} vs {len(bb)}")


def _projection_change_audit(current: pd.DataFrame, heldout: pd.DataFrame) -> pd.DataFrame:
    keys = IDENTITY
    a = current[keys + ["ensemble_proj"]].rename(columns={"ensemble_proj": "ensemble_proj_current"})
    b = heldout[keys + ["ensemble_proj"]].rename(columns={"ensemble_proj": "ensemble_proj_heldout"})
    z = a.merge(b, on=keys, validate="one_to_one")
    z["proj_abs_delta"] = (
        pd.to_numeric(z["ensemble_proj_heldout"], errors="coerce")
        - pd.to_numeric(z["ensemble_proj_current"], errors="coerce")
    ).abs()
    non_target = z.loc[~z["market"].isin(TARGET_MARKETS)]
    if (non_target["proj_abs_delta"] > 1e-12).any():
        bad = non_target.loc[non_target["proj_abs_delta"] > 1e-12, keys + ["proj_abs_delta"]].head(10)
        raise RuntimeError(f"heldout means changed non-target rows: {bad.to_dict('records')}")
    return z.groupby("market", as_index=False).agg(
        rows=("proj_abs_delta", "size"),
        changed_rows=("proj_abs_delta", lambda s: int((s > 1e-12).sum())),
        mean_abs_projection_delta=("proj_abs_delta", "mean"),
        max_abs_projection_delta=("proj_abs_delta", "max"),
    )


def _identity_from_detail(detail: pd.DataFrame) -> pd.DataFrame:
    cols = IDENTITY + ["line"]
    return detail[cols].sort_values(cols).reset_index(drop=True)


def _assert_four_cell_row_parity(details: dict[str, pd.DataFrame]) -> None:
    names = list(details)
    base = _identity_from_detail(details[names[0]])
    for name in names[1:]:
        other = _identity_from_detail(details[name])
        if len(base) != len(other) or not base.equals(other):
            raise RuntimeError(f"ALL_NO_FILTER row identity differs between {names[0]} and {name}")


def _summary_diff(observed: pd.DataFrame, reference: pd.DataFrame, label: str) -> pd.DataFrame:
    o = observed.copy().sort_values(SUMMARY_KEYS).reset_index(drop=True)
    r = reference.copy().sort_values(SUMMARY_KEYS).reset_index(drop=True)
    z = r.merge(o, on=SUMMARY_KEYS, how="outer", suffixes=("_reference", "_observed"), indicator=True)
    z.insert(0, "label", label)
    for col in SUMMARY_NUMERIC:
        rc = f"{col}_reference"
        oc = f"{col}_observed"
        if rc in z and oc in z:
            z[f"{col}_delta"] = pd.to_numeric(z[oc], errors="coerce") - pd.to_numeric(z[rc], errors="coerce")
    return z


def _assert_summary_reproduces(observed: pd.DataFrame, reference: pd.DataFrame, label: str, tol: float = 1e-10) -> pd.DataFrame:
    diff = _summary_diff(observed, reference, label)
    if not diff["_merge"].eq("both").all():
        raise RuntimeError(f"{label} summary keys do not reproduce canonical reference")
    for col in SUMMARY_NUMERIC:
        d = pd.to_numeric(diff[f"{col}_delta"], errors="coerce").to_numpy(dtype=float)
        if not np.all(np.isnan(d) | (np.abs(d) <= tol)):
            delta = np.nanmax(np.abs(d))
            raise RuntimeError(f"{label} failed canonical reproduction at {col}; max_abs_delta={delta}")
    return diff


def _tag_summary(summary: pd.DataFrame, cell: str, weight_arm: str, prob_arm: str) -> pd.DataFrame:
    x = summary.copy()
    x.insert(0, "cell", cell)
    x.insert(1, "weight_arm", weight_arm)
    x.insert(2, "probability_arm", prob_arm)
    return x


def _tag_diag(detail: pd.DataFrame, cell: str, weight_arm: str, prob_arm: str) -> pd.DataFrame:
    x = _probability_diagnostics(detail, f"{cell}_{prob_arm}")
    x.insert(0, "cell", cell)
    x.insert(1, "weight_arm", weight_arm)
    x.insert(2, "probability_arm", prob_arm)
    return x


def _strong_side_counts(detail: pd.DataFrame, cell: str) -> pd.DataFrame:
    x = detail.loc[detail["signal"].eq("STRONG_EDGE")].copy()
    rows = []
    for market in list(x["market"].dropna().unique()) + ["ALL_MARKETS"]:
        g = x if market == "ALL_MARKETS" else x.loc[x["market"].eq(market)]
        for side in ["OVER", "UNDER"]:
            s = g.loc[g["side"].eq(side)]
            rows.append({
                "cell": cell,
                "market": market,
                "side": side,
                "strong_rows": int(len(s)),
                "wins": int(s["bet_result"].eq("WIN").sum()),
                "losses": int(s["bet_result"].eq("LOSS").sum()),
                "win_rate": float(s["bet_result"].eq("WIN").mean()) if len(s) else np.nan,
                "units": float(pd.to_numeric(s["unit_result"], errors="coerce").sum()) if len(s) else 0.0,
                "roi_per_unit": float(pd.to_numeric(s["unit_result"], errors="coerce").mean()) if len(s) else np.nan,
            })
    return pd.DataFrame(rows)


def _interaction_deltas(cell_summary: pd.DataFrame, probability_diag: pd.DataFrame) -> pd.DataFrame:
    strong = cell_summary.loc[cell_summary["tier"].eq("STRONG_ONLY_PLAY_TIER")].copy()
    rows = []
    for market in strong["market"].unique():
        g = strong.loc[strong["market"].eq(market)].set_index("cell")
        if not {"A0", "A1", "B0", "B1"}.issubset(g.index):
            continue
        row = {"market": market, "tier": "STRONG_ONLY_PLAY_TIER"}
        for metric in ["model_mae", "roi_per_unit", "win_rate", "matched_rows"]:
            v = {c: float(g.loc[c, metric]) for c in ("A0", "A1", "B0", "B1")}
            row[f"{metric}_weight_effect_legacy_A1_minus_A0"] = v["A1"] - v["A0"]
            row[f"{metric}_translator_effect_current_B0_minus_A0"] = v["B0"] - v["A0"]
            row[f"{metric}_translator_effect_heldout_B1_minus_A1"] = v["B1"] - v["A1"]
            row[f"{metric}_weight_effect_empirical_B1_minus_B0"] = v["B1"] - v["B0"]
            row[f"{metric}_interaction"] = (v["B1"] - v["A1"]) - (v["B0"] - v["A0"])
        rows.append(row)
    diag = probability_diag.set_index(["market", "cell"])
    for row in rows:
        market = row["market"]
        for metric in ("brier_over", "log_loss_over", "strong_coverage"):
            if not all((market, c) in diag.index for c in ("A0", "A1", "B0", "B1")):
                continue
            v = {c: float(diag.loc[(market, c), metric]) for c in ("A0", "A1", "B0", "B1")}
            row[f"{metric}_weight_effect_legacy_A1_minus_A0"] = v["A1"] - v["A0"]
            row[f"{metric}_translator_effect_current_B0_minus_A0"] = v["B0"] - v["A0"]
            row[f"{metric}_translator_effect_heldout_B1_minus_A1"] = v["B1"] - v["A1"]
            row[f"{metric}_weight_effect_empirical_B1_minus_B0"] = v["B1"] - v["B0"]
            row[f"{metric}_interaction"] = (v["B1"] - v["A1"]) - (v["B0"] - v["A0"])
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--projection-trace", type=Path, required=True)
    ap.add_argument("--props", type=Path, required=True)
    ap.add_argument("--distribution-dir", type=Path, required=True)
    ap.add_argument("--heldout-weights", type=Path, required=True)
    ap.add_argument("--current-weights", type=Path, required=True)
    ap.add_argument("--reference-grade-dir", type=Path, required=True)
    ap.add_argument("--a1-reference", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    current_trace = _read(a.projection_trace, "clean projection trace")
    props = _read(a.props, "historical props")
    current_weights = _read(a.current_weights, "current ensemble weights")
    heldout_weights = _read(a.heldout_weights, "2023 heldout ensemble weights")
    overlay = build_overlay_weights(current_weights, heldout_weights)
    heldout_trace, semantics_audit = build_published_a1_trace(current_trace, heldout_weights)
    _assert_trace_identity(current_trace, heldout_trace)
    projection_audit = _projection_change_audit(current_trace, heldout_trace)

    a0_detail, a0_summary = legacy_grade(current_trace, props, proj_col="ensemble_proj")
    a1_detail, a1_summary = legacy_grade(heldout_trace, props, proj_col="ensemble_proj")
    b0_detail, b0_summary, b0_legacy_summary, *_ = grade_empirical(current_trace, props, distribution_dir=a.distribution_dir, proj_col="ensemble_proj")
    b1_detail, b1_summary, b1_legacy_summary, *_ = grade_empirical(heldout_trace, props, distribution_dir=a.distribution_dir, proj_col="ensemble_proj")
    _assert_four_cell_row_parity({"A0": a0_detail, "A1": a1_detail, "B0": b0_detail, "B1": b1_detail})

    ref_a0 = _read(a.reference_grade_dir / "legacy_component_sd_summary.csv", "A0 canonical reference")
    ref_b0 = _read(a.reference_grade_dir / "empirical_fair_prob_summary.csv", "B0 canonical reference")
    ref_a1 = _read(a.a1_reference, "A1 canonical reference")
    reproduction = []
    for observed, reference, label in [
        (a0_summary, ref_a0, "A0"),
        (b0_summary, ref_b0, "B0"),
        (a1_summary, ref_a1, "A1"),
        (b0_legacy_summary, ref_a0, "B0_internal_legacy"),
        (b1_legacy_summary, ref_a1, "B1_internal_legacy"),
    ]:
        reproduction.append(_assert_summary_reproduces(observed, reference, label))

    summaries = pd.concat([
        _tag_summary(a0_summary, "A0", "CURRENT", "LEGACY_COMPONENT_SD"),
        _tag_summary(a1_summary, "A1", "HELDOUT_2023_PUBLISHED", "LEGACY_COMPONENT_SD"),
        _tag_summary(b0_summary, "B0", "CURRENT", "EMPIRICAL_MC"),
        _tag_summary(b1_summary, "B1", "HELDOUT_2023_PUBLISHED", "EMPIRICAL_MC"),
    ], ignore_index=True)
    diagnostics = pd.concat([
        _tag_diag(a0_detail, "A0", "CURRENT", "LEGACY_COMPONENT_SD"),
        _tag_diag(a1_detail, "A1", "HELDOUT_2023_PUBLISHED", "LEGACY_COMPONENT_SD"),
        _tag_diag(b0_detail, "B0", "CURRENT", "EMPIRICAL_MC"),
        _tag_diag(b1_detail, "B1", "HELDOUT_2023_PUBLISHED", "EMPIRICAL_MC"),
    ], ignore_index=True)
    sides = pd.concat([
        _strong_side_counts(a0_detail, "A0"),
        _strong_side_counts(a1_detail, "A1"),
        _strong_side_counts(b0_detail, "B0"),
        _strong_side_counts(b1_detail, "B1"),
    ], ignore_index=True)
    deltas = _interaction_deltas(summaries, diagnostics)
    focal = deltas.loc[deltas["market"].eq("rush_rec_yards")].copy()
    if len(focal) != 1:
        raise RuntimeError("missing unique rush_rec_yards interaction row")

    overlay.to_csv(a.out_dir / "combined_frozen_weights_v1.csv", index=False)
    heldout_trace.to_csv(a.out_dir / "heldout_projection_trace_v1.csv", index=False)
    semantics_audit.to_csv(a.out_dir / "published_a1_application_semantics_audit.csv", index=False)
    projection_audit.to_csv(a.out_dir / "projection_change_audit.csv", index=False)
    pd.concat(reproduction, ignore_index=True).to_csv(a.out_dir / "canonical_reproduction_audit.csv", index=False)
    summaries.to_csv(a.out_dir / "interaction_cell_summary.csv", index=False)
    diagnostics.to_csv(a.out_dir / "interaction_probability_diagnostics.csv", index=False)
    sides.to_csv(a.out_dir / "interaction_strong_side_counts.csv", index=False)
    deltas.to_csv(a.out_dir / "interaction_deltas.csv", index=False)
    focal.to_csv(a.out_dir / "rush_rec_yards_focal_interaction.csv", index=False)
    b1_detail.to_csv(a.out_dir / "B1_empirical_heldout_detail.csv", index=False)

    print("=== 2x2 STRONG CELL SUMMARY ===")
    print(summaries.loc[summaries["tier"].eq("STRONG_ONLY_PLAY_TIER")].to_string(index=False))
    print("\n=== RUSH_REC_YARDS FOCAL INTERACTION ===")
    print(focal.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
