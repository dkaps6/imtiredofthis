#!/usr/bin/env python3
"""Pool RB R24 folds and apply the pre-frozen R24 promotion gates."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def metric(actual, pred) -> dict:
    z = pd.DataFrame(
        {"actual": pd.to_numeric(actual, errors="coerce"), "pred": pd.to_numeric(pred, errors="coerce")}
    ).dropna()
    z = z[np.isfinite(z.actual) & np.isfinite(z.pred)]
    e = z.pred - z.actual
    ae = e.abs()
    return {
        "n": int(len(z)),
        "mae": float(ae.mean()),
        "rmse": float(np.sqrt(np.mean(e.to_numpy() ** 2))),
        "bias": float(e.mean()),
        "pearson": float(z.pred.corr(z.actual, method="pearson")) if len(z) > 1 else np.nan,
        "spearman": float(z.pred.corr(z.actual, method="spearman")) if len(z) > 1 else np.nan,
        "median_abs_error": float(ae.median()),
        "p75_abs_error": float(ae.quantile(0.75)),
        "p90_abs_error": float(ae.quantile(0.90)),
        "miss20_rate": float((ae >= 20).mean()),
        "miss30_rate": float((ae >= 30).mean()),
        "miss40_rate": float((ae >= 40).mean()),
    }


def pct_change(candidate: float, baseline: float) -> float:
    return (candidate - baseline) / baseline if baseline > 0 else 0.0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="+", required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()

    frames = []
    conservation = []
    for d in a.dirs:
        root = Path(d)
        frames.append(pd.read_csv(root / "r24_predictions.csv", low_memory=False))
        conservation.append(pd.read_csv(root / "r24_conservation_audit.csv", low_memory=False))
    x = pd.concat(frames, ignore_index=True)
    ca = pd.concat(conservation, ignore_index=True)

    if int(x.sportsbook_inputs_used.fillna(0).sum()) != 0:
        raise RuntimeError("sportsbook integrity violation")
    if int(x.future_outcomes_used.fillna(0).sum()) != 0:
        raise RuntimeError("future-outcome integrity violation")
    if float(ca.room_mass_gap.abs().max()) >= 1e-10:
        raise RuntimeError("RB-room conservation violation")
    if float(x.production_ypt_gap.abs().max()) != 0.0:
        raise RuntimeError("production receiving-efficiency parity violation")

    rows: list[dict] = []
    for season in sorted(x.season.unique()):
        g = x[x.season.eq(season)]
        for variant in ("baseline", "candidate"):
            for market, actual_col in (
                ("targets", "actual_targets"),
                ("receptions", "actual_receptions"),
                ("rec_yards", "actual_rec_yards"),
            ):
                rows.append(
                    {
                        "scope": str(int(season)),
                        "role": "ALL",
                        "variant": variant,
                        "market": market,
                        **metric(g[actual_col], g[f"{variant}_{market}"]),
                    }
                )
    for role, g in x.groupby("role"):
        for variant in ("baseline", "candidate"):
            rows.append(
                {
                    "scope": "POOLED",
                    "role": role,
                    "variant": variant,
                    "market": "rec_yards",
                    **metric(g.actual_rec_yards, g[f"{variant}_rec_yards"]),
                }
            )
    for variant in ("baseline", "candidate"):
        for market, actual_col in (
            ("targets", "actual_targets"),
            ("receptions", "actual_receptions"),
            ("rec_yards", "actual_rec_yards"),
        ):
            rows.append(
                {
                    "scope": "POOLED",
                    "role": "ALL",
                    "variant": variant,
                    "market": market,
                    **metric(x[actual_col], x[f"{variant}_{market}"]),
                }
            )
    summary = pd.DataFrame(rows)

    def row(scope: str | int, variant: str, market: str, role: str = "ALL"):
        return summary[
            summary.scope.eq(str(scope))
            & summary.variant.eq(variant)
            & summary.market.eq(market)
            & summary.role.eq(role)
        ].iloc[0]

    bt, ct = row("POOLED", "baseline", "targets"), row("POOLED", "candidate", "targets")
    br, cr = row("POOLED", "baseline", "receptions"), row("POOLED", "candidate", "receptions")
    by, cy = row("POOLED", "baseline", "rec_yards"), row("POOLED", "candidate", "rec_yards")

    season_changes: dict[str, float] = {}
    season_improves = 0
    max_season_worsen = 0.0
    for season in sorted(x.season.unique()):
        b = row(int(season), "baseline", "rec_yards")
        c = row(int(season), "candidate", "rec_yards")
        change = pct_change(float(c.mae), float(b.mae))
        season_changes[str(int(season))] = change
        season_improves += int(float(c.mae) < float(b.mae))
        max_season_worsen = max(max_season_worsen, change)

    role_changes: dict[str, float] = {}
    role_gate = True
    for role in ("RB1", "RB2+"):
        b = row("POOLED", "baseline", "rec_yards", role)
        c = row("POOLED", "candidate", "rec_yards", role)
        change = pct_change(float(c.mae), float(b.mae))
        role_changes[role] = change
        role_gate &= change <= 0.0075 + 1e-12

    integrity = {
        "sportsbook_zero": True,
        "future_2026_zero": True,
        "strict_prior_contract": True,
        "rb_room_mass_conserved": float(ca.room_mass_gap.abs().max()) < 1e-10,
        "production_efficiency_ypt_exact": float(x.production_ypt_gap.abs().max()) == 0.0,
    }
    scientific = {
        "targets_mae_improves": float(ct.mae) < float(bt.mae),
        "targets_rmse_nonworse": float(ct.rmse) <= float(bt.rmse) + 1e-12,
        "receptions_mae_improves": float(cr.mae) < float(br.mae),
        "receptions_rmse_nonworse": float(cr.rmse) <= float(br.rmse) + 1e-12,
        "rec_yards_mae_improves_0p5pct": float(cy.mae) <= float(by.mae) * 0.995 + 1e-12,
        "rec_yards_rmse_nonworse": float(cy.rmse) <= float(by.rmse) + 1e-12,
        "directional_replication_2of3_max_worsen_0p75pct": season_improves >= 2
        and max_season_worsen <= 0.0075 + 1e-12,
        "p90_protected_1pct": float(cy.p90_abs_error) <= float(by.p90_abs_error) * 1.01 + 1e-12,
        "miss30_protected_0p5pp": float(cy.miss30_rate) <= float(by.miss30_rate) + 0.005 + 1e-12,
        "yards_bias_protected_0p75yd": abs(float(cy.bias)) <= abs(float(by.bias)) + 0.75 + 1e-12,
        "receptions_bias_protected_0p10": abs(float(cr.bias)) <= abs(float(br.bias)) + 0.10 + 1e-12,
        "role_robustness_0p75pct": bool(role_gate),
        "mechanism_coherence": float(ct.mae) < float(bt.mae)
        and float(cr.mae) < float(br.mae)
        and bool(integrity["rb_room_mass_conserved"]),
    }
    passed = all(integrity.values()) and all(scientific.values())
    out = {
        "candidate": "RB_R24_ENTITLEMENT_PRODUCTION_EFFICIENCY_V1",
        "disposition": "PASS_FROZEN_SCIENTIFIC_GATES_INTEGRATION_AUTHORIZED"
        if passed
        else "MIXED_OR_FAIL_NO_PROMOTION",
        "pass": bool(passed),
        "integrity_gates": integrity,
        "scientific_gates": scientific,
        "season_rec_yards_mae_pct_changes": season_changes,
        "role_rec_yards_mae_pct_changes": role_changes,
        "pooled": {
            "targets": {
                "baseline_mae": float(bt.mae),
                "candidate_mae": float(ct.mae),
                "baseline_rmse": float(bt.rmse),
                "candidate_rmse": float(ct.rmse),
            },
            "receptions": {
                "baseline_mae": float(br.mae),
                "candidate_mae": float(cr.mae),
                "baseline_rmse": float(br.rmse),
                "candidate_rmse": float(cr.rmse),
                "baseline_bias": float(br.bias),
                "candidate_bias": float(cr.bias),
            },
            "rec_yards": {
                "baseline_mae": float(by.mae),
                "candidate_mae": float(cy.mae),
                "mae_pct_change": pct_change(float(cy.mae), float(by.mae)),
                "baseline_rmse": float(by.rmse),
                "candidate_rmse": float(cy.rmse),
                "baseline_bias": float(by.bias),
                "candidate_bias": float(cy.bias),
                "baseline_p90": float(by.p90_abs_error),
                "candidate_p90": float(cy.p90_abs_error),
                "baseline_miss30": float(by.miss30_rate),
                "candidate_miss30": float(cy.miss30_rate),
            },
        },
    }
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    summary.to_csv(a.out.with_suffix(".csv"), index=False)
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
