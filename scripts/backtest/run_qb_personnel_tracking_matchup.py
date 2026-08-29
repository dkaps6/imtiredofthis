#!/usr/bin/env python3
"""Robust authoritative runner for Migration 75.

This runner keeps the frozen M75 hypotheses/gates unchanged while making source
qualification explicit: an unavailable new source is reported and skipped rather
than crashing the independent source families. The four precommitted matchup
interactions are all-or-none; M75 never silently fits a post-hoc subset.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.backtest import audit_qb_personnel_tracking_matchup as m

FROZEN_INTERACTIONS = ["x_sep_ypt", "x_yacoe_yac", "x_adot_adot", "x_top1_weak_ypt"]


def usable_cols(train: pd.DataFrame, cols: list[str]) -> list[str]:
    out = []
    for c in cols:
        if c not in train:
            continue
        s = m.num(train[c])
        if int(s.notna().sum()) >= 30 and int(s.nunique(dropna=True)) > 1:
            out.append(c)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--seasons", default="2022,2023,2024,2025")
    a = ap.parse_args()
    out = a.out_dir
    out.mkdir(parents=True, exist_ok=True)

    base = m.lower(pd.read_csv(a.canonical, low_memory=False))
    required = {"season", "week", "team", "opponent", "player_clean_key", "actual_pass_yards", "actual_attempts", "pred_pass_yards", "pred_attempts"}
    missing = sorted(required - set(base.columns))
    if missing:
        raise RuntimeError(f"M75 canonical frontier missing required columns: {missing}")
    if base.duplicated(["season", "week", "team", "player_clean_key"]).any():
        raise RuntimeError("M75 canonical frontier has duplicate stable-QB keys")
    counts = m.num(base.season).value_counts().to_dict()
    if int(counts.get(2024, 0)) < 250 or int(counts.get(2025, 0)) < 250:
        raise RuntimeError(f"M75 premarket stable-QB population unexpectedly small: {counts}")
    prohibited = [c for c in base.columns if any(k in c.lower() for k in ["market", "spread", "moneyline", "total_line"])]
    if prohibited:
        raise RuntimeError(f"M75 canonical market boundary violated: {prohibited}")
    base["team"] = base.team.map(m.canon)
    base["opponent"] = base.opponent.map(m.canon)
    seasons = [int(v) for v in a.seasons.split(",") if v.strip()]

    ngs, pfr, players, ngs_err, pfr_err = m.load_sources(seasons)
    ngs_week, ngs_meta = m.build_ngs_team_week(ngs)
    db_week, db_meta = m.build_pfr_secondary_week(pfr, players)
    features = m.build_features(base, ngs_week, db_week)

    train = features[m.num(features.season).eq(2024)].copy().reset_index(drop=True)
    test = features[m.num(features.season).eq(2025)].copy().reset_index(drop=True)
    train["ypa_resid"] = m.num(train.actual_pass_yards) / m.num(train.actual_attempts) - m.num(train.pred_pass_yards) / m.num(train.pred_attempts)
    test["ypa_resid"] = m.num(test.actual_pass_yards) / m.num(test.actual_attempts) - m.num(test.pred_pass_yards) / m.num(test.pred_attempts)

    off_all = [c for c in features if c.startswith("off_ngs_")]
    def_all = [c for c in features if c.startswith("def_db_")]
    off = usable_cols(train, off_all)
    deff = usable_cols(train, def_all)
    xx_usable = usable_cols(train, FROZEN_INTERACTIONS)

    ngs_qualified = bool(ngs_meta.get("usable")) and bool(off)
    pfr_qualified = bool(db_meta.get("usable")) and bool(deff)
    interaction_qualified = (
        ngs_qualified
        and pfr_qualified
        and len(xx_usable) == len(FROZEN_INTERACTIONS)
        and set(xx_usable) == set(FROZEN_INTERACTIONS)
    )
    xx = list(FROZEN_INTERACTIONS) if interaction_qualified else []

    source_df = pd.DataFrame([
        {
            "source": "nflverse_ngs_receiving_weekly",
            "usable": ngs_qualified,
            "rows_raw": len(ngs),
            "rows_team_week": len(ngs_week),
            "usable_feature_count": len(off),
            "usable_features": "|".join(off),
            "error": ngs_err,
            "detail": str(ngs_meta),
        },
        {
            "source": "nflverse_pfr_advstats_def_weekly",
            "usable": pfr_qualified,
            "rows_raw": len(pfr),
            "rows_team_week": len(db_week),
            "usable_feature_count": len(deff),
            "usable_features": "|".join(deff),
            "error": pfr_err,
            "detail": str(db_meta),
        },
    ])
    source_df.to_csv(out / "m75_source_manifest.csv", index=False)
    ngs_week.to_csv(out / "m75_ngs_receiver_team_week.csv", index=False)
    db_week.to_csv(out / "m75_pfr_secondary_team_week.csv", index=False)
    features.to_csv(out / "m75_game_features.csv", index=False)

    families: dict[str, list[str]] = {}
    family_status = []
    if ngs_qualified:
        families["ngs_receiving_tracking"] = off
        family_status.append({"family": "ngs_receiving_tracking", "status": "eligible", "reason": "source_qualified"})
    else:
        family_status.append({"family": "ngs_receiving_tracking", "status": "skipped", "reason": "ngs_source_or_features_unavailable"})
    if pfr_qualified:
        families["pfr_secondary_coverage"] = deff
        family_status.append({"family": "pfr_secondary_coverage", "status": "eligible", "reason": "source_qualified"})
    else:
        family_status.append({"family": "pfr_secondary_coverage", "status": "skipped", "reason": "pfr_secondary_source_or_features_unavailable"})
    if interaction_qualified:
        families["tracking_x_secondary"] = xx
        families["combined_personnel_tracking"] = list(dict.fromkeys(off + deff + xx))
        family_status += [
            {"family": "tracking_x_secondary", "status": "eligible", "reason": "all_four_frozen_interactions_qualified"},
            {"family": "combined_personnel_tracking", "status": "eligible", "reason": "both_sources_and_all_four_interactions_qualified"},
        ]
    else:
        missing_interactions = [c for c in FROZEN_INTERACTIONS if c not in xx_usable]
        reason = "requires_both_sources_and_all_four_interactions"
        if missing_interactions:
            reason += ":" + "|".join(missing_interactions)
        family_status += [
            {"family": "tracking_x_secondary", "status": "skipped", "reason": reason},
            {"family": "combined_personnel_tracking", "status": "skipped", "reason": reason},
        ]
    pd.DataFrame(family_status).to_csv(out / "m75_family_status.csv", index=False)

    rows = []
    for family, cols in families.items():
        cov = float(test[cols].notna().mean().median()) if cols else 0.0
        for kind in ["ridge", "hgb"]:
            model = m.make_model(kind)
            ok = train.ypa_resid.notna()
            model.fit(train.loc[ok, cols], train.loc[ok, "ypa_resid"])
            pred = np.asarray(model.predict(test[cols]), dtype=float)
            ev = m.evaluate(test, pred)
            full = bool(
                cov >= m.MIN_COVERAGE
                and ev["ypa_residual_corr"] >= m.MIN_YPA_RESID_CORR
                and ev["ypa_mae_gain"] >= m.MIN_YPA_MAE_GAIN
                and ev["pass_mae_gain"] >= m.MIN_PASS_MAE_GAIN
                and ev["pass_corr_gain"] >= m.MIN_PASS_CORR_GAIN
                and ev["corrected_100plus"] <= ev["base_100plus"]
            )
            support = bool(
                cov >= m.MIN_COVERAGE
                and ev["ypa_residual_corr"] >= m.SUPPORT_CORR
                and ev["pass_mae_gain"] >= m.SUPPORT_PASS_GAIN
                and ev["corrected_100plus"] <= ev["base_100plus"]
            )
            rows.append({
                "family": family, "model": kind, "feature_count": len(cols),
                "coverage": cov, **ev, "full_gate": full, "support_gate": support,
            })
    results = pd.DataFrame(rows)
    results.to_csv(out / "m75_model_results.csv", index=False)

    supported = []
    if len(results):
        for family in results.family.unique():
            q = results[results.family.eq(family)]
            if len(q) != 2:
                continue
            for _, winner in q.iterrows():
                other = q[q.model.ne(winner.model)]
                if bool(winner.full_gate) and len(other) and bool(other.iloc[0].support_gate):
                    supported.append(family)
                    break

    if supported:
        verdict = "m75_personnel_tracking_new_information_signal"
    elif ngs_qualified and pfr_qualified:
        verdict = "m75_sources_qualified_no_predictive_breakthrough"
    elif ngs_qualified:
        verdict = "m75_receiver_tracking_qualified_secondary_contract_incomplete"
    else:
        verdict = "m75_free_personnel_sources_not_qualified"

    pd.DataFrame([{
        "canonical_rows": len(base),
        "train_rows_2024": len(train),
        "evaluation_rows_2025": len(test),
        "ngs_source_qualified": ngs_qualified,
        "pfr_secondary_source_qualified": pfr_qualified,
        "all_four_interactions_qualified": interaction_qualified,
        "supported_families": "|".join(sorted(set(supported))),
        "m75_interpretation": verdict,
        "production_actionable": False,
    }]).to_csv(out / "m75_precommitted_interpretation.csv", index=False)

    print("=== M75 SOURCES ===")
    print(source_df.to_string(index=False))
    print("=== M75 FAMILY STATUS ===")
    print(pd.DataFrame(family_status).to_string(index=False))
    print("=== M75 RESULTS ===")
    print(results.to_string(index=False) if len(results) else "no eligible model families")
    print("=== M75 INTERPRETATION ===")
    print(pd.read_csv(out / "m75_precommitted_interpretation.csv").to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
