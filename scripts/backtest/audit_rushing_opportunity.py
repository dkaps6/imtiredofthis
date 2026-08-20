#!/usr/bin/env python3
"""Migration 24: diagnose MC rushing-opportunity allocation without changing production."""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _corr(a, b):
    x = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(x) < 2 or x.a.nunique() < 2 or x.b.nunique() < 2:
        return np.nan
    return float(x.a.corr(x.b))


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--predictions", type=Path, default=Path("data/backtests/component_predictions.csv"))
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rushing_opportunity_diagnostics"))
    args = p.parse_args()
    if not args.predictions.exists() or args.predictions.stat().st_size == 0:
        raise RuntimeError(f"missing predictions: {args.predictions}")

    x = pd.read_csv(args.predictions)
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "market" not in x.columns:
        raise RuntimeError("component predictions missing market")
    r = x.loc[x.market.astype(str).eq("rush_att")].copy()
    if r.empty:
        raise RuntimeError("component predictions contain no rush_att rows")

    for c in ["actual", "actual_opportunities", "mc_proj", "rules_rush_share", "bayes_rush_share", "rush_share", "rules_plays_est", "rules_pass_rate"]:
        if c not in r.columns:
            r[c] = np.nan
        r[c] = _num(r[c])

    r["actual_rush_att"] = r["actual_opportunities"].where(r["actual_opportunities"].notna(), r["actual"])
    r["share_source"] = np.select(
        [r.rules_rush_share.notna(), r.bayes_rush_share.notna(), r.rush_share.notna()],
        ["rules_rush_share", "bayes_rush_share", "rush_share"],
        default="missing",
    )
    r["effective_rush_share"] = r.rules_rush_share.where(r.rules_rush_share.notna(), r.bayes_rush_share.where(r.bayes_rush_share.notna(), r.rush_share))
    r["team_expected_rushes"] = r.rules_plays_est * (1.0 - r.rules_pass_rate)
    r["deterministic_carries_from_share"] = r.team_expected_rushes * r.effective_rush_share
    r["mc_error"] = r.mc_proj - r.actual_rush_att
    r["share_error_proxy"] = r.deterministic_carries_from_share - r.actual_rush_att

    group_cols = [c for c in ["season", "week", "team"] if c in r.columns]
    if group_cols:
        team = r.groupby(group_cols, dropna=False).agg(
            modeled_players=("player_clean_key", "nunique") if "player_clean_key" in r.columns else ("market", "size"),
            modeled_share_sum=("effective_rush_share", "sum"),
            actual_rushes_modeled=("actual_rush_att", "sum"),
            mc_rushes_modeled=("mc_proj", "sum"),
            expected_team_rushes=("team_expected_rushes", "first"),
        ).reset_index()
        team["residual_share"] = 1.0 - team.modeled_share_sum
        team["actual_minus_expected_team"] = team.actual_rushes_modeled - team.expected_team_rushes
    else:
        team = pd.DataFrame()

    summary_rows = [{
        "scope": "all",
        "n": int(r.actual_rush_att.notna().sum()),
        "mc_mae": float((r.mc_proj-r.actual_rush_att).abs().mean()),
        "mc_bias": float((r.mc_proj-r.actual_rush_att).mean()),
        "mc_corr": _corr(r.mc_proj, r.actual_rush_att),
        "effective_share_corr_actual": _corr(r.effective_rush_share, r.actual_rush_att),
        "deterministic_share_carries_corr_actual": _corr(r.deterministic_carries_from_share, r.actual_rush_att),
        "missing_effective_share_rate": float(r.effective_rush_share.isna().mean()),
        "zero_effective_share_rate": float(r.effective_rush_share.fillna(0).eq(0).mean()),
        "mean_effective_share": float(r.effective_rush_share.mean()),
    }]
    if "position" in r.columns:
        for pos, g in r.groupby(r.position.fillna("UNKNOWN").astype(str), dropna=False):
            summary_rows.append({
                "scope": f"position:{pos}", "n": int(g.actual_rush_att.notna().sum()),
                "mc_mae": float((g.mc_proj-g.actual_rush_att).abs().mean()),
                "mc_bias": float((g.mc_proj-g.actual_rush_att).mean()),
                "mc_corr": _corr(g.mc_proj, g.actual_rush_att),
                "effective_share_corr_actual": _corr(g.effective_rush_share, g.actual_rush_att),
                "deterministic_share_carries_corr_actual": _corr(g.deterministic_carries_from_share, g.actual_rush_att),
                "missing_effective_share_rate": float(g.effective_rush_share.isna().mean()),
                "zero_effective_share_rate": float(g.effective_rush_share.fillna(0).eq(0).mean()),
                "mean_effective_share": float(g.effective_rush_share.mean()),
            })
    summary = pd.DataFrame(summary_rows)

    cols = [c for c in ["season","week","team","opponent","player","player_clean_key","position","mc_proj","actual_rush_att","mc_error","rules_plays_est","rules_pass_rate","team_expected_rushes","rules_rush_share","bayes_rush_share","rush_share","effective_rush_share","deterministic_carries_from_share","share_error_proxy","share_source"] if c in r.columns]
    players = r[cols].copy().sort_values("mc_error", key=lambda s: s.abs(), ascending=False)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.out_dir / "rushing_opportunity_summary.csv", index=False)
    players.to_csv(args.out_dir / "rushing_opportunity_player_diagnostics.csv", index=False)
    team.to_csv(args.out_dir / "rushing_opportunity_team_diagnostics.csv", index=False)

    print("\n[rush-opportunity] summary")
    print(summary.to_string(index=False))
    if not team.empty:
        print("\n[rush-opportunity] team-share health")
        print(team[[c for c in ["modeled_share_sum","residual_share","actual_rushes_modeled","mc_rushes_modeled","expected_team_rushes"] if c in team.columns]].describe().to_string())
    print("\n[rush-opportunity] largest player errors")
    print(players.head(30).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
