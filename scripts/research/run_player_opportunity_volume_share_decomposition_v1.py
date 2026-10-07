#!/usr/bin/env python3
"""Decompose individual-player opportunity error into team-volume vs player-share error.

Frozen by PLAYER_OPPORTUNITY_VOLUME_VS_SHARE_DECOMPOSITION_V1_CONTRACT.md.
The prediction parent is immutable ACT-only evidence. Completed-game team
volumes are attached only after that parent is loaded and certified.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.pbp import get_pbp

SEASON = 2026
WEEKS = (1, 2, 3, 4)
TOL = 1e-10
HIGH_BIN = {
    ("QB", "pass_attempts"): "41_PLUS",
    ("RB", "carries"): "15_PLUS",
    ("FB", "carries"): "15_PLUS",
    ("RB", "targets"): "09_PLUS",
    ("FB", "targets"): "09_PLUS",
    ("WR", "targets"): "09_PLUS",
    ("TE", "targets"): "09_PLUS",
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _num(df: pd.DataFrame, col: str, default=0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def _load_actual_team_volumes() -> pd.DataFrame:
    """Outcome-side team official pass/rush opportunities from completed PBP."""
    x = get_pbp(SEASON, min_rows=1).copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "season_type" in x.columns:
        reg = x.loc[x["season_type"].astype(str).str.upper().eq("REG")].copy()
        if not reg.empty:
            x = reg
    required = {"week", "posteam"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"PBP missing actual team-volume columns: {sorted(missing)}")

    x["week"] = pd.to_numeric(x["week"], errors="coerce")
    x = x.loc[x["week"].isin(WEEKS)].copy()
    x["week"] = x["week"].astype(int)
    x["team"] = x["posteam"].map(canon_team)
    x = x.loc[x["team"].astype(str).ne("")].copy()

    pass_attempt = _num(x, "pass_attempt", 0.0).eq(1)
    sack = _num(x, "sack", 0.0).eq(1)
    rush_attempt = _num(x, "rush_attempt", 0.0).eq(1)
    x["official_pass_attempt"] = (pass_attempt & ~sack).astype(int)
    x["team_rush_attempt"] = rush_attempt.astype(int)

    out = (
        x.groupby(["week", "team"], as_index=False)
        .agg(
            actual_team_official_pass_attempts=("official_pass_attempt", "sum"),
            actual_team_rush_attempts=("team_rush_attempt", "sum"),
        )
        .sort_values(["week", "team"])
        .reset_index(drop=True)
    )
    if out.empty:
        raise RuntimeError("actual team-volume PBP aggregation produced zero rows")
    if out.duplicated(["week", "team"]).any():
        raise RuntimeError("actual team-volume PBP aggregation has duplicate team-week")
    return out


def _actual_team_volume(row: pd.Series) -> float:
    if str(row["opportunity_type"]) in {"pass_attempts", "targets"}:
        return float(row["actual_team_official_pass_attempts"])
    if str(row["opportunity_type"]) == "carries":
        return float(row["actual_team_rush_attempts"])
    raise RuntimeError(f"unknown opportunity type: {row['opportunity_type']}")


def build_rows(parent: pd.DataFrame, team_actual: pd.DataFrame) -> pd.DataFrame:
    required = {
        "season", "week", "event_id", "team", "opponent", "player",
        "player_clean_key", "position_family", "opportunity_type",
        "predicted_team_opportunity_mean", "final_player_probability",
        "expected_opportunities_from_probability", "actual_opportunities",
        "actual_opportunity_bin", "sportsbook_inputs_used_upstream",
        "rb_week5_room_allocation_shadow_applied",
    }
    missing = required - set(parent.columns)
    if missing:
        raise RuntimeError(f"ACT-only parent missing columns: {sorted(missing)}")

    x = parent.copy()
    if not pd.to_numeric(x["season"], errors="coerce").eq(SEASON).all():
        raise RuntimeError("ACT-only parent season drift")
    if set(pd.to_numeric(x["week"], errors="coerce").dropna().astype(int)) != set(WEEKS):
        raise RuntimeError("ACT-only parent week coverage drift")
    if x["sportsbook_inputs_used_upstream"].astype(bool).any():
        raise RuntimeError("sportsbook input flag present in ACT-only parent")
    if x["rb_week5_room_allocation_shadow_applied"].astype(bool).any():
        raise RuntimeError("Week-5 RB room shadow leaked into ACT-only parent")
    key = ["season", "week", "event_id", "team", "player_clean_key", "opportunity_type"]
    if x.duplicated(key).any():
        raise RuntimeError("ACT-only parent contains duplicate player-opportunity identity")

    # Prediction parent is certified before completed-game PBP is merged.
    x["_prediction_parent_frozen"] = True

    a = team_actual.copy()
    x["team"] = x["team"].map(canon_team)
    a["team"] = a["team"].map(canon_team)
    x = x.merge(a, on=["week", "team"], how="left", validate="many_to_one")
    if x[["actual_team_official_pass_attempts", "actual_team_rush_attempts"]].isna().any().any():
        bad = x.loc[
            x[["actual_team_official_pass_attempts", "actual_team_rush_attempts"]].isna().any(axis=1),
            ["week", "team", "player", "opportunity_type"],
        ].head(20)
        raise RuntimeError(f"missing completed team volume: {bad.to_dict('records')}")

    x["predicted_team_volume"] = pd.to_numeric(
        x["predicted_team_opportunity_mean"], errors="coerce"
    )
    x["predicted_player_share"] = pd.to_numeric(
        x["final_player_probability"], errors="coerce"
    )
    x["actual_player_opportunity"] = pd.to_numeric(
        x["actual_opportunities"], errors="coerce"
    )
    x["actual_team_volume"] = x.apply(_actual_team_volume, axis=1)

    if x["predicted_team_volume"].isna().any() or x["predicted_player_share"].isna().any():
        raise RuntimeError("non-finite predicted team volume/share")
    if x["actual_player_opportunity"].isna().any() or x["actual_team_volume"].isna().any():
        raise RuntimeError("non-finite actual opportunity/team volume")
    if x["actual_team_volume"].le(0).any():
        bad = x.loc[x["actual_team_volume"].le(0), ["week", "team", "opportunity_type"]].drop_duplicates().head(20)
        raise RuntimeError(f"non-positive actual team volume: {bad.to_dict('records')}")
    if x["predicted_player_share"].lt(-TOL).any() or x["predicted_player_share"].gt(1 + TOL).any():
        raise RuntimeError("predicted player share outside [0,1]")

    x["actual_player_share"] = x["actual_player_opportunity"] / x["actual_team_volume"]
    if x["actual_player_share"].lt(-TOL).any() or x["actual_player_share"].gt(1 + TOL).any():
        bad = x.loc[
            x["actual_player_share"].lt(-TOL) | x["actual_player_share"].gt(1 + TOL),
            ["week", "team", "player", "opportunity_type", "actual_player_opportunity",
             "actual_team_volume", "actual_player_share"],
        ].head(20)
        raise RuntimeError(f"actual player share outside [0,1]: {bad.to_dict('records')}")

    x["baseline_expected_opportunity"] = (
        x["predicted_team_volume"] * x["predicted_player_share"]
    )
    parent_expected = pd.to_numeric(
        x["expected_opportunities_from_probability"], errors="coerce"
    )
    base_gap = (x["baseline_expected_opportunity"] - parent_expected).abs()
    if float(base_gap.max()) > TOL:
        raise RuntimeError(
            "baseline decomposition does not reproduce parent expectation "
            f"max_gap={float(base_gap.max())}"
        )

    x["oracle_team_volume_opportunity"] = (
        x["actual_team_volume"] * x["predicted_player_share"]
    )
    x["oracle_player_share_opportunity"] = (
        x["predicted_team_volume"] * x["actual_player_share"]
    )
    x["full_oracle_opportunity"] = (
        x["actual_team_volume"] * x["actual_player_share"]
    )
    oracle_gap = (x["full_oracle_opportunity"] - x["actual_player_opportunity"]).abs()
    if float(oracle_gap.max()) > TOL:
        raise RuntimeError(f"full oracle identity failed max_gap={float(oracle_gap.max())}")

    for stage, col in [
        ("baseline", "baseline_expected_opportunity"),
        ("oracle_team", "oracle_team_volume_opportunity"),
        ("oracle_share", "oracle_player_share_opportunity"),
    ]:
        x[f"{stage}_error"] = x[col] - x["actual_player_opportunity"]
        x[f"{stage}_absolute_error"] = x[f"{stage}_error"].abs()
        x[f"{stage}_squared_error"] = x[f"{stage}_error"] ** 2

    x["team_volume_error"] = x["predicted_team_volume"] - x["actual_team_volume"]
    x["absolute_team_volume_error"] = x["team_volume_error"].abs()
    x["player_share_error"] = x["predicted_player_share"] - x["actual_player_share"]
    x["absolute_player_share_error"] = x["player_share_error"].abs()
    x["sportsbook_inputs_used_upstream"] = False
    x["actual_team_volume_loaded_after_parent_freeze"] = x["_prediction_parent_frozen"]
    return x.drop(columns=["_prediction_parent_frozen"])


def _stage_metrics(g: pd.DataFrame, stage: str) -> dict:
    err = pd.to_numeric(g[f"{stage}_error"], errors="coerce")
    return {
        f"{stage}_mae": float(err.abs().mean()),
        f"{stage}_bias": float(err.mean()),
        f"{stage}_rmse": float(np.sqrt(np.mean(np.square(err)))),
    }


def _correlation(a: pd.Series, b: pd.Series) -> float:
    x = pd.to_numeric(a, errors="coerce")
    y = pd.to_numeric(b, errors="coerce")
    ok = x.notna() & y.notna()
    if ok.sum() < 3 or x.loc[ok].nunique() <= 1 or y.loc[ok].nunique() <= 1:
        return np.nan
    return float(x.loc[ok].corr(y.loc[ok]))


def _summarize_group(g: pd.DataFrame) -> dict:
    rec: dict = {"rows": int(len(g))}
    for stage in ("baseline", "oracle_team", "oracle_share"):
        rec.update(_stage_metrics(g, stage))

    base = rec["baseline_mae"]
    rec["team_oracle_mae_improvement"] = base - rec["oracle_team_mae"]
    rec["share_oracle_mae_improvement"] = base - rec["oracle_share_mae"]
    rec["team_oracle_fraction_baseline_mae_removed"] = (
        rec["team_oracle_mae_improvement"] / base if base > 0 else np.nan
    )
    rec["share_oracle_fraction_baseline_mae_removed"] = (
        rec["share_oracle_mae_improvement"] / base if base > 0 else np.nan
    )

    team_err = pd.to_numeric(g["team_volume_error"], errors="coerce")
    share_err = pd.to_numeric(g["player_share_error"], errors="coerce")
    rec["team_volume_mae"] = float(team_err.abs().mean())
    rec["team_volume_bias"] = float(team_err.mean())
    rec["player_share_mae"] = float(share_err.abs().mean())
    rec["player_share_bias"] = float(share_err.mean())
    rec["baseline_error_vs_linked_yards_error_pearson"] = _correlation(
        g["baseline_error"], g.get("linked_yards_error", pd.Series(np.nan, index=g.index))
    )
    rec["baseline_error_vs_linked_count_error_pearson"] = _correlation(
        g["baseline_error"], g.get("linked_count_error", pd.Series(np.nan, index=g.index))
    )
    rec["team_volume_error_vs_baseline_error_pearson"] = _correlation(
        g["team_volume_error"], g["baseline_error"]
    )
    rec["player_share_error_vs_baseline_error_pearson"] = _correlation(
        g["player_share_error"], g["baseline_error"]
    )
    return rec


def build_summary(rows: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    summaries = []
    high_rows = []
    for (pos, opp), g in rows.groupby(["position_family", "opportunity_type"], dropna=False):
        summaries.append({
            "position_family": str(pos),
            "opportunity_type": str(opp),
            "subset": "ALL",
            **_summarize_group(g),
        })
        active = g.loc[pd.to_numeric(g["actual_player_opportunity"], errors="coerce").gt(0)]
        summaries.append({
            "position_family": str(pos),
            "opportunity_type": str(opp),
            "subset": "ACTUAL_OPPORTUNITY_GT_ZERO",
            **_summarize_group(active),
        })

        label = HIGH_BIN.get((str(pos), str(opp)), "")
        h = g.loc[g["actual_opportunity_bin"].astype(str).eq(label)].copy()
        if not h.empty:
            high_rows.append({
                "position_family": str(pos),
                "opportunity_type": str(opp),
                "high_bin": label,
                **_summarize_group(h),
            })
    return pd.DataFrame(summaries), pd.DataFrame(high_rows)


def run(*, parent_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Freeze/certify prediction parent before touching completed-game PBP.
    parent = _read(parent_path, "ACT-only opportunity parent")
    parent_digest_proxy = {
        "rows": int(len(parent)),
        "season_values": sorted(pd.to_numeric(parent.get("season"), errors="coerce").dropna().astype(int).unique().tolist()),
        "week_values": sorted(pd.to_numeric(parent.get("week"), errors="coerce").dropna().astype(int).unique().tolist()),
    }

    team_actual = _load_actual_team_volumes()
    rows = build_rows(parent, team_actual)
    summary, high = build_summary(rows)

    rows_path = out_dir / "player_opportunity_volume_share_rows.csv"
    summary_path = out_dir / "player_opportunity_volume_share_summary.csv"
    high_path = out_dir / "player_opportunity_volume_share_high_workload.csv"
    rows.to_csv(rows_path, index=False)
    summary.to_csv(summary_path, index=False)
    high.to_csv(high_path, index=False)

    overall = summary.loc[summary["subset"].eq("ALL")].copy()
    active = summary.loc[summary["subset"].eq("ACTUAL_OPPORTUNITY_GT_ZERO")].copy()
    payload = {
        "version": "PLAYER_OPPORTUNITY_VOLUME_VS_SHARE_DECOMPOSITION_V1",
        "season": SEASON,
        "weeks": list(WEEKS),
        "parent": parent_digest_proxy,
        "rows": int(len(rows)),
        "unique_player_weeks": int(
            rows[["week", "team", "player_clean_key"]].drop_duplicates().shape[0]
        ),
        "overall_summary": overall.to_dict("records"),
        "active_summary": active.to_dict("records"),
        "high_workload_summary": high.to_dict("records"),
        "max_baseline_parent_gap": float(
            (
                rows["baseline_expected_opportunity"]
                - pd.to_numeric(rows["expected_opportunities_from_probability"], errors="coerce")
            ).abs().max()
        ),
        "max_full_oracle_actual_gap": float(
            (rows["full_oracle_opportunity"] - rows["actual_player_opportunity"]).abs().max()
        ),
        "actual_team_volume_loaded_after_parent_freeze": bool(
            rows["actual_team_volume_loaded_after_parent_freeze"].all()
        ),
        "parameters_fit": 0,
        "automatic_promotion": False,
        "sportsbook_inputs_used_upstream": False,
        "paid_odds_api_used": False,
        "disposition": "DECOMPOSITION_COMPLETE_RAW_RESULT_REQUIRES_INTERPRETATION",
    }
    (out_dir / "player_opportunity_volume_share_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(parent_path=a.parent, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
