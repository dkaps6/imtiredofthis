#!/usr/bin/env python3
"""Decompose ACT-only player opportunity error into team-volume vs player-share error."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

SEASON = 2026
WEEKS = (1, 2, 3, 4)
TOL = 1e-10
MC_TOL = 0.25

HIGH_BIN = {
    ("QB", "pass_attempts"): "41_PLUS",
    ("RB", "carries"): "15_PLUS",
    ("FB", "carries"): "15_PLUS",
    ("RB", "targets"): "09_PLUS",
    ("FB", "targets"): "09_PLUS",
    ("WR", "targets"): "09_PLUS",
    ("TE", "targets"): "09_PLUS",
}


def _to_pandas(obj) -> pd.DataFrame:
    return obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _num(s) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def build_actual_team_volumes() -> pd.DataFrame:
    import nflreadpy as nfl

    raw = _to_pandas(nfl.load_pbp(seasons=[SEASON]))
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    if "season_type" in x.columns:
        reg = x["season_type"].astype(str).str.upper().eq("REG")
        if reg.any():
            x = x.loc[reg].copy()
    required = {"week", "posteam", "qb_dropback", "rush_attempt", "pass_attempt", "sack"}
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"PBP missing columns: {sorted(missing)}")

    x["week"] = _num(x["week"])
    x = x.loc[x["week"].isin(WEEKS)].copy()
    x["team"] = x["posteam"].map(canon_team)
    x = x.loc[x["team"].astype(str).ne("")].copy()

    for c in ("qb_dropback", "rush_attempt", "pass_attempt", "sack", "two_point_attempt"):
        if c not in x.columns:
            x[c] = 0.0
        x[c] = _num(x[c]).fillna(0.0)

    x = x.loc[~x["two_point_attempt"].eq(1)].copy()
    x["off_play"] = (x["qb_dropback"].eq(1) | x["rush_attempt"].eq(1)).astype(int)
    x["dropback_play"] = x["qb_dropback"].eq(1).astype(int)
    x["official_pass_attempt"] = (
        x["pass_attempt"].eq(1) & ~x["sack"].eq(1)
    ).astype(int)
    x["non_dropback_play"] = (
        x["off_play"].eq(1) & ~x["qb_dropback"].eq(1)
    ).astype(int)

    out = (
        x.groupby(["week", "team"], as_index=False)
        .agg(
            actual_team_official_pass_attempts=("official_pass_attempt", "sum"),
            actual_team_dropbacks=("dropback_play", "sum"),
            actual_team_non_dropback_plays=("non_dropback_play", "sum"),
            actual_team_offensive_plays=("off_play", "sum"),
        )
    )
    out["season"] = SEASON
    if out.duplicated(["season", "week", "team"]).any():
        raise RuntimeError("duplicate actual team-volume identity")
    return out


def _actual_volume_column(row: pd.Series) -> str:
    pos = str(row["position_family"])
    opp = str(row["opportunity_type"])
    if pos == "QB" and opp == "pass_attempts":
        return "actual_team_official_pass_attempts"
    if opp == "targets":
        return "actual_team_dropbacks"
    if opp == "carries":
        return "actual_team_non_dropback_plays"
    raise RuntimeError(f"unsupported opportunity family pos={pos} opp={opp}")


def _summary(g: pd.DataFrame) -> dict:
    actual = _num(g["actual_opportunities"])
    model = _num(g["model_expected"])
    vol = _num(g["actual_volume_diagnostic"])
    share = _num(g["actual_share_diagnostic"])

    def stats(pred: pd.Series, prefix: str) -> dict:
        err = pred - actual
        return {
            f"{prefix}_mae": float(err.abs().mean()),
            f"{prefix}_bias": float(err.mean()),
            f"{prefix}_rmse": float(np.sqrt(np.mean(np.square(err)))),
        }

    out = {"rows": int(len(g))}
    out.update(stats(model, "model"))
    out.update(stats(vol, "actual_volume"))
    out.update(stats(share, "actual_share"))

    model_mae = out["model_mae"]
    out["mae_improvement_actual_volume"] = model_mae - out["actual_volume_mae"]
    out["mae_improvement_actual_share"] = model_mae - out["actual_share_mae"]
    out["fraction_mae_removed_actual_volume"] = (
        out["mae_improvement_actual_volume"] / model_mae if model_mae > 0 else np.nan
    )
    out["fraction_mae_removed_actual_share"] = (
        out["mae_improvement_actual_share"] / model_mae if model_mae > 0 else np.nan
    )

    pteam = _num(g["predicted_team_opportunity_mean"])
    ateam = _num(g["actual_team_volume"])
    pshare = _num(g["final_player_probability"])
    ashare = _num(g["actual_player_share"])
    out["predicted_team_volume_mean"] = float(pteam.mean())
    out["actual_team_volume_mean"] = float(ateam.mean())
    out["team_volume_mae"] = float((pteam - ateam).abs().mean())
    out["predicted_player_share_mean"] = float(pshare.mean())
    out["actual_player_share_mean"] = float(ashare.mean())
    out["player_share_mae"] = float((pshare - ashare).abs().mean())
    opp_err = model - actual
    share_err = pshare - ashare
    volume_err = pteam - ateam
    out["share_error_vs_opportunity_error_pearson"] = (
        float(share_err.corr(opp_err)) if share_err.nunique() > 1 and opp_err.nunique() > 1 else np.nan
    )
    out["team_volume_error_vs_opportunity_error_pearson"] = (
        float(volume_err.corr(opp_err)) if volume_err.nunique() > 1 and opp_err.nunique() > 1 else np.nan
    )
    return out


def run(*, rows_path: Path, out_dir: Path) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = _read(rows_path, "ACT-only opportunity rows")

    required = {
        "season", "week", "team", "player", "player_clean_key",
        "position_family", "opportunity_type", "actual_opportunities",
        "actual_opportunity_bin", "predicted_team_opportunity_mean",
        "final_player_probability", "predicted_opportunities",
        "expected_opportunities_from_probability",
        "sportsbook_inputs_used_upstream",
    }
    missing = required - set(rows.columns)
    if missing:
        raise RuntimeError(f"ACT-only rows missing columns: {sorted(missing)}")
    if rows["sportsbook_inputs_used_upstream"].astype(bool).any():
        raise RuntimeError("sportsbook leakage in parent ACT-only rows")
    if set(_num(rows["week"]).dropna().astype(int).unique()) - set(WEEKS):
        raise RuntimeError("parent rows contain week outside frozen W1-4")

    for c in (
        "predicted_team_opportunity_mean", "final_player_probability",
        "predicted_opportunities", "expected_opportunities_from_probability",
        "actual_opportunities",
    ):
        rows[c] = _num(rows[c])

    rows["model_expected"] = (
        rows["predicted_team_opportunity_mean"] * rows["final_player_probability"]
    )
    parity_gap = (
        rows["model_expected"] - rows["expected_opportunities_from_probability"]
    ).abs()
    if float(parity_gap.max()) > TOL:
        raise RuntimeError(
            f"deterministic expectation parity failed max_gap={float(parity_gap.max())}"
        )
    sampled_gap = (
        rows["predicted_opportunities"] - rows["model_expected"]
    ).abs()
    if float(sampled_gap.max()) > MC_TOL:
        raise RuntimeError(
            f"sampled allocation parity exceeded certified tolerance max_gap={float(sampled_gap.max())}"
        )

    team = build_actual_team_volumes()
    rows = rows.merge(
        team,
        on=["season", "week", "team"],
        how="left",
        validate="many_to_one",
    )
    if rows["actual_team_official_pass_attempts"].isna().any():
        bad = rows.loc[
            rows["actual_team_official_pass_attempts"].isna(),
            ["week", "team"],
        ].drop_duplicates().head(20)
        raise RuntimeError(f"missing actual team PBP volume: {bad.to_dict('records')}")

    rows["actual_team_volume_source"] = [
        _actual_volume_column(r) for _, r in rows.iterrows()
    ]
    rows["actual_team_volume"] = [
        float(r[col]) for (_, r), col in zip(
            rows.iterrows(), rows["actual_team_volume_source"]
        )
    ]
    if (rows["actual_team_volume"] <= 0).any():
        bad = rows.loc[
            rows["actual_team_volume"] <= 0,
            ["week", "team", "position_family", "opportunity_type", "actual_team_volume"],
        ].head(20)
        raise RuntimeError(f"nonpositive actual team volume: {bad.to_dict('records')}")

    rows["actual_player_share"] = (
        rows["actual_opportunities"] / rows["actual_team_volume"]
    )
    rows["actual_volume_diagnostic"] = (
        rows["actual_team_volume"] * rows["final_player_probability"]
    )
    rows["actual_share_diagnostic"] = (
        rows["predicted_team_opportunity_mean"] * rows["actual_player_share"]
    )
    rows["full_identity"] = (
        rows["actual_team_volume"] * rows["actual_player_share"]
    )
    rows["full_identity_gap"] = (
        rows["full_identity"] - rows["actual_opportunities"]
    ).abs()
    if float(rows["full_identity_gap"].max()) > TOL:
        raise RuntimeError(
            f"full identity failed max_gap={float(rows['full_identity_gap'].max())}"
        )

    rows["model_error"] = rows["model_expected"] - rows["actual_opportunities"]
    rows["actual_volume_error"] = rows["actual_volume_diagnostic"] - rows["actual_opportunities"]
    rows["actual_share_error"] = rows["actual_share_diagnostic"] - rows["actual_opportunities"]
    rows["team_volume_error"] = (
        rows["predicted_team_opportunity_mean"] - rows["actual_team_volume"]
    )
    rows["player_share_error"] = (
        rows["final_player_probability"] - rows["actual_player_share"]
    )
    rows["parameters_fit"] = 0
    rows["automatic_promotion"] = False
    rows["sportsbook_inputs_used_upstream"] = False

    group_rows = []
    for (pos, opp), g in rows.groupby(["position_family", "opportunity_type"], dropna=False):
        group_rows.append({
            "position_family": str(pos),
            "opportunity_type": str(opp),
            "scope": "ALL",
            **_summary(g),
        })
        for week, w in g.groupby("week"):
            group_rows.append({
                "position_family": str(pos),
                "opportunity_type": str(opp),
                "scope": f"W{int(week)}",
                **_summary(w),
            })
    summary = pd.DataFrame(group_rows)

    high_rows = []
    for (pos, opp), g in rows.groupby(["position_family", "opportunity_type"], dropna=False):
        high_label = HIGH_BIN.get((str(pos), str(opp)))
        if not high_label:
            continue
        h = g.loc[g["actual_opportunity_bin"].astype(str).eq(high_label)].copy()
        if h.empty:
            continue
        rec = {
            "position_family": str(pos),
            "opportunity_type": str(opp),
            "high_bin": high_label,
            "rows": int(len(h)),
        }
        actual = h["actual_opportunities"]
        for col, prefix in (
            ("model_expected", "model"),
            ("actual_volume_diagnostic", "actual_volume"),
            ("actual_share_diagnostic", "actual_share"),
        ):
            err = h[col] - actual
            rec[f"{prefix}_bias"] = float(err.mean())
            rec[f"{prefix}_mae"] = float(err.abs().mean())
        high_rows.append(rec)
    high = pd.DataFrame(high_rows)

    rows.to_csv(out_dir / "player_opportunity_volume_share_rows.csv", index=False)
    summary.to_csv(out_dir / "player_opportunity_volume_share_group_summary.csv", index=False)
    high.to_csv(out_dir / "player_opportunity_volume_share_high_workload_summary.csv", index=False)

    overall = summary.loc[summary["scope"].eq("ALL")].copy()
    payload = {
        "version": "PLAYER_OPPORTUNITY_VOLUME_VS_SHARE_DECOMPOSITION_V1",
        "season": SEASON,
        "weeks": list(WEEKS),
        "rows": int(len(rows)),
        "unique_player_weeks": int(
            rows[["week", "team", "player_clean_key"]].drop_duplicates().shape[0]
        ),
        "max_deterministic_expectation_parity_gap": float(parity_gap.max()),
        "max_sampled_allocation_gap": float(sampled_gap.max()),
        "max_full_identity_gap": float(rows["full_identity_gap"].max()),
        "group_summary": overall.to_dict("records"),
        "high_workload_summary": high.to_dict("records"),
        "parameters_fit": 0,
        "automatic_promotion": False,
        "sportsbook_inputs_used_upstream": False,
        "paid_odds_api_used": False,
        "disposition": "DIAGNOSTIC_COMPLETE_RAW_RESULT_REQUIRES_INTERPRETATION",
    }
    (out_dir / "player_opportunity_volume_share_decomposition_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
    return payload


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--rows", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    a = p.parse_args()
    run(rows_path=a.rows, out_dir=a.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
