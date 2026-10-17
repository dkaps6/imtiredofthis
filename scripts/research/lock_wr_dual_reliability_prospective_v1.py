#!/usr/bin/env python3
"""Build a frozen WR dual-reliability prospective lock from pregame-only inputs.

This is implementation machinery for the already-frozen
WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_CONFIRMATION_V1 plan.

The builder is generic for target Weeks 7-10, but it fails closed unless the
strict-prior current-season histories are complete through target_week - 1.
It never reads target-week outcomes and never changes the WR-R15 projection.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

VERSION = "WR_DUAL_RELIABILITY_TRUST_PROSPECTIVE_LOCK_V1"
SEASON = 2026
TARGET_WEEKS = (7, 8, 9, 10)
PRIOR_WINDOW = 8
MIN_PRIOR = 4
TOL = 1e-10


def num(s: Any) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def read_csv(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or path.stat().st_size <= 0:
        raise RuntimeError(f"missing/empty {label}: {path}")
    x = pd.read_csv(path, low_memory=False)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _validate_history(
    history: pd.DataFrame,
    *,
    target_week: int,
    error_col: str,
    label: str,
) -> pd.DataFrame:
    need = {"season", "week", "player_clean_key", error_col}
    if need - set(history.columns):
        raise RuntimeError(f"{label} missing {sorted(need-set(history.columns))}")
    x = history.copy()
    x["season"] = num(x["season"]).astype("Int64")
    x["week"] = num(x["week"]).astype("Int64")
    x = x.loc[x["season"].eq(SEASON)].copy()
    x["player_clean_key"] = x["player_clean_key"].astype("string").fillna("").str.strip()
    x[error_col] = num(x[error_col])
    if x.empty:
        raise RuntimeError(f"zero 2026 rows in {label}")
    if x["player_clean_key"].eq("").any():
        raise RuntimeError(f"blank player identity in {label}")
    if x.duplicated(["season", "week", "player_clean_key"]).any():
        raise RuntimeError(f"duplicate player-week in {label}")
    if x[error_col].isna().any() or (~np.isfinite(x[error_col])).any():
        raise RuntimeError(f"nonfinite {error_col} in {label}")
    bad = sorted(
        x.loc[num(x["week"]).ge(target_week), "week"]
        .dropna().astype(int).unique().tolist()
    )
    if bad:
        raise RuntimeError(
            f"target/future outcomes present in {label}: weeks={bad}"
        )
    max_week = int(num(x["week"]).max())
    if max_week != target_week - 1:
        raise RuntimeError(
            f"{label} incomplete for W{target_week}: max history week "
            f"{max_week}, require {target_week-1}"
        )
    return x.sort_values(["player_clean_key", "week"]).reset_index(drop=True)


def build_lock(
    cohort: pd.DataFrame,
    opportunity_history: pd.DataFrame,
    efficiency_history: pd.DataFrame,
    *,
    target_week: int,
) -> tuple[pd.DataFrame, dict]:
    if target_week not in TARGET_WEEKS:
        raise RuntimeError(f"target week must be one of {TARGET_WEEKS}")

    need = {
        "season", "week", "event_id", "team", "opponent",
        "player", "player_clean_key", "wr_r15_projection",
    }
    if need - set(cohort.columns):
        raise RuntimeError(f"cohort missing {sorted(need-set(cohort.columns))}")

    c = cohort.copy()
    c["season"] = num(c["season"]).astype("Int64")
    c["week"] = num(c["week"]).astype("Int64")
    c = c.loc[c["season"].eq(SEASON) & c["week"].eq(target_week)].copy()
    c["player_clean_key"] = c["player_clean_key"].astype("string").fillna("").str.strip()
    c["wr_r15_projection"] = num(c["wr_r15_projection"])
    if c.empty:
        raise RuntimeError(f"zero Week-{target_week} WR cohort rows")
    if c["player_clean_key"].eq("").any():
        raise RuntimeError("blank Week target player identity")
    if c["player_clean_key"].duplicated().any():
        raise RuntimeError("duplicate Week target player identity")
    if c["wr_r15_projection"].isna().any() or (~np.isfinite(c["wr_r15_projection"])).any():
        raise RuntimeError("nonfinite frozen WR-R15 projection")

    opp = _validate_history(
        opportunity_history,
        target_week=target_week,
        error_col="opportunity_error",
        label="opportunity history",
    )
    eff = _validate_history(
        efficiency_history,
        target_week=target_week,
        error_col="efficiency_error",
        label="efficiency history",
    )

    rows = []
    for r in c.itertuples(index=False):
        pk = str(r.player_clean_key)
        oh = (
            opp.loc[opp["player_clean_key"].eq(pk)]
            .sort_values("week")
            .tail(PRIOR_WINDOW)
        )
        eh = (
            eff.loc[eff["player_clean_key"].eq(pk)]
            .sort_values("week")
            .tail(PRIOR_WINDOW)
        )
        opp_supported = len(oh) >= MIN_PRIOR
        eff_supported = len(eh) >= MIN_PRIOR
        supported = opp_supported and eff_supported

        rows.append({
            "season": SEASON,
            "week": target_week,
            "event_id": str(r.event_id),
            "team": str(r.team),
            "opponent": str(r.opponent),
            "player": str(r.player),
            "player_clean_key": pk,
            "wr_r15_projection": float(r.wr_r15_projection),
            "opportunity_prior_games": int(len(oh)),
            "efficiency_prior_games": int(len(eh)),
            "opportunity_player_mae": (
                float(np.mean(np.abs(oh["opportunity_error"].to_numpy(float))))
                if opp_supported else np.nan
            ),
            "efficiency_player_mae": (
                float(np.mean(np.abs(eh["efficiency_error"].to_numpy(float))))
                if eff_supported else np.nan
            ),
            "opportunity_history_min_week": (
                int(oh.iloc[0]["week"]) if len(oh) else np.nan
            ),
            "opportunity_history_max_week": (
                int(oh.iloc[-1]["week"]) if len(oh) else np.nan
            ),
            "efficiency_history_min_week": (
                int(eh.iloc[0]["week"]) if len(eh) else np.nan
            ),
            "efficiency_history_max_week": (
                int(eh.iloc[-1]["week"]) if len(eh) else np.nan
            ),
            "supported": bool(supported),
        })

    out = pd.DataFrame(rows)
    sup = out.loc[out["supported"]].copy()
    if sup.empty:
        raise RuntimeError("zero supported dual-reliability WR rows")

    sup["opportunity_fragility_score"] = sup["opportunity_player_mae"].rank(
        method="average", pct=True
    )
    sup["efficiency_fragility_score"] = sup["efficiency_player_mae"].rank(
        method="average", pct=True
    )
    sup["dual_fragility_score"] = (
        sup["opportunity_fragility_score"] + sup["efficiency_fragility_score"]
    ) / 2.0
    sup["dual_trust_bucket"] = "MIDDLE"
    sup.loc[
        sup["dual_fragility_score"].le(0.25), "dual_trust_bucket"
    ] = "DUAL_HIGH_TRUST"
    sup.loc[
        sup["dual_fragility_score"].ge(0.75), "dual_trust_bucket"
    ] = "DUAL_LOW_TRUST"

    keep = [
        "player_clean_key",
        "opportunity_fragility_score",
        "efficiency_fragility_score",
        "dual_fragility_score",
        "dual_trust_bucket",
    ]
    out = out.merge(sup[keep], on="player_clean_key", how="left", validate="one_to_one")
    out["dual_trust_bucket"] = out["dual_trust_bucket"].fillna("UNSUPPORTED")
    out = out.sort_values(["team", "player_clean_key"]).reset_index(drop=True)

    for col in (
        "opportunity_fragility_score",
        "efficiency_fragility_score",
        "dual_fragility_score",
    ):
        vals = num(out.loc[out["supported"], col])
        if vals.isna().any() or (vals < -TOL).any() or (vals > 1.0 + TOL).any():
            raise RuntimeError(f"invalid prospective reliability rank {col}")

    payload = out.to_csv(
        index=False, float_format="%.12f", lineterminator="\n"
    ).encode()
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()

    result = {
        "version": VERSION,
        "status": f"WEEK{target_week}_WR_DUAL_RELIABILITY_PREGAME_LOCK_FROZEN",
        "season": SEASON,
        "week": target_week,
        "fixed_confirmation_weeks": list(TARGET_WEEKS),
        "current_wr_cohort": int(len(out)),
        "supported_wr": int(out["supported"].sum()),
        "unsupported_wr": int((~out["supported"]).sum()),
        "dual_bucket_counts": {
            str(k): int(v)
            for k, v in out["dual_trust_bucket"].value_counts().to_dict().items()
        },
        "history_source_max_week_2026": target_week - 1,
        "prior_window": PRIOR_WINDOW,
        "min_prior_games": MIN_PRIOR,
        "row_csv_sha256": digest,
        "dual_formula": "(opportunity_fragility_score + efficiency_fragility_score) / 2",
        "rank_method": "average_percentile",
        "week6_composite_backfilled": False,
        "target_week_outcomes_read": 0,
        "sportsbook_inputs_used": 0,
        "oddsapi_calls": 0,
        "models_fit": 0,
        "projection_distribution_values_changed": 0,
        "production_changed": False,
    }
    return out, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", type=Path, required=True)
    ap.add_argument("--opportunity-history", type=Path, required=True)
    ap.add_argument("--efficiency-history", type=Path, required=True)
    ap.add_argument("--target-week", type=int, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows, result = build_lock(
        read_csv(args.cohort, "target-week cohort"),
        read_csv(args.opportunity_history, "opportunity history"),
        read_csv(args.efficiency_history, "efficiency history"),
        target_week=args.target_week,
    )
    rows.to_csv(
        args.out_dir / f"wr_week{args.target_week}_dual_reliability_lock.csv",
        index=False,
        float_format="%.12f",
    )
    (
        args.out_dir / f"wr_week{args.target_week}_dual_reliability_lock.json"
    ).write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
