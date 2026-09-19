#!/usr/bin/env python3
"""Outcome-free redundancy audit for Football Context role/room candidates.

Reconstructs the canonical PlayerForm production opportunity evidence at each
historical player-game cutoff (prior-season aggregate + current-season games
strictly before the target week), then asks how reconstructible each context
candidate is from that already-existing production evidence.

No target-game outcomes, future games, sportsbook data, or model residuals are
read. Reconstruction coefficients are fit on seasons <= 2023 and evaluated on
2024-2025 to avoid declaring novelty from in-sample fit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd

KEY = ["season", "week", "team", "player_identity_key"]
CANDIDATES = [
    "prior3_tgt_share_game_mean", "prior5_tgt_share_game_mean",
    "prior3_rush_share_game_mean", "prior5_rush_share_game_mean",
    "prior_tgt_share_game_top1", "prior_tgt_share_game_top2",
    "prior_rush_share_game_top1", "prior_rush_share_game_top2",
]
PRODUCTION_STATE_PREFIX = "prod_"


def _safe_spearman(a: pd.Series, b: pd.Series) -> float:
    z = pd.DataFrame({"a": pd.to_numeric(a, errors="coerce"), "b": pd.to_numeric(b, errors="coerce")}).dropna()
    return float(z["a"].corr(z["b"], method="spearman")) if len(z) >= 3 else np.nan


def build_production_opportunity_state(history: pd.DataFrame) -> pd.DataFrame:
    """Historical snapshot of the opportunity evidence already used by PlayerForm."""
    h = history.copy()
    h.columns = [str(c).strip().lower() for c in h.columns]
    req = {*KEY, "targets", "rushes", "team_targets", "team_rushes"}
    miss = req - set(h.columns)
    if miss:
        raise RuntimeError(f"history missing production-state columns: {sorted(miss)}")
    if h.duplicated(KEY).any():
        raise RuntimeError("history has duplicate canonical player-game keys")
    h["season"] = pd.to_numeric(h["season"], errors="coerce").astype("Int64")
    h["week"] = pd.to_numeric(h["week"], errors="coerce").astype("Int64")
    h = h.sort_values(["season", "week", "player_identity_key", "team"]).reset_index(drop=True)

    out = h[KEY].copy()
    for prefix, num_col, den_col in [
        ("tgt", "targets", "team_targets"), ("rush", "rushes", "team_rushes")
    ]:
        num = pd.to_numeric(h[num_col], errors="coerce").fillna(0.0)
        den = pd.to_numeric(h[den_col], errors="coerce").fillna(0.0)
        grp = [h["season"], h["player_identity_key"]]
        prior_num = num.groupby(grp).cumsum() - num
        prior_den = den.groupby(grp).cumsum() - den
        current_games = h.groupby(["season", "player_identity_key"], sort=False).cumcount()
        current_share = np.where(prior_den > 0, prior_num / prior_den, np.nan)
        out[f"prod_{prefix}_current_share"] = current_share
        out[f"prod_{prefix}_current_games"] = current_games.astype(float)

        agg = h.assign(_num=num, _den=den).groupby(["season", "player_identity_key"], dropna=False).agg(
            prior_games=("week", "nunique"), prior_num=("_num", "sum"), prior_den=("_den", "sum")
        ).reset_index()
        agg["prior_share"] = np.where(agg["prior_den"] > 0, agg["prior_num"] / agg["prior_den"], np.nan)
        agg["season"] = agg["season"].astype(int) + 1
        lookup = agg.rename(columns={
            "prior_games": f"prod_{prefix}_prior_games", "prior_share": f"prod_{prefix}_prior_share"
        })[["season", "player_identity_key", f"prod_{prefix}_prior_games", f"prod_{prefix}_prior_share"]]
        out = out.merge(lookup, on=["season", "player_identity_key"], how="left", validate="many_to_one")
        out[f"prod_{prefix}_prior_games"] = pd.to_numeric(out[f"prod_{prefix}_prior_games"], errors="coerce").fillna(0.0)
        cg = out[f"prod_{prefix}_current_games"]
        pv = pd.to_numeric(out[f"prod_{prefix}_prior_share"], errors="coerce")
        cv = pd.to_numeric(out[f"prod_{prefix}_current_share"], errors="coerce")
        w = cg / (cg + 4.0)
        blend = pv.copy()
        blend.loc[cv.notna() & pv.isna()] = cv.loc[cv.notna() & pv.isna()]
        both = pv.notna() & cv.notna()
        blend.loc[both] = (1.0 - w.loc[both]) * pv.loc[both] + w.loc[both] * cv.loc[both]
        out[f"prod_{prefix}_playerform_blend"] = blend
    return out


def _holdout_r2(df: pd.DataFrame, target: str, inputs: list[str]) -> tuple[float, int, int]:
    z = df[["season", target, *inputs]].copy()
    for c in [target, *inputs]:
        z[c] = pd.to_numeric(z[c], errors="coerce")
    z = z.dropna()
    train = z[z["season"] <= 2023]
    test = z[z["season"] >= 2024]
    if len(train) < 500 or len(test) < 200:
        return np.nan, len(train), len(test)
    Xtr = np.column_stack([np.ones(len(train)), train[inputs].to_numpy(float)])
    Xte = np.column_stack([np.ones(len(test)), test[inputs].to_numpy(float)])
    beta, *_ = np.linalg.lstsq(Xtr, train[target].to_numpy(float), rcond=None)
    pred = Xte @ beta
    y = test[target].to_numpy(float)
    sst = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - pred) ** 2)) / sst if sst > 0 else np.nan
    return r2, len(train), len(test)


def audit_redundancy(history: pd.DataFrame, context: pd.DataFrame) -> pd.DataFrame:
    state = build_production_opportunity_state(history)
    c = context.copy()
    c.columns = [str(x).strip().lower() for x in c.columns]
    if c.duplicated(KEY).any():
        raise RuntimeError("context has duplicate canonical player-game keys")
    # Context may be an enriched fixture in tests or a future upstream table that
    # already carries production-state columns. The audit owns those columns, so
    # discard any copies before the canonical one-to-one join. This prevents
    # pandas _x/_y suffixing from silently removing the expected field names.
    c = c.drop(columns=[col for col in c.columns if col.startswith(PRODUCTION_STATE_PREFIX)], errors="ignore")
    x = c.merge(state, on=KEY, how="left", validate="one_to_one")
    rows = []
    for feature in CANDIDATES:
        if feature not in x.columns:
            continue
        domain = "tgt" if "tgt" in feature else "rush"
        inputs = [
            f"prod_{domain}_prior_share", f"prod_{domain}_prior_games",
            f"prod_{domain}_current_share", f"prod_{domain}_current_games",
            f"prod_{domain}_playerform_blend",
        ]
        r2, ntr, nte = _holdout_r2(x, feature, inputs)
        rho = _safe_spearman(x[feature], x[f"prod_{domain}_playerform_blend"])
        if np.isfinite(r2) and r2 >= 0.90:
            disposition = "HIGHLY_RECONSTRUCTIBLE_REDUNDANT"
        elif np.isfinite(r2) and r2 >= 0.75:
            disposition = "PARTIALLY_RECONSTRUCTIBLE_REVIEW"
        else:
            disposition = "INCREMENTAL_INFORMATION_SURVIVES_REDUNDANCY_GATE"
        rows.append({
            "feature_name": feature,
            "production_domain": domain,
            "production_blend_spearman": rho,
            "holdout_reconstructibility_r2": r2,
            "reconstruction_train_rows": ntr,
            "reconstruction_holdout_rows": nte,
            "train_seasons": "2019-2023",
            "holdout_seasons": "2024-2025",
            "redundancy_r2_high_threshold": 0.90,
            "redundancy_r2_review_threshold": 0.75,
            "redundancy_disposition": disposition,
            "outcomes_read": False,
            "sportsbook_read": False,
        })
    return pd.DataFrame(rows).sort_values(["redundancy_disposition", "feature_name"]).reset_index(drop=True)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--context", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    a = p.parse_args()
    out = audit_redundancy(pd.read_csv(a.history), pd.read_csv(a.context))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(a.out, index=False)
    print(out.to_string(index=False))
    print(f"[role_room_redundancy] rows={len(out)} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
