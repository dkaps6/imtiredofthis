#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TOL = 1e-12
BOOT_REPS = 10_000
BOOT_SEED = 42029


def num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def mean_side(gap: pd.Series) -> pd.Series:
    g = num(gap)
    out = pd.Series("ON_LINE", index=g.index, dtype="string")
    out.loc[g > TOL] = "OVER"
    out.loc[g < -TOL] = "UNDER"
    return out


def classify(df: pd.DataFrame) -> pd.Series:
    mg = num(df["mc_gap"])
    fg = num(df["final_gap"])
    out = pd.Series("", index=df.index, dtype="string")

    crossed = (mg * fg) < 0
    out.loc[crossed] = "CROSSED_LINE"

    final_on = out.eq("") & (fg.abs() <= TOL) & (mg.abs() > TOL)
    out.loc[final_on] = "FINAL_ON_LINE"

    mc_on = out.eq("") & (mg.abs() <= TOL) & (fg.abs() > TOL)
    out.loc[mc_on] = "MC_ON_LINE"

    unchanged_on = out.eq("") & (mg.abs() <= TOL) & (fg.abs() <= TOL)
    out.loc[unchanged_on] = "UNCHANGED_ON_LINE"

    same_sign = out.eq("") & ((mg > TOL) & (fg > TOL) | (mg < -TOL) & (fg < -TOL))
    strengthened = same_sign & (fg.abs() > mg.abs() + TOL)
    weakened = same_sign & (fg.abs() < mg.abs() - TOL)
    out.loc[strengthened] = "SAME_SIDE_STRENGTHENED"
    out.loc[weakened] = "SAME_SIDE_WEAKENED"
    out.loc[out.eq("") & same_sign] = "SAME_SIDE_NO_MATERIAL_DISTANCE_CHANGE"

    if out.eq("").any():
        sample = df.loc[out.eq(""), ["mc_gap", "final_gap"]].head(10).to_dict("records")
        raise RuntimeError(f"unclassified rows: {sample}")
    return out


def metrics(q: pd.DataFrame) -> dict:
    n = int(len(q))
    if not n:
        return {
            "rows": 0,
            "clusters": 0,
            "win_rate": np.nan,
            "units": np.nan,
            "roi": np.nan,
            "final_mae": np.nan,
            "mc_mae": np.nan,
            "model_closer_rate": np.nan,
            "paired_abs_improvement": np.nan,
            "selected_matches_mc_mean_side": np.nan,
            "selected_matches_final_mean_side": np.nan,
        }
    decided = q["bet_result"].isin(["WIN", "LOSS"])
    return {
        "rows": n,
        "clusters": int(q["event_id"].astype(str).nunique()),
        "win_rate": float(q.loc[decided, "bet_result"].eq("WIN").mean()) if decided.any() else np.nan,
        "units": float(num(q["unit_result"]).sum()),
        "roi": float(num(q["unit_result"]).mean()),
        "final_mae": float((num(q["model_proj"]) - num(q["actual"])).abs().mean()),
        "mc_mae": float((num(q["mc_proj"]) - num(q["actual"])).abs().mean()),
        "model_closer_rate": float(q["model_closer_than_vegas"].astype(bool).mean()),
        "paired_abs_improvement": float(
            ((num(q["mc_proj"]) - num(q["actual"])).abs()
             - (num(q["model_proj"]) - num(q["actual"])).abs()).mean()
        ),
        "selected_matches_mc_mean_side": float(
            q["side"].astype(str).str.upper().eq(q["mc_mean_side"]).mean()
        ),
        "selected_matches_final_mean_side": float(
            q["side"].astype(str).str.upper().eq(q["final_mean_side"]).mean()
        ),
    }


def contrast_row(df: pd.DataFrame, group_type: str, group_value: str) -> dict:
    crossed = df.loc[df["authority_state"].eq("CROSSED_LINE")]
    other = df.loc[~df["authority_state"].eq("CROSSED_LINE")]
    a = metrics(crossed)
    b = metrics(other)
    return {
        "group_type": group_type,
        "group_value": group_value,
        "crossed_rows": a["rows"],
        "crossed_clusters": a["clusters"],
        "noncrossed_rows": b["rows"],
        "noncrossed_clusters": b["clusters"],
        "crossed_win_rate": a["win_rate"],
        "noncrossed_win_rate": b["win_rate"],
        "win_rate_diff_crossed_minus_noncrossed": a["win_rate"] - b["win_rate"],
        "crossed_roi": a["roi"],
        "noncrossed_roi": b["roi"],
        "crossed_final_mae": a["final_mae"],
        "noncrossed_final_mae": b["final_mae"],
        "final_mae_diff_crossed_minus_noncrossed": a["final_mae"] - b["final_mae"],
        "crossed_model_closer_rate": a["model_closer_rate"],
        "noncrossed_model_closer_rate": b["model_closer_rate"],
        "model_closer_diff_crossed_minus_noncrossed": (
            a["model_closer_rate"] - b["model_closer_rate"]
        ),
    }


def cluster_bootstrap(df: pd.DataFrame) -> dict:
    # Resample game clusters by multiplicity, then calculate row-weighted
    # crossed/non-crossed metrics from pre-aggregated game totals. This is
    # algebraically equivalent to concatenating each sampled game's rows but
    # avoids materializing 10,000 full DataFrames.
    q = df.copy()
    q["_cross"] = q["authority_state"].eq("CROSSED_LINE")
    q["_win"] = q["bet_result"].eq("WIN").astype(float)
    q["_closer"] = q["model_closer_than_vegas"].astype(bool).astype(float)
    q["_final_abs_err"] = (num(q["model_proj"]) - num(q["actual"])).abs()

    agg = (
        q.groupby(["event_id", "_cross"], dropna=False)
        .agg(
            n=("_win", "size"),
            wins=("_win", "sum"),
            closer=("_closer", "sum"),
            final_abs=("_final_abs_err", "sum"),
        )
        .reset_index()
    )
    agg["event_id"] = agg["event_id"].astype(str)
    keys = sorted(q["event_id"].astype(str).unique().tolist())
    k = len(keys)
    if k == 0:
        raise RuntimeError("no game clusters for bootstrap")

    arrays = {}
    for crossed in (False, True):
        sub = agg.loc[agg["_cross"].eq(crossed)].set_index("event_id")
        for col in ("n", "wins", "closer", "final_abs"):
            arrays[(crossed, col)] = np.array(
                [float(sub[col].get(key, 0.0)) for key in keys], dtype=float
            )

    rng = np.random.default_rng(BOOT_SEED)
    weights = rng.multinomial(k, [1.0 / k] * k, size=BOOT_REPS)

    def rate(crossed: bool, numerator: str) -> np.ndarray:
        nume = weights @ arrays[(crossed, numerator)]
        deno = weights @ arrays[(crossed, "n")]
        return np.divide(
            nume,
            deno,
            out=np.full_like(nume, np.nan, dtype=float),
            where=deno > 0,
        )

    wr = rate(True, "wins") - rate(False, "wins")
    closer = rate(True, "closer") - rate(False, "closer")
    mae = rate(True, "final_abs") - rate(False, "final_abs")

    def ci(vals):
        arr = np.asarray(vals, dtype=float)
        arr = arr[np.isfinite(arr)]
        if not len(arr):
            raise RuntimeError("bootstrap produced no valid primary contrasts")
        return {
            "mean": float(np.mean(arr)),
            "lo": float(np.quantile(arr, 0.025)),
            "hi": float(np.quantile(arr, 0.975)),
            "reps": int(len(arr)),
        }

    return {
        "win_rate_diff": ci(wr),
        "model_closer_diff": ci(closer),
        "final_mae_diff": ci(mae),
    }

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    df = pd.read_csv(a.input)
    expected_rows = 1240
    if len(df) != expected_rows:
        raise RuntimeError(f"authority artifact row count drift: {len(df)} != {expected_rows}")

    required = {
        "vegas_line", "mc_proj", "model_proj", "actual", "vegas_odds",
        "event_id", "position", "market", "side", "bet_result",
        "unit_result", "model_closer_than_vegas",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"missing required columns: {missing}")

    finite = (
        num(df["vegas_line"]).notna()
        & num(df["mc_proj"]).notna()
        & num(df["model_proj"]).notna()
        & num(df["actual"]).notna()
        & num(df["vegas_odds"]).notna()
    )
    nonmissing = (
        df["event_id"].notna()
        & df["position"].notna()
        & df["market"].notna()
        & df["side"].notna()
        & df["bet_result"].notna()
    )
    eligible = df.loc[finite & nonmissing].copy()
    excluded = df.loc[~(finite & nonmissing)].copy()

    if num(eligible["unit_result"]).isna().any():
        raise RuntimeError("eligible rows contain non-finite unit_result")
    if eligible["model_closer_than_vegas"].isna().any():
        raise RuntimeError("eligible rows contain missing model_closer_than_vegas")

    eligible["mc_gap"] = num(eligible["mc_proj"]) - num(eligible["vegas_line"])
    eligible["final_gap"] = num(eligible["model_proj"]) - num(eligible["vegas_line"])
    eligible["authority_move"] = num(eligible["model_proj"]) - num(eligible["mc_proj"])
    eligible["mc_mean_side"] = mean_side(eligible["mc_gap"])
    eligible["final_mean_side"] = mean_side(eligible["final_gap"])
    eligible["authority_state"] = classify(eligible)

    states = []
    state_order = [
        "CROSSED_LINE",
        "FINAL_ON_LINE",
        "MC_ON_LINE",
        "UNCHANGED_ON_LINE",
        "SAME_SIDE_STRENGTHENED",
        "SAME_SIDE_WEAKENED",
        "SAME_SIDE_NO_MATERIAL_DISTANCE_CHANGE",
    ]
    for state in state_order:
        states.append({"authority_state": state, **metrics(eligible.loc[eligible["authority_state"].eq(state)])})
    state_df = pd.DataFrame(states)

    contrasts = [contrast_row(eligible, "ALL", "ALL")]
    for market, q in eligible.groupby("market", dropna=False):
        contrasts.append(contrast_row(q, "MARKET", str(market)))
    for position, q in eligible.groupby("position", dropna=False):
        contrasts.append(contrast_row(q, "POSITION", str(position)))
    contrast_df = pd.DataFrame(contrasts)

    primary = contrasts[0]
    boot = cluster_bootstrap(eligible)

    signal = (
        primary["crossed_rows"] >= 50
        and primary["crossed_clusters"] >= 15
        and primary["win_rate_diff_crossed_minus_noncrossed"] <= -0.05
        and primary["model_closer_diff_crossed_minus_noncrossed"] <= -0.08
        and (
            boot["win_rate_diff"]["hi"] < 0
            or boot["model_closer_diff"]["hi"] < 0
        )
    )
    disposition = (
        "AUTHORITY_LINE_CONFLICT_CURRENT_SEASON_SIGNAL"
        if signal
        else "NO_CLEAR_CURRENT_SEASON_AUTHORITY_LINE_CONFLICT_SIGNAL"
    )

    a.out_dir.mkdir(parents=True, exist_ok=True)
    eligible.to_csv(a.out_dir / "authority_line_conflict_row_detail.csv", index=False)
    state_df.to_csv(a.out_dir / "authority_line_conflict_state_summary.csv", index=False)
    contrast_df.to_csv(a.out_dir / "authority_line_conflict_contrasts.csv", index=False)

    report = [
        "=== WEEKS 1-3 PROJECTION AUTHORITY LINE-CONFLICT V1 ===",
        f"source_rows={len(df)} eligible_rows={len(eligible)} excluded_rows={len(excluded)}",
        f"unique_game_clusters={eligible['event_id'].astype(str).nunique()}",
        "",
        "STATE SUMMARY",
        state_df.to_string(index=False),
        "",
        "PRIMARY / DESCRIPTIVE CONTRASTS",
        contrast_df.to_string(index=False),
        "",
        "PRIMARY CLUSTER BOOTSTRAP",
        str(boot),
        "",
        f"DISPOSITION={disposition}",
        "NO_PRODUCTION_CHANGE_AUTHORIZED",
    ]
    text = "\n".join(report) + "\n"
    (a.out_dir / "authority_line_conflict_report.txt").write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
