"""Migration 94D: joint RB opportunity engine.

M93B found a real but incomplete pregame backfield-concentration signal. M94C
found a real but incomplete football-only team-rush/game-environment signal.
M94D tests the interaction directly while keeping M91 rushing efficiency and
receiving projections frozen:

    M94C team rushing opportunity
      x M93B pregame backfield concentration probability
      -> conditional RB-pool share sharpening
      -> player carries

Architecture choices are selected only on 2024 W13-18 after fitting the M93B
classifier on 2024 W1-12. The selected coupling is then frozen, the classifier
is refit on all 2024, and 2025 is untouched temporal validation.

No sportsbook fields are used and no production file is modified.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from scripts.backtest.evaluate_rb_team_rush_volume import (
    TEAM_KEYS,
    PLAYER_KEYS,
    _metrics,
    _player_candidate,
)

CONC_GATES = (0.50, 0.60, 0.70)
TEAM_RUSH_GATES = (26.0, 28.0, 30.0)
GAMMAS = (1.20, 1.40, 1.60, 1.80)
CONTINUOUS_CENTERS = (26.0, 28.0, 30.0)
CONTINUOUS_STRENGTHS = (0.40, 0.80, 1.20, 1.60)


def _find_one(root: Path, name: str) -> Path:
    hits = list(root.rglob(name))
    if len(hits) != 1:
        raise RuntimeError(f"expected exactly one {name} under {root}; found {len(hits)}")
    return hits[0]


def _lower(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def _read_pred(root: Path, season: int) -> pd.DataFrame:
    path = root / str(season) / "component_predictions.csv"
    if not path.exists():
        hits = [p for p in root.rglob("component_predictions.csv") if f"/{season}/" in str(p).replace("\\", "/")]
        if len(hits) != 1:
            raise RuntimeError(f"M94D could not locate {season} component_predictions.csv")
        path = hits[0]
    return _lower(pd.read_csv(path, low_memory=False))


def _m93b_feature_cols(x: pd.DataFrame) -> list[str]:
    blocked = {
        "season", "week", "team", "lead_player", "lead_key", "lead_role",
        "actual_team_top_share",
    }
    cols: list[str] = []
    for c in x.columns:
        if c in blocked:
            continue
        v = pd.to_numeric(x[c], errors="coerce")
        if v.notna().any():
            cols.append(c)
    return sorted(cols)


def _new_concentration_classifier() -> Pipeline:
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("scale", StandardScaler()),
        ("model", LogisticRegression(C=0.20, max_iter=1000, random_state=93)),
    ])


def _concentration_scores(m93b_root: Path) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    f24 = _lower(pd.read_csv(_find_one(m93b_root, "m93b_features_2024.csv"), low_memory=False))
    f25 = _lower(pd.read_csv(_find_one(m93b_root, "m93b_features_2025.csv"), low_memory=False))
    disp = _lower(pd.read_csv(_find_one(m93b_root, "m93b_disposition.csv"), low_memory=False))
    target_threshold = float(pd.to_numeric(disp["target_threshold"], errors="coerce").iloc[0])
    cols = sorted(set(_m93b_feature_cols(f24)) & set(_m93b_feature_cols(f25)))
    if not cols:
        raise RuntimeError("M94D found no common M93B concentration features")

    w24 = pd.to_numeric(f24["week"], errors="coerce")
    train = f24.loc[w24.le(12)].copy()
    hold = f24.loc[w24.ge(13)].copy()
    y_train = pd.to_numeric(train["actual_team_top_share"], errors="coerce").ge(target_threshold).astype(int)
    clf_hold = _new_concentration_classifier()
    clf_hold.fit(train[cols], y_train)
    s24 = hold[TEAM_KEYS + ["lead_key", "baseline_lead_share", "baseline_gap12", "baseline_hhi"]].copy()
    s24["concentration_probability"] = clf_hold.predict_proba(hold[cols])[:, 1]

    y_all = pd.to_numeric(f24["actual_team_top_share"], errors="coerce").ge(target_threshold).astype(int)
    clf25 = _new_concentration_classifier()
    clf25.fit(f24[cols], y_all)
    s25 = f25[TEAM_KEYS + ["lead_key", "baseline_lead_share", "baseline_gap12", "baseline_hhi"]].copy()
    s25["concentration_probability"] = clf25.predict_proba(f25[cols])[:, 1]
    return s24, s25, target_threshold


def _m94c_rb_frame(pred: pd.DataFrame, team_trace: pd.DataFrame) -> pd.DataFrame:
    team = team_trace[TEAM_KEYS + ["candidate_team_rush_att", "baseline_team_rush_att"]].copy()
    rb = _player_candidate(pred, team, "candidate_team_rush_att")
    rb = rb.rename(columns={
        "candidate_rush_att": "m94c_rush_att",
        "candidate_rush_yards": "m94c_rush_yards",
        "candidate_rush_rec_yards": "m94c_rush_rec_yards",
    })
    return rb


def _sigmoid(x: pd.Series | np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=float)
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20.0, 20.0)))


def _apply_joint(
    rb: pd.DataFrame,
    team_trace: pd.DataFrame,
    scores: pd.DataFrame,
    config: dict[str, float | str],
) -> pd.DataFrame:
    out = rb.copy()
    team_cols = TEAM_KEYS + [
        "candidate_team_rush_att", "baseline_team_rush_att",
        "pred_lead_play_share", "pred_trail_play_share", "pred_off_plays",
    ]
    keep = [c for c in team_cols if c in team_trace.columns]
    out = out.merge(team_trace[keep].drop_duplicates(TEAM_KEYS), on=TEAM_KEYS, how="left", validate="many_to_one")
    out = out.merge(scores[TEAM_KEYS + ["concentration_probability", "lead_key"]], on=TEAM_KEYS, how="left", validate="many_to_one")
    out["concentration_probability"] = pd.to_numeric(out["concentration_probability"], errors="coerce").fillna(0.0).clip(0.0, 1.0)
    out["joint_is_lead"] = out["player_clean_key"].astype(str).eq(out["lead_key"].astype(str))

    rb_pool = out.groupby(TEAM_KEYS)["m94c_rush_att"].transform(lambda s: s.sum(min_count=1))
    base_pool = out.groupby(TEAM_KEYS)["base_rush_att"].transform(lambda s: s.sum(min_count=1))
    base_share = np.where(base_pool.gt(0), out["base_rush_att"] / base_pool, 0.0)
    out["m94c_rb_pool"] = rb_pool
    out["base_rb_pool_share"] = base_share

    team_rush = pd.to_numeric(out["candidate_team_rush_att"], errors="coerce")
    base_team = pd.to_numeric(out["baseline_team_rush_att"], errors="coerce").replace(0, np.nan)
    out["team_volume_ratio"] = (team_rush / base_team).replace([np.inf, -np.inf], np.nan).fillna(1.0)

    mode = str(config["mode"])
    if mode == "gate":
        active = (
            out["concentration_probability"].ge(float(config["conc_gate"]))
            & team_rush.ge(float(config["team_rush_gate"]))
        )
        gamma = np.where(active, float(config["gamma"]), 1.0)
        joint_score = out["concentration_probability"] * _sigmoid((team_rush - float(config["team_rush_gate"])) / 3.0)
    elif mode == "continuous":
        vf = _sigmoid((team_rush - float(config["center"])) / 3.0)
        joint_score = out["concentration_probability"] * vf
        gamma = 1.0 + float(config["strength"]) * joint_score
        active = gamma > 1.05
    else:
        raise ValueError(f"unknown M94D mode: {mode}")

    out["joint_score"] = np.asarray(joint_score, dtype=float)
    out["joint_active"] = np.asarray(active, dtype=bool)
    out["joint_gamma"] = np.asarray(gamma, dtype=float)
    raw = np.power(np.clip(np.asarray(base_share, dtype=float), 1e-12, None), out["joint_gamma"].to_numpy(dtype=float))
    out["_raw_share"] = raw
    denom = out.groupby(TEAM_KEYS)["_raw_share"].transform("sum")
    out["candidate_rb_pool_share"] = np.where(denom.gt(0), out["_raw_share"] / denom, base_share)
    out["candidate_rush_att"] = out["candidate_rb_pool_share"] * rb_pool

    ypc = np.where(
        pd.to_numeric(out["base_rush_att"], errors="coerce").gt(0.5),
        pd.to_numeric(out["base_rush_yards"], errors="coerce") / pd.to_numeric(out["base_rush_att"], errors="coerce"),
        np.nan,
    )
    ypc = pd.Series(ypc, index=out.index).clip(lower=0.0, upper=12.0)
    out["candidate_rush_yards"] = np.where(ypc.notna(), out["candidate_rush_att"] * ypc, out["base_rush_yards"])
    out["candidate_rush_rec_yards"] = out["base_rush_rec_yards"] + out["candidate_rush_yards"] - out["base_rush_yards"]
    return out.drop(columns=["_raw_share"])


def _slices(x: pd.DataFrame) -> dict[str, pd.Series]:
    a = pd.to_numeric(x["actual_rush_att"], errors="coerce")
    return {
        "all_rb": pd.Series(True, index=x.index),
        "actual_0_5": a.le(5),
        "actual_6_10": a.between(6, 10),
        "actual_11_14": a.between(11, 14),
        "actual_15_plus": a.ge(15),
        "actual_20_plus": a.ge(20),
        "actual_25_plus": a.ge(25),
        "bellcow_60": x["bellcow_60"].fillna(False),
    }


def _score(x: pd.DataFrame, scope: str) -> pd.DataFrame:
    rows: list[dict] = []
    markets = [
        ("rush_att", "actual_rush_att", "base_rush_att", "m94c_rush_att", "candidate_rush_att"),
        ("rush_yards", "actual_rush_yards", "base_rush_yards", "m94c_rush_yards", "candidate_rush_yards"),
        ("rush_rec_yards", "actual_rush_rec_yards", "base_rush_rec_yards", "m94c_rush_rec_yards", "candidate_rush_rec_yards"),
    ]
    for slice_name, mask in _slices(x).items():
        g = x.loc[mask]
        for market, actual, m91, m94c, m94d in markets:
            a = _metrics(g[actual], g[m91])
            b = _metrics(g[actual], g[m94c])
            c = _metrics(g[actual], g[m94d])
            rows.append({
                "season_scope": scope, "slice": slice_name, "market": market, "n": c["n"],
                "m91_mae": a["mae"], "m94c_mae": b["mae"], "m94d_mae": c["mae"],
                "gain_vs_m91": a["mae"] - c["mae"], "gain_vs_m94c": b["mae"] - c["mae"],
                "m91_bias": a["bias"], "m94c_bias": b["bias"], "m94d_bias": c["bias"],
                "m91_corr": a["correlation"], "m94c_corr": b["correlation"], "m94d_corr": c["correlation"],
            })
    return pd.DataFrame(rows)


def _v(t: pd.DataFrame, slice_name: str, market: str, col: str) -> float:
    q = t.loc[t["slice"].eq(slice_name) & t["market"].eq(market), col]
    return float(q.iloc[0]) if len(q) else np.nan


def _grid(hold_rb: pd.DataFrame, hold_team: pd.DataFrame, hold_scores: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, float | str]]:
    configs: list[dict[str, float | str]] = []
    for cg in CONC_GATES:
        for tg in TEAM_RUSH_GATES:
            for gamma in GAMMAS:
                configs.append({"mode": "gate", "conc_gate": cg, "team_rush_gate": tg, "gamma": gamma})
    for center in CONTINUOUS_CENTERS:
        for strength in CONTINUOUS_STRENGTHS:
            configs.append({"mode": "continuous", "center": center, "strength": strength})

    rows: list[dict] = []
    for i, cfg in enumerate(configs):
        trace = _apply_joint(hold_rb, hold_team, hold_scores, cfg)
        score = _score(trace, "2024_w13_18_holdout")
        all_att = _v(score, "all_rb", "rush_att", "gain_vs_m94c")
        all_ry = _v(score, "all_rb", "rush_yards", "gain_vs_m94c")
        low = _v(score, "actual_0_5", "rush_att", "gain_vs_m94c")
        mid6 = _v(score, "actual_6_10", "rush_att", "gain_vs_m94c")
        mid11 = _v(score, "actual_11_14", "rush_att", "gain_vs_m94c")
        t20 = _v(score, "actual_20_plus", "rush_att", "gain_vs_m94c")
        t25 = _v(score, "actual_25_plus", "rush_att", "gain_vs_m94c")
        t20y = _v(score, "actual_20_plus", "rush_yards", "gain_vs_m94c")
        eligible = (
            all_att >= 0.0
            and all_ry >= -0.10
            and low >= -0.10
            and mid6 >= -0.15
            and mid11 >= -0.15
            and t20 > 0.0
        )
        objective = all_att + 0.60 * t20 + 0.30 * (0.0 if not np.isfinite(t25) else t25) + 0.02 * t20y + 0.02 * all_ry
        rows.append({
            "config_id": i, **cfg, "eligible": int(eligible), "objective": objective,
            "all_att_gain_vs_m94c": all_att, "all_rush_yards_gain_vs_m94c": all_ry,
            "low_0_5_att_gain_vs_m94c": low, "mid_6_10_att_gain_vs_m94c": mid6,
            "mid_11_14_att_gain_vs_m94c": mid11, "tail_20_att_gain_vs_m94c": t20,
            "tail_25_att_gain_vs_m94c": t25, "tail_20_rush_yards_gain_vs_m94c": t20y,
            "fraction_active": float(trace[TEAM_KEYS + ["joint_active"]].drop_duplicates()["joint_active"].mean()),
        })
    grid = pd.DataFrame(rows)
    eligible = grid.loc[grid["eligible"].eq(1)].copy()
    src = eligible if len(eligible) else grid
    pick = src.sort_values(["objective", "tail_20_att_gain_vs_m94c", "all_att_gain_vs_m94c"], ascending=False).iloc[0]
    cfg = {k: pick[k] for k in ["mode", "conc_gate", "team_rush_gate", "gamma", "center", "strength"] if k in pick.index and pd.notna(pick[k])}
    return grid, cfg


def _tail_diag(x: pd.DataFrame) -> pd.DataFrame:
    a = pd.to_numeric(x["actual_rush_att"], errors="coerce")
    rows: list[dict] = []
    for name, mask in {"actual_20_plus": a.ge(20), "actual_25_plus": a.ge(25)}.items():
        g = x.loc[mask]
        row: dict[str, float | int | str] = {"slice": name, "n": len(g)}
        for label, col in [("actual", "actual_rush_att"), ("m91", "base_rush_att"), ("m94c", "m94c_rush_att"), ("m94d", "candidate_rush_att")]:
            v = pd.to_numeric(g[col], errors="coerce").dropna()
            row[f"{label}_mean"] = float(v.mean()) if len(v) else np.nan
            row[f"{label}_median"] = float(v.median()) if len(v) else np.nan
            row[f"{label}_min"] = float(v.min()) if len(v) else np.nan
            row[f"{label}_max"] = float(v.max()) if len(v) else np.nan
        row["m91_under_rate"] = float((pd.to_numeric(g["base_rush_att"], errors="coerce") < pd.to_numeric(g["actual_rush_att"], errors="coerce")).mean()) if len(g) else np.nan
        row["m94c_under_rate"] = float((pd.to_numeric(g["m94c_rush_att"], errors="coerce") < pd.to_numeric(g["actual_rush_att"], errors="coerce")).mean()) if len(g) else np.nan
        row["m94d_under_rate"] = float((pd.to_numeric(g["candidate_rush_att"], errors="coerce") < pd.to_numeric(g["actual_rush_att"], errors="coerce")).mean()) if len(g) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _joint_calibration(x: pd.DataFrame) -> pd.DataFrame:
    lead = x.loc[x["joint_is_lead"]].copy()
    if lead.empty:
        return pd.DataFrame()
    lead["joint_bucket"] = pd.qcut(lead["joint_score"].rank(method="first"), q=5, labels=["q1", "q2", "q3", "q4", "q5"])
    rows = []
    for b, g in lead.groupby("joint_bucket", observed=True):
        actual = pd.to_numeric(g["actual_rush_att"], errors="coerce")
        rows.append({
            "joint_bucket": str(b), "n": len(g), "mean_joint_score": float(g["joint_score"].mean()),
            "actual_carries_mean": float(actual.mean()), "actual_20_plus_rate": float(actual.ge(20).mean()),
            "actual_25_plus_rate": float(actual.ge(25).mean()), "m94d_carries_mean": float(pd.to_numeric(g["candidate_rush_att"], errors="coerce").mean()),
        })
    return pd.DataFrame(rows)


def _legacy_guard(pred25: pd.DataFrame, rb25: pd.DataFrame) -> pd.DataFrame:
    all_ry = pred25.loc[pred25["market"].astype(str).str.lower().eq("rush_yards"), PLAYER_KEYS + ["position", "actual", "ml_proj"]].copy()
    all_ry = all_ry.merge(rb25[PLAYER_KEYS + ["candidate_rush_yards"]], on=PLAYER_KEYS, how="left", validate="many_to_one")
    all_ry["m94d"] = np.where(
        all_ry["position"].astype(str).str.upper().eq("RB") & all_ry["candidate_rush_yards"].notna(),
        all_ry["candidate_rush_yards"], all_ry["ml_proj"],
    )
    b = _metrics(all_ry["actual"], all_ry["ml_proj"])
    c = _metrics(all_ry["actual"], all_ry["m94d"])
    return pd.DataFrame([{
        "n": b["n"], "m91_all_player_rush_yards_mae": b["mae"], "m94d_all_player_rush_yards_mae": c["mae"],
        "mae_gain": b["mae"] - c["mae"], "m91_corr": b["correlation"], "m94d_corr": c["correlation"],
    }])


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--m91-root", type=Path, required=True)
    p.add_argument("--m93b-root", type=Path, required=True)
    p.add_argument("--m94c-root", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, default=Path("data/backtests/rb_m94d"))
    args = p.parse_args()

    pred24 = _read_pred(args.m91_root, 2024)
    pred25 = _read_pred(args.m91_root, 2025)
    team24 = _lower(pd.read_csv(_find_one(args.m94c_root, "m94c_2024_holdout_trace.csv"), low_memory=False))
    team25 = _lower(pd.read_csv(_find_one(args.m94c_root, "m94c_2025_team_trace.csv"), low_memory=False))
    score24, score25, target_threshold = _concentration_scores(args.m93b_root)

    rb24 = _m94c_rb_frame(pred24, team24)
    rb25 = _m94c_rb_frame(pred25, team25)
    grid, config = _grid(rb24, team24, score24)
    val = _apply_joint(rb25, team25, score25, config)
    hold = _apply_joint(rb24, team24, score24, config)

    hold_summary = _score(hold, "2024_w13_18_holdout")
    val_summary = _score(val, "2025_validation")
    summary = pd.concat([hold_summary, val_summary], ignore_index=True)
    tail = _tail_diag(val)
    calibration = _joint_calibration(val)
    guard = _legacy_guard(pred25, val)

    pass_gate = (
        _v(val_summary, "all_rb", "rush_att", "gain_vs_m94c") > 0
        and _v(val_summary, "actual_20_plus", "rush_att", "gain_vs_m94c") > 0
        and _v(val_summary, "actual_25_plus", "rush_att", "gain_vs_m94c") > 0
        and _v(val_summary, "all_rb", "rush_yards", "gain_vs_m94c") >= 0
        and _v(val_summary, "actual_20_plus", "rush_yards", "gain_vs_m94c") > 0
        and _v(val_summary, "actual_0_5", "rush_att", "gain_vs_m94c") >= -0.10
        and _v(val_summary, "actual_6_10", "rush_att", "gain_vs_m94c") >= -0.15
        and _v(val_summary, "actual_11_14", "rush_att", "gain_vs_m94c") >= -0.15
        and float(guard["mae_gain"].iloc[0]) >= 0
    )
    disposition = pd.DataFrame([{
        **config,
        "m93b_target_threshold": target_threshold,
        "development_train_weeks": "2024_01-12_concentration_fit",
        "development_holdout_weeks": "2024_13-18_joint_selection",
        "validation_season": 2025,
        "validation_pass": int(pass_gate),
        "disposition": "ADVANCE_JOINT_OPPORTUNITY_SIGNAL" if pass_gate else "DO_NOT_ADVANCE_M94D_TO_PRODUCTION",
        "note": "Research only; M91 efficiency/receiving frozen; no sportsbook inputs.",
    }])

    args.out_dir.mkdir(parents=True, exist_ok=True)
    grid.to_csv(args.out_dir / "m94d_2024_joint_grid.csv", index=False)
    summary.to_csv(args.out_dir / "m94d_rb_summary.csv", index=False)
    tail.to_csv(args.out_dir / "m94d_2025_tail_diagnostics.csv", index=False)
    calibration.to_csv(args.out_dir / "m94d_2025_joint_calibration.csv", index=False)
    guard.to_csv(args.out_dir / "m94d_legacy_guard.csv", index=False)
    disposition.to_csv(args.out_dir / "m94d_disposition.csv", index=False)
    score24.to_csv(args.out_dir / "m94d_2024_concentration_scores.csv", index=False)
    score25.to_csv(args.out_dir / "m94d_2025_concentration_scores.csv", index=False)
    hold.to_csv(args.out_dir / "m94d_2024_holdout_rb_trace.csv", index=False)
    val.to_csv(args.out_dir / "m94d_2025_rb_trace.csv", index=False)

    print("[rb_m94d] selected 2024 joint coupling")
    print(disposition.to_string(index=False))
    print("\n[rb_m94d] 2025 RB validation vs M91 and M94C")
    print(val_summary.loc[val_summary["slice"].isin(["all_rb", "actual_0_5", "actual_6_10", "actual_11_14", "actual_15_plus", "actual_20_plus", "actual_25_plus", "bellcow_60"])].to_string(index=False))
    print("\n[rb_m94d] 2025 carry tails")
    print(tail.to_string(index=False))
    print("\n[rb_m94d] joint-score calibration")
    print(calibration.to_string(index=False))
    print("\n[rb_m94d] legacy guard")
    print(guard.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
