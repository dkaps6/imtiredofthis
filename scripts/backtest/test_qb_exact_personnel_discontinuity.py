#!/usr/bin/env python3
"""Migration 77: QB exact-personnel discontinuity correction.

Single pre-registered predictive test using genuinely new pregame information
qualified by M76. The frozen canonical-v3 football-only baseline is never rebuilt.

Design:
- 2024 = development/training only.
- 2025 = untouched final evaluation for this migration.
- One model family only: standardized Ridge(alpha=10, fit_intercept=False).
- Separate corrections for attempt residual and YPA residual.
- Personnel features only; no market, game-line, generic injury, DBR, pace,
  playcaller, aggregate-defense, or baseline-model remix features.
- No post-result retuning/model substitution on this feature universe.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

from scripts.backtest import audit_qb_40s_information_frontier as m76

EXPECTED_CANONICAL_SHA256 = "c4a481a760657bb7516d52b9aa9ba84af096063a4e78660b12a541f59fd7b742"
EXPECTED_ROWS = 884
EXPECTED_SEASONS = {2024: 444, 2025: 440}

# Frozen before first M77 result.
RIDGE_ALPHA = 10.0
BOOTSTRAP_N = 2000
BOOTSTRAP_SEED = 77
MIN_EVAL_FEATURE_COVERAGE = 0.85
PROMOTION_MAE_GAIN = 1.50
PROMOTION_CORR_GAIN = 0.02
DEV_SPLIT_WEEK = 9

FEATURES = [
    "off_ol_turnover",
    "off_ol_added_ratio",
    "off_ol_replacement_deficit",
    "off_ol_role_delta",
    "off_skill_turnover",
    "off_skill_added_ratio",
    "off_skill_replacement_deficit",
    "off_skill_role_delta",
    "def_db_turnover",
    "def_db_added_ratio",
    "def_db_replacement_deficit",
    "def_db_role_delta",
    "def_rush_turnover",
    "def_rush_added_ratio",
    "def_rush_replacement_deficit",
    "def_rush_role_delta",
    "def_rush_pressure_quality",
    "def_rush_pressure_delta",
    "def_rush_sack_quality",
    "def_rush_sack_delta",
    "ol_turnover_x_rush_pressure",
    "ol_replacement_x_rush_pressure",
    "skill_turnover_x_db_turnover",
]

MARKET_TOKENS = (
    "market", "spread", "moneyline", "sportsbook", "implied_total",
    "game_total", "team_total", "vegas",
)


def num(x):
    return pd.to_numeric(x, errors="coerce")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require_canonical(path: Path) -> pd.DataFrame:
    digest = sha256_file(path)
    if digest != EXPECTED_CANONICAL_SHA256:
        raise RuntimeError(f"canonical-v3 SHA drift: {digest}")
    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(c).strip().lower() for c in df.columns]
    if len(df) != EXPECTED_ROWS:
        raise RuntimeError(f"expected {EXPECTED_ROWS} rows, got {len(df)}")
    counts = {int(k): int(v) for k, v in num(df["season"]).value_counts().to_dict().items()}
    if counts != EXPECTED_SEASONS:
        raise RuntimeError(f"canonical season-count drift: {counts}")
    bad = [c for c in df.columns if any(tok in c for tok in MARKET_TOKENS)]
    if bad:
        raise RuntimeError(f"market boundary violated by canonical columns: {bad}")
    required = {
        "season", "week", "team", "opponent", "player_clean_key",
        "actual_pass_yards", "actual_attempts", "pred_pass_yards", "pred_attempts",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"canonical missing columns: {missing}")
    df["season"] = num(df["season"]).astype(int)
    df["week"] = num(df["week"]).astype(int)
    df["team"] = df["team"].map(m76.team_value)
    df["opponent"] = df["opponent"].map(m76.team_value)
    df["base_ypa"] = num(df["pred_pass_yards"]) / num(df["pred_attempts"]).replace(0, np.nan)
    df["actual_ypa_calc"] = num(df["actual_pass_yards"]) / num(df["actual_attempts"]).replace(0, np.nan)
    return df


def pct_fraction(s: pd.Series) -> pd.Series:
    raw = s.astype("string").str.strip()
    has_pct = raw.str.endswith("%", na=False)
    vals = pd.to_numeric(raw.str.rstrip("%"), errors="coerce")
    vals.loc[has_pct] = vals.loc[has_pct] / 100.0
    vals.loc[vals > 1.5] = vals.loc[vals > 1.5] / 100.0
    return vals.clip(lower=0.0, upper=1.0)


def first_metric(df: pd.DataFrame, names: list[str]) -> str | None:
    for c in names:
        if c in df.columns:
            return c
    return None


def prepare_snap_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    season = m76.first_col(raw, ["season"])
    week = m76.first_col(raw, ["week"])
    player = m76.first_col(raw, ["gsis_id"])
    if not all([season, week, player]):
        return pd.DataFrame()
    off_pct = first_metric(raw, ["offense_pct", "off_pct"])
    def_pct = first_metric(raw, ["defense_pct", "def_pct"])
    out = pd.DataFrame({
        "season": num(raw[season]),
        "week": num(raw[week]),
        "player_id": m76.clean_id(raw[player]),
        "off_pct": pct_fraction(raw[off_pct]) if off_pct else np.nan,
        "def_pct": pct_fraction(raw[def_pct]) if def_pct else np.nan,
    })
    out = out.dropna(subset=["season", "week", "player_id"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def prepare_pfr_history(raw: pd.DataFrame) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()
    season = m76.first_col(raw, ["season"])
    week = m76.first_col(raw, ["week"])
    player = m76.first_col(raw, ["gsis_id"])
    if not all([season, week, player]):
        return pd.DataFrame()

    def choose(tokens: tuple[str, ...]) -> str | None:
        exact = [c for c in raw.columns if c.startswith("def_") and any(t in c for t in tokens)]
        if exact:
            return exact[0]
        loose = [c for c in raw.columns if any(t in c for t in tokens)]
        return loose[0] if loose else None

    press = choose(("pressure", "pressures"))
    sacks = choose(("sack", "sacks"))
    hurr = choose(("hurr", "hurried"))
    blitz = choose(("blitz",))
    out = pd.DataFrame({
        "season": num(raw[season]),
        "week": num(raw[week]),
        "player_id": m76.clean_id(raw[player]),
        "pressures": num(raw[press]) if press else np.nan,
        "sacks": num(raw[sacks]) if sacks else np.nan,
        "hurries": num(raw[hurr]) if hurr else np.nan,
        "blitzes": num(raw[blitz]) if blitz else np.nan,
    })
    out = out.dropna(subset=["season", "week", "player_id"])
    out["season"] = out["season"].astype(int)
    out["week"] = out["week"].astype(int)
    return out.sort_values(["player_id", "season", "week"]).reset_index(drop=True)


def history_index(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {}
    return {str(pid): g.reset_index(drop=True) for pid, g in df.groupby("player_id", sort=False)}


def prior_rows(index: dict[str, pd.DataFrame], player: str, season: int, week: int, last_n: int = 4) -> pd.DataFrame:
    g = index.get(str(player))
    if g is None or g.empty:
        return pd.DataFrame()
    q = g.loc[(g["season"] < season) | ((g["season"] == season) & (g["week"] < week))]
    if q.empty:
        return q
    return q.sort_values(["season", "week"]).tail(last_n)


def prior_role(index: dict[str, pd.DataFrame], player: str, season: int, week: int, side: str) -> float:
    q = prior_rows(index, player, season, week)
    col = "off_pct" if side == "off" else "def_pct"
    if q.empty or col not in q:
        return 0.0
    v = num(q[col]).dropna()
    return float(v.mean()) if len(v) else 0.0


def prior_pfr(index: dict[str, pd.DataFrame], player: str, season: int, week: int, metric: str) -> float:
    q = prior_rows(index, player, season, week)
    if q.empty or metric not in q:
        return 0.0
    v = num(q[metric]).dropna()
    return float(v.mean()) if len(v) else 0.0


def starter_set(depth: pd.DataFrame, group: str) -> set[str]:
    if depth.empty:
        return set()
    q = depth.loc[depth["group"].eq(group) & num(depth["rank"]).le(1), "player_id"].dropna()
    return set(q.astype(str))


def jaccard_retention(current: set[str], previous: set[str]) -> float:
    union = current | previous
    if not union:
        return np.nan
    return len(current & previous) / len(union)


def group_discontinuity(
    current: set[str],
    previous: set[str],
    snap_idx: dict[str, pd.DataFrame],
    season: int,
    week: int,
    side: str,
) -> dict[str, float]:
    if not current or not previous:
        return {"turnover": np.nan, "added_ratio": np.nan, "replacement_deficit": np.nan, "role_delta": np.nan}
    added = current - previous
    retention = jaccard_retention(current, previous)
    current_roles = [prior_role(snap_idx, p, season, week, side) for p in current]
    previous_roles = [prior_role(snap_idx, p, season, week, side) for p in previous]
    added_roles = [prior_role(snap_idx, p, season, week, side) for p in added]
    replacement_deficit = (
        sum(1.0 - min(max(x, 0.0), 1.0) for x in added_roles) / max(len(current), 1)
        if added else 0.0
    )
    return {
        "turnover": float(1.0 - retention),
        "added_ratio": float(len(added) / max(len(current), 1)),
        "replacement_deficit": float(replacement_deficit),
        "role_delta": float(np.mean(current_roles) - np.mean(previous_roles)),
    }


def rusher_quality(players: set[str], pfr_idx: dict[str, pd.DataFrame], season: int, week: int) -> tuple[float, float]:
    if not players:
        return np.nan, np.nan
    pressures = [prior_pfr(pfr_idx, p, season, week, "pressures") for p in players]
    sacks = [prior_pfr(pfr_idx, p, season, week, "sacks") for p in players]
    return float(np.sum(pressures)), float(np.sum(sacks))


def build_team_snapshots(base: pd.DataFrame, depth: dict[int, pd.DataFrame], schedule: pd.DataFrame):
    targets = m76.target_team_weeks(base, schedule)
    snapshots: dict[tuple[int, int, str], pd.DataFrame] = {}
    for r in targets.itertuples(index=False):
        key = (int(r.season), int(r.week), str(r.team))
        snapshots[key] = m76.latest_depth_for_target(
            depth[int(r.season)], int(r.season), int(r.week), str(r.team), r.kickoff
        )

    prev_key: dict[tuple[int, int, str], tuple[int, int, str] | None] = {}
    for (_, _), g in targets.sort_values(["season", "team", "week"]).groupby(["season", "team"], sort=False):
        keys = [(int(x.season), int(x.week), str(x.team)) for x in g.itertuples(index=False)]
        for i, key in enumerate(keys):
            prev_key[key] = keys[i - 1] if i > 0 else None
    return snapshots, prev_key


def load_personnel_sources(base: pd.DataFrame, out_dir: Path):
    as_of = datetime.now(timezone.utc).isoformat()
    source_meta: list[dict] = []

    schedule = m76.load_schedule(source_meta, as_of)
    depth_raw = {
        s: m76.download_table("nflverse_depth_charts", s, m76.release_urls("depth", s), source_meta, as_of)
        for s in [2024, 2025]
    }
    depth = {s: m76.prepare_depth(depth_raw[s], s) for s in [2024, 2025]}

    snap_parts = [
        m76.download_table("pfr_snap_counts", s, m76.release_urls("snaps", s), source_meta, as_of)
        for s in [2023, 2024, 2025]
    ]
    pfr_parts = [
        m76.download_table("pfr_individual_pass_rush", s, m76.release_urls("pfr", s), source_meta, as_of)
        for s in [2023, 2024, 2025]
    ]
    snap_raw = pd.concat([x for x in snap_parts if not x.empty], ignore_index=True) if any(not x.empty for x in snap_parts) else pd.DataFrame()
    pfr_raw = pd.concat([x for x in pfr_parts if not x.empty], ignore_index=True) if any(not x.empty for x in pfr_parts) else pd.DataFrame()
    snap_raw, pfr_raw, bridge_detail = m76.build_id_bridge(snap_raw, pfr_raw, source_meta, as_of)

    snap_hist = prepare_snap_history(snap_raw)
    pfr_hist = prepare_pfr_history(pfr_raw)

    pd.DataFrame(source_meta).to_csv(out_dir / "m77_source_snapshot_hashes.csv", index=False)
    (out_dir / "m77_source_contract.json").write_text(
        json.dumps({
            "as_of_utc": as_of,
            "bridge": bridge_detail,
            "snap_rows": int(len(snap_hist)),
            "pfr_rows": int(len(pfr_hist)),
            "target_game_outcomes_as_features": False,
            "injury_feed_used": False,
            "sportsbook_used": False,
        }, indent=2),
        encoding="utf-8",
    )
    return schedule, depth, snap_hist, pfr_hist


def build_features(base: pd.DataFrame, schedule: pd.DataFrame, depth: dict[int, pd.DataFrame], snap_hist: pd.DataFrame, pfr_hist: pd.DataFrame) -> pd.DataFrame:
    depth_snap, prev_key = build_team_snapshots(base, depth, schedule)
    snap_idx = history_index(snap_hist)
    pfr_idx = history_index(pfr_hist)

    rows = []
    for r in base.itertuples(index=False):
        season, week = int(r.season), int(r.week)
        off_key = (season, week, str(r.team))
        def_key = (season, week, str(r.opponent))
        off_prev_key = prev_key.get(off_key)
        def_prev_key = prev_key.get(def_key)

        current_off = depth_snap.get(off_key, pd.DataFrame())
        current_def = depth_snap.get(def_key, pd.DataFrame())
        previous_off = depth_snap.get(off_prev_key, pd.DataFrame()) if off_prev_key else pd.DataFrame()
        previous_def = depth_snap.get(def_prev_key, pd.DataFrame()) if def_prev_key else pd.DataFrame()

        off_ol_cur, off_ol_prev = starter_set(current_off, "OL"), starter_set(previous_off, "OL")
        off_sk_cur, off_sk_prev = starter_set(current_off, "WR_TE_RB"), starter_set(previous_off, "WR_TE_RB")
        db_cur, db_prev = starter_set(current_def, "DB"), starter_set(previous_def, "DB")
        rush_cur, rush_prev = starter_set(current_def, "PASS_RUSH"), starter_set(previous_def, "PASS_RUSH")

        covered = bool(off_ol_cur and off_ol_prev and off_sk_cur and off_sk_prev and db_cur and db_prev and rush_cur and rush_prev)

        ol = group_discontinuity(off_ol_cur, off_ol_prev, snap_idx, season, week, "off")
        sk = group_discontinuity(off_sk_cur, off_sk_prev, snap_idx, season, week, "off")
        db = group_discontinuity(db_cur, db_prev, snap_idx, season, week, "def")
        rush = group_discontinuity(rush_cur, rush_prev, snap_idx, season, week, "def")
        cur_press, cur_sack = rusher_quality(rush_cur, pfr_idx, season, week)
        prev_press, prev_sack = rusher_quality(rush_prev, pfr_idx, season, week)

        rec = {
            "season": season,
            "week": week,
            "team": r.team,
            "opponent": r.opponent,
            "player_clean_key": r.player_clean_key,
            "personnel_feature_covered": covered,
            "off_ol_turnover": ol["turnover"],
            "off_ol_added_ratio": ol["added_ratio"],
            "off_ol_replacement_deficit": ol["replacement_deficit"],
            "off_ol_role_delta": ol["role_delta"],
            "off_skill_turnover": sk["turnover"],
            "off_skill_added_ratio": sk["added_ratio"],
            "off_skill_replacement_deficit": sk["replacement_deficit"],
            "off_skill_role_delta": sk["role_delta"],
            "def_db_turnover": db["turnover"],
            "def_db_added_ratio": db["added_ratio"],
            "def_db_replacement_deficit": db["replacement_deficit"],
            "def_db_role_delta": db["role_delta"],
            "def_rush_turnover": rush["turnover"],
            "def_rush_added_ratio": rush["added_ratio"],
            "def_rush_replacement_deficit": rush["replacement_deficit"],
            "def_rush_role_delta": rush["role_delta"],
            "def_rush_pressure_quality": cur_press,
            "def_rush_pressure_delta": cur_press - prev_press if pd.notna(cur_press) and pd.notna(prev_press) else np.nan,
            "def_rush_sack_quality": cur_sack,
            "def_rush_sack_delta": cur_sack - prev_sack if pd.notna(cur_sack) and pd.notna(prev_sack) else np.nan,
        }
        rec["ol_turnover_x_rush_pressure"] = rec["off_ol_turnover"] * rec["def_rush_pressure_quality"] if covered else np.nan
        rec["ol_replacement_x_rush_pressure"] = rec["off_ol_replacement_deficit"] * rec["def_rush_pressure_quality"] if covered else np.nan
        rec["skill_turnover_x_db_turnover"] = rec["off_skill_turnover"] * rec["def_db_turnover"] if covered else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)


def metrics(actual, pred) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "rmse": np.nan, "bias": np.nan, "corr": np.nan, "tail100": 0}
    err = z["pred"] - z["actual"]
    return {
        "n": int(len(z)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "bias": float(err.mean()),
        "corr": float(z["actual"].corr(z["pred"])) if len(z) > 2 else np.nan,
        "tail100": int(err.abs().ge(100).sum()),
    }


def component_metrics(actual, pred) -> dict:
    z = pd.DataFrame({"actual": num(actual), "pred": num(pred)}).dropna()
    if z.empty:
        return {"n": 0, "mae": np.nan, "bias": np.nan, "corr": np.nan}
    err = z["pred"] - z["actual"]
    return {
        "n": int(len(z)),
        "mae": float(err.abs().mean()),
        "bias": float(err.mean()),
        "corr": float(z["actual"].corr(z["pred"])) if len(z) > 2 else np.nan,
    }


def bootstrap_gain(actual, base_pred, new_pred, seed: int) -> dict:
    z = pd.DataFrame({"actual": num(actual), "base": num(base_pred), "new": num(new_pred)}).dropna()
    if z.empty:
        return {"gain": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    gain_rows = (z["base"] - z["actual"]).abs() - (z["new"] - z["actual"]).abs()
    rng = np.random.default_rng(seed)
    n = len(gain_rows)
    vals = gain_rows.to_numpy()
    sims = np.empty(BOOTSTRAP_N, dtype=float)
    for i in range(BOOTSTRAP_N):
        sims[i] = vals[rng.integers(0, n, size=n)].mean()
    return {"gain": float(vals.mean()), "ci_low": float(np.quantile(sims, 0.025)), "ci_high": float(np.quantile(sims, 0.975))}


class PersonnelCorrection:
    def __init__(self):
        self.scaler = StandardScaler()
        self.attempt = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)
        self.ypa = Ridge(alpha=RIDGE_ALPHA, fit_intercept=False)

    def fit(self, df: pd.DataFrame):
        X = df[FEATURES].astype(float)
        Xs = self.scaler.fit_transform(X)
        att_y = num(df["actual_attempts"]) - num(df["pred_attempts"])
        ypa_y = num(df["actual_ypa_calc"]) - num(df["base_ypa"])
        self.attempt.fit(Xs, att_y)
        self.ypa.fit(Xs, ypa_y)
        return self

    def predict(self, df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        Xs = self.scaler.transform(df[FEATURES].astype(float))
        return self.attempt.predict(Xs), self.ypa.predict(Xs)


def eligible(df: pd.DataFrame) -> pd.Series:
    return df["personnel_feature_covered"].fillna(False) & df[FEATURES].notna().all(axis=1)


def apply_model(model: PersonnelCorrection, df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["m77_attempt_correction"] = 0.0
    out["m77_ypa_correction"] = 0.0
    mask = eligible(out)
    if mask.any():
        ac, yc = model.predict(out.loc[mask])
        out.loc[mask, "m77_attempt_correction"] = ac
        out.loc[mask, "m77_ypa_correction"] = yc
    out["m77_pred_attempts"] = num(out["pred_attempts"]) + out["m77_attempt_correction"]
    out["m77_pred_ypa"] = num(out["base_ypa"]) + out["m77_ypa_correction"]
    out["m77_pred_pass_yards"] = out["m77_pred_attempts"] * out["m77_pred_ypa"]
    return out


def out_abs_max(s: pd.Series) -> float:
    v = num(s).abs().dropna()
    return float(v.max()) if len(v) else 0.0


def evaluate_split(df: pd.DataFrame, label: str) -> dict:
    base_pass = metrics(df["actual_pass_yards"], df["pred_pass_yards"])
    new_pass = metrics(df["actual_pass_yards"], df["m77_pred_pass_yards"])
    base_att = component_metrics(df["actual_attempts"], df["pred_attempts"])
    new_att = component_metrics(df["actual_attempts"], df["m77_pred_attempts"])
    base_ypa = component_metrics(df["actual_ypa_calc"], df["base_ypa"])
    new_ypa = component_metrics(df["actual_ypa_calc"], df["m77_pred_ypa"])
    pass_boot = bootstrap_gain(df["actual_pass_yards"], df["pred_pass_yards"], df["m77_pred_pass_yards"], seed=77)
    att_boot = bootstrap_gain(df["actual_attempts"], df["pred_attempts"], df["m77_pred_attempts"], seed=177)
    ypa_boot = bootstrap_gain(df["actual_ypa_calc"], df["base_ypa"], df["m77_pred_ypa"], seed=277)
    return {
        "split": label,
        "n": int(len(df)),
        "feature_coverage": float(eligible(df).mean()),
        "base_pass_mae": base_pass["mae"],
        "m77_pass_mae": new_pass["mae"],
        "pass_mae_gain": base_pass["mae"] - new_pass["mae"],
        "pass_gain_ci_low": pass_boot["ci_low"],
        "pass_gain_ci_high": pass_boot["ci_high"],
        "base_pass_rmse": base_pass["rmse"],
        "m77_pass_rmse": new_pass["rmse"],
        "base_pass_bias": base_pass["bias"],
        "m77_pass_bias": new_pass["bias"],
        "base_pass_corr": base_pass["corr"],
        "m77_pass_corr": new_pass["corr"],
        "pass_corr_gain": new_pass["corr"] - base_pass["corr"],
        "base_tail100": base_pass["tail100"],
        "m77_tail100": new_pass["tail100"],
        "base_attempt_mae": base_att["mae"],
        "m77_attempt_mae": new_att["mae"],
        "attempt_mae_gain": base_att["mae"] - new_att["mae"],
        "attempt_gain_ci_low": att_boot["ci_low"],
        "attempt_gain_ci_high": att_boot["ci_high"],
        "base_ypa_mae": base_ypa["mae"],
        "m77_ypa_mae": new_ypa["mae"],
        "ypa_mae_gain": base_ypa["mae"] - new_ypa["mae"],
        "ypa_gain_ci_low": ypa_boot["ci_low"],
        "ypa_gain_ci_high": ypa_boot["ci_high"],
        "max_abs_attempt_correction": out_abs_max(df["m77_attempt_correction"]),
        "max_abs_ypa_correction": out_abs_max(df["m77_ypa_correction"]),
    }


def fit_dev_and_final(data: pd.DataFrame):
    dev_train = data.loc[(data["season"] == 2024) & (data["week"] <= DEV_SPLIT_WEEK) & eligible(data)].copy()
    dev_eval = data.loc[(data["season"] == 2024) & (data["week"] > DEV_SPLIT_WEEK)].copy()
    if len(dev_train) < 100:
        raise RuntimeError(f"insufficient M77 dev training rows: {len(dev_train)}")
    dev_model = PersonnelCorrection().fit(dev_train)
    dev_pred = apply_model(dev_model, dev_eval)

    train = data.loc[(data["season"] == 2024) & eligible(data)].copy()
    eval25 = data.loc[data["season"] == 2025].copy()
    if len(train) < 250:
        raise RuntimeError(f"insufficient M77 2024 training rows: {len(train)}")
    final_model = PersonnelCorrection().fit(train)
    eval_pred = apply_model(final_model, eval25)
    return dev_pred, final_model, eval_pred, train


def coefficient_table(model: PersonnelCorrection) -> pd.DataFrame:
    return pd.DataFrame({
        "feature": FEATURES,
        "attempt_coef_standardized": model.attempt.coef_,
        "ypa_coef_standardized": model.ypa.coef_,
    })


def promotion_gate(dev_result: dict, eval_result: dict) -> tuple[pd.DataFrame, str]:
    component_supported = (
        (eval_result["attempt_mae_gain"] > 0 and eval_result["attempt_gain_ci_low"] > 0)
        or (eval_result["ypa_mae_gain"] > 0 and eval_result["ypa_gain_ci_low"] > 0)
    )
    checks = [
        ("eval_feature_coverage", eval_result["feature_coverage"], f">={MIN_EVAL_FEATURE_COVERAGE}", eval_result["feature_coverage"] >= MIN_EVAL_FEATURE_COVERAGE),
        ("dev_directional_pass_mae", dev_result["pass_mae_gain"], ">0", dev_result["pass_mae_gain"] > 0),
        ("eval_pass_mae_gain", eval_result["pass_mae_gain"], f">={PROMOTION_MAE_GAIN}", eval_result["pass_mae_gain"] >= PROMOTION_MAE_GAIN),
        ("eval_pass_gain_ci_low", eval_result["pass_gain_ci_low"], ">0", eval_result["pass_gain_ci_low"] > 0),
        ("eval_pass_corr_gain", eval_result["pass_corr_gain"], f">={PROMOTION_CORR_GAIN}", eval_result["pass_corr_gain"] >= PROMOTION_CORR_GAIN),
        ("eval_tail100_nonincrease", eval_result["m77_tail100"] - eval_result["base_tail100"], "<=0", eval_result["m77_tail100"] <= eval_result["base_tail100"]),
        ("component_supported_bootstrap", 1.0 if component_supported else 0.0, "==1", component_supported),
    ]
    gate = pd.DataFrame(checks, columns=["gate", "value", "threshold", "passed"])
    status = "PROMOTE_CANDIDATE" if bool(gate["passed"].all()) else "REJECTED_FOR_PROMOTION"
    return gate, status


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--canonical", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    base = require_canonical(Path(args.canonical))
    schedule, depth, snap_hist, pfr_hist = load_personnel_sources(base, out_dir)
    feat = build_features(base, schedule, depth, snap_hist, pfr_hist)

    data = base.merge(feat, on=["season", "week", "team", "opponent", "player_clean_key"], how="left", validate="one_to_one")
    if len(data) != EXPECTED_ROWS:
        raise RuntimeError(f"feature merge changed canonical population: {len(data)}")

    bad_features = [c for c in FEATURES if any(tok in c for tok in MARKET_TOKENS)]
    if bad_features:
        raise RuntimeError(f"market-like M77 feature names: {bad_features}")

    dev_pred, final_model, eval_pred, train = fit_dev_and_final(data)
    dev_result = evaluate_split(dev_pred, "2024_weeks_10_18_dev")
    eval_result = evaluate_split(eval_pred, "2025_untouched")
    results = pd.DataFrame([dev_result, eval_result])
    gate, status = promotion_gate(dev_result, eval_result)

    interpretation = pd.DataFrame([{
        "migration": "M77",
        "status": status,
        "production_actionable": bool(status == "PROMOTE_CANDIDATE"),
        "train_rows_2024_eligible": int(len(train)),
        "eval_rows_2025": int(len(eval_pred)),
        "eval_feature_coverage": eval_result["feature_coverage"],
        "base_2025_pass_mae": eval_result["base_pass_mae"],
        "m77_2025_pass_mae": eval_result["m77_pass_mae"],
        "pass_mae_gain": eval_result["pass_mae_gain"],
        "pass_corr_gain": eval_result["pass_corr_gain"],
        "tail100_change": int(eval_result["m77_tail100"] - eval_result["base_tail100"]),
        "attempt_mae_gain": eval_result["attempt_mae_gain"],
        "ypa_mae_gain": eval_result["ypa_mae_gain"],
        "next_step_if_promoted": "M78_integrate_frozen_personnel_correction_into_2026_pregame_pipeline",
        "next_step_if_rejected": "seek_genuinely_new_pregame_information_do_not_remix_M77_features",
    }])

    manifest = pd.DataFrame({
        "feature": FEATURES,
        "information_family": ["exact_personnel_discontinuity"] * len(FEATURES),
        "pregame_only": [True] * len(FEATURES),
        "target_game_outcome_used": [False] * len(FEATURES),
        "sportsbook_used": [False] * len(FEATURES),
    })

    results.to_csv(out_dir / "m77_results.csv", index=False)
    gate.to_csv(out_dir / "m77_promotion_gate.csv", index=False)
    interpretation.to_csv(out_dir / "m77_interpretation.csv", index=False)
    coefficient_table(final_model).to_csv(out_dir / "m77_coefficients.csv", index=False)
    manifest.to_csv(out_dir / "m77_feature_manifest.csv", index=False)

    pred_cols = [
        "season", "week", "team", "opponent", "player_clean_key",
        "actual_pass_yards", "pred_pass_yards", "m77_pred_pass_yards",
        "actual_attempts", "pred_attempts", "m77_pred_attempts",
        "actual_ypa_calc", "base_ypa", "m77_pred_ypa",
        "personnel_feature_covered", "m77_attempt_correction", "m77_ypa_correction",
    ] + FEATURES
    eval_pred[pred_cols].to_csv(out_dir / "m77_2025_predictions.csv", index=False)

    print("=== M77 INTERPRETATION ===")
    print(interpretation.to_string(index=False))
    print("\n=== M77 PROMOTION GATE ===")
    print(gate.to_string(index=False))
    print("\n=== M77 RESULTS ===")
    print(results.to_string(index=False))
    print("\n=== M77 FEATURE COVERAGE ===")
    print(data.groupby("season")["personnel_feature_covered"].mean().to_string())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
