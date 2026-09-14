#!/usr/bin/env python3
"""WR-R16 Stage A: strictly-prior QB/WR delivery-state diagnostic.

Research-only. This file implements only the 2023 development stage from
WR_R16_QB_WR_DELIVERY_STATE_V1_PLAN.md. It deliberately does not grade 2024.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

AUTHORITY_VARIANT = "WR_R15_WR1_ANCHORED_PARTICIPATION"
EXPECTED_ROWS = {2023: 2076, 2024: 2117}
PRIOR_GAMES = 8
RECENT_GAMES = 3
MIN_PRIOR_TARGET_GAMES = 4
MIN_COVERAGE = 0.60
MIN_SPEARMAN = 0.08
MIN_RESIDUAL_GAP = 5.0
MIN_TAIL_RATIO = 1.20
MIN_SLICE_N = 150
SIGNAL_PRIORITY = ["DEEP_DELIVERY", "COMPLETED_AIR", "DELIVERY_CPOE", "DELIVERY_MOMENTUM"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        if key:
            return str(key)
    except Exception:
        pass
    return ""


def _team(value) -> str:
    return canon_team(value)


def _regular_only(x: pd.DataFrame) -> pd.DataFrame:
    q = x.copy()
    c = "season_type" if "season_type" in q.columns else "game_type" if "game_type" in q.columns else None
    if c:
        s = q[c].astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            q = q.loc[keep].copy()
    return q


def load_authority(path: Path) -> pd.DataFrame:
    x = pd.read_csv(path, low_memory=False)
    required = {
        "variant", "team", "player_clean_key", "player", "wr_rank", "pred_targets",
        "mc_receptions", "mc_rec_yards", "season", "week", "actual_targets", "actual_rec_yards",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"WR-R16 authority missing columns: {missing}")
    x = x.loc[x["variant"].astype(str).eq(AUTHORITY_VARIANT)].copy()
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x["team"] = x["team"].map(_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    counts = x.groupby("season").size().to_dict()
    if counts != EXPECTED_ROWS:
        raise RuntimeError(f"WR-R16 authority row-count parity failed: {counts} != {EXPECTED_ROWS}")
    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(10).to_dict("records")
        raise RuntimeError(f"WR-R16 duplicate authority identities: {bad}")
    x["yard_residual"] = _num(x["actual_rec_yards"]) - _num(x["mc_rec_yards"])
    return x.sort_values(keys).reset_index(drop=True)


def load_pbp(seasons: Iterable[int]) -> pd.DataFrame:
    import nflreadpy as nfl

    frames = []
    for season in seasons:
        obj = nfl.load_pbp(seasons=[int(season)])
        q = obj.to_pandas() if hasattr(obj, "to_pandas") else pd.DataFrame(obj)
        if not q.empty:
            frames.append(_regular_only(q))
    if not frames:
        raise RuntimeError("WR-R16 historical PBP source returned zero rows")
    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_target_pbp(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    needed = [
        "season", "week", "game_id", "posteam", "receiver_player_name", "receiver_player_id",
        "pass_attempt", "sack", "two_point_attempt", "complete_pass", "passing_yards",
        "air_yards", "yards_after_catch", "cpoe",
    ]
    for c in needed:
        if c not in x.columns:
            x[c] = np.nan
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    x["team"] = x["posteam"].map(_team)
    official = _num(x["pass_attempt"]).fillna(0).eq(1)
    official &= ~_num(x["sack"]).fillna(0).eq(1)
    official &= ~_num(x["two_point_attempt"]).fillna(0).eq(1)
    x = x.loc[official & x["season"].notna() & x["week"].notna() & x["team"].ne("")].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["complete"] = _num(x["complete_pass"]).fillna(0).eq(1)
    x["air"] = _num(x["air_yards"])
    x["cpoe_num"] = _num(x["cpoe"])
    x["completed_air"] = np.where(x["complete"], x["air"].fillna(0.0), 0.0)
    x["deep15"] = x["air"].ge(15)
    x["deep15_complete"] = x["deep15"] & x["complete"]

    # Team passing-attempt table: every official attempt is eligible history.
    team_attempts = x[[
        "season", "week", "game_id", "team", "cpoe_num", "air", "completed_air", "deep15", "deep15_complete"
    ]].copy()

    # Receiver target table: require a receiver identity. Canonical player keys are
    # derived from the recorded receiver name only; target-game PBP is never used
    # to select an expected QB or alter the authority identity.
    names = x["receiver_player_name"].astype(str).replace({"nan": "", "None": "", "<NA>": ""})
    x["receiver_key"] = names.map(_key)
    targets = x.loc[x["receiver_key"].ne("")].copy()
    targets = targets[[
        "season", "week", "game_id", "team", "receiver_key", "cpoe_num", "air", "completed_air",
        "complete", "passing_yards", "yards_after_catch",
    ]]
    return targets, team_attempts


def _before(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return frame.loc[(frame["season"] < int(season)) | ((frame["season"] == int(season)) & (frame["week"] < int(week)))].copy()


def _last_games(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    games = (
        frame[["season", "week", "game_id"]]
        .drop_duplicates()
        .sort_values(["season", "week", "game_id"])
        .tail(int(n))
    )
    return frame.merge(games, on=["season", "week", "game_id"], how="inner", validate="many_to_one")


def _safe_mean(s: pd.Series) -> float:
    q = _num(s).dropna()
    return float(q.mean()) if len(q) else np.nan


def _receiver_state(targets: pd.DataFrame, key: str, season: int, week: int) -> dict:
    h = _before(targets.loc[targets["receiver_key"].eq(str(key))], season, week)
    h8 = _last_games(h, PRIOR_GAMES)
    games8 = h8[["season", "week", "game_id"]].drop_duplicates()
    if len(games8) < MIN_PRIOR_TARGET_GAMES:
        return {"wr_prior_target_games": int(len(games8))}
    h3 = _last_games(h8, RECENT_GAMES)
    air8 = _num(h8["air"])
    completed8 = _num(h8["completed_air"]).fillna(0.0)
    completed3 = _num(h3["completed_air"]).fillna(0.0)
    out = {
        "wr_prior_target_games": int(len(games8)),
        "wr_target_cpoe_mean8": _safe_mean(h8["cpoe_num"]),
        "wr_completed_air_yards_per_target8": float(completed8.sum() / len(h8)) if len(h8) else np.nan,
        "wr_target_depth_sd8": float(air8.std(ddof=1)) if air8.notna().sum() >= 2 else np.nan,
    }
    cpoe3 = _safe_mean(h3["cpoe_num"])
    completed_air3 = float(completed3.sum() / len(h3)) if len(h3) else np.nan
    out["wr_cpoe_recent3_minus8"] = cpoe3 - out["wr_target_cpoe_mean8"] if np.isfinite(cpoe3) and np.isfinite(out["wr_target_cpoe_mean8"]) else np.nan
    out["wr_completed_air_recent3_minus8"] = completed_air3 - out["wr_completed_air_yards_per_target8"] if np.isfinite(completed_air3) and np.isfinite(out["wr_completed_air_yards_per_target8"]) else np.nan
    return out


def _team_state(attempts: pd.DataFrame, team: str, season: int, week: int) -> dict:
    h = _before(attempts.loc[attempts["team"].eq(str(team))], season, week)
    h8 = _last_games(h, PRIOR_GAMES)
    games8 = h8[["season", "week", "game_id"]].drop_duplicates()
    if len(games8) < MIN_PRIOR_TARGET_GAMES:
        return {"team_prior_pass_games": int(len(games8))}
    h3 = _last_games(h8, RECENT_GAMES)
    deep8 = h8.loc[h8["deep15"].fillna(False)]
    out = {
        "team_prior_pass_games": int(len(games8)),
        "team_cpoe_mean8": _safe_mean(h8["cpoe_num"]),
        "team_air_per_attempt_mean8": _safe_mean(h8["air"]),
        "team_deep15_completion_rate8": float(deep8["deep15_complete"].mean()) if len(deep8) else np.nan,
        "team_completed_air_per_attempt8": float(_num(h8["completed_air"]).fillna(0.0).mean()) if len(h8) else np.nan,
    }
    cpoe3 = _safe_mean(h3["cpoe_num"])
    compair3 = float(_num(h3["completed_air"]).fillna(0.0).mean()) if len(h3) else np.nan
    out["team_cpoe_recent3_minus8"] = cpoe3 - out["team_cpoe_mean8"] if np.isfinite(cpoe3) and np.isfinite(out["team_cpoe_mean8"]) else np.nan
    out["team_completed_air_recent3_minus8"] = compair3 - out["team_completed_air_per_attempt8"] if np.isfinite(compair3) and np.isfinite(out["team_completed_air_per_attempt8"]) else np.nan
    return out


def build_feature_panel(authority: pd.DataFrame, targets: pd.DataFrame, attempts: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in authority.itertuples(index=False):
        base = {
            "season": int(r.season), "week": int(r.week), "team": str(r.team),
            "player_clean_key": str(r.player_clean_key), "player": str(r.player),
            "wr_rank": int(r.wr_rank), "actual_rec_yards": float(r.actual_rec_yards),
            "mc_rec_yards": float(r.mc_rec_yards), "yard_residual": float(r.yard_residual),
        }
        base.update(_receiver_state(targets, base["player_clean_key"], base["season"], base["week"]))
        base.update(_team_state(attempts, base["team"], base["season"], base["week"]))
        rows.append(base)
    return pd.DataFrame(rows)


def _zfit(dev: pd.DataFrame, col: str) -> tuple[float, float]:
    q = _num(dev[col]).dropna()
    if len(q) < 2:
        raise RuntimeError(f"WR-R16 cannot fit standardization for {col}")
    mu = float(q.mean())
    sd = float(q.std(ddof=0))
    if not np.isfinite(sd) or sd <= 0:
        raise RuntimeError(f"WR-R16 invalid standardization sd for {col}: {sd}")
    return mu, sd


def add_signals(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    x = panel.copy()
    dev = x.loc[x["season"].eq(2023)].copy()
    zcols = [
        "wr_target_cpoe_mean8", "team_cpoe_mean8",
        "wr_completed_air_yards_per_target8", "team_completed_air_per_attempt8",
    ]
    params = {}
    for c in zcols:
        mu, sd = _zfit(dev, c)
        params[c] = {"mean": mu, "sd": sd}
        x[f"z_{c}"] = (_num(x[c]) - mu) / sd
    x["DELIVERY_CPOE"] = x["z_wr_target_cpoe_mean8"] + x["z_team_cpoe_mean8"]
    x["COMPLETED_AIR"] = x["z_wr_completed_air_yards_per_target8"] + x["z_team_completed_air_per_attempt8"]
    x["DEEP_DELIVERY"] = _num(x["wr_completed_air_yards_per_target8"]) * _num(x["team_deep15_completion_rate8"])
    x["DELIVERY_MOMENTUM"] = _num(x["wr_cpoe_recent3_minus8"]) + _num(x["team_cpoe_recent3_minus8"])
    return x, params


def _spearman(a: pd.Series, b: pd.Series) -> float:
    z = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(z) < 3 or z["a"].nunique() < 2 or z["b"].nunique() < 2:
        return np.nan
    return float(z["a"].rank().corr(z["b"].rank()))


def _ratio(a: float, b: float) -> float:
    if not np.isfinite(a) or not np.isfinite(b) or b <= 0:
        return np.nan
    return float(a / b)


def score_development(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    dev_all = panel.loc[panel["season"].eq(2023)].copy()
    if len(dev_all) != EXPECTED_ROWS[2023]:
        raise RuntimeError("WR-R16 development authority count drift")
    records = []
    thresholds = {}
    for signal in SIGNAL_PRIORITY:
        d = dev_all.loc[_num(dev_all[signal]).notna() & _num(dev_all["yard_residual"]).notna()].copy()
        coverage = len(d) / len(dev_all)
        if d.empty:
            records.append({"signal": signal, "n": 0, "coverage": coverage, "supported": False})
            continue
        q25 = float(_num(d[signal]).quantile(0.25, interpolation="linear"))
        q75 = float(_num(d[signal]).quantile(0.75, interpolation="linear"))
        thresholds[signal] = {"q25": q25, "q75": q75}
        low = d.loc[_num(d[signal]).le(q25)]
        high = d.loc[_num(d[signal]).ge(q75)]
        gap = float(_num(high["yard_residual"]).mean() - _num(low["yard_residual"]).mean())
        rho = _spearman(d[signal], d["yard_residual"])
        hi100 = float((_num(high["actual_rec_yards"]) >= 100).mean()) if len(high) else np.nan
        lo100 = float((_num(low["actual_rec_yards"]) >= 100).mean()) if len(low) else np.nan
        hi30 = float((_num(high["yard_residual"]).abs() >= 30).mean()) if len(high) else np.nan
        lo30 = float((_num(low["yard_residual"]).abs() >= 30).mean()) if len(low) else np.nan
        ratio100 = _ratio(hi100, lo100)
        ratio30 = _ratio(hi30, lo30)
        slice_gaps = {}
        coherent = True
        for label, mask in {
            "WR1": _num(d["wr_rank"]).eq(1),
            "WR2PLUS": _num(d["wr_rank"]).ge(2),
        }.items():
            s = d.loc[mask].copy()
            slo = s.loc[_num(s[signal]).le(q25)]
            shi = s.loc[_num(s[signal]).ge(q75)]
            sgap = float(_num(shi["yard_residual"]).mean() - _num(slo["yard_residual"]).mean()) if len(slo) and len(shi) else np.nan
            slice_gaps[label] = {"n": int(len(s)), "gap": sgap}
            if len(s) >= MIN_SLICE_N and (not np.isfinite(sgap) or sgap <= 0):
                coherent = False
        tail_ok = (np.isfinite(ratio100) and ratio100 >= MIN_TAIL_RATIO) or (np.isfinite(ratio30) and ratio30 >= MIN_TAIL_RATIO)
        supported = bool(
            coverage >= MIN_COVERAGE
            and np.isfinite(rho) and rho >= MIN_SPEARMAN
            and np.isfinite(gap) and gap >= MIN_RESIDUAL_GAP
            and tail_ok
            and coherent
        )
        records.append({
            "signal": signal, "n": int(len(d)), "coverage": float(coverage),
            "spearman": rho, "q25": q25, "q75": q75,
            "residual_gap_q4_minus_q1": gap,
            "rate100_high": hi100, "rate100_low": lo100, "ratio100_high_low": ratio100,
            "miss30_high": hi30, "miss30_low": lo30, "ratio30_high_low": ratio30,
            "wr1_n": slice_gaps["WR1"]["n"], "wr1_gap": slice_gaps["WR1"]["gap"],
            "wr2plus_n": slice_gaps["WR2PLUS"]["n"], "wr2plus_gap": slice_gaps["WR2PLUS"]["gap"],
            "supported": supported,
        })
    score = pd.DataFrame(records)
    advancing = next((s for s in SIGNAL_PRIORITY if bool(score.loc[score.signal.eq(s), "supported"].iloc[0])), None)
    return score, {"advancing_signal": advancing, "thresholds": thresholds}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()

    authority = load_authority(args.authority)
    pbp = load_pbp([2022, 2023, 2024])
    targets, attempts = prepare_target_pbp(pbp)
    panel = build_feature_panel(authority, targets, attempts)
    panel, standardization = add_signals(panel)

    # Scientific Stage A reads 2023 outcomes only. 2024 rows are written with
    # feature columns for source/identity audit, but 2024 residual/outcome columns
    # are blanked before persistence to prevent accidental result exposure.
    persisted = panel.copy()
    holdout = persisted["season"].eq(2024)
    for c in ["actual_rec_yards", "mc_rec_yards", "yard_residual"]:
        persisted.loc[holdout, c] = np.nan

    score, decision = score_development(panel)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    persisted.to_csv(args.out_dir / "wr_r16_feature_panel_stage_a.csv", index=False)
    score.to_csv(args.out_dir / "wr_r16_stage_a_scorecard.csv", index=False)
    with (args.out_dir / "wr_r16_stage_a_contract.json").open("w") as f:
        json.dump({
            "authority_variant": AUTHORITY_VARIANT,
            "expected_rows": EXPECTED_ROWS,
            "signal_priority": SIGNAL_PRIORITY,
            "standardization_2023": standardization,
            "decision": decision,
            "sportsbook_inputs": 0,
            "production_changed": False,
            "holdout_outcomes_persisted": False,
        }, f, indent=2, sort_keys=True)

    advancing = decision["advancing_signal"]
    disposition = "WR_DELIVERY_STATE_DEVELOPMENT_SUPPORTED" if advancing else "NO_ACTIONABLE_WR_DELIVERY_STATE_SIGNAL"
    print(json.dumps({"disposition": disposition, "advancing_signal": advancing}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
