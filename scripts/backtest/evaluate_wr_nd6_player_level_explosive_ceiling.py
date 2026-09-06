#!/usr/bin/env python3
"""WR-ND6 frozen player-level explosive ceiling diagnostic.

Diagnostic only. No model fitting and no production change. The canonical WR
casebook is rebuilt by the already-frozen ND5 harness from exact M38, then ND6
adds only strictly-prior nflverse play-by-play explosive traits.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team

EXPECTED_M38_REC_N = 4647
EXPECTED_M38_REC_MAE = 17.099904733366
EXPECTED_WR_ROWS = 2130
EXPECTED_TARGET_MAE = 2.076010432545868
TARGET_MAE_TOL = 0.01

SIGNALS = {
    "PLAYER_EXP20_PER_TARGET_PRIOR8": {"column": "player_exp20_per_target_prior8", "side": "player"},
    "PLAYER_EXP40_PER_TARGET_PRIOR8": {"column": "player_exp40_per_target_prior8", "side": "player"},
    "PLAYER_YAC_PER_RECEPTION_PRIOR8": {"column": "player_yac_per_reception_prior8", "side": "player"},
    "PLAYER_AIR_PER_TARGET_PRIOR8": {"column": "player_air_per_target_prior8", "side": "player"},
    "DEF_EXP20_PER_ATT_ALLOWED_PRIOR8": {"column": "def_exp20_per_att_allowed_prior8", "side": "defense"},
    "DEF_EXP40_PER_ATT_ALLOWED_PRIOR8": {"column": "def_exp40_per_att_allowed_prior8", "side": "defense"},
    "DEF_YAC_PER_COMPLETION_ALLOWED_PRIOR8": {"column": "def_yac_per_completion_allowed_prior8", "side": "defense"},
    "DEF_AIR_PER_ATT_ALLOWED_PRIOR8": {"column": "def_air_per_att_allowed_prior8", "side": "defense"},
}


def _read(path: Path, label: str) -> pd.DataFrame:
    if not path.exists() or not path.stat().st_size:
        raise RuntimeError(f"missing {label}: {path}")
    x = pd.read_csv(path)
    x.columns = [str(c).strip().lower() for c in x.columns]
    return x


def _to_pandas(value) -> pd.DataFrame:
    if isinstance(value, pd.DataFrame):
        return value.copy()
    if hasattr(value, "to_pandas"):
        return value.to_pandas()
    return pd.DataFrame(value)


def _series(frame: pd.DataFrame, column: str, default=np.nan) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(default, index=frame.index, dtype=float)
    return pd.to_numeric(frame[column], errors="coerce")


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if np.isfinite(num) and np.isfinite(den) and den > 0 else np.nan


def _parent_m38_check(predictions: pd.DataFrame) -> dict:
    p = predictions.copy()
    g = p.loc[p["market"].astype(str).eq("rec_yards")].copy()
    g["actual"] = _series(g, "actual")
    g["mc_proj"] = _series(g, "mc_proj")
    g = g.loc[g["actual"].notna() & g["mc_proj"].notna()].copy()
    err = g["mc_proj"] - g["actual"]
    out = {
        "n": int(len(g)),
        "mae": float(err.abs().mean()),
        "rmse": float(np.sqrt(np.mean(np.square(err)))),
        "bias": float(err.mean()),
        "correlation": float(g["mc_proj"].corr(g["actual"])),
    }
    if out["n"] != EXPECTED_M38_REC_N or abs(out["mae"] - EXPECTED_M38_REC_MAE) > 1e-9:
        raise RuntimeError(f"exact M38 parent drift: {out}")
    return out


def _canonical_casebook(casebook: pd.DataFrame, predictions: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    x = casebook.copy()
    if len(x) != EXPECTED_WR_ROWS:
        raise RuntimeError(f"ND6 WR casebook drift: expected {EXPECTED_WR_ROWS}, got {len(x)}")
    if "raw_target_error" not in x.columns:
        raise RuntimeError("ND6 casebook missing raw_target_error")
    target_mae = float(_series(x, "raw_target_error").abs().mean())
    if abs(target_mae - EXPECTED_TARGET_MAE) > TARGET_MAE_TOL:
        raise RuntimeError(f"ND6 target reconstruction drift: {target_mae}")

    keys = ["season", "week", "team", "player_clean_key"]
    for c in ["season", "week"]:
        x[c] = _series(x, c)
    x["team"] = x["team"].fillna("").map(canon_team)
    x["opponent"] = x["opponent"].fillna("").map(canon_team)
    x["player_id"] = x["player_id"].astype("string").str.strip()

    r = predictions.loc[predictions["market"].astype(str).eq("rec_yards")].copy()
    for c in ["season", "week"]:
        r[c] = _series(r, c)
    r["team"] = r["team"].fillna("").map(canon_team)
    if "player_clean_key" not in r.columns:
        raise RuntimeError("component predictions missing player_clean_key")
    r["m38_rec_proj"] = _series(r, "mc_proj")
    r["m38_actual_rec_yards"] = _series(r, "actual")
    r = r[keys + ["m38_rec_proj", "m38_actual_rec_yards"]].dropna(subset=["m38_rec_proj"])
    if r.duplicated(keys).any():
        dup = r.loc[r.duplicated(keys, keep=False), keys].head(20)
        raise RuntimeError(f"duplicate M38 rec-yard keys:\n{dup.to_string(index=False)}")

    x = x.merge(r, on=keys, how="left", validate="one_to_one")
    if x["m38_rec_proj"].isna().any():
        bad = x.loc[x["m38_rec_proj"].isna(), keys].head(20)
        raise RuntimeError(f"ND6 casebook missing M38 rec projections:\n{bad.to_string(index=False)}")

    actual = _series(x, "actual_rec_yards") if "actual_rec_yards" in x.columns else _series(x, "m38_actual_rec_yards")
    x["actual_rec_yards"] = actual
    if x["actual_rec_yards"].isna().any():
        raise RuntimeError("ND6 casebook has missing actual receiving yards")
    x["rec_yards_residual"] = x["actual_rec_yards"] - x["m38_rec_proj"]
    x["under25"] = x["rec_yards_residual"].ge(25.0)
    x["under50"] = x["rec_yards_residual"].ge(50.0)
    x["actual100"] = x["actual_rec_yards"].ge(100.0)
    return x, target_mae


def _prepare_pbp(raw: pd.DataFrame) -> pd.DataFrame:
    p = raw.copy()
    p.columns = [str(c).strip().lower() for c in p.columns]
    required = {"season", "week", "game_id", "posteam", "defteam", "complete_pass", "receiver_player_id"}
    missing = required - set(p.columns)
    if missing:
        raise RuntimeError(f"ND6 PBP missing required columns: {sorted(missing)}")
    attempt_col = "official_pass_attempt" if "official_pass_attempt" in p.columns else "pass_attempt"
    if attempt_col not in p.columns:
        raise RuntimeError("ND6 PBP missing official_pass_attempt/pass_attempt")
    p["season"] = _series(p, "season")
    p["week"] = _series(p, "week")
    p["posteam"] = p["posteam"].fillna("").map(canon_team)
    p["defteam"] = p["defteam"].fillna("").map(canon_team)
    p["_attempt"] = _series(p, attempt_col, 0.0).fillna(0.0).eq(1.0)
    p["_complete"] = _series(p, "complete_pass", 0.0).fillna(0.0).eq(1.0)
    p["receiver_player_id"] = p["receiver_player_id"].astype("string").str.strip()
    p["_rec_yards"] = _series(p, "receiving_yards") if "receiving_yards" in p.columns else _series(p, "passing_yards")
    p["_air"] = _series(p, "air_yards")
    p["_yac"] = _series(p, "yards_after_catch")
    if "season_type" in p.columns:
        reg = p["season_type"].fillna("").astype(str).str.upper().eq("REG")
        if reg.any():
            p = p.loc[reg].copy()
    return p.loc[p["_attempt"] & p["season"].notna() & p["week"].notna() & p["posteam"].ne("") & p["defteam"].ne("")].copy()


def _player_games(pbp: pd.DataFrame) -> pd.DataFrame:
    x = pbp.loc[pbp["receiver_player_id"].notna() & pbp["receiver_player_id"].ne("")].copy()
    rows = []
    keys = ["season", "week", "game_id", "posteam", "defteam", "receiver_player_id"]
    for key, g in x.groupby(keys, dropna=False, sort=True):
        season, week, game_id, team, opponent, player_id = key
        targets = float(len(g))
        recs = float(g["_complete"].sum())
        yards = g["_rec_yards"].fillna(0.0)
        exp20 = float((g["_complete"] & yards.ge(20.0)).sum())
        exp40 = float((g["_complete"] & yards.ge(40.0)).sum())
        yac = float(np.where(g["_complete"], g["_yac"].fillna(0.0), 0.0).sum())
        air = float(g["_air"].fillna(0.0).sum())
        rows.append({
            "season": int(season), "week": int(week), "game_id": str(game_id),
            "team": team, "opponent": opponent, "player_id": str(player_id),
            "targets": targets, "receptions": recs, "exp20": exp20, "exp40": exp40,
            "yac": yac, "air_yards": air,
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.sort_values(["season", "week", "game_id", "player_id"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


def _defense_games(pbp: pd.DataFrame) -> pd.DataFrame:
    rows = []
    keys = ["season", "week", "game_id", "defteam", "posteam"]
    for key, g in pbp.groupby(keys, dropna=False, sort=True):
        season, week, game_id, defense, offense = key
        attempts = float(len(g))
        completions = float(g["_complete"].sum())
        yards = g["_rec_yards"].fillna(0.0)
        rows.append({
            "season": int(season), "week": int(week), "game_id": str(game_id),
            "defense": defense, "offense": offense,
            "attempts": attempts,
            "completions": completions,
            "exp20": float((g["_complete"] & yards.ge(20.0)).sum()),
            "exp40": float((g["_complete"] & yards.ge(40.0)).sum()),
            "yac": float(np.where(g["_complete"], g["_yac"].fillna(0.0), 0.0).sum()),
            "air_yards": float(g["_air"].fillna(0.0).sum()),
        })
    out = pd.DataFrame(rows)
    if len(out):
        out.sort_values(["season", "week", "game_id", "defense"], inplace=True)
        out.reset_index(drop=True, inplace=True)
    return out


def _prior_mask(frame: pd.DataFrame, season: int, week: int) -> pd.Series:
    return frame["season"].lt(int(season)) | (frame["season"].eq(int(season)) & frame["week"].lt(int(week)))


def _attach_features(casebook: pd.DataFrame, player_games: pd.DataFrame, defense_games: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in casebook.iterrows():
        season, week = int(r["season"]), int(r["week"])
        pg = player_games.loc[
            player_games["player_id"].astype(str).eq(str(r["player_id"])) & _prior_mask(player_games, season, week)
        ].sort_values(["season", "week", "game_id"]).tail(8)
        dg = defense_games.loc[
            defense_games["defense"].eq(r["opponent"]) & _prior_mask(defense_games, season, week)
        ].sort_values(["season", "week", "game_id"]).tail(8)

        pt = float(pg["targets"].sum()) if len(pg) else 0.0
        pr = float(pg["receptions"].sum()) if len(pg) else 0.0
        da = float(dg["attempts"].sum()) if len(dg) else 0.0
        dc = float(dg["completions"].sum()) if len(dg) else 0.0
        rows.append({
            "player_prior_games": int(len(pg)),
            "player_exp20_per_target_prior8": _safe_div(float(pg["exp20"].sum()) if len(pg) else np.nan, pt),
            "player_exp40_per_target_prior8": _safe_div(float(pg["exp40"].sum()) if len(pg) else np.nan, pt),
            "player_yac_per_reception_prior8": _safe_div(float(pg["yac"].sum()) if len(pg) else np.nan, pr),
            "player_air_per_target_prior8": _safe_div(float(pg["air_yards"].sum()) if len(pg) else np.nan, pt),
            "def_prior_games": int(len(dg)),
            "def_exp20_per_att_allowed_prior8": _safe_div(float(dg["exp20"].sum()) if len(dg) else np.nan, da),
            "def_exp40_per_att_allowed_prior8": _safe_div(float(dg["exp40"].sum()) if len(dg) else np.nan, da),
            "def_yac_per_completion_allowed_prior8": _safe_div(float(dg["yac"].sum()) if len(dg) else np.nan, dc),
            "def_air_per_att_allowed_prior8": _safe_div(float(dg["air_yards"].sum()) if len(dg) else np.nan, da),
        })
    return pd.concat([casebook.reset_index(drop=True), pd.DataFrame(rows)], axis=1)


def _threshold(frame: pd.DataFrame, col: str) -> dict:
    v = _series(frame, col).dropna()
    if len(v) < 4:
        return {"q25": np.nan, "q75": np.nan}
    return {"q25": float(v.quantile(0.25)), "q75": float(v.quantile(0.75))}


def _masks(frame: pd.DataFrame, col: str, threshold: dict) -> tuple[pd.Series, pd.Series, pd.Series]:
    v = _series(frame, col)
    valid = v.notna()
    return valid & v.ge(threshold["q75"]), valid & v.le(threshold["q25"]), valid


def _gap(frame: pd.DataFrame, col: str, threshold: dict) -> float:
    high, low, _ = _masks(frame, col, threshold)
    if int(high.sum()) == 0 or int(low.sum()) == 0:
        return np.nan
    return float(frame.loc[high, "rec_yards_residual"].mean() - frame.loc[low, "rec_yards_residual"].mean())


def _enrichment(frame: pd.DataFrame, event: str, high: pd.Series, valid: pd.Series) -> float:
    overall = float(frame.loc[valid, event].astype(bool).mean()) if int(valid.sum()) else np.nan
    high_rate = float(frame.loc[high, event].astype(bool).mean()) if int(high.sum()) else np.nan
    return float(high_rate / overall) if np.isfinite(overall) and overall > 0 and np.isfinite(high_rate) else np.nan


def _score_signal(casebook: pd.DataFrame, name: str, spec: dict) -> tuple[dict, list[dict]]:
    col = spec["column"]
    threshold = _threshold(casebook, col)
    high, low, valid = _masks(casebook, col, threshold)
    coverage = float(valid.mean()) if len(casebook) else 0.0
    corr = casebook.loc[valid, [col, "rec_yards_residual"]].dropna()
    spearman = float(corr[col].corr(corr["rec_yards_residual"], method="spearman")) if len(corr) > 2 and corr[col].nunique() > 1 else np.nan
    overall_gap = _gap(casebook, col, threshold)
    under25 = _enrichment(casebook, "under25", high, valid)
    under50 = _enrichment(casebook, "under50", high, valid)
    actual100 = _enrichment(casebook, "actual100", high, valid)

    slices = {
        "ALL_WR": pd.Series(True, index=casebook.index),
        "W2_18": casebook["week"].between(2, 18),
        "W13_18": casebook["week"].between(13, 18),
        "WR1": casebook["m38_wr_role"].astype(str).eq("WR1"),
        "WR2": casebook["m38_wr_role"].astype(str).eq("WR2"),
        "WR3": casebook["m38_wr_role"].astype(str).eq("WR3"),
        "WR4+": casebook["m38_wr_role"].astype(str).eq("WR4+"),
    }
    slice_rows = []
    gaps = {}
    for slice_name, mask in slices.items():
        sub = casebook.loc[mask].copy()
        h, l, v = _masks(sub, col, threshold)
        gap = _gap(sub, col, threshold)
        gaps[slice_name] = gap
        slice_rows.append({
            "signal": name, "slice": slice_name, "n": int(len(sub)),
            "coverage": float(v.mean()) if len(sub) else np.nan,
            "high_n": int(h.sum()), "low_n": int(l.sum()), "rec_yards_residual_gap": gap,
        })
    positive_count = int(sum(np.isfinite(gaps[r]) and gaps[r] > 0 for r in ("WR1", "WR2", "WR3")))
    passed = bool(
        coverage >= 0.75
        and np.isfinite(spearman) and spearman >= 0.08
        and np.isfinite(overall_gap) and overall_gap >= 4.0
        and np.isfinite(under25) and under25 >= 1.25
        and ((np.isfinite(under50) and under50 >= 1.25) or (np.isfinite(actual100) and actual100 >= 1.25))
        and np.isfinite(gaps["W2_18"]) and gaps["W2_18"] > 0
        and np.isfinite(gaps["W13_18"]) and gaps["W13_18"] > 0
        and positive_count >= 2
    )
    return {
        "signal": name, "side": spec["side"], "column": col,
        "coverage": coverage,
        "spearman_rec_yards_residual": spearman,
        "high_low_rec_yards_residual_gap": overall_gap,
        "under25_enrichment": under25,
        "under50_enrichment": under50,
        "actual100_enrichment": actual100,
        "w2_18_gap": gaps["W2_18"], "w13_18_gap": gaps["W13_18"],
        "wr1_gap": gaps["WR1"], "wr2_gap": gaps["WR2"], "wr3_gap": gaps["WR3"],
        "wr1_wr2_wr3_positive_count": positive_count,
        "threshold_low": threshold["q25"], "threshold_high": threshold["q75"],
        "gate_passed": passed,
    }, slice_rows


def _disposition(summary: pd.DataFrame) -> tuple[str, list[str]]:
    winners = summary.loc[summary["gate_passed"].astype(bool), ["signal", "side"]]
    names = winners["signal"].astype(str).tolist()
    players = winners.loc[winners["side"].eq("player"), "signal"].astype(str).tolist()
    defenses = winners.loc[winners["side"].eq("defense"), "signal"].astype(str).tolist()
    if players and defenses:
        return "PLAYER_AND_DEFENSE_CEILING_SIGNALS", names
    if len(players) == 1:
        return f"{players[0]}_PLAYER_CEILING_SIGNAL", names
    if len(players) > 1:
        return "MULTIPLE_PLAYER_CEILING_SIGNALS", names
    if len(defenses) == 1:
        return f"{defenses[0]}_DEFENSE_CEILING_SIGNAL", names
    if len(defenses) > 1:
        return "MULTIPLE_DEFENSE_CEILING_SIGNALS", names
    return "NO_ACTIONABLE_EXPLOSIVE_CEILING_SIGNAL", []


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, default=2025)
    ap.add_argument("--prior-season", type=int, default=2024)
    ap.add_argument("--predictions", type=Path, required=True)
    ap.add_argument("--casebook", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    predictions = _read(a.predictions, "exact M38 component predictions")
    parent_check = _parent_m38_check(predictions)
    base_casebook = _read(a.casebook, "canonical ND5 WR casebook")
    casebook, target_mae = _canonical_casebook(base_casebook, predictions)

    import nflreadpy as nfl
    raw = _to_pandas(nfl.load_pbp(seasons=[int(a.prior_season), int(a.season)]))
    pbp = _prepare_pbp(raw)
    pg = _player_games(pbp)
    dg = _defense_games(pbp)
    casebook = _attach_features(casebook, pg, dg)

    signal_rows = []
    slice_rows = []
    for name, spec in SIGNALS.items():
        row, slices = _score_signal(casebook, name, spec)
        signal_rows.append(row)
        slice_rows.extend(slices)
    signal_summary = pd.DataFrame(signal_rows)
    slice_summary = pd.DataFrame(slice_rows)
    disposition, winners = _disposition(signal_summary)

    coverage = {spec["column"]: float(_series(casebook, spec["column"]).notna().mean()) for spec in SIGNALS.values()}
    result = {
        "migration": "WR-ND6",
        "season": int(a.season), "prior_season": int(a.prior_season),
        "evaluation_rows": int(len(casebook)),
        "m38_parent_check": parent_check,
        "target_reconstruction_mae": target_mae,
        "frozen_history_window_games": 8,
        "frozen_gate": {
            "coverage_min": 0.75, "spearman_min": 0.08, "residual_gap_min_yards": 4.0,
            "under25_enrichment_min": 1.25, "under50_or_actual100_enrichment_min": 1.25,
            "w2_18_positive": True, "w13_18_positive": True, "wr1_wr2_wr3_positive_min_count": 2,
        },
        "feature_coverage": coverage,
        "sportsbook_inputs_used": False,
        "postseason_participation_used": False,
        "wr_cb_assignment_used": False,
        "model_fitting_used": False,
        "production_changed": False,
        "player_defense_interactions_tested": False,
        "winners": winners,
        "disposition": disposition,
    }

    a.out_dir.mkdir(parents=True, exist_ok=True)
    casebook.to_csv(a.out_dir / "wr_nd6_casebook.csv", index=False)
    pg.to_csv(a.out_dir / "wr_nd6_player_game_explosive_history.csv", index=False)
    dg.to_csv(a.out_dir / "wr_nd6_defense_game_explosive_history.csv", index=False)
    signal_summary.to_csv(a.out_dir / "wr_nd6_signal_summary.csv", index=False)
    slice_summary.to_csv(a.out_dir / "wr_nd6_slice_summary.csv", index=False)
    (a.out_dir / "wr_nd6_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print("[wr-nd6] signal summary")
    print(signal_summary.to_string(index=False))
    print("[wr-nd6] result")
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
