#!/usr/bin/env python3
"""WR-R17 Stage A evaluator, mechanically corrected pre-result implementation.

Implements only the frozen 2023 development stage. The v1b suffix records a
pre-result mechanical correction: preserve every target event, including
repeated same-depth targets within a game. No frozen science/gates changed.
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
DEV_SEASON = 2023
PRIOR_GAMES = 8
MIN_PRIOR_TARGET_GAMES = 4
MIN_PRIOR_TARGET_EVENTS = 12
MIN_COVERAGE = 0.60
MIN_SPEARMAN = 0.08
MIN_RESIDUAL_GAP = 5.0
MIN_TAIL_RATIO = 1.20
MIN_SLICE_N = 150
SIGNAL_PRIORITY = ["DEPTH_SD8", "DEPTH_IQR8", "DEEP15_TARGET_SHARE8"]


def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def _clean_id(value) -> str:
    if value is None or pd.isna(value):
        return ""
    s = str(value).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def _name_key(value) -> str:
    try:
        _, key = canonicalize_player_name_safe(value)
        return str(key) if key else ""
    except Exception:
        return ""


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
        "entitlement_tgt_share", "mc_receptions", "mc_rec_yards", "season", "week",
        "actual_targets", "actual_rec_yards",
    }
    missing = sorted(required - set(x.columns))
    if missing:
        raise RuntimeError(f"WR-R17 authority missing columns: {missing}")
    x = x.loc[x["variant"].astype(str).eq(AUTHORITY_VARIANT)].copy()
    x["season"] = _num(x["season"]).astype(int)
    x["week"] = _num(x["week"]).astype(int)
    x["team"] = x["team"].map(canon_team)
    x["player_clean_key"] = x["player_clean_key"].astype(str)
    counts = x.groupby("season").size().to_dict()
    if counts != EXPECTED_ROWS:
        raise RuntimeError(f"WR-R17 authority row-count parity failed: {counts} != {EXPECTED_ROWS}")
    keys = ["season", "week", "team", "player_clean_key"]
    if x.duplicated(keys).any():
        bad = x.loc[x.duplicated(keys, keep=False), keys].head(10).to_dict("records")
        raise RuntimeError(f"WR-R17 duplicate authority identities: {bad}")
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
        raise RuntimeError("WR-R17 historical PBP source returned zero rows")
    return pd.concat(frames, ignore_index=True, sort=False)


def prepare_target_events(raw: pd.DataFrame) -> pd.DataFrame:
    """Return every valid targeted-pass event, retaining stable ID + name alias."""
    x = raw.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    needed = [
        "season", "week", "game_id", "posteam", "receiver_player_name",
        "receiver_player_id", "pass_attempt", "sack", "two_point_attempt", "air_yards",
    ]
    for c in needed:
        if c not in x.columns:
            x[c] = np.nan
    x["season"] = _num(x["season"])
    x["week"] = _num(x["week"])
    x["team"] = x["posteam"].map(canon_team)
    official = _num(x["pass_attempt"]).fillna(0).eq(1)
    official &= ~_num(x["sack"]).fillna(0).eq(1)
    official &= ~_num(x["two_point_attempt"]).fillna(0).eq(1)
    x = x.loc[official & x["season"].notna() & x["week"].notna() & x["team"].ne("")].copy()
    x["season"] = x["season"].astype(int)
    x["week"] = x["week"].astype(int)
    x["receiver_id"] = x["receiver_player_id"].map(_clean_id)
    x["receiver_name_key"] = x["receiver_player_name"].map(_name_key)
    x["air"] = _num(x["air_yards"])
    x = x.loc[(x["receiver_id"].ne("") | x["receiver_name_key"].ne("")) & x["air"].notna()].copy()
    # Synthetic/stable row id guarantees repeated equal-depth targets remain separate events.
    x["target_event_seq"] = np.arange(len(x), dtype=np.int64)
    return x[[
        "season", "week", "game_id", "team", "receiver_id", "receiver_name_key", "air", "target_event_seq"
    ]].reset_index(drop=True)


def _before(frame: pd.DataFrame, season: int, week: int) -> pd.DataFrame:
    return frame.loc[
        (frame["season"] < int(season))
        | ((frame["season"] == int(season)) & (frame["week"] < int(week)))
    ].copy()


def _last_games(frame: pd.DataFrame, n: int) -> pd.DataFrame:
    if frame.empty:
        return frame.copy()
    games = (
        frame[["season", "week", "game_id"]].drop_duplicates()
        .sort_values(["season", "week", "game_id"], kind="mergesort").tail(int(n))
    )
    return frame.merge(games, on=["season", "week", "game_id"], how="inner", validate="many_to_one")


def resolve_prior_receiver_history(
    targets: pd.DataFrame, authority_name_key: str, season: int, week: int
) -> tuple[pd.DataFrame, dict]:
    """Resolve alias -> stable receiver ID using strictly-prior events only."""
    prior = _before(targets, season, week)
    alias = prior.loc[prior["receiver_name_key"].eq(str(authority_name_key))].copy()
    ids = sorted({v for v in alias["receiver_id"].map(_clean_id).tolist() if v})
    audit = {
        "identity_mode": "unmatched", "resolved_receiver_id": "",
        "prior_alias_rows": int(len(alias)), "stable_ids_for_alias": int(len(ids)),
        "fallback_event_count": 0,
    }
    if len(ids) > 1:
        audit["identity_mode"] = "ambiguous"
        return prior.iloc[0:0].copy(), audit
    if len(ids) == 1:
        rid = ids[0]
        by_id = prior.loc[prior["receiver_id"].eq(rid)].copy()
        fallback = prior.loc[
            prior["receiver_id"].eq("") & prior["receiver_name_key"].eq(str(authority_name_key))
        ].copy()
        history = pd.concat([by_id, fallback], ignore_index=True)
        # by_id and fallback are disjoint by construction. Never dedupe target events.
        audit.update({
            "identity_mode": "id", "resolved_receiver_id": rid,
            "fallback_event_count": int(len(fallback)),
        })
        return history, audit
    if len(alias):
        audit["identity_mode"] = "name_fallback"
        audit["fallback_event_count"] = int(len(alias))
        return alias, audit
    return prior.iloc[0:0].copy(), audit


def receiver_state(targets: pd.DataFrame, authority_name_key: str, season: int, week: int) -> dict:
    history, audit = resolve_prior_receiver_history(targets, authority_name_key, season, week)
    h8 = _last_games(history, PRIOR_GAMES)
    games8 = h8[["season", "week", "game_id"]].drop_duplicates() if not h8.empty else pd.DataFrame()
    air = _num(h8["air"]).dropna() if not h8.empty else pd.Series(dtype=float)
    out = {
        **audit,
        "prior_target_games": int(len(games8)), "prior_target_events": int(len(air)),
        "DEPTH_SD8": np.nan, "DEPTH_IQR8": np.nan, "DEEP15_TARGET_SHARE8": np.nan,
        "mean_air_yards_per_target8": np.nan,
        "history_max_season": np.nan, "history_max_week": np.nan,
    }
    if len(games8):
        latest = games8.sort_values(["season", "week", "game_id"], kind="mergesort").iloc[-1]
        out["history_max_season"] = int(latest["season"])
        out["history_max_week"] = int(latest["week"])
    if audit["identity_mode"] in {"ambiguous", "unmatched"}:
        return out
    if len(games8) < MIN_PRIOR_TARGET_GAMES or len(air) < MIN_PRIOR_TARGET_EVENTS:
        return out
    out["DEPTH_SD8"] = float(air.std(ddof=0))
    out["DEPTH_IQR8"] = float(
        air.quantile(0.75, interpolation="linear") - air.quantile(0.25, interpolation="linear")
    )
    out["DEEP15_TARGET_SHARE8"] = float(air.ge(15.0).mean())
    out["mean_air_yards_per_target8"] = float(air.mean())
    return out


def build_development_panel(authority: pd.DataFrame, targets: pd.DataFrame) -> pd.DataFrame:
    dev = authority.loc[authority["season"].eq(DEV_SEASON)].copy()
    if len(dev) != EXPECTED_ROWS[DEV_SEASON]:
        raise RuntimeError("WR-R17 development authority count drift")
    rows = []
    for r in dev.itertuples(index=False):
        key = str(r.player_clean_key)
        display_key = _name_key(r.player)
        base = {
            "season": int(r.season), "week": int(r.week), "team": str(r.team),
            "player_clean_key": key, "player": str(r.player), "display_name_key": display_key,
            "authority_display_key_match": bool(not display_key or display_key == key),
            "wr_rank": int(r.wr_rank), "wr_rank_bucket": "WR1" if int(r.wr_rank) == 1 else "WR2PLUS",
            "pred_targets": float(r.pred_targets), "entitlement_tgt_share": float(r.entitlement_tgt_share),
            "mc_rec_yards": float(r.mc_rec_yards), "actual_rec_yards": float(r.actual_rec_yards),
            "yard_residual": float(r.yard_residual),
        }
        base.update(receiver_state(targets, key, base["season"], base["week"]))
        rows.append(base)
    panel = pd.DataFrame(rows)
    has_hist = panel["history_max_season"].notna()
    before = (panel["history_max_season"] < panel["season"]) | (
        (panel["history_max_season"] == panel["season"]) & (panel["history_max_week"] < panel["week"])
    )
    if int((has_hist & ~before).sum()):
        raise RuntimeError("WR-R17 target-game leakage assertion failed")
    return panel


def _spearman(a: pd.Series, b: pd.Series) -> float:
    z = pd.DataFrame({"a": _num(a), "b": _num(b)}).dropna()
    if len(z) < 3 or z["a"].nunique() < 2 or z["b"].nunique() < 2:
        return np.nan
    return float(z["a"].rank().corr(z["b"].rank()))


def _dir_ratio(high: float, low: float, direction: int) -> float:
    n, d = (high, low) if direction > 0 else (low, high)
    return float(n / d) if np.isfinite(n) and np.isfinite(d) and d > 0 else np.nan


def _same_sign(value: float, direction: int) -> bool:
    return bool(np.isfinite(value) and value != 0 and int(np.sign(value)) == int(direction))


def score_development(panel: pd.DataFrame) -> tuple[pd.DataFrame, dict, str | None]:
    if len(panel) != EXPECTED_ROWS[DEV_SEASON]:
        raise RuntimeError("WR-R17 development panel count drift")
    records, thresholds, advancing = [], {}, None
    for signal in SIGNAL_PRIORITY:
        d = panel.loc[_num(panel[signal]).notna() & _num(panel["yard_residual"]).notna()].copy()
        coverage = float(len(d) / len(panel))
        rec = {"signal": signal, "n": int(len(d)), "coverage": coverage, "supported": False}
        if d.empty:
            records.append(rec)
            continue
        q25 = float(d[signal].quantile(0.25, interpolation="linear"))
        q75 = float(d[signal].quantile(0.75, interpolation="linear"))
        thresholds[signal] = {"q25": q25, "q75": q75}
        low, high = d.loc[d[signal].le(q25)], d.loc[d[signal].ge(q75)]
        rho = _spearman(d[signal], d["yard_residual"])
        gap = float(high["yard_residual"].mean() - low["yard_residual"].mean())
        direction = 0 if not np.isfinite(rho) or rho == 0 else int(np.sign(rho))
        actual100 = _num(d["actual_rec_yards"]).ge(100.0)
        miss30 = _num(d["yard_residual"]).abs().ge(30.0)
        hi100 = float(actual100.loc[high.index].mean()) if len(high) else np.nan
        lo100 = float(actual100.loc[low.index].mean()) if len(low) else np.nan
        himiss = float(miss30.loc[high.index].mean()) if len(high) else np.nan
        lomiss = float(miss30.loc[low.index].mean()) if len(low) else np.nan
        r100 = _dir_ratio(hi100, lo100, direction) if direction else np.nan
        rmiss = _dir_ratio(himiss, lomiss, direction) if direction else np.nan
        slice_values, slice_ok = {}, True
        for bucket in ["WR1", "WR2PLUS"]:
            s = d.loc[d["wr_rank_bucket"].eq(bucket)]
            required = len(s) >= MIN_SLICE_N
            slo, shi = s.loc[s[signal].le(q25)], s.loc[s[signal].ge(q75)]
            sgap = float(shi["yard_residual"].mean() - slo["yard_residual"].mean()) if len(slo) and len(shi) else np.nan
            coherent = _same_sign(sgap, direction) if required and direction else (not required)
            slice_values[bucket] = (int(len(s)), sgap, required, coherent)
            if required:
                slice_ok &= coherent
        identity_ok = bool(d["identity_mode"].isin(["id", "name_fallback"]).all())
        tail_ok = bool((np.isfinite(r100) and r100 >= MIN_TAIL_RATIO) or (np.isfinite(rmiss) and rmiss >= MIN_TAIL_RATIO))
        supported = bool(
            coverage >= MIN_COVERAGE and np.isfinite(rho) and abs(rho) >= MIN_SPEARMAN
            and np.isfinite(gap) and abs(gap) >= MIN_RESIDUAL_GAP and direction and _same_sign(gap, direction)
            and tail_ok and slice_ok and identity_ok
        )
        rec.update({
            "spearman": rho, "q25": q25, "q75": q75, "q4_minus_q1_residual_gap": gap,
            "expected_direction": direction, "q4_actual100_rate": hi100, "q1_actual100_rate": lo100,
            "actual100_directional_rate_ratio": r100, "q4_abs_miss30_rate": himiss,
            "q1_abs_miss30_rate": lomiss, "miss30_directional_rate_ratio": rmiss,
            "wr1_n": slice_values["WR1"][0], "wr1_gap": slice_values["WR1"][1],
            "wr1_required": slice_values["WR1"][2], "wr1_coherent": slice_values["WR1"][3],
            "wr2plus_n": slice_values["WR2PLUS"][0], "wr2plus_gap": slice_values["WR2PLUS"][1],
            "wr2plus_required": slice_values["WR2PLUS"][2], "wr2plus_coherent": slice_values["WR2PLUS"][3],
            "identity_audit_pass": identity_ok, "supported": supported,
        })
        records.append(rec)
        if advancing is None and supported:
            advancing = signal
    return pd.DataFrame(records), thresholds, advancing


def role_endogeneity_robustness(panel: pd.DataFrame, signal: str, raw_direction: int) -> dict:
    d = panel[[signal, "entitlement_tgt_share", "mean_air_yards_per_target8", "yard_residual", "wr_rank_bucket"]].copy()
    for c in [signal, "entitlement_tgt_share", "mean_air_yards_per_target8", "yard_residual"]:
        d[c] = _num(d[c])
    d = d.dropna(subset=[signal, "entitlement_tgt_share", "mean_air_yards_per_target8", "yard_residual"])
    if len(d) < 10:
        raise RuntimeError("WR-R17 role robustness has insufficient development rows")
    X = np.column_stack([
        np.ones(len(d)), d["entitlement_tgt_share"].to_numpy(float),
        d["wr_rank_bucket"].eq("WR2PLUS").astype(float).to_numpy(),
        d["mean_air_yards_per_target8"].to_numpy(float),
    ])
    y = d[signal].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    d["role_orthogonal_signal"] = y - X @ beta
    rho = _spearman(d["role_orthogonal_signal"], d["yard_residual"])
    q25 = float(d["role_orthogonal_signal"].quantile(.25, interpolation="linear"))
    q75 = float(d["role_orthogonal_signal"].quantile(.75, interpolation="linear"))
    lo, hi = d.loc[d["role_orthogonal_signal"].le(q25)], d.loc[d["role_orthogonal_signal"].ge(q75)]
    gap = float(hi["yard_residual"].mean() - lo["yard_residual"].mean())
    rp, gp = _same_sign(rho, raw_direction), _same_sign(gap, raw_direction)
    return {
        "signal": signal, "n": int(len(d)),
        "ols_formula": "signal ~ entitlement_tgt_share + C(wr_rank_bucket) + mean_air_yards_per_target8",
        "coefficients": {"intercept": float(beta[0]), "entitlement_tgt_share": float(beta[1]),
                         "wr_rank_bucket_WR2PLUS": float(beta[2]), "mean_air_yards_per_target8": float(beta[3])},
        "role_orthogonal_spearman": rho, "role_orthogonal_q25": q25, "role_orthogonal_q75": q75,
        "role_orthogonal_q4_minus_q1_residual_gap": gap, "raw_direction": int(raw_direction),
        "spearman_preserves_raw_direction": bool(rp), "gap_preserves_raw_direction": bool(gp),
        "role_mediated_warning": bool(not rp and not gp and np.isfinite(rho) and np.isfinite(gap)),
    }


def identity_audit(panel: pd.DataFrame) -> dict:
    counts = panel["identity_mode"].value_counts(dropna=False).to_dict()
    return {
        "development_rows": int(len(panel)), "expected_development_rows": EXPECTED_ROWS[DEV_SEASON],
        "identity_mode_counts": {str(k): int(v) for k, v in counts.items()},
        "rows_with_4plus_prior_target_games": int(panel["prior_target_games"].ge(MIN_PRIOR_TARGET_GAMES).sum()),
        "rows_with_12plus_prior_target_events": int(panel["prior_target_events"].ge(MIN_PRIOR_TARGET_EVENTS).sum()),
        "rows_with_valid_depth_signal": int(panel["DEPTH_SD8"].notna().sum()),
        "authority_display_key_mismatch_rows": int((~panel["authority_display_key_match"]).sum()),
        "target_game_leakage_rows": 0, "sportsbook_inputs": 0,
        "history_seasons_loaded": [2022, 2023], "holdout_2024_scored": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--authority", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    authority = load_authority(args.authority)
    targets = prepare_target_events(load_pbp([2022, 2023]))
    panel = build_development_panel(authority, targets)
    metrics, thresholds, advancing = score_development(panel)
    robustness = None
    disposition = "NO_ACTIONABLE_WR_TARGET_DEPTH_DISTRIBUTION_SIGNAL"
    if advancing:
        row = metrics.loc[metrics["signal"].eq(advancing)].iloc[0]
        robustness = role_endogeneity_robustness(panel, advancing, int(row["expected_direction"]))
        disposition = "WR_TARGET_DEPTH_DISTRIBUTION_DEVELOPMENT_SUPPORTED"
    out = args.out_dir
    out.mkdir(parents=True, exist_ok=True)
    panel.to_csv(out / "wr_r17_stage_a_feature_panel_2023.csv", index=False)
    metrics.to_csv(out / "wr_r17_stage_a_metrics_2023.csv", index=False)
    (out / "wr_r17_stage_a_thresholds_2023.json").write_text(json.dumps(thresholds, indent=2, sort_keys=True) + "\n")
    (out / "wr_r17_stage_a_identity_audit.json").write_text(json.dumps(identity_audit(panel), indent=2, sort_keys=True) + "\n")
    if robustness:
        (out / "wr_r17_stage_a_role_endogeneity_robustness.json").write_text(json.dumps(robustness, indent=2, sort_keys=True) + "\n")
    result = {
        "specification": "WR_R17_TARGET_DEPTH_DISTRIBUTION_V1", "implementation": "v1b_event_preserving",
        "stage": "A_2023_DEVELOPMENT_ONLY", "disposition": disposition, "advancing_signal": advancing,
        "signal_priority": SIGNAL_PRIORITY, "holdout_2024_scored": False,
        "role_mediated_warning": bool(robustness and robustness["role_mediated_warning"]),
        "authority_expected_rows": EXPECTED_ROWS,
    }
    (out / "wr_r17_stage_a_result.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print("=== WR-R17 STAGE A / 2023 ONLY ===")
    print(metrics.to_string(index=False))
    print(json.dumps(result, indent=2, sort_keys=True))
    print("2024 holdout was not scored.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
