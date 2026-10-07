#!/usr/bin/env python3
"""Freeze Week-5 player target-depth distribution scales and expose mean-neutral draw transform."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.player_form_v2 import _normalize_weekly, _to_pandas

VERSION = "PLAYER_TARGET_DEPTH_DISTRIBUTION_SHADOW_V1"
SEASON = 2026
WEEK = 5
PARENT_RUN = 37654382316
PARENT_ARTIFACT = 11497153776
PARENT_ROW_DIGEST = "sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3"
PARENT_ARTIFACT_DIGEST = "sha256:c12878ed97a4543a907ac54f4cbadadda7178db068491373091d2c048f176e88"
DEPTH_ANCHOR = {
    "WR": 10.02786868561449,
    "TE": 6.735519692444827,
}


def num(x):
    return pd.to_numeric(x, errors="coerce")


def clean(v) -> str:
    if v is None or pd.isna(v):
        return ""
    s = str(v).strip()
    return "" if s.lower() in {"", "nan", "none", "<na>"} else s


def _to_lower_frame(x) -> pd.DataFrame:
    y = _to_pandas(x).copy()
    y.columns = [str(c).strip().lower() for c in y.columns]
    return y


def _regular_only(x: pd.DataFrame) -> pd.DataFrame:
    y = x.copy()
    c = "season_type" if "season_type" in y.columns else "game_type" if "game_type" in y.columns else None
    if c:
        s = y[c].astype(str).str.upper()
        keep = s.isin(["REG", "REGULAR", "RS", ""])
        if keep.any():
            y = y.loc[keep].copy()
    return y


def depth_scale(position: str, prior8_target_depth_sd: float | None) -> float:
    pos = str(position or "").upper().strip()
    anchor = DEPTH_ANCHOR.get(pos)
    if anchor is None:
        return 1.0
    try:
        value = float(prior8_target_depth_sd)
    except Exception:
        return 1.0
    if not np.isfinite(value) or value <= 0:
        return 1.0
    return float(np.sqrt(value / float(anchor)))


def _mean_align_nonnegative(draws: np.ndarray, exact_mean: float) -> np.ndarray:
    a = np.asarray(draws, dtype=float).copy()
    mu = float(exact_mean)
    if a.ndim != 1 or len(a) == 0:
        raise ValueError("draws must be a non-empty 1D array")
    if not np.isfinite(a).all() or (a < 0).any():
        raise ValueError("draws must be finite and non-negative")
    if not np.isfinite(mu) or mu < 0:
        raise ValueError("exact_mean must be finite and non-negative")
    if mu == 0:
        return np.zeros_like(a)
    m = float(a.mean())
    if not np.isfinite(m) or m <= 0:
        raise ValueError("cannot align a zero-mean draw array to a positive mean")
    out = a * (mu / m)
    return out


def mean_neutral_distribution_shadow(
    draws: np.ndarray,
    *,
    exact_mean: float,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return paired mean-aligned baseline and target-depth shadow draws."""
    s = float(scale)
    if not np.isfinite(s) or s <= 0:
        raise ValueError("scale must be finite and positive")
    mu = float(exact_mean)
    baseline = _mean_align_nonnegative(np.asarray(draws, dtype=float), mu)
    if abs(s - 1.0) <= 1e-15:
        return baseline, baseline.copy()
    pre = mu + s * (baseline - mu)
    clipped = np.clip(pre, 0.0, None)
    candidate = _mean_align_nonnegative(clipped, mu)
    if abs(float(baseline.mean()) - mu) > 1e-10:
        raise RuntimeError("baseline mean preservation failed")
    if abs(float(candidate.mean()) - mu) > 1e-10:
        raise RuntimeError("candidate mean preservation failed")
    return baseline, candidate


def _identity_map() -> dict[str, str]:
    import nflreadpy as nfl

    season_maps: list[pd.DataFrame] = []
    for season in (2022, 2023, 2024, 2025, 2026):
        raw = _to_pandas(nfl.load_player_stats(seasons=[season], summary_level="week"))
        x = _normalize_weekly(raw, season)
        if "week" in x.columns and season == SEASON:
            x = x.loc[num(x["week"]).lt(WEEK)].copy()
        z = x[["season", "player_clean_key", "player_id"]].copy()
        z["player_id"] = z["player_id"].map(clean)
        z = z.loc[z["player_clean_key"].astype(str).ne("") & z["player_id"].ne("")].drop_duplicates()
        g = (
            z.groupby(["season", "player_clean_key"])["player_id"]
            .agg(lambda s: sorted(set(map(str, s))))
            .reset_index()
        )
        g["id_count"] = g["player_id"].map(len)
        g["receiver_id"] = g["player_id"].map(lambda v: v[0] if len(v) == 1 else "")
        season_maps.append(g[["season", "player_clean_key", "receiver_id", "id_count"]])

    allx = pd.concat(season_maps, ignore_index=True)
    out: dict[str, str] = {}
    for key, g in allx.groupby("player_clean_key", sort=False):
        good = g.loc[num(g["id_count"]).eq(1) & g["receiver_id"].astype(str).ne("")].sort_values("season")
        if good.empty:
            continue
        out[str(key)] = str(good.iloc[-1]["receiver_id"])
    return out


def _load_target_events() -> pd.DataFrame:
    import nflreadpy as nfl

    frames = []
    for season in (2022, 2023, 2024, 2025, 2026):
        x = _regular_only(_to_lower_frame(nfl.load_pbp(seasons=[season])))
        for c in (
            "season", "week", "game_id", "receiver_player_id",
            "pass_attempt", "sack", "two_point_attempt", "air_yards",
        ):
            if c not in x.columns:
                x[c] = np.nan
        x["season"] = num(x["season"]).fillna(season).astype(int)
        x["week"] = num(x["week"]).astype("Int64")
        if season == SEASON:
            x = x.loc[x["week"].lt(WEEK)].copy()
        x["receiver_id"] = x["receiver_player_id"].map(clean)
        raw = num(x["pass_attempt"]).fillna(0).eq(1)
        sack = num(x["sack"]).fillna(0).eq(1)
        two = num(x["two_point_attempt"]).fillna(0).eq(1)
        x["target_event"] = raw & ~sack & ~two & x["receiver_id"].ne("")
        x["air_yards"] = num(x["air_yards"])
        frames.append(
            x.loc[x["target_event"] & x["air_yards"].notna(),
                  ["season", "week", "game_id", "receiver_id", "air_yards"]]
        )
    out = pd.concat(frames, ignore_index=True, sort=False)
    violation = (out["season"].gt(SEASON)) | (out["season"].eq(SEASON) & num(out["week"]).ge(WEEK))
    if bool(violation.any()):
        raise RuntimeError("same/future Week-5 target event entered source history")
    return out


def _event_index(events: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        str(pid): g.sort_values(["season", "week", "game_id"], kind="mergesort").copy()
        for pid, g in events.groupby("receiver_id", sort=False)
    }


def feature_for(index: dict[str, pd.DataFrame], receiver_id: str) -> dict | None:
    g = index.get(str(receiver_id))
    if g is None or g.empty:
        return None
    h = g.loc[
        (num(g["season"]).lt(SEASON))
        | (num(g["season"]).eq(SEASON) & num(g["week"]).lt(WEEK))
    ].copy()
    if h.empty:
        return None
    games = (
        h[["season", "week", "game_id"]]
        .drop_duplicates()
        .sort_values(["season", "week", "game_id"], kind="mergesort")
        .tail(8)
    )
    if len(games) < 4:
        return None
    keys = set(
        zip(
            games["season"].astype(int),
            games["week"].astype(int),
            games["game_id"].astype(str),
        )
    )
    mask = [
        (int(s), int(w), str(gid)) in keys
        for s, w, gid in zip(h["season"], h["week"], h["game_id"])
    ]
    q = h.loc[mask].copy()
    air = num(q["air_yards"]).dropna().to_numpy(float)
    if len(air) < 10:
        return None
    max_season = int(q["season"].max())
    max_week = int(q.loc[q["season"].eq(max_season), "week"].max())
    if max_season > SEASON or (max_season == SEASON and max_week >= WEEK):
        raise RuntimeError("same/future target-depth feature detected")
    return {
        "prior_receiver_games": int(len(games)),
        "prior_finite_air_targets": int(len(air)),
        "prior8_target_depth_sd": float(np.std(air, ddof=0)),
        "feature_max_season": max_season,
        "feature_max_week": max_week,
    }


def _canonical_lock_bytes(df: pd.DataFrame) -> bytes:
    cols = [
        "season", "week", "event_id", "team", "opponent", "player",
        "player_clean_key", "position_family", "receiver_id", "identity_route",
        "feature_available", "feature_route", "prior_receiver_games",
        "prior_finite_air_targets", "prior8_target_depth_sd",
        "feature_max_season", "feature_max_week", "position_depth_sd_anchor",
        "depth_distribution_scale",
    ]
    x = (
        df[cols]
        .copy()
        .sort_values(["event_id", "team", "position_family", "player_clean_key"], kind="mergesort")
        .reset_index(drop=True)
    )
    return x.to_csv(index=False, float_format="%.12f", lineterminator="\n").encode("utf-8")


def build_lock(parent: pd.DataFrame, events: pd.DataFrame, identity_fallback: dict[str, str]) -> tuple[pd.DataFrame, dict]:
    x = parent.copy()
    x.columns = [str(c).strip().lower() for c in x.columns]
    required = {
        "season", "week", "event_id", "team", "opponent", "player",
        "player_clean_key", "position_family", "receiver_id",
    }
    missing = required - set(x.columns)
    if missing:
        raise RuntimeError(f"parent lock missing columns: {sorted(missing)}")
    if len(x) != 277:
        raise RuntimeError(f"parent row universe drifted: expected 277 got {len(x)}")
    if not num(x["season"]).eq(SEASON).all() or not num(x["week"]).eq(WEEK).all():
        raise RuntimeError("parent is not the frozen 2026 Week-5 universe")
    if int(x["position_family"].astype(str).eq("WR").sum()) != 167:
        raise RuntimeError("parent WR count drifted")
    if int(x["position_family"].astype(str).eq("TE").sum()) != 110:
        raise RuntimeError("parent TE count drifted")

    idx = _event_index(events)
    cache: dict[str, dict | None] = {}
    rows = []
    violations = 0
    for r in x.itertuples(index=False):
        rid = clean(getattr(r, "receiver_id", ""))
        route = "PARENT_RECEIVER_ID"
        if not rid:
            rid = clean(identity_fallback.get(str(r.player_clean_key), ""))
            route = "STRICT_PRIOR_IDENTITY_FALLBACK" if rid else "UNRESOLVED"
        if rid not in cache:
            cache[rid] = feature_for(idx, rid) if rid else None
        feat = cache.get(rid)
        pos = str(r.position_family).upper().strip()
        anchor = float(DEPTH_ANCHOR[pos])
        available = feat is not None
        if available:
            if feat["feature_max_season"] > SEASON or (
                feat["feature_max_season"] == SEASON and feat["feature_max_week"] >= WEEK
            ):
                violations += 1
            sd = float(feat["prior8_target_depth_sd"])
            scale = depth_scale(pos, sd)
            feature_route = "TARGET_DEPTH_DISPERSION_AVAILABLE"
        else:
            sd = np.nan
            scale = 1.0
            feature_route = "TARGET_DEPTH_DISPERSION_UNAVAILABLE_NO_CHANGE"
        rows.append({
            "season": SEASON,
            "week": WEEK,
            "event_id": str(r.event_id),
            "team": str(r.team),
            "opponent": str(r.opponent),
            "player": str(r.player),
            "player_clean_key": str(r.player_clean_key),
            "position_family": pos,
            "receiver_id": rid,
            "identity_route": route,
            "feature_available": bool(available),
            "feature_route": feature_route,
            "prior_receiver_games": int(feat["prior_receiver_games"]) if available else np.nan,
            "prior_finite_air_targets": int(feat["prior_finite_air_targets"]) if available else np.nan,
            "prior8_target_depth_sd": sd,
            "feature_max_season": int(feat["feature_max_season"]) if available else np.nan,
            "feature_max_week": int(feat["feature_max_week"]) if available else np.nan,
            "position_depth_sd_anchor": anchor,
            "depth_distribution_scale": float(scale),
        })

    out = pd.DataFrame(rows)
    if violations:
        raise RuntimeError(f"same/future feature violations={violations}")
    if not np.isfinite(num(out["depth_distribution_scale"])).all() or (num(out["depth_distribution_scale"]) <= 0).any():
        raise RuntimeError("invalid depth-distribution scale")
    payload = _canonical_lock_bytes(out)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    available = out["feature_available"].astype(bool)
    result = {
        "version": VERSION,
        "status": "WEEK5_PREGAME_DISTRIBUTION_LOCK_FROZEN",
        "season": SEASON,
        "week": WEEK,
        "parent_run": PARENT_RUN,
        "parent_artifact": PARENT_ARTIFACT,
        "parent_artifact_digest": PARENT_ARTIFACT_DIGEST,
        "parent_row_digest": PARENT_ROW_DIGEST,
        "row_csv_digest": digest,
        "locked_players": int(len(out)),
        "locked_wr": int(out["position_family"].eq("WR").sum()),
        "locked_te": int(out["position_family"].eq("TE").sum()),
        "resolved_receiver_ids": int(out["receiver_id"].astype(str).ne("").sum()),
        "resolved_receiver_id_fraction": float(out["receiver_id"].astype(str).ne("").mean()),
        "feature_available_players": int(available.sum()),
        "feature_available_fraction": float(available.mean()),
        "feature_available_wr": int((available & out["position_family"].eq("WR")).sum()),
        "feature_available_te": int((available & out["position_family"].eq("TE")).sum()),
        "changed_uncertainty_players": int((num(out["depth_distribution_scale"]) - 1.0).abs().gt(1e-12).sum()),
        "scale_min": float(num(out["depth_distribution_scale"]).min()),
        "scale_p10": float(num(out["depth_distribution_scale"]).quantile(0.10)),
        "scale_p50": float(num(out["depth_distribution_scale"]).quantile(0.50)),
        "scale_p90": float(num(out["depth_distribution_scale"]).quantile(0.90)),
        "scale_max": float(num(out["depth_distribution_scale"]).max()),
        "wr_depth_sd_anchor": float(DEPTH_ANCHOR["WR"]),
        "te_depth_sd_anchor": float(DEPTH_ANCHOR["TE"]),
        "same_or_future_feature_violations": 0,
        "parameters_fit": 0,
        "sportsbook_inputs_used": 0,
        "week5_outcomes_read": 0,
        "football_means_changed": False,
        "target_entitlement_changed": False,
        "team_volume_changed": False,
        "production_changed": False,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    return out, result


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--parent-lock", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    raw = a.parent_lock.read_bytes()
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    if digest != PARENT_ROW_DIGEST:
        raise RuntimeError(f"parent row digest mismatch: {digest}")
    parent = pd.read_csv(a.parent_lock, low_memory=False)
    ids = _identity_map()
    events = _load_target_events()
    lock, result = build_lock(parent, events, ids)
    a.out_dir.mkdir(parents=True, exist_ok=True)
    payload = _canonical_lock_bytes(lock)
    (a.out_dir / "player_target_depth_distribution_shadow_week5_lock.csv").write_bytes(payload)
    (a.out_dir / "player_target_depth_distribution_shadow_week5_lock.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
