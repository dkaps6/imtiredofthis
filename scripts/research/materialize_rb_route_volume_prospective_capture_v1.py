#!/usr/bin/env python3
"""Materialize immutable RB route-volume prospective research captures.

This is source-validation infrastructure only. It does not fit a model, grade
outcomes, use sportsbook data, or modify production inputs.

Provider acquisition is intentionally separate. This script consumes small
normalized provider exports so dynamic-page parsing cannot silently redefine the
scientific contract.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from scripts._opponent_map import canon_team
from scripts.utils.canonical_names import canonicalize_player_name_safe

ELIGIBLE_POSITIONS = {"RB", "HB", "FB"}
FORBIDDEN_COLUMNS = {
    "actual", "bet_result", "unit_result", "vegas_line", "vegas_odds",
    "fair_prob", "edge_pct", "edge_abs", "decision",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_player_key(value: Any) -> str:
    _, key = canonicalize_player_name_safe(value)
    return str(key or "").strip()


def _position(value: Any) -> str:
    return str(value or "").strip().upper()


def _parse_utc(value: str) -> str:
    text = str(value).strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    dt = datetime.fromisoformat(text)
    if dt.tzinfo is None:
        raise ValueError("captured timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def _reject_forbidden(df: pd.DataFrame, source: str) -> None:
    bad = sorted(FORBIDDEN_COLUMNS.intersection(df.columns))
    if bad:
        raise RuntimeError(f"{source} input contains forbidden outcome/sportsbook fields: {bad}")


def _base_normalize(df: pd.DataFrame, source: str) -> pd.DataFrame:
    _reject_forbidden(df, source)
    required = {"player", "team", "position"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"{source} missing required columns: {missing}")

    x = df.copy()
    x["player"] = x["player"].astype("string").fillna("").str.strip()
    x["team"] = x["team"].map(canon_team)
    x["position"] = x["position"].map(_position)
    x = x.loc[x["position"].isin(ELIGIBLE_POSITIONS)].copy()
    x["player_clean_key"] = x["player"].map(_canonical_player_key)

    unresolved = x["player_clean_key"].eq("") | x["team"].astype("string").fillna("").str.strip().eq("")
    if unresolved.any():
        sample = x.loc[unresolved, ["player", "team", "position"]].head(20).to_dict("records")
        raise RuntimeError(f"{source} unresolved canonical identity: {sample}")

    if x.duplicated(["team", "player_clean_key"]).any():
        sample = x.loc[
            x.duplicated(["team", "player_clean_key"], keep=False),
            ["player", "team", "position", "player_clean_key"],
        ].head(20).to_dict("records")
        raise RuntimeError(f"{source} duplicate team/player identity rows: {sample}")
    return x


def normalize_heatradar(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"player", "team", "position", "routes"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"heatradar missing normalized columns: {missing}")
    x = _base_normalize(raw, "heatradar")
    x["routes"] = pd.to_numeric(x["routes"], errors="coerce")
    if x["routes"].isna().any() or (x["routes"] < 0).any():
        raise RuntimeError("heatradar has invalid routes")
    if "route_pct" in x.columns:
        x["route_pct"] = pd.to_numeric(x["route_pct"], errors="coerce")
    else:
        x["route_pct"] = np.nan
    keep = ["player", "player_clean_key", "team", "position", "routes", "route_pct"]
    return x[keep].sort_values(["team", "player_clean_key"]).reset_index(drop=True)


def normalize_statrankings(path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    required = {"player", "team", "position", "cumulative_routes"}
    missing = sorted(required - set(raw.columns))
    if missing:
        raise RuntimeError(f"statrankings missing normalized columns: {missing}")
    x = _base_normalize(raw, "statrankings")
    x["cumulative_routes"] = pd.to_numeric(x["cumulative_routes"], errors="coerce")
    if x["cumulative_routes"].isna().any() or (x["cumulative_routes"] < 0).any():
        raise RuntimeError("statrankings has invalid cumulative_routes")
    keep = ["player", "player_clean_key", "team", "position", "cumulative_routes"]
    return x[keep].sort_values(["team", "player_clean_key"]).reset_index(drop=True)


def derive_week(current: pd.DataFrame, prior: pd.DataFrame) -> pd.DataFrame:
    cur = current.rename(columns={
        "player": "player_current",
        "team": "team_current",
        "position": "position_current",
        "cumulative_routes": "cumulative_routes_current",
    })
    prv = prior.rename(columns={
        "player": "player_prior",
        "team": "team_prior",
        "position": "position_prior",
        "cumulative_routes": "cumulative_routes_prior",
    })
    z = cur.merge(
        prv[[
            "player_clean_key", "player_prior", "team_prior", "position_prior",
            "cumulative_routes_prior",
        ]],
        on="player_clean_key",
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    z["derivation_status"] = "READY"
    z.loc[z["_merge"].eq("left_only"), "derivation_status"] = "NO_PRIOR_SNAPSHOT"
    z.loc[z["_merge"].eq("right_only"), "derivation_status"] = "MISSING_CURRENT_SNAPSHOT"

    both = z["_merge"].eq("both")
    team_changed = both & z["team_current"].ne(z["team_prior"])
    z.loc[team_changed, "derivation_status"] = "TEAM_CHANGED"

    z["derived_week_routes"] = np.nan
    ready = z["derivation_status"].eq("READY")
    delta = (
        pd.to_numeric(z["cumulative_routes_current"], errors="coerce")
        - pd.to_numeric(z["cumulative_routes_prior"], errors="coerce")
    )
    negative = ready & (delta < 0)
    z.loc[negative, "derivation_status"] = "NEGATIVE_PROVIDER_DELTA"
    ready = z["derivation_status"].eq("READY")
    z.loc[ready, "derived_week_routes"] = delta.loc[ready]

    # Missing rows never become statistical zeroes.
    assert z.loc[~z["derivation_status"].eq("READY"), "derived_week_routes"].isna().all()
    return z.drop(columns=["_merge"]).sort_values(
        ["derivation_status", "player_clean_key"], na_position="last"
    ).reset_index(drop=True)


def parity(heatradar: pd.DataFrame, derived: pd.DataFrame) -> pd.DataFrame:
    d = derived.loc[derived["derivation_status"].eq("READY")].copy()
    d = d[["player_clean_key", "team_current", "derived_week_routes"]].rename(
        columns={"team_current": "team"}
    )
    h = heatradar[["player", "player_clean_key", "team", "routes"]].copy()
    z = h.merge(
        d,
        on=["player_clean_key", "team"],
        how="outer",
        validate="one_to_one",
        indicator=True,
    )
    z["parity_status"] = np.select(
        [
            z["_merge"].eq("both"),
            z["_merge"].eq("left_only"),
            z["_merge"].eq("right_only"),
        ],
        ["MATCHABLE", "HEATRADAR_ONLY", "STATRANKINGS_ONLY"],
        default="UNKNOWN",
    )
    z["abs_route_gap"] = np.nan
    both = z["parity_status"].eq("MATCHABLE")
    z.loc[both, "abs_route_gap"] = (
        pd.to_numeric(z.loc[both, "routes"], errors="coerce")
        - pd.to_numeric(z.loc[both, "derived_week_routes"], errors="coerce")
    ).abs()
    z["exact_route_match"] = False
    z.loc[both, "exact_route_match"] = z.loc[both, "abs_route_gap"].eq(0)
    return z.drop(columns=["_merge"]).sort_values(
        ["parity_status", "team", "player_clean_key"], na_position="last"
    ).reset_index(drop=True)


def parity_summary(p: pd.DataFrame) -> dict:
    matched = p.loc[p["parity_status"].eq("MATCHABLE")].copy()
    gaps = pd.to_numeric(matched["abs_route_gap"], errors="coerce").dropna()
    return {
        "cross_source_matched_rows": int(len(matched)),
        "exact_route_count_agreement_rate": (
            float(matched["exact_route_match"].mean()) if len(matched) else None
        ),
        "median_abs_route_gap": float(gaps.median()) if len(gaps) else None,
        "p90_abs_route_gap": float(gaps.quantile(0.90)) if len(gaps) else None,
        "heatradar_only_rows": int(p["parity_status"].eq("HEATRADAR_ONLY").sum()),
        "statrankings_only_rows": int(p["parity_status"].eq("STATRANKINGS_ONLY").sum()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--season", type=int, required=True)
    ap.add_argument("--through-week", type=int, required=True)
    ap.add_argument("--captured-utc", required=True)
    ap.add_argument("--heatradar-csv", type=Path, required=True)
    ap.add_argument("--statrankings-csv", type=Path, required=True)
    ap.add_argument("--prior-statrankings-csv", type=Path)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()

    if a.season != 2026:
        raise RuntimeError("V1 prospective capture is frozen to season 2026")
    if a.through_week < 1:
        raise RuntimeError("through-week must be >=1")
    captured_utc = _parse_utc(a.captured_utc)

    if a.out_dir.exists() and any(a.out_dir.iterdir()):
        raise RuntimeError("immutable capture directory already exists and is nonempty")
    a.out_dir.mkdir(parents=True, exist_ok=True)

    heat = normalize_heatradar(a.heatradar_csv)
    stat = normalize_statrankings(a.statrankings_csv)
    heat.to_csv(a.out_dir / "heatradar_rb_routes_normalized.csv", index=False)
    stat.to_csv(a.out_dir / "statrankings_rb_routes_cumulative_normalized.csv", index=False)

    derived = None
    p = None
    if a.prior_statrankings_csv is not None:
        prior = normalize_statrankings(a.prior_statrankings_csv)
        derived = derive_week(stat, prior)
        derived.to_csv(a.out_dir / "statrankings_rb_routes_weekly_derived.csv", index=False)
        p = parity(heat, derived)
        p.to_csv(a.out_dir / "rb_route_cross_source_parity.csv", index=False)

    manifest = {
        "contract": "RB_ROUTE_VOLUME_PROSPECTIVE_CAPTURE_V1",
        "season": int(a.season),
        "through_week": int(a.through_week),
        "captured_utc": captured_utc,
        "outcomes_used": False,
        "sportsbook_inputs_used": False,
        "paid_source_used": False,
        "heatradar_input_sha256": _sha256(a.heatradar_csv),
        "statrankings_input_sha256": _sha256(a.statrankings_csv),
        "prior_statrankings_input_sha256": (
            _sha256(a.prior_statrankings_csv) if a.prior_statrankings_csv else None
        ),
        "heatradar_rb_rows": int(len(heat)),
        "statrankings_rb_rows": int(len(stat)),
        "statrankings_derivation_status_counts": (
            derived["derivation_status"].value_counts(dropna=False).to_dict()
            if derived is not None else None
        ),
        "parity": parity_summary(p) if p is not None else None,
    }
    (a.out_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
