"""RB rush+receiving conservation V2 research-gated pricing adapter.

Non-Week-1 only. Builds pathwise rush+receiving draws from the two standalone
component distributions after each component is aligned to its own final generic
ensemble mean. Sportsbook line/odds columns are never read.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.modeling.ensemble_v2 import apply_ensemble
from scripts.simulation_v2 import MARKET_MAP, _player_key, lookup

VERSION = "RB_RUSH_REC_CONSERVATION_V2"
AUDIT_CSV = Path("data/rb_rush_rec_conservation_v2_integration_audit.csv")
AUDIT_JSON = Path("data/rb_rush_rec_conservation_v2_integration_audit.json")

_ALLOWED = [
    "event_id", "player", "player_clean_key", "team", "opponent",
    "position_group", "position", "alignment_position",
    "season", "week", "market", "ml_proj", "state_proj",
]


def _position(row: pd.Series) -> str:
    for c in ("position_group", "position", "alignment_position"):
        v = row.get(c)
        if v is None or pd.isna(v):
            continue
        p = str(v).upper().strip()
        if p in {"HB", "TB"} or p.startswith("RB"):
            return "RB"
        if p.startswith("FB"):
            return "FB"
        return p
    return ""


def _canonical_market(v) -> str:
    s = str(v or "").lower()
    return MARKET_MAP.get(s, s)


def _scale_to_mean(arr: np.ndarray, target: float, *, label: str) -> np.ndarray:
    x = np.asarray(arr, dtype=float)
    if not np.isfinite(x).all() or (x < 0).any():
        raise RuntimeError(f"{label}: component draws must be finite/nonnegative")
    raw = float(np.mean(x))
    if raw > 0 and np.isfinite(target):
        return x * max(0.0, float(target) / raw)
    if abs(float(target)) <= 1e-12:
        return np.zeros_like(x)
    raise RuntimeError(f"{label}: cannot align positive target={target} from raw mean={raw}")


def _ensemble_mean(row: pd.Series, outcomes: np.ndarray, weights: pd.DataFrame, market: str) -> float:
    mc = float(np.mean(np.asarray(outcomes, dtype=float)))
    c = pd.DataFrame([{
        "market": market,
        "mc_proj": mc,
        "ml_proj": row.get("ml_proj"),
        "state_proj": row.get("state_proj"),
    }])
    ens = apply_ensemble(c, weights=weights).iloc[0]
    out = float(ens["ensemble_proj"])
    if not np.isfinite(out):
        raise RuntimeError(f"{market}: non-finite component ensemble mean")
    return out


def build_candidate_map(metrics: pd.DataFrame, sims, weights: pd.DataFrame) -> tuple[dict, dict]:
    missing = [c for c in ("event_id", "market", "week") if c not in metrics.columns]
    if missing:
        raise RuntimeError(f"V2 metrics missing required columns: {missing}")

    cols = [c for c in _ALLOWED if c in metrics.columns]
    f = metrics[cols].copy()
    f.columns = [str(c).strip().lower() for c in f.columns]
    f["_canonical_market"] = f["market"].map(_canonical_market)
    f["_position_family"] = f.apply(_position, axis=1)
    f["_pkey"] = f.apply(_player_key, axis=1)
    f["_event"] = f["event_id"].astype(str)
    f["_week"] = pd.to_numeric(f["week"], errors="coerce")

    # Candidate is explicitly non-Week-1 and RB/FB only.
    combos = f.loc[
        f["_canonical_market"].eq("rush_rec_yards")
        & f["_position_family"].isin({"RB", "FB"})
        & f["_week"].ne(1)
    ].copy()

    if combos.empty:
        payload = {
            "version": VERSION,
            "disposition": "RB_RUSH_REC_CONSERVATION_V2_NO_ELIGIBLE_ROWS",
            "players": 0,
            "sportsbook_inputs_used": 0,
            "week1_rows_changed": 0,
        }
        AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(columns=["event_id","player_clean_key"]).to_csv(AUDIT_CSV, index=False)
        AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        return {}, payload

    # One football component authority per player-game-market; sportsbook
    # duplication must not alter ml/state component projections.
    key = ["_event", "_pkey", "_canonical_market"]
    for col in ("ml_proj", "state_proj"):
        if col in f.columns:
            spread = f.groupby(key, dropna=False)[col].nunique(dropna=False)
            bad = spread[spread.gt(1)]
            if len(bad):
                raise RuntimeError(f"V2 {col} varies across duplicate offers: {bad.head(20).to_dict()}")

    reps = f.sort_values(["_event","_pkey","_canonical_market"]).drop_duplicates(key, keep="first")
    rep = {}
    for _, r in reps.iterrows():
        rep[(str(r["_event"]), str(r["_pkey"]), str(r["_canonical_market"]))] = r.copy()

    result = {}
    audit_rows = []
    seen = set()
    for _, combo in combos.iterrows():
        event = str(combo["_event"])
        pkey = str(combo["_pkey"])
        k = (event, pkey)
        if k in seen:
            continue
        seen.add(k)

        rush_row = rep.get((event, pkey, "rush_yards"))
        rec_row = rep.get((event, pkey, "rec_yards"))
        if rush_row is None or rec_row is None:
            raise RuntimeError(f"V2 missing standalone component row event={event} player={pkey}")

        rush_series = rush_row.copy()
        rec_series = rec_row.copy()
        rush_draws = lookup(sims, rush_series, "rush_yards")
        rec_draws = lookup(sims, rec_series, "rec_yards")
        combo_before = lookup(sims, combo, "rush_rec_yards")
        if rush_draws is None or rec_draws is None or combo_before is None:
            raise RuntimeError(f"V2 missing simulation component event={event} player={pkey}")

        rush_draws = np.asarray(rush_draws, float)
        rec_draws = np.asarray(rec_draws, float)
        combo_before = np.asarray(combo_before, float)
        if not (len(rush_draws) == len(rec_draws) == len(combo_before)):
            raise RuntimeError(f"V2 component draw length mismatch event={event} player={pkey}")

        rush_target = _ensemble_mean(rush_series, rush_draws, weights, "rush_yards")
        rec_target = _ensemble_mean(rec_series, rec_draws, weights, "rec_yards")
        adj_rush = _scale_to_mean(rush_draws, rush_target, label=f"{event}/{pkey}/rush")
        adj_rec = _scale_to_mean(rec_draws, rec_target, label=f"{event}/{pkey}/rec")
        conserved = adj_rush + adj_rec
        target = rush_target + rec_target
        mean_gap = float(np.mean(conserved) - target)
        path_gap = float(np.max(np.abs(conserved - (adj_rush + adj_rec))))
        if abs(mean_gap) > 1e-10 or path_gap > 1e-10:
            raise RuntimeError(f"V2 conservation failure event={event} player={pkey} mean_gap={mean_gap} path_gap={path_gap}")
        if not np.isfinite(conserved).all() or (conserved < 0).any():
            raise RuntimeError(f"V2 invalid conserved draws event={event} player={pkey}")

        result[k] = {
            "draws": conserved,
            "target_mean": float(target),
            "rush_target_mean": float(rush_target),
            "rec_target_mean": float(rec_target),
        }
        audit_rows.append({
            "event_id": event,
            "player_clean_key": pkey,
            "player": combo.get("player"),
            "team": combo.get("team"),
            "week": int(combo["_week"]),
            "rush_target_mean": float(rush_target),
            "rec_target_mean": float(rec_target),
            "candidate_combo_mean": float(np.mean(conserved)),
            "raw_combo_mean": float(np.mean(combo_before)),
            "mean_conservation_gap": mean_gap,
            "pathwise_identity_max_gap": path_gap,
            "finite_nonnegative": True,
            "sportsbook_inputs_used": 0,
        })

    audit = pd.DataFrame(audit_rows)
    AUDIT_CSV.parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(AUDIT_CSV, index=False)
    payload = {
        "version": VERSION,
        "disposition": "RB_RUSH_REC_CONSERVATION_V2_CANDIDATE_BUILT",
        "players": int(len(audit)),
        "max_mean_conservation_gap": float(audit["mean_conservation_gap"].abs().max()) if len(audit) else 0.0,
        "max_pathwise_identity_gap": float(audit["pathwise_identity_max_gap"].abs().max()) if len(audit) else 0.0,
        "sportsbook_inputs_used": 0,
        "week1_rows_changed": 0,
    }
    AUDIT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return result, payload
