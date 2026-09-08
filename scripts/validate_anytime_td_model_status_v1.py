#!/usr/bin/env python3
"""Audit the current anytime-TD production lane without pretending it is validated.

The old scorer that used market-derived team touchdown opportunity belongs to the
retired alternate engine and is not part of current Full Slate.  Current ATD is
sportsbook-independent football logic, but it has not yet earned a dedicated
walk-forward probability-model certification.  This audit makes both facts
explicit and fails only if the execution/provenance contract is violated.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data")
OUTPUTS = Path("outputs")
OUT_CSV = DATA / "anytime_td_model_status_v1.csv"
OUT_JSON = DATA / "anytime_td_model_status_v1.json"


def main() -> int:
    priced_path = OUTPUTS / "props_priced_clean.csv"
    prior_status_path = DATA / "player_scoring_td_prior_status.json"
    if not priced_path.exists() or priced_path.stat().st_size <= 0:
        raise RuntimeError("ATD status requires priced output")
    if not prior_status_path.exists() or prior_status_path.stat().st_size <= 0:
        raise RuntimeError("ATD status requires player_scoring_td_prior_status.json")

    priced = pd.read_csv(priced_path, low_memory=False)
    priced.columns = [str(c).strip().lower() for c in priced.columns]
    atd = priced.loc[priced["source_market"].astype(str).eq("player_anytime_td")].copy()
    if atd.empty:
        raise RuntimeError("ATD status found zero priced anytime-TD rows")
    prior = json.loads(prior_status_path.read_text(encoding="utf-8"))

    if prior.get("sportsbook_inputs_used") is not False:
        raise RuntimeError("current ATD scoring prior is not certified sportsbook-independent")
    if int(pd.to_numeric(atd.get("ml_applied", 0), errors="coerce").fillna(0).sum()) != 0:
        raise RuntimeError("current ATD unexpectedly consumed the generic yardage ML lane")
    if int(pd.to_numeric(atd.get("state_applied", 0), errors="coerce").fillna(0).sum()) != 0:
        raise RuntimeError("current ATD unexpectedly consumed the generic State lane")
    fair = pd.to_numeric(atd["fair_prob"], errors="coerce")
    if fair.isna().any() or not np.isfinite(fair).all() or not fair.between(0, 1, inclusive="both").all():
        raise RuntimeError("ATD current execution produced invalid probabilities")

    unique = atd.loc[atd["side"].astype(str).eq("OVER")].drop_duplicates(["event_id", "player", "book", "vegas_line"])
    rows = pd.DataFrame([{
        "execution_status": "PASS",
        "scientific_status": "DEDICATED_ATD_WALK_FORWARD_CERTIFICATION_REQUIRED",
        "old_market_assisted_scorer_active": 0,
        "current_sportsbook_inputs_used_for_football_probability": 0,
        "current_method": "offensive_td_rate + rz_share/script modifiers + joint MC Bernoulli",
        "priced_atd_book_line_rows": int(len(unique)),
        "priced_atd_players": int(unique["player"].nunique()),
        "player_history_rate_rows": int(prior.get("player_history_rate_rows", 0)),
        "position_prior_fallback_rows": int(prior.get("position_prior_fallback_rows", 0)),
        "dedicated_probability_model_certified": 0,
        "dedicated_brier_logloss_calibration_audited": 0,
    }])
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    rows.to_csv(OUT_CSV, index=False)
    payload = rows.iloc[0].to_dict()
    payload.update({
        "disposition": "ATD_EXECUTION_PROVEN_SCIENCE_NOT_YET_CERTIFIED",
        "next_required_research": "football-only team TD opportunity -> red-zone/end-zone pool -> player entitlement -> probability calibration",
        "sportsbook_role": "downstream comparison only",
    })
    OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print("[anytime_td_model_status] " + json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
