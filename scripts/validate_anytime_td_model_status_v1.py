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
        # Zero ATD rows has two very different causes and only one is a defect.
        #
        # player_anytime_td was removed from the paid pull on 2026-11-04 by
        # standing instruction, in favour of the alternate-line ladders. When
        # the market is not requested there is simply nothing to audit, and the
        # provenance contract this script enforces -- sportsbook-independent
        # prior, no generic ML/State lane, valid probabilities -- holds
        # vacuously. Failing there blocks every paid slate for the absence of a
        # market nobody asked for; 2026 Week 9 run 42104253081 died exactly
        # that way after the board was already priced.
        #
        # If the market IS configured and still produced nothing, that is a
        # real provider or pricing gap and must still fail closed. The
        # configured market list is the authority for telling the two apart; if
        # it cannot be read, fail closed rather than assume.
        try:
            from scripts.fetch_props_oddsapi import MARKETS as _CONFIGURED_MARKETS
        except Exception as exc:
            raise RuntimeError(
                "ATD status found zero priced anytime-TD rows and the configured "
                f"market list could not be read to explain it: {exc}"
            ) from exc
        if "player_anytime_td" in set(_CONFIGURED_MARKETS):
            raise RuntimeError(
                "ATD status found zero priced anytime-TD rows even though "
                "player_anytime_td IS in the configured market pull"
            )
        rows = pd.DataFrame([{
            "execution_status": "PASS",
            "scientific_status": "DEDICATED_ATD_WALK_FORWARD_CERTIFICATION_REQUIRED",
            "atd_market_offered": 0,
            "old_market_assisted_scorer_active": 0,
            "current_sportsbook_inputs_used_for_football_probability": 0,
            "current_method": "not priced this run; player_anytime_td is not in the configured market pull",
            "priced_atd_book_line_rows": 0,
            "priced_atd_players": 0,
            "player_history_rate_rows": 0,
            "position_prior_fallback_rows": 0,
            "dedicated_probability_model_certified": 0,
            "dedicated_brier_logloss_calibration_audited": 0,
        }])
        OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
        rows.to_csv(OUT_CSV, index=False)
        payload = rows.iloc[0].to_dict()
        payload.update({
            "disposition": "ATD_MARKET_NOT_OFFERED_BY_CONFIGURATION",
            "next_required_research": "none while the market is not pulled; re-audit before reinstating player_anytime_td",
            "sportsbook_role": "not fetched",
        })
        OUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("[anytime_td_model_status] " + json.dumps(payload, sort_keys=True))
        return 0
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
        "atd_market_offered": 1,
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
