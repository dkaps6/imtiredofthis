#!/usr/bin/env python3
"""M80 authoritative hardening: semantic FTN novelty + event prevalence.

Imports v2 (which fixes route/shell blank-string presence), then replaces only
the FTN source-audit job. This prevents exact-new-column-name from being treated
as sufficient novelty: every FTN candidate is crosswalked to the closest M1-M79
mechanism and classified before M81.

Still ZERO predictive fitting and ZERO 2025 outcome selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import audit_qb_research_frontier_batch_v2 as v2

m80 = v2.m80

SEMANTIC = {
    "is_motion": (
        "ADVANCE_NEW_MECHANISM", "M67/M68 generic formation/opening tendency",
        "explicit pre-snap motion usage/response was not modeled",
        "TACTICAL_CALL_STRUCTURE",
    ),
    "is_screen_pass": (
        "ADVANCE_NEW_MECHANISM", "M70 YAC/explosive decomposition; M67/M68 tendencies",
        "screen call identity is a tactical play type, not generic YAC or pass tendency",
        "TACTICAL_CALL_STRUCTURE",
    ),
    "is_rpo": (
        "ADVANCE_NEW_MECHANISM", "M67 generic personnel/formation/pass tendency",
        "explicit RPO call identity was not modeled",
        "TACTICAL_CALL_STRUCTURE",
    ),
    "is_trick_play": (
        "DIAGNOSTIC_ONLY_RARE", "none",
        "genuinely new tag but too specialized to pre-register as a standalone QB correction family",
        "NONE",
    ),
    "is_qb_out_of_pocket": (
        "ADVANCE_NEW_MECHANISM", "M45/M56/M70 generic pressure and efficiency mechanisms",
        "QB pocket-exit response is distinct from aggregate pressure rate",
        "PRESSURE_RESPONSE",
    ),
    "is_interception_worthy": (
        "ADVANCE_NEW_OBSERVABLE", "M70 interception/CPOE outcome decomposition; M71 volatility",
        "charted decision-error rate is not ordinary INT rate, YPA volatility, or CPOE",
        "THROW_DECISION_QUALITY",
    ),
    "is_throw_away": (
        "ADVANCE_NEW_OBSERVABLE", "M45/M56/M70 generic pressure",
        "throwaway response separates pressure survival from sacks/incompletions",
        "PRESSURE_RESPONSE",
    ),
    "read_thrown": (
        "ADVANCE_NEW_OBSERVABLE", "none M1-M79",
        "QB read-progression distribution was not modeled",
        "THROW_DECISION_QUALITY",
    ),
    "is_catchable_ball": (
        "ADVANCE_NEW_OBSERVABLE", "M70 completion/CPOE decomposition",
        "manual catchability attribution is distinct from outcome completion rate/CPOE",
        "THROW_DECISION_QUALITY",
    ),
    "is_contested_ball": (
        "HOLD_OVERLAP_PRIOR_FAMILY", "M75 separation/cushion x secondary; M72 receiver matchup",
        "contested-target environment is too close to the already-negative separation/coverage family for standalone retest",
        "NONE",
    ),
    "is_created_reception": (
        "HOLD_OVERLAP_PRIOR_FAMILY", "M72 receiver explosive/YAC; M75 YACOE/receiver tracking",
        "receiver-created completion concept overlaps prior receiver-creation/after-catch information",
        "NONE",
    ),
    "is_drop": (
        "ADVANCE_NEW_OBSERVABLE", "M34 catch conversion and M70 completion decomposition did not chart drops",
        "manual receiver-drop attribution separates receiver failure from QB throw quality",
        "RECEIVER_ERROR_ATTRIBUTION",
    ),
    "n_blitzers": (
        "ADVANCE_NEW_MECHANISM", "M16/M22-M23/M45/M56/M69/M72 aggregate pressure family",
        "exact blitz construction/count is materially different from pressure rate or pass-rush strength",
        "PRESSURE_RESPONSE",
    ),
    "is_qb_fault_sack": (
        "ADVANCE_NEW_OBSERVABLE", "M9 sack/dropback conversion; M45/M56/M70 pressure",
        "manual QB-attributable sack responsibility was not modeled",
        "PRESSURE_RESPONSE",
    ),
}

M81_FAMILIES = {
    "TACTICAL_CALL_STRUCTURE": ["is_motion", "is_screen_pass", "is_rpo"],
    "PRESSURE_RESPONSE": ["n_blitzers", "is_qb_out_of_pocket", "is_throw_away", "is_qb_fault_sack"],
    "THROW_DECISION_QUALITY": ["is_interception_worthy", "read_thrown", "is_catchable_ball"],
    "RECEIVER_ERROR_ATTRIBUTION": ["is_drop"],
}


def _present(series: pd.Series) -> pd.Series:
    return v2._present(series)


def _event_stats(series: pd.Series) -> dict:
    present = _present(series)
    s = series.loc[present]
    if s.empty:
        return {"populated_rows": 0, "distinct_values": 0, "event_or_nonzero_rate": np.nan, "mean_numeric": np.nan}
    numeric = pd.to_numeric(s, errors="coerce")
    if numeric.notna().mean() >= 0.95:
        nonzero = float(numeric.ne(0).mean())
        return {
            "populated_rows": int(len(s)),
            "distinct_values": int(numeric.nunique(dropna=True)),
            "event_or_nonzero_rate": nonzero,
            "mean_numeric": float(numeric.mean()),
        }
    z = s.astype("string").str.strip().str.lower()
    true_tokens = {"1", "true", "t", "yes", "y"}
    false_tokens = {"0", "false", "f", "no", "n"}
    known = z.isin(true_tokens | false_tokens)
    rate = float(z.loc[known].isin(true_tokens).mean()) if known.any() else np.nan
    return {
        "populated_rows": int(len(s)),
        "distinct_values": int(z.nunique(dropna=True)),
        "event_or_nonzero_rate": rate,
        "mean_numeric": np.nan,
    }


def job_ftn(out):
    out.mkdir(parents=True, exist_ok=True)
    inventory = []
    source = []
    all_fields = {**m80.FTN_DUPLICATE_OR_CLOSED, **m80.FTN_NOVEL_CANDIDATES}
    for season in [2022, 2023, 2024, 2025]:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        try:
            df, meta = m80.read_url_parquet(url)
            df = m80.lower(df)
            weeks = sorted(pd.to_numeric(df.get("week"), errors="coerce").dropna().astype(int).unique().tolist()) if "week" in df else []
            source.append({
                "season": season, "rows": len(df), "weeks": len(weeks),
                "min_week": min(weeks) if weeks else np.nan,
                "max_week": max(weeks) if weeks else np.nan,
                **meta, "status": "OK",
            })
            for field, old_reason in all_fields.items():
                coverage = m80.field_coverage(df, field)
                stats = _event_stats(df[field]) if field in df.columns else _event_stats(pd.Series(dtype=object))
                if field in m80.FTN_DUPLICATE_OR_CLOSED:
                    disposition = "DUPLICATE_CLOSED"
                    closest = old_reason
                    new_obs = "none; explicitly excluded from M81"
                    family = "NONE"
                else:
                    disposition, closest, new_obs, family = SEMANTIC[field]
                inventory.append({
                    "season": season,
                    "field": field,
                    "coverage": coverage,
                    "semantic_disposition": disposition,
                    "closest_prior_m1_m79": closest,
                    "materially_new_observable": new_obs,
                    "m81_family": family,
                    **stats,
                })
        except Exception as exc:
            source.append({
                "season": season, "rows": 0, "weeks": 0, "min_week": np.nan, "max_week": np.nan,
                "url": url, "bytes": 0, "sha256": "", "status": f"ERROR:{type(exc).__name__}:{exc}",
            })

    s = pd.DataFrame(source)
    inv = pd.DataFrame(inventory)
    s.to_csv(out / "m80_ftn_source_audit.csv", index=False)
    inv.to_csv(out / "m80_ftn_field_inventory.csv", index=False)

    family_rows = []
    for family, fields in M81_FAMILIES.items():
        y = inv.loc[(inv.season == 2025) & inv.field.isin(fields)].copy()
        family_rows.append({
            "family": family,
            "fields": ";".join(fields),
            "field_count": len(fields),
            "all_fields_present_2025": bool(len(y) == len(fields) and (y.coverage >= 0.80).all()),
            "min_2025_coverage": float(y.coverage.min()) if len(y) else 0.0,
            "predictive_model_fit": False,
            "m81_status": "PREREGISTERED_DEVELOPMENT_FAMILY",
        })
    families = pd.DataFrame(family_rows)
    families.to_csv(out / "m80_ftn_m81_families.csv", index=False)

    ok_2425 = bool(
        not s.empty
        and all((s.loc[s.season.eq(y), "rows"] > 0).any() for y in [2024, 2025])
    )
    adv = inv.loc[
        (inv.season == 2025)
        & inv.semantic_disposition.isin(["ADVANCE_NEW_MECHANISM", "ADVANCE_NEW_OBSERVABLE"])
        & (inv.coverage >= 0.80)
    ]
    ready_families = int(families.all_fields_present_2025.sum()) if len(families) else 0
    decision = pd.DataFrame([{
        "candidate": "FTN_NOVEL_ONLY",
        "historical_2024_2025_available": ok_2425,
        "semantically_advanceable_fields_2025": int(adv.field.nunique()),
        "preregistered_m81_families": int(len(families)),
        "m81_families_source_ready_2025": ready_families,
        "in_season_update_contract": True,
        "predictive_model_fit": False,
        "advance_to_m81_development": bool(ok_2425 and ready_families >= 1),
        "notes": "M81 may use only the four preregistered mechanism families. DUPLICATE_CLOSED, HOLD_OVERLAP_PRIOR_FAMILY, and DIAGNOSTIC_ONLY_RARE fields are excluded from predictive development. All FTN predictors must be strictly-prior historical aggregates; target-game charting is forbidden.",
    }])
    decision.to_csv(out / "m80_ftn_decision.csv", index=False)
    print(s.to_string(index=False))
    print(inv.loc[inv.season.eq(2025)].to_string(index=False))
    print(families.to_string(index=False))
    print(decision.to_string(index=False))


m80.job_ftn = job_ftn

if __name__ == "__main__":
    raise SystemExit(m80.main())
