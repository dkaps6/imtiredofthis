#!/usr/bin/env python3
"""M80 authoritative hardening: semantic FTN novelty + strict source contract.

Imports v2 (which fixes route/shell blank-string presence), then replaces only
the FTN source-audit job. Exact-new-column-name is not sufficient novelty: every
FTN candidate is crosswalked to the closest M1-M79 mechanism before M81.

FTN advancement additionally requires:
- complete regular-season week coverage for required history seasons,
- candidate-field coverage in every required history season, not just 2025,
- explicit fail-closed source-error disposition.

Still ZERO predictive fitting and ZERO 2025 outcome selection.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import audit_qb_research_frontier_batch_v2 as v2

m80 = v2.m80
REQUIRED_FTN_SEASONS = (2023, 2024, 2025)
REQUIRED_REG_WEEKS = set(range(1, 19))
MIN_FIELD_COVERAGE = 0.80

SEMANTIC = {
    "is_motion": (
        "ADVANCE_NEW_MECHANISM", "M67/M68 generic formation/opening tendency",
        "explicit pre-snap motion usage/response was not modeled", "TACTICAL_CALL_STRUCTURE",
    ),
    "is_screen_pass": (
        "ADVANCE_NEW_MECHANISM", "M70 YAC/explosive decomposition; M67/M68 tendencies",
        "screen call identity is a tactical play type, not generic YAC or pass tendency", "TACTICAL_CALL_STRUCTURE",
    ),
    "is_rpo": (
        "ADVANCE_NEW_MECHANISM", "M67 generic personnel/formation/pass tendency",
        "explicit RPO call identity was not modeled", "TACTICAL_CALL_STRUCTURE",
    ),
    "is_trick_play": (
        "DIAGNOSTIC_ONLY_RARE", "none",
        "genuinely new tag but too specialized to pre-register as a standalone QB correction family", "NONE",
    ),
    "is_qb_out_of_pocket": (
        "ADVANCE_NEW_MECHANISM", "M45/M56/M70 generic pressure and efficiency mechanisms",
        "QB pocket-exit response is distinct from aggregate pressure rate", "PRESSURE_RESPONSE",
    ),
    "is_interception_worthy": (
        "ADVANCE_NEW_OBSERVABLE", "M70 interception/CPOE outcome decomposition; M71 volatility",
        "charted decision-error rate is not ordinary INT rate, YPA volatility, or CPOE", "THROW_DECISION_QUALITY",
    ),
    "is_throw_away": (
        "ADVANCE_NEW_OBSERVABLE", "M45/M56/M70 generic pressure",
        "throwaway response separates pressure survival from sacks/incompletions", "PRESSURE_RESPONSE",
    ),
    "read_thrown": (
        "ADVANCE_NEW_OBSERVABLE", "none M1-M79",
        "QB read-progression distribution was not modeled", "THROW_DECISION_QUALITY",
    ),
    "is_catchable_ball": (
        "ADVANCE_NEW_OBSERVABLE", "M70 completion/CPOE decomposition",
        "manual catchability attribution is distinct from outcome completion rate/CPOE", "THROW_DECISION_QUALITY",
    ),
    "is_contested_ball": (
        "HOLD_OVERLAP_PRIOR_FAMILY", "M75 separation/cushion x secondary; M72 receiver matchup",
        "contested-target environment is too close to the already-negative separation/coverage family for standalone retest", "NONE",
    ),
    "is_created_reception": (
        "HOLD_OVERLAP_PRIOR_FAMILY", "M72 receiver explosive/YAC; M75 YACOE/receiver tracking",
        "receiver-created completion concept overlaps prior receiver-creation/after-catch information", "NONE",
    ),
    "is_drop": (
        "ADVANCE_NEW_OBSERVABLE", "M34 catch conversion and M70 completion decomposition did not chart drops",
        "manual receiver-drop attribution separates receiver failure from QB throw quality", "RECEIVER_ERROR_ATTRIBUTION",
    ),
    "n_blitzers": (
        "ADVANCE_NEW_MECHANISM", "M16/M22-M23/M45/M56/M69/M72 aggregate pressure family",
        "exact blitz construction/count is materially different from pressure rate or pass-rush strength", "PRESSURE_RESPONSE",
    ),
    "is_qb_fault_sack": (
        "ADVANCE_NEW_OBSERVABLE", "M9 sack/dropback conversion; M45/M56/M70 pressure",
        "manual QB-attributable sack responsibility was not modeled", "PRESSURE_RESPONSE",
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
        return {
            "populated_rows": int(len(s)),
            "distinct_values": int(numeric.nunique(dropna=True)),
            "event_or_nonzero_rate": float(numeric.ne(0).mean()),
            "mean_numeric": float(numeric.mean()),
        }
    z = s.astype("string").str.strip().str.lower()
    true_tokens = {"1", "true", "t", "yes", "y"}
    false_tokens = {"0", "false", "f", "no", "n"}
    known = z.isin(true_tokens | false_tokens)
    return {
        "populated_rows": int(len(s)),
        "distinct_values": int(z.nunique(dropna=True)),
        "event_or_nonzero_rate": float(z.loc[known].isin(true_tokens).mean()) if known.any() else np.nan,
        "mean_numeric": np.nan,
    }


def _regular_rows(df: pd.DataFrame) -> tuple[pd.DataFrame, list[int]]:
    if "week" not in df.columns:
        raise RuntimeError("FTN source missing required week column")
    week = pd.to_numeric(df["week"], errors="coerce")
    reg = df.loc[week.between(1, 18, inclusive="both")].copy()
    weeks = sorted(pd.to_numeric(reg["week"], errors="coerce").dropna().astype(int).unique().tolist())
    return reg, weeks


def job_ftn(out):
    out.mkdir(parents=True, exist_ok=True)
    inventory, source = [], []
    all_fields = {**m80.FTN_DUPLICATE_OR_CLOSED, **m80.FTN_NOVEL_CANDIDATES}

    for season in [2022, 2023, 2024, 2025]:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/ftn_charting/ftn_charting_{season}.parquet"
        try:
            df, meta = m80.read_url_parquet(url)
            df = m80.lower(df)
            reg, reg_weeks = _regular_rows(df)
            all_weeks = sorted(pd.to_numeric(df["week"], errors="coerce").dropna().astype(int).unique().tolist())
            reg_complete = REQUIRED_REG_WEEKS.issubset(set(reg_weeks))
            source.append({
                "season": season,
                "rows": len(df),
                "regular_rows": len(reg),
                "weeks": len(all_weeks),
                "regular_weeks": len(reg_weeks),
                "regular_weeks_complete_1_18": bool(reg_complete),
                "min_week": min(all_weeks) if all_weeks else np.nan,
                "max_week": max(all_weeks) if all_weeks else np.nan,
                **meta,
                "status": "OK" if reg_complete else "ERROR:INCOMPLETE_REGULAR_WEEK_SPAN",
            })
            if not reg_complete:
                print(f"[m80_ftn_source_error] season={season} missing_regular_weeks={sorted(REQUIRED_REG_WEEKS-set(reg_weeks))}")

            for field, old_reason in all_fields.items():
                coverage = m80.field_coverage(reg, field)
                stats = _event_stats(reg[field]) if field in reg.columns else _event_stats(pd.Series(dtype=object))
                if field in m80.FTN_DUPLICATE_OR_CLOSED:
                    disposition, closest, new_obs, family = (
                        "DUPLICATE_CLOSED", old_reason, "none; explicitly excluded from M81", "NONE"
                    )
                else:
                    disposition, closest, new_obs, family = SEMANTIC[field]
                inventory.append({
                    "season": season,
                    "scope": "REG_WEEKS_1_18",
                    "field": field,
                    "coverage": coverage,
                    "semantic_disposition": disposition,
                    "closest_prior_m1_m79": closest,
                    "materially_new_observable": new_obs,
                    "m81_family": family,
                    **stats,
                })
        except Exception as exc:
            msg = f"ERROR:{type(exc).__name__}:{exc}"
            print(f"[m80_ftn_source_error] season={season} url={url} error={msg}")
            source.append({
                "season": season, "rows": 0, "regular_rows": 0, "weeks": 0, "regular_weeks": 0,
                "regular_weeks_complete_1_18": False, "min_week": np.nan, "max_week": np.nan,
                "url": url, "bytes": 0, "sha256": "", "status": msg,
            })

    s = pd.DataFrame(source)
    inv = pd.DataFrame(inventory, columns=[
        "season", "scope", "field", "coverage", "semantic_disposition",
        "closest_prior_m1_m79", "materially_new_observable", "m81_family",
        "populated_rows", "distinct_values", "event_or_nonzero_rate", "mean_numeric",
    ])
    s.to_csv(out / "m80_ftn_source_audit.csv", index=False)
    inv.to_csv(out / "m80_ftn_field_inventory.csv", index=False)

    required_source = s.loc[s.season.isin(REQUIRED_FTN_SEASONS)].copy()
    source_error_present = bool(
        len(required_source) != len(REQUIRED_FTN_SEASONS)
        or required_source.status.astype(str).ne("OK").any()
        or ~required_source.regular_weeks_complete_1_18.fillna(False).all()
    )
    source_contract_ok = bool(
        not source_error_present
        and (required_source.rows > 0).all()
        and (required_source.regular_rows > 0).all()
    )
    source_error_summary = ";".join(
        f"{int(r.season)}:{r.status}" for _, r in required_source.iterrows() if str(r.status) != "OK"
    ) or "NONE"

    family_rows = []
    for family, fields in M81_FAMILIES.items():
        per_season = []
        for season in REQUIRED_FTN_SEASONS:
            y = inv.loc[(inv.season == season) & inv.field.isin(fields)].copy()
            per_season.append(bool(len(y) == len(fields) and (y.coverage >= MIN_FIELD_COVERAGE).all()))
        yall = inv.loc[inv.season.isin(REQUIRED_FTN_SEASONS) & inv.field.isin(fields)].copy()
        family_rows.append({
            "family": family,
            "fields": ";".join(fields),
            "field_count": len(fields),
            "coverage_required_seasons": ";".join(map(str, REQUIRED_FTN_SEASONS)),
            "all_fields_ready_every_required_season": bool(all(per_season)),
            "min_required_season_coverage": float(yall.coverage.min()) if len(yall) else 0.0,
            "regular_weeks_1_18_source_complete": source_contract_ok,
            "predictive_model_fit": False,
            "m81_status": "PREREGISTERED_DEVELOPMENT_FAMILY" if all(per_season) and source_contract_ok else "HOLD_SOURCE_CONTRACT",
        })
    families = pd.DataFrame(family_rows)
    families.to_csv(out / "m80_ftn_m81_families.csv", index=False)

    advanceable = inv.loc[
        inv.season.isin(REQUIRED_FTN_SEASONS)
        & inv.semantic_disposition.isin(["ADVANCE_NEW_MECHANISM", "ADVANCE_NEW_OBSERVABLE"])
    ].copy()
    field_ready = (
        advanceable.groupby("field")["coverage"]
        .agg(["count", "min"])
        .reset_index()
    ) if len(advanceable) else pd.DataFrame(columns=["field", "count", "min"])
    field_ready = field_ready.loc[
        field_ready["count"].eq(len(REQUIRED_FTN_SEASONS)) & field_ready["min"].ge(MIN_FIELD_COVERAGE)
    ]
    ready_families = int(families.all_fields_ready_every_required_season.sum()) if len(families) else 0
    advance = bool(source_contract_ok and ready_families >= 1)

    decision = pd.DataFrame([{
        "candidate": "FTN_NOVEL_ONLY",
        "required_history_seasons": ";".join(map(str, REQUIRED_FTN_SEASONS)),
        "required_regular_weeks": "1-18",
        "source_contract_required_seasons_ok": source_contract_ok,
        "source_error_present": source_error_present,
        "source_error_summary": source_error_summary,
        "semantically_advanceable_fields_all_required_seasons": int(field_ready.field.nunique()),
        "preregistered_m81_families": int(len(families)),
        "m81_families_source_ready_all_required_seasons": ready_families,
        "in_season_update_contract": True,
        "predictive_model_fit": False,
        "advance_to_m81_development": advance,
        "status": "QUALIFIED_FOR_M81_DEVELOPMENT" if advance else ("SOURCE_ERROR" if source_error_present else "HOLD_SOURCE_COVERAGE"),
        "notes": "M81 may use only the four preregistered mechanism families. Every family must have >=80% field coverage in each required season and complete regular weeks 1-18. DUPLICATE_CLOSED, HOLD_OVERLAP_PRIOR_FAMILY, and DIAGNOSTIC_ONLY_RARE fields are excluded. All FTN predictors must be strictly-prior historical aggregates; target-game charting is forbidden.",
    }])
    decision.to_csv(out / "m80_ftn_decision.csv", index=False)

    print("[m80_ftn_source_audit]")
    print(s.to_string(index=False))
    print("[m80_ftn_semantic_inventory_2025]")
    print(inv.loc[inv.season.eq(2025)].to_string(index=False))
    print("[m80_ftn_m81_families]")
    print(families.to_string(index=False))
    print("[m80_ftn_decision]")
    print(decision.to_string(index=False))


m80.job_ftn = job_ftn

if __name__ == "__main__":
    raise SystemExit(m80.main())
