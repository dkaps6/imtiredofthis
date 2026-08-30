#!/usr/bin/env python3
"""M80 hardening wrapper.

Preserves the frozen M80 diagnostic/source-frontier contract while correcting
route+coverage same-play presence semantics: empty strings are not observations.
No predictive modeling or candidate-selection rules are changed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

import audit_qb_research_frontier_batch as m80


def _present(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.notna()
    if pd.api.types.is_numeric_dtype(series):
        return pd.to_numeric(series, errors="coerce").notna()
    z = series.astype("string").str.strip()
    return z.notna() & z.ne("") & z.str.lower().ne("nan") & z.str.lower().ne("none")


def job_route(out):
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for season in [2024, 2025]:
        url = f"https://github.com/nflverse/nflverse-data/releases/download/pbp_participation/pbp_participation_{season}.parquet"
        try:
            df, meta = m80.read_url_parquet(url)
            df = m80.lower(df)
            route = m80.field_coverage(df, "route")
            shell = m80.field_coverage(df, "defense_coverage_type")
            manzone = m80.field_coverage(df, "defense_man_zone_type")
            route_mask = _present(df["route"]) if "route" in df.columns else pd.Series(False, index=df.index)
            shell_mask = _present(df["defense_coverage_type"]) if "defense_coverage_type" in df.columns else pd.Series(False, index=df.index)
            both = float((route_mask & shell_mask).mean()) if len(df) else 0.0
            rows.append({
                "season": season,
                "rows": len(df),
                "route_coverage": route,
                "coverage_shell_coverage": shell,
                "man_zone_coverage": manzone,
                "route_and_shell_same_play": both,
                **meta,
                "status": "OK",
            })
        except Exception as exc:
            rows.append({
                "season": season,
                "rows": 0,
                "route_coverage": 0.0,
                "coverage_shell_coverage": 0.0,
                "man_zone_coverage": 0.0,
                "route_and_shell_same_play": 0.0,
                "url": url,
                "bytes": 0,
                "sha256": "",
                "status": f"ERROR:{type(exc).__name__}:{exc}",
            })
    audit = pd.DataFrame(rows)
    audit.to_csv(out / "m80_route_shell_source_audit.csv", index=False)

    # Historical feasibility is descriptive only. Deployment remains the binding
    # gate because 2023+ nflverse participation arrives after the postseason.
    hist = bool(
        len(audit) == 2
        and (audit.rows > 0).all()
        and (audit.route_coverage > 0.20).all()
        and (audit.coverage_shell_coverage > 0.20).all()
        and (audit.route_and_shell_same_play > 0.10).all()
    )
    decision = pd.DataFrame([{
        "candidate": "ROUTE_X_COVERAGE_SHELL",
        "historical_science_feasible": hist,
        "coverage_shell_itself_is_new": False,
        "route_interaction_is_new": True,
        "in_season_2026_source_available": False,
        "predictive_model_fit": False,
        "advance_to_m81_development": False,
        "status": "HOLD_FOR_DEPLOYABLE_LIVE_SOURCE" if hist else "NO_GO_SOURCE",
        "notes": "M14 already recovered shells; corrected simultaneous-presence metric excludes blank route/shell strings. nflverse 2023+ participation is postseason-only, so do not build an undeployable predictive winner unless an in-season route source is found.",
    }])
    decision.to_csv(out / "m80_route_shell_decision.csv", index=False)
    print(audit.to_string(index=False))
    print(decision.to_string(index=False))


m80.job_route = job_route

if __name__ == "__main__":
    raise SystemExit(m80.main())
