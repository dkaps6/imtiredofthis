"""Phase-0 fidelity reporting for BDB 2024 contact artifacts.

Data-quality only: no predictive metrics, threshold tuning, model comparison, or
sportsbook logic. The frozen contact detector is treated as immutable input.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPORT_VERSION = "BDB_2024_CONTACT_FIDELITY_REPORT_V1"


def _rate(series: pd.Series) -> float | None:
    if len(series) == 0:
        return None
    return float(pd.to_numeric(series, errors="coerce").fillna(0).mean())


def _numeric_summary(series: pd.Series) -> dict:
    x = pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if x.empty:
        return {"n": 0, "missing": int(len(series)), "min": None, "p05": None, "median": None, "p95": None, "max": None}
    return {
        "n": int(len(x)),
        "missing": int(len(series) - len(x)),
        "min": float(x.min()),
        "p05": float(x.quantile(0.05)),
        "median": float(x.median()),
        "p95": float(x.quantile(0.95)),
        "max": float(x.max()),
    }


def build_fidelity_report(
    enriched: pd.DataFrame,
    defenders: pd.DataFrame,
    dispositions: pd.DataFrame,
    qa_summary: dict,
) -> dict:
    """Summarize coverage, abstention, source overlap, and geometry sanity."""
    attempted = int(len(dispositions))
    if "benchmark_disposition" in dispositions.columns:
        disp_counts = dispositions["benchmark_disposition"].fillna("<missing>").astype(str).value_counts().sort_index()
    else:
        disp_counts = pd.Series(dtype=int)
    scoreable = int(disp_counts.get("SCOREABLE", len(enriched)))

    reason_col = next((c for c in ("benchmark_reason", "reason", "disposition_reason") if c in dispositions.columns), None)
    reason_counts = (
        dispositions.loc[dispositions.get("benchmark_disposition", pd.Series(index=dispositions.index, dtype=object)).ne("SCOREABLE"), reason_col]
        .fillna("<missing>").astype(str).value_counts().sort_index()
        if reason_col else pd.Series(dtype=int)
    )

    geometry_resolved = pd.to_numeric(enriched.get("contactGeometryDefendersResolved", pd.Series(dtype=float)), errors="coerce").fillna(0)
    overlap_cols = [
        "first_contact_primary_tackle_overlap",
        "first_contact_assist_overlap",
        "first_contact_missed_tackle_overlap",
        "first_contact_any_source_label_overlap",
    ]
    overlaps = {c: _rate(enriched[c]) if c in enriched.columns else None for c in overlap_cols}

    numeric_cols = [
        "firstContactCarrierSidelineDistanceYards",
        "firstContactMaxClosingSpeedProxyYdsPerSec",
        "firstContactMeanClosingSpeedProxyYdsPerSec",
        "firstContactMinPursuitAngleErrorDeg",
        "firstContactMeanPursuitAngleErrorDeg",
    ]
    distributions = {c: _numeric_summary(enriched[c]) for c in numeric_cols if c in enriched.columns}
    if "closingSpeedProxyYdsPerSec" in defenders.columns:
        distributions["defender_closingSpeedProxyYdsPerSec"] = _numeric_summary(defenders["closingSpeedProxyYdsPerSec"])
    if "pursuitAngleErrorDeg" in defenders.columns:
        distributions["defender_pursuitAngleErrorDeg"] = _numeric_summary(defenders["pursuitAngleErrorDeg"])

    source_windows = qa_summary.get("source_window_diagnostics", {})
    return {
        "report_version": REPORT_VERSION,
        "contact_detector_changed": False,
        "scope": "data_fidelity_only",
        "attempted_plays": attempted,
        "scoreable_plays": scoreable,
        "scoreable_rate": (scoreable / attempted) if attempted else None,
        "disposition_counts": {str(k): int(v) for k, v in disp_counts.items()},
        "abstention_reason_counts": {str(k): int(v) for k, v in reason_counts.items()},
        "geometry_resolution": {
            "plays_with_one_or_more_resolved_contact_defenders": int((geometry_resolved > 0).sum()),
            "rate_over_enriched_plays": float((geometry_resolved > 0).mean()) if len(geometry_resolved) else None,
            "resolved_defender_rows": int(len(defenders)),
        },
        "source_label_overlap_rates": overlaps,
        "geometry_distribution_sanity": distributions,
        "source_window_diagnostics": source_windows,
        "interpretation_guardrails": [
            "Overlap with tackle/assist/missed-tackle labels is reconciliation, not predictive accuracy.",
            "BDB 2024 tracking is event-window filtered; absent events do not prove absent football events.",
            "Geometry missingness is reported rather than imputed.",
            "No detector threshold or feature definition is tuned from this report.",
        ],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Build BDB 2024 Phase-0 contact fidelity report.")
    ap.add_argument("--artifact-dir", type=Path, required=True)
    args = ap.parse_args()
    enriched = pd.read_csv(args.artifact_dir / "rb_contact_features_enriched_v1.csv")
    defenders_path = args.artifact_dir / "rb_contact_defender_geometry_v1.csv"
    defenders = pd.read_csv(defenders_path) if defenders_path.exists() else pd.DataFrame()
    dispositions = pd.read_csv(args.artifact_dir / "benchmark_dispositions.csv")
    qa_summary = json.loads((args.artifact_dir / "qa_summary.json").read_text(encoding="utf-8"))
    report = build_fidelity_report(enriched, defenders, dispositions, qa_summary)
    out = args.artifact_dir / "contact_fidelity_report_v1.json"
    out.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"report_version": REPORT_VERSION, "output": str(out), "attempted_plays": report["attempted_plays"], "scoreable_plays": report["scoreable_plays"], "contact_detector_changed": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
