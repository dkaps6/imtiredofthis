"""Versioned Phase-0 enrichment for already-detected BDB 2024 contacts.

This module does not change contact detection. It attaches deterministic geometry
and source-label reconciliation to SCOREABLE V1 contact rows.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scripts.data_frontier.bdb_2024_contact_geometry import (
        closing_speed_proxy,
        pursuit_angle_error_deg,
        sideline_distance_yards,
        source_overlap_breakdown,
    )
except ModuleNotFoundError:
    from bdb_2024_contact_geometry import (
        closing_speed_proxy,
        pursuit_angle_error_deg,
        sideline_distance_yards,
        source_overlap_breakdown,
    )

ENRICHMENT_VERSION = "BDB_2024_CONTACT_ENRICHMENT_V1"


def _ids(value: object) -> set[int]:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return set()
    out: set[int] = set()
    for token in str(value).split("|"):
        token = token.strip()
        if token:
            out.add(int(float(token)))
    return out


def enrich_contact_features(features: pd.DataFrame, tracking: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return play-level enriched features and defender-level contact geometry."""
    play_rows: list[dict] = []
    defender_rows: list[dict] = []
    groups = {(int(g), int(p)): x for (g, p), x in tracking.groupby(["gameId", "playId"], sort=False)}

    for _, f in features.iterrows():
        gid, pid = int(f["gameId"]), int(f["playId"])
        frame = int(f["firstContactFrameId"])
        carrier_id = int(f["ballCarrierId"])
        contact_ids = _ids(f.get("firstContactDefenderIds"))
        primary_ids = _ids(f.get("sourcePrimaryTacklerIds"))
        assist_ids = _ids(f.get("sourceAssistIds"))
        missed_ids = _ids(f.get("sourceMissedTacklerIds"))
        pt = groups.get((gid, pid), pd.DataFrame())
        at_frame = pt.loc[pt["frameId"].eq(frame)] if len(pt) else pd.DataFrame()
        carrier = at_frame.loc[at_frame["nflId"].eq(carrier_id)] if len(at_frame) else pd.DataFrame()

        closing_values: list[float] = []
        pursuit_values: list[float] = []
        if not carrier.empty:
            c = carrier.iloc[0].to_dict()
            for did in sorted(contact_ids):
                drow = at_frame.loc[at_frame["nflId"].eq(did)]
                if drow.empty:
                    continue
                d = drow.iloc[0].to_dict()
                closing = closing_speed_proxy(c, d)
                pursuit = pursuit_angle_error_deg(c, d)
                if closing is not None:
                    closing_values.append(float(closing))
                if pursuit is not None:
                    pursuit_values.append(float(pursuit))
                defender_rows.append({
                    "gameId": gid,
                    "playId": pid,
                    "firstContactFrameId": frame,
                    "ballCarrierId": carrier_id,
                    "defenderId": did,
                    "closingSpeedProxyYdsPerSec": closing,
                    "pursuitAngleErrorDeg": pursuit,
                    "isSourcePrimaryTackler": int(did in primary_ids),
                    "isSourceAssist": int(did in assist_ids),
                    "isSourceMissedTackler": int(did in missed_ids),
                    "enrichmentVersion": ENRICHMENT_VERSION,
                })
            sideline = sideline_distance_yards(c.get("y"))
        else:
            sideline = None

        row = f.to_dict()
        row.update(source_overlap_breakdown(contact_ids, primary_ids, assist_ids, missed_ids))
        row.update({
            "firstContactCarrierSidelineDistanceYards": sideline,
            "firstContactMaxClosingSpeedProxyYdsPerSec": max(closing_values) if closing_values else np.nan,
            "firstContactMeanClosingSpeedProxyYdsPerSec": float(np.mean(closing_values)) if closing_values else np.nan,
            "firstContactMinPursuitAngleErrorDeg": min(pursuit_values) if pursuit_values else np.nan,
            "firstContactMeanPursuitAngleErrorDeg": float(np.mean(pursuit_values)) if pursuit_values else np.nan,
            "contactGeometryDefendersResolved": len(closing_values),
            "enrichmentVersion": ENRICHMENT_VERSION,
            "contactDetectorChanged": False,
        })
        play_rows.append(row)

    return pd.DataFrame(play_rows), pd.DataFrame(defender_rows)


def main() -> int:
    ap = argparse.ArgumentParser(description="Enrich frozen BDB 2024 V1 contact artifacts with deterministic geometry.")
    ap.add_argument("--artifact-dir", type=Path, required=True)
    args = ap.parse_args()
    features = pd.read_csv(args.artifact_dir / "rb_contact_features.csv")
    tracking = pd.read_csv(args.artifact_dir / "normalized" / "tracking.csv")
    enriched, defenders = enrich_contact_features(features, tracking)
    enriched.to_csv(args.artifact_dir / "rb_contact_features_enriched_v1.csv", index=False)
    defenders.to_csv(args.artifact_dir / "rb_contact_defender_geometry_v1.csv", index=False)
    print({"enrichment_version": ENRICHMENT_VERSION, "plays": len(enriched), "contact_defender_rows": len(defenders), "contact_detector_changed": False})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
