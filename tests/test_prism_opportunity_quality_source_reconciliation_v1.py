import pandas as pd

from scripts.research.reconcile_prism_opportunity_quality_source_v1 import reconcile


def _parent():
    rows=[]
    for season in (2024,2025):
        for week in range(5,19):
            for team in ("AAA","BBB"):
                rows.append({
                    "season":season,"week":week,"team":team,
                    "team_rushes":20,"team_inside10_rushes":2,
                    "team_targets":30,"team_redzone_targets":4,
                })
    return pd.DataFrame(rows)


def _raw():
    p=_parent()
    return p.rename(columns={
        "team_rushes":"team_rush_attempts",
        "team_inside10_rushes":"inside10_rush",
        "team_redzone_targets":"red_zone_targets",
    })


def test_reconciliation_requires_exact_count_parity():
    metrics,diff=reconcile(_parent(),_raw())
    # Synthetic key population is intentionally smaller than the real frozen
    # 832-row population, so field parity passes while exact key gate does not.
    assert metrics["exact_field_parity"] is True
    assert metrics["mismatch_rows"] == 0
    assert diff.empty


def test_reconciliation_keeps_single_count_drift_visible():
    raw=_raw()
    raw.loc[0,"inside10_rush"] += 1
    metrics,diff=reconcile(_parent(),raw)
    assert metrics["exact_field_parity"] is False
    assert metrics["mismatch_rows"] == 1
    assert len(diff) == 1
    assert metrics["field_metrics"]["team_inside10_rushes"]["mismatch_rows"] == 1
