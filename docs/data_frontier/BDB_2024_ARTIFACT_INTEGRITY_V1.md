# BDB 2024 Artifact Integrity V1

**Scope:** Phase-0 data engineering only. No predictive experiment, detector retuning, production science, sportsbook logic, paid run, or Issue #535 direction.

**Parent checkpoint:** `9d394648582f6809f90462612dfad2a9aa986259` (`Data frontier: add BDB contact fidelity report`).

## Purpose

The fidelity report is only meaningful if its input artifacts are structurally trustworthy. `scripts/data_frontier/bdb_2024_artifact_integrity.py` therefore adds a fail-closed gate before any real-corpus fidelity interpretation.

## Hard checks

The V1 gate checks:

- unique `(gameId, playId)` keys in enriched play artifacts and benchmark dispositions;
- exact key parity between `SCOREABLE` dispositions and enriched play rows;
- unique `(gameId, playId, defenderNflId)` defender geometry rows;
- defender-row referential integrity back to enriched plays;
- finite geometry values where values are present;
- pursuit-angle bounds `[0, 180]` degrees;
- carrier sideline-distance bounds `[0, 26.65]` yards;
- presence of the upstream feature version in `qa_summary.json`;
- explicit `contact_detector_changed == false` preservation;
- normalized tracking/plays/tackles manifest metadata presence.

A failure returns process exit code `2` and writes the complete failure list to `artifact_integrity_v1.json`. The gate must not silently drop, impute, or repair corrupt rows.

## QA coverage

`tests/test_bdb_2024_artifact_integrity.py` adds focused source-level cases for:

1. clean-artifact pass;
2. duplicate enriched keys and orphan defender rows;
3. scoreable/enriched parity failure plus invalid geometry ranges;
4. missing feature version and detector-mutation guardrail failure.

These tests are committed as source QA. This checkpoint does not claim a verified runtime pass unless a later environment actually executes them.

## Real-corpus boundary

No real BDB 2024 corpus result is claimed. Authenticated Kaggle competition access/rule acceptance remains outside this autonomous lane. The integrity gate is intentionally useful before those files are available: when lawful inputs arrive, corrupt artifacts will fail before the fidelity report is interpreted.

## Next checkpoint

Wire the integrity gate into a single Phase-0 benchmark orchestration entry point with explicit stage order:

`normalized/source QA -> frozen contact features -> geometry enrichment -> integrity gate -> fidelity report`

The orchestration layer must stop on integrity failure and record stage/version lineage. It must not introduce detector tuning or predictive evaluation.
