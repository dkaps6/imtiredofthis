# Phase-0 adversarial provenance/readiness checkpoint V1

**Scope:** isolated data engineering only. No predictive experiment, detector retuning, production integration, sportsbook logic, Issue #535 direction, frozen WR/RB/QB/TE plan change, or paid production run.

**Starting branch head verified:** `7105ac9fdf519e8c430fc84e59d3b0cdaa64f9e6` on `data-frontier-phase0-bdb-contact-v1`.

## Review finding

The existing orchestration test's nominal-success expectation had become stale after provenance sealing and post-fidelity verification were added. It still expected only four stages and did not synthesize the provenance/fidelity artifacts now read by the orchestrator. That test would no longer represent the actual six-stage fail-closed contract.

## Changes

Updated `tests/test_bdb_2024_phase0_orchestration.py` to assert the current ordered contract:

1. source normalization/QA and frozen contact derivation,
2. geometry enrichment,
3. structural integrity,
4. provenance seal,
5. fidelity,
6. post-fidelity provenance verification.

The tests now explicitly assert that integrity failure prevents both provenance and fidelity, that the nominal path seals before fidelity and verifies afterward, and that a fidelity/provenance SHA mismatch fails closed.

Added `docs/data_frontier/phase0_readiness_manifest_v1.json`, a machine-readable separation between implemented engineering gates and gates that still require execution in a verified runtime or lawful access to the real BDB 2024 corpus.

## Lineage

- Verified start: `7105ac9fdf519e8c430fc84e59d3b0cdaa64f9e6`
- Orchestration-test hardening commit: `e6d2ecd7edf74494f428ceff9323c82c21a711f9`
- Readiness-manifest commit: `255dceda44f53e558b5007d598a0f71ca7a6892f`

## Validation boundary

No executed pytest/full-corpus result is claimed from this connector-only runtime. No real BDB data were downloaded and no competition terms were accepted on the user's behalf.

## Next checkpoint

Run the focused Phase-0 tests in a verified checkout/runtime. If they pass, the remaining material Phase-0 blocker is lawful real-corpus execution: schema contract, structural integrity, provenance sealing/reverification, and fidelity reporting, all without tuning the frozen detector.
