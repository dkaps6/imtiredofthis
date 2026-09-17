# Phase-0 BDB 2024 Orchestration Checkpoint V1

**Scope:** isolated data engineering only. No predictive experiment, model retuning, production science, sportsbook logic, paid production run, or Issue #535 direction.

## Starting checkpoint

Verified branch `data-frontier-phase0-bdb-contact-v1` at `c4ec6891b85b4acd0272904503c9599194677453` before this run.

## Work completed

Added `scripts/data_frontier/run_bdb_2024_phase0.py`, a single fail-closed entry point with enforced ordering:

1. source loading + normalization + source-window QA + frozen V1 contact derivation;
2. deterministic geometry enrichment;
3. structural artifact-integrity gate;
4. fidelity report only after the integrity gate exits successfully.

The orchestrator writes `phase0_pipeline_status.json`. A failed stage stops later stages and returns nonzero. In particular, fidelity reporting cannot run after a failed structural-integrity gate.

## QA finding fixed during orchestration review

The integrity gate declared defender uniqueness on `(gameId, playId, defenderNflId)`, while the committed enrichment artifact actually persists the identifier as `defenderId`. That schema mismatch would have made every non-empty defender artifact fail the required-key gate even when structurally valid.

The integrity gate is now bound to the persisted enrichment schema: `(gameId, playId, defenderId)`. This is a structural QA correction only; it does not alter contact detection, geometry definitions, thresholds, or source labels.

## Frozen detector boundary

The orchestrator consumes the existing frozen contact detector through `bdb_2024_artifact_qa.py`. No contact threshold or detector definition was changed. Pipeline and artifacts continue to emit `contact_detector_changed=false`.

## Access boundary

No real BDB competition corpus was downloaded or executed in this checkpoint. Authenticated competition access/rule acceptance remains a user-controlled boundary. No real-corpus fidelity numbers are claimed.

## Next checkpoint

Add orchestration-focused tests that prove fail-closed stage ordering, especially that a failed integrity stage prevents fidelity output. Then add artifact provenance/hashing so a fidelity report can be tied to exact normalized inputs and upstream artifact bytes before any real-corpus run.
