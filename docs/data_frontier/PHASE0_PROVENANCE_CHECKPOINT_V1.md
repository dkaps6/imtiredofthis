# Phase-0 BDB 2024 provenance checkpoint V1

**Scope:** isolated data engineering only. No predictive experiment, model retuning, production integration, sportsbook logic, Issue #535 direction, or frozen WR/RB/QB/TE plan changes.

**Starting branch head verified:** `f6553c26195ea799199911c35acb253a4540e785` on `data-frontier-phase0-bdb-contact-v1`.

## What changed

Added `scripts/data_frontier/bdb_2024_artifact_provenance.py` and wired it into the one-command Phase-0 orchestrator **after structural integrity and before fidelity reporting**.

The provenance seal SHA-256 hashes the exact normalized tracking/plays/tackles tables plus contact features, benchmark dispositions, QA/source manifests, enriched contact features, structural-integrity report, and defender geometry when present. It also emits an aggregate `artifact_set_sha256` over the file-hash manifest.

The stage refuses to seal if required artifacts are missing or structural integrity did not pass. A separate verify mode detects byte-level mutation after sealing.

The successful pipeline status now records the provenance-manifest path and aggregate artifact-set SHA-256.

Added focused source tests for: successful seal/verification followed by mutation detection; refusal after failed structural integrity; and refusal when a normalized source artifact is missing.

## Guardrails

The frozen contact detector and its threshold/rule were not modified. No fidelity threshold was tuned. No real BDB corpus result is claimed. Authenticated Kaggle competition access remains outside this implementation.

## Next checkpoint

Bind the fidelity report itself to the upstream provenance seal by recording the artifact-set SHA-256 inside the fidelity output and add a post-fidelity finalization step that verifies upstream hashes did not change while fidelity was being computed. This closes the remaining provenance gap between the sealed inputs and the report artifact itself.
