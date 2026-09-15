# WR Phase 4B Identity Temporal Ablation V3 — Result

## Purpose
Resolve Claude's Issue #535 question about whether Phase 4B must require strictly-prior roster evidence for alias->GSIS identity lookup.

This is a **source-only identity audit**. It does not load receiving-yard projection/outcome fields and does not run any Phase 4B attribution outcome.

## Canonical lineage
- Branch: `research-wr-phase4-authority-exact-opportunity-attribution-v1`
- Script: `scripts/research/audit_wr_phase4b_identity_temporal_ablation_v3.py`
- Script commit: `897abb6950ee897ab4723c8c82dee247762e6449`
- Workflow: `.github/workflows/research-wr-phase4b-identity-temporal-ablation-v3.yml`
- Workflow head: `76fa99b4b85792ed2297a5222e15c46c87e27401`
- Run: `34971134534`
- Job: `104387443638`
- Artifact: `10397407493`
- Artifact name: `wr-phase4b-identity-temporal-ablation-v3`
- Artifact digest: `sha256:ea175f8b86cf4faba88cfd22559a0826e61480d3ecd84d688048e6815f806b44`

Exact WR-R15 authority remained pinned to:
- run `34238301577`
- artifact `10061328722`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

## Compared identity rules
### Existing V2 strict-prior identity rule
Roster alias evidence must precede `(season, week)`.

### V3 static-identity rule
For **alias->GSIS identity metadata only**, allow all audited weekly roster alias evidence from 2022-2024. Keep the deterministic hierarchy unchanged:
1. current-team exact full alias
2. current-team suffix-insensitive alias
3. globally unique exact full alias
4. globally unique suffix-insensitive alias

Ambiguity still fails closed. No fuzzy matching.

The PBP target-count rule is unchanged:
`REG week1-18 AND pass_attempt==1 AND two_point_attempt!=1 AND no_play!=1 AND receiver_player_id nonnull`.

No predictive feature temporal rule is relaxed by this audit.

## Result
### Layer 4 WR2+ domain
Strict-prior:
- rows: 5,321
- resolved: 5,244 = 98.5529%
- unresolved: 75
- ambiguous: 2

Static identity:
- rows: 5,321
- resolved: 5,320 = 99.9812%
- unresolved: 0
- ambiguous: 1
- recovered from prior unresolved/ambiguous set: 76 rows
- PBP team-game missing: 0

### Layers 2/3 canonical domain
Strict-prior:
- rows: 6,012
- resolved: 5,933 = 98.6860%
- unresolved: 77
- ambiguous: 2

Static identity:
- rows: 6,012
- resolved: 6,011 = 99.9834%
- unresolved: 0
- ambiguous: 1
- recovered from prior unresolved/ambiguous set: 78 rows
- PBP team-game missing: 0

### Exact target parity against frozen R15 authority
Strict-prior identity:
- resolved R15 rows: 4,154 / 4,193
- exact target parity: 4,154 / 4,154
- failures: 0
- max absolute delta: 0.0

Static identity:
- resolved R15 rows: **4,193 / 4,193**
- exact target parity: **4,193 / 4,193**
- failures: **0**
- max absolute delta: **0.0**

This resolves all 39 previously unresolved R15 authority rows with no target-count discrepancy.

## Remaining ambiguity
One Layer-4 / Layer-2/3 row remains intentionally fail-closed:
- 2024 Week 15 NYJ — `Brandon Smith`
- status: `AMBIGUOUS_IDENTITY`
- reason: multiple same-team exact-name GSIS candidates under the deterministic roster alias rule.

This row is not part of the unresolved R15 candidate set; the R15 authority candidate cohort is fully resolved and exact-parity validated.

## Boundaries
- receiving-yard fields loaded: false
- Phase 4B attribution outcomes run: false
- predictive feature temporal rules changed: false
- sportsbook inputs: 0
- zero imputation: false
- fuzzy matching: false
- challenger authorized: false
- production change: false

## Scientific/source interpretation
Claude's concern is supported. Strict-prior roster evidence was unnecessarily excluding a systematic identity population (debut / first-observed alias rows) in a retrospective attribution diagnostic. Alias->GSIS is static identity metadata, not a predictive football feature. Allowing complete audited roster evidence for that identity lookup materially improves coverage while preserving exact target parity across all 4,193 R15 authority rows.

**Recommended Phase 4B source rule:** use the V3 static-identity alias->GSIS rule for historical attribution, while keeping all actual target counts PBP-only and preserving strict temporal rules for any predictive feature/model input.

No Phase 4B yard-attribution outcome is authorized by this result alone; final implementation review remains required before exposure.
