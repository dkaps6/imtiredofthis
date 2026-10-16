# PLAYER RECEIVER CONVERSION STATE V1 — SOURCE AUDIT RESULT

Date: 2026-10-13

Status: **SOURCE READY — PREDICTIVE MECHANISM MAY BE FROZEN SEPARATELY**

Parent contract:
- `docs/research/PLAYER_RECEIVER_CONVERSION_STATE_V1_SOURCE_ANTIRETEST.md`

## Canonical authority

- workflow: `Research Player Receiver Conversion Source Audit V1`
- run: `38409898663` — **SUCCESS**
- exact head: `686a38ef1a1ecb03288661a83e93fe5fddd2323c`
- artifact: `11820921870`
- digest:
  `sha256:8967a609e4fe08de5affd936c01d7ee65e60ccf350a2fc4f48f83f945cc4a07a`
- strict repository audit: **PASS**

Overall disposition:

`PLAYER_RECEIVER_CONVERSION_SOURCE_READY`

The first run (`38409756672`) failed before source scoring because the workflow
installed pyarrow 26 against the repository's NumPy 1.26.4 environment. The
rerun pinned the established compatible `pyarrow==18.1.0` only. No source
threshold, football definition, feature, or gate changed.

## Frozen gates and results

All gates passed independently in 2023, 2024, and 2025.

| Season | FTN→PBP join | Receiver ID | Roster position | Drop coverage | Catchable coverage | WR/TE 4+ games | WR/TE 8+ games |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2023 | 95.73% | 100% | 100% | 100% | 100% | 267 | 219 |
| 2024 | 95.58% | 100% | 100% | 100% | 100% | 264 | 201 |
| 2025 | 95.51% | 100% | 100% | 100% | 100% | 273 | 204 |

Frozen thresholds:
- FTN→PBP join >=95%;
- stable receiver ID >=95%;
- roster position mapping >=95%;
- charted field coverage >=80%;
- >=100 WR/TE receivers with 4+ attributed games;
- >=75 WR/TE receivers with 8+ attributed games.

Regular-season Weeks 1-18 were complete in both FTN and PBP for each season.

## What this result means

A distinct receiver-level historical information source exists with enough
three-season coverage to test:

- receiver-attributed drops;
- receiver-target catchability;
- stable receiver identity;
- WR/TE position identity.

This is materially narrower than M81's failed QB/offense-level FTN aggregation.
M81 remains closed and is not rescued.

This result does **not** establish predictive value.

## What is now authorized

One separately preregistered historical mechanism experiment may test whether
strictly-prior **receiver-level** charted conversion state contains incremental
pregame information beyond the canonical catch-rate mean.

The next experiment must:
- be WR/TE only;
- keep M34/M70/M71/M81/R7/WR-R3/target-depth closures intact;
- use receiver-level charting rather than QB/offense aggregate retesting;
- freeze train/confirmation seasons, exact features, canonical target, support,
  clustering/inference, and pass/fail rules before scoring;
- use no sportsbook/OddsAPI information;
- make no production change;
- not treat source readiness as a model PASS.

## Integrity

- sportsbook inputs: 0
- OddsAPI calls: 0
- predictive models fit: 0
- candidate mean changes: 0
- production changed: false
