# RB R21 Week-1 Prospective Forecast Lock V1 — Result

Date: 2026-09-08/09 UTC
Branch: `research-cross-position-catastrophic-casebook-v1`
Frozen plan: `docs/migrations/RB_R21_2026_PROSPECTIVE_TAIL_FORECAST_LOCK_AND_GRADE_V1_PLAN.md`
Plan commit: `cdd503919e56860c6f17df12e01f4f34072d888c`
Implementation commit/workflow head: `eefa0a4fc0b272de03db804acc4e77d7cbe39b92`
Workflow: `.github/workflows/research-rb-r21-2026-prospective-tail-forecast-lock-v1.yml`
Run: `34294227588`
Job: `102287208335`
Artifact: `10082525892`
Artifact name: `rb-r21-week1-prospective-forecast-lock-v1`
Artifact digest: `sha256:302df98f83c461d8abd74da9985dacc80c9477596b8bba85cc4d4a27c4fbc6f3`
Disposition: `RB_R21_WEEK1_PROSPECTIVE_FORECAST_LOCK_PASS_SHADOW_ONLY`
PASS: true

## Prospective timing proof

Frozen first-kickoff cutoff: `2026-09-10T00:20:00Z`.

R21 run started: `2026-09-09T00:15:29Z`.

Therefore the Week-1 CONTROL/SHADOW forecast lock was created before any 2026 regular-season outcome.

## What was locked

- 94 RB player rows.
- 10,000 receiving-yard draws per player for CONTROL.
- 10,000 receiving-yard draws per player for SHADOW.
- Exact sorted player/event ledger.
- Player-level quantiles/probabilities and R19 risk/state fields.
- Raw float64 matrix hashes.

CONTROL matrix SHA256 (raw little-endian float64 bytes):
`0149d2bc2fcd4cd9f401b1a46c1c27ecc9489c3ce00c40ec8f33880bb8b5181e`

SHADOW matrix SHA256 (raw little-endian float64 bytes):
`6be1c67952f465ffb6d01bf3bbdbbdb0ea3e5758684e10c2f075715f58b582d9`

Forecast ledger SHA256:
`b6f528217f1f14057730f52a64174f59497354256a5037ccbf4a43c8bd4a980a`

Compressed draw archive SHA256:
`8a255f76ac5a327833c220bd134d31d73ad06b566672dd38584fb0c7220918a6`

## Parent R20 integrity

R20 parent run: `34291433027`
R20 parent artifact: `10081502774`
R20 parent digest: `sha256:853c0d3aea971c058ae6cc3b80c99ae2a5a0f681fae642835bbfa544da8283ca`
R20 parent head: `587bf2a89ca16f11361016df3915361390289a7e`

R21 re-executed the unchanged R20 evaluator only to capture the exact arrays. The re-execution result hash exactly matched the immutable parent result hash:
`955a647375f5c27920e24790c65b5605a76ecf99da8b5ebe1cc756eccc2c11d5`.

Parent player-level summary parity max absolute delta: `7.105427357601002e-15`.

Maximum CONTROL-vs-SHADOW mean delta: `3.552713678800501e-15` yards.

## Frozen Phase-A gate result

Every frozen lock gate passed:

- pre-kickoff lock: PASS
- R20 parent exact: PASS
- unchanged R20 re-execution: PASS
- R20 parent draw-summary parity: PASS
- exact 94 x 10,000 RB shape: PASS
- unique player keys: PASS
- finite/nonnegative draws: PASS
- mean parity: PASS
- draw hashes recorded: PASS
- sportsbook zero upstream: PASS
- 2026 outcome reads zero: PASS
- production parameter changes zero: PASS

## Governance

This result seals Week-1 prospective forecasts only. It does **not** activate the RB receiving-yard tail adapter in production.

The frozen R21 plan controls later grading. Week 1 is an early prospective checkpoint. A production-promotion decision is not eligible until the shadow ledger contains at least 4 completed 2026 regular-season weeks, 250 eligible locked RB player-games, 15 observed 30+ underprojection events, and 5 observed 50+ underprojection events, followed by a separate explicit governed promotion ledger/commit.

R21 continues the exact lineage:

`R8 identity -> R9 shrinkage -> R10/R11 weekly state -> R12 state-gated identity -> R13-R15B efficiency rejection -> R16 predictable upside tail -> R17 mean-preserving distribution -> R18 canonical MC adapter -> R19 deployable strict-prior scorer -> R20 real 2026 full-slate shadow integration -> R21 pre-outcome prospective forecast lock`.

Important scope note: R21 grades the RB **receiving-yard distribution**. It does not separately promote or claim to solve RB receptions/target entitlement.
