# TE-R5 Week 1 Full-Stack Confirmation V1 — Result

Date: 2026-09-08/09 UTC
Branch: `production-week1-player-prop-readiness-v1`
Frozen plan: `docs/production/TE_R5_WEEK1_FULL_STACK_CONFIRMATION_V1_PLAN.md`
Plan commit: `47559f0007fc5acf022dcd81f874c627d08e95ab`
Confirmation head: `a8cabb66ba4402f4e20355c300a2a88f385da5e5`
Workflow: `.github/workflows/te-r5-week1-full-stack-confirmation-v1.yml`
Run: `34297504346`
Job: `102297192845`
Artifact: `10083700709`
Artifact name: `te-r5-week1-full-stack-confirmation-v1`
Artifact digest: `sha256:42d0522e239983306060cb518a4e2f7f67c937b8cc42757bf78654b387683092`
Disposition: `TE_R5_WEEK1_FULL_STACK_CONFIRMATION_PASS_PROMOTION_ELIGIBLE`
PASS: true

## What was confirmed

The previously supported TE-R5 mechanism was reconstructed as a deployable 2026 Week-1 scorer and run against the current canonical Week-1 Full Slate architecture.

The mechanism remains exactly the frozen two-stage opportunity model:

1. TE-R3 estimates the finite team TE target pool.
2. TE-R5 allocates that pool across current TEs using strict-prior participation/history and baseline TE-room shares.

No TE catch-rate or YPT model was added. TE-R5 changes target opportunity only; receptions and receiving yards respond through the canonical joint simulation.

## Current Week-1 slate

- 469 total modeled players
- 32 teams
- 96 TE rows

## Historical/deployable parity

- R3 parent max absolute team-pool delta: `5.3290705182007514e-14`
- R5 parent max absolute player-target delta: `1.509903313490213e-14`
- R3 serialization roundtrip max delta: `0.0`
- R5 serialization roundtrip max delta: `0.0`

## Opportunity conservation

- maximum team TE-pool gap: `3.552713678800501e-15` expected targets
- maximum TE-room share gap from 1.0: `2.220446049250313e-16`

The canonical modeled-target residual-bucket contract required an exact inverse solve because `simulation_v2.py` caps modeled target mass at 0.95. The confirmation solved the TE raw mass so that the post-normalization expected TE pool exactly equals the frozen R3/R5 candidate pool without directly modifying WR/RB/FB target shares.

## Protected-model integrity

All final gates passed, including:

- immutable parent artifact metadata
- parent R4 strict-prior eligibility
- parent R5 scientific PASS
- historical R3 and R5 parity
- model serialization parity
- zero same/future historical leakage
- 32-team Week-1 current universe
- finite live features
- strict-prior participation only
- explicit new-team/rookie availability states
- exact candidate TE-pool conservation
- exact TE-room share conservation
- finite/nonnegative candidate shares
- non-TE raw target shares exact
- M38 WR direct hierarchy output exact
- canonical simulation deterministic
- non-TE rushing football input/path exact
- promoted QB/RB production paths present
- zero sportsbook inputs upstream
- zero 2026 current/future outcomes
- zero production parameter changes during confirmation
- TE-only target-entitlement change

Production-readiness audit before confirmation: `[READY] no findings`.
Production-readiness audit after confirmation: `[READY] no findings`.

### RNG audit note

An intermediate confirmation attempt compared non-TE rushing Monte Carlo arrays draw-for-draw. That failed even though no rushing input changed because canonical `simulation_v2.py` uses one shared NumPy generator; changing target multinomial probabilities shifts later RNG consumption.

The frozen gate was `non-TE rushing components exact`, so the final confirmation audited the actual rushing football input/path at exact equality: play/pass-volume inputs, rush shares, YPC inputs, volatility, and player-position path. Across all 469 rows / 373 non-TE rows there were zero mismatched audited rushing columns. The underlying frozen TE-R5 model, coefficients, caps, conservation equations, and scientific gates were not changed.

## Week-1 examples from the shadow confirmation

These are examples of the confirmed candidate's effect before any production promotion:

- Trey McBride: expected targets `4.708 -> 6.434`; receptions mean `3.458 -> 4.747`; receiving-yards mean `34.44 -> 47.34`.
- Kyle Pitts: expected targets `4.895 -> 6.226`; receiving-yards mean `37.82 -> 48.01`.
- Tyler Warren: expected targets `3.966 -> 5.352`; receiving-yards mean `28.96 -> 39.02`.
- Travis Kelce: expected targets `3.982 -> 4.967`; receiving-yards mean `30.63 -> 37.98`.

These are not sportsbook-informed projections. They are football-only candidate outputs from the frozen confirmation.

## Production boundary

This PASS makes TE-R5 **promotion-eligible for 2026 Week 1**. It is not yet production-active merely because this confirmation passed.

The next TE step is a separate explicit production commit that:

1. installs the confirmed serialized R3/R5 scorer and strict-prior live feature builder;
2. applies TE-R5 before canonical target multinomial allocation using the exact confirmed residual-bucket inverse;
3. exposes version/provenance audits;
4. runs the canonical no-odds Full Slate validation;
5. only then permits controlled live-odds pricing.
