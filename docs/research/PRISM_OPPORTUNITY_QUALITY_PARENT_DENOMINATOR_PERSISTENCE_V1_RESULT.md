# PRISM Opportunity Quality — Parent Denominator Persistence V1 Result

Date: 2026-10-16

Status: **PARITY PASS / NUMERIC CANONICAL PREGAME DENOMINATOR PERSISTED / OUTPUT-LINEAGE ONLY / NO PREDICTIVE SCORING**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1`

Frozen addendum:
`docs/research/PRISM_OPPORTUNITY_QUALITY_PARENT_DENOMINATOR_PERSISTENCE_V1_ADDENDUM.md`

Freeze commit:
`3a980c9ba821c5e0a3b41231314db5c21bb0adef`

## Canonical execution

Run:
- `38991915503` — **SUCCESS**
- job `117275946409`
- exact head `02e96ad9faabee62c1ecb31303b48746e4945a8c`
- artifact `12085194208`
- digest `sha256:c2be00311f9cf2149ebef36d419851214add08cae83bbc3b947990ce60a5e569`

Prior canonical Stage-2 source authority:
- run `38955368643`
- artifact `12070806828`
- digest `sha256:e91a108495d7a2b6eb289e0618ab6af09e3d6cdf6895e534742a07340e4a2a83`

Disposition:

`PRISM_PARENT_DENOMINATOR_PERSISTENCE_PARITY_PASS`

## Exact parity

Rows:
**14,284**

Required parity:
- key sets exact: **true**
- denominator finite flags exact: **true**
- ready cells exact: **true**
- scientific source disposition exact: **true**
- maximum absolute canonical entitlement-q drift: **2.0816681711721685e-16**

The refreshed source result remains:

`PLAYER_ACTIVE_ROLE_CONDITIONAL_ALLOCATION_SOURCE_READY_PARTIAL`

Ready source cells remain:
- QB passing
- RB rushing
- RB receiving
- TE receiving

Blocked source cell remains:
- WR receiving in 2025 because independent WR-R15 OOS authority is unavailable/forbidden.

No scientific source gate moved.

## What was persisted

The existing numeric canonical pregame parent opportunity denominator is now retained in the player-level source lineage as:

`authority_denominator_mean`

Semantics remain unchanged:

RB rushing:
`rules_plays_est * (1 - rules_pass_rate)`

Receiving:
`rules_plays_est * rules_pass_rate`

This result does **not** fit a new denominator, share, or opportunity-quality coefficient.

## Why this matters for PRISM

Opportunity quality must sit on top of the same finite opportunity quantity already used by the player model.

PRISM can now test a quality/topology mechanism while preserving:
1. the canonical pregame opportunity denominator;
2. the canonical named-player entitlement;
3. the qualified active-role / conditional-allocation bridge;
4. finite team opportunity conservation.

This closes the source-lineage seam that was blocking the next opportunity-quality experiment.

## Integrity

- target-game outcomes used as predictors: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- 2026 outcomes: **0**
- predictive candidates scored: **0**
- new share models fit: **0**
- denominator formulas changed: **0**
- entitlement formulas changed: **0**
- production changed: **false**

## Exact next action

Freeze **one** opportunity-quality mechanism before scoring.

It must be materially different from the closed recent-player high-value-share persistence V1.

The next mechanism should isolate the **team creation of high-value opportunity conditional on the existing canonical opportunity quantity** before adding opponent/environment transmission.

Allowed source-ready quality pools:
- RB rushing: team inside-10 rushing opportunities;
- RB receiving: team red-zone targets;
- TE receiving: team red-zone targets.

The player-side allocation must continue to use current canonical entitlement / qualified Stage-2 logic. Do **not** replace it with recent player scoring-area share.

Use:
- 2024 development;
- 2025 independent confirmation.

Do not use:
- sportsbook information;
- OddsAPI;
- 2026 outcomes;
- TD outcomes to rescue the mechanism;
- after-result bands/windows/subgroups;
- opponent/environment transmission in the first quality mechanism.

Fail closed rather than weaken a frozen gate.
