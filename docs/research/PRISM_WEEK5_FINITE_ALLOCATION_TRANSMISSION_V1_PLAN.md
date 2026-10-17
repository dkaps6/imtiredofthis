# PRISM Week-5 Finite Allocation Transmission V1 — Frozen Diagnostic Plan

Date: 2026-10-17

Status: **FROZEN DIAGNOSTIC / DETERMINISTIC TRANSMISSION AUDIT / NO PREDICTIVE CANDIDATE**

Model:
**PRISM — Player Role & Individual Situation Model**

Version:
`PRISM_WEEK5_FINITE_ALLOCATION_TRANSMISSION_V1_AUDIT`

## 1. Governing player-first question

The Week-4/5 all-row replay identified a second high-priority architecture question:

> Where does strong upstream named-player role knowledge get attenuated before final market opportunity?

The canonical example is Brock Bowers Week 5:
- upstream `rules_tgt_share` approximately matched his realized target share;
- the final target probability was materially lower;
- therefore the forensic question is not automatically “PRISM did not know Bowers’ role.”

This audit isolates the deterministic finite-opportunity transmission layer only.

## 2. Frozen hypothesis class

This is **not a new projection candidate**.

It asks whether independent pregame player-share estimates collectively overbook a finite team opportunity pool and are then compressed by the canonical conservation mechanics before simulation.

Canonical deterministic mechanics to reproduce exactly:

Receiving:
1. read `rules_tgt_share`;
2. apply the already-promoted WR hierarchy transform while conserving total WR target mass;
3. clip individual probabilities to [0, 0.95];
4. when modeled player target mass exceeds 0.95, multiply all modeled player shares by `0.95 / team_mass`;
5. assign residual probability to the unmodeled bucket.

Rushing:
1. read `rules_rush_share`;
2. retain the canonical top five team rushing shares;
3. clip individual probabilities to [0, 0.95];
4. when retained modeled rush mass exceeds 0.95, multiply all retained shares by `0.95 / retained_mass`;
5. assign residual probability to the unmodeled bucket.

The audit must match `scripts/simulation_v2.py` semantics. It may not invent a different normalization rule.

## 3. Frozen authorities

Week-5 pregame player-state trace:
- run `37717016519`;
- artifact `11524296325`;
- digest `sha256:e043133d4b9a48ce472dde8810d4d77d636f4ce2fbcee14708b23fe0ea35f6a0`.

Week-5 exact frozen player-failure casebook:
- run `38397361891`;
- artifact `11812403368`;
- digest `sha256:4cdef29912f35368473d57e15b38efdc3dd0f75e897bbbdc3b97fe1ff8f67670`.

The original Week-5 Full Slate parent artifact `11585973062` is no longer retrievable. Therefore:
- do not reconstruct unsupported all-row final component states;
- use the preserved player-state trace for deterministic team-room mass;
- use the frozen top-40 casebook only where exact final target/rush probability is preserved.

## 4. Required outputs

Team-room audit:
- raw modeled target mass;
- post-WR-hierarchy target mass;
- target global scale factor;
- target cap applied;
- all-player raw rush mass;
- top-five retained rush mass;
- rush global scale factor;
- rush cap applied.

Named-player transmission:
- upstream rules share;
- post-WR-hierarchy share where relevant;
- deterministic final probability implied by the finite-pool normalization;
- scale factor applied.

Exact casebook validation:
- exact upstream rules share;
- exact preserved final target/rush probability;
- ratio final/upstream;
- clean non-WR target-cap cases;
- WR hierarchy + cap composite cases;
- rush top-five truncation / cap cases.

## 5. Primary audit questions

1. How often does the target pool exceed 0.95 before final allocation?
2. How often does the retained top-five rushing pool exceed 0.95?
3. What is the distribution of deterministic compression factors?
4. In clean non-WR target cases, does the exact frozen final probability equal the global-cap path implied by the preserved team room?
5. In retained rush cases, does exact frozen final probability agree with the top-five + global-cap path?
6. Does the Brock Bowers case reproduce exactly as a clean non-WR global-cap attenuation example?

## 6. Interpretation boundary

If overbooking/compression is confirmed, the allowed conclusion is:

> PRISM’s independently constructed player shares are not always jointly coherent with the finite team opportunity pool before the conservation safeguard. The safeguard can therefore attenuate an upstream named-player role read.

That does **not** imply:
- remove the 0.95 cap;
- allow team opportunity probabilities to sum above 1;
- boost every high-role player;
- replace entitlements with raw recent share;
- reopen RB Vacancy;
- alter WR hierarchy;
- alter qualified Stage-2 science;
- fit a correction from Week-5 outcomes.

Finite opportunity conservation is a required football constraint. Any future repair must improve the **joint entitlement/allocation state before normalization**, not disable conservation.

## 7. Protected science

Remain unchanged:
- Stage-1 active role QUALIFIED ALL;
- Stage-2 conditional allocation QUALIFIED for RB rushing / RB receiving / TE receiving;
- QB exact Stage-2 CLOSED;
- WR Stage-2 source-blocked/not scored;
- raw recent-share replacement CLOSED;
- RB carry/snap protected;
- WR/TE trajectory protected;
- Room Competition State SOURCE_BLOCKED;
- Week-6 locks immutable/ungraded;
- sportsbook downstream only.

## 8. No predictive transport claim from expired source

A desirable follow-up would be to transport the already-qualified Stage-2 allocations through this exact Week-5 deterministic transmission seam.

However, that requires the exact pregame Stage-2 q/availability authority at the relevant historical target game.

If the exact artifact/source is unavailable, fail closed. Do not approximate it from Week-5 outcomes or substitute a later reconstruction.

## 9. Integrity at freeze

- new projection candidate: **0**;
- fitted coefficients: **0**;
- threshold searches: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- Week-6 outcomes: **0**;
- Week-6 lock changes: **0**;
- production changes: **false**.

Disposition at freeze:
`PRISM_WEEK5_FINITE_ALLOCATION_TRANSMISSION_V1_AUDIT_FROZEN`.
