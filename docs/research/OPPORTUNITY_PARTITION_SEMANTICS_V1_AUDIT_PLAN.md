# Opportunity Partition Semantics V1 — Read-Only Audit Plan

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

This audit is prompted by a production-semantics inconsistency discovered while
reviewing the successful Hierarchical Receiver Reconciliation V1 diagnostic.

It is not a rescue of One-Pass V1 or of hierarchical reconciliation.

## Core football identity

The current team pass-opportunity state is a **dropback** state, not an official
pass-attempt state.

Historical context explicitly defines:

`dropbacks = official pass attempts + sacks + QB scrambles`

and stores:

`pass_attempts_per_dropback = official pass attempts / dropbacks`

Production QB pricing already corrects for this distinction before passing-yard
pricing.

## Current simulation seam under audit

Canonical `simulation_v2` currently does:

1. draw `pass_att = Binomial(plays, pass_rate)`;
2. downstream M89/M90 treats that count as dropbacks and converts QB passing
   opportunity to official attempts;
3. receiver targets are allocated directly from the **unconverted** `pass_att`;
4. `rush_att = plays - pass_att`.

Therefore the same dropback state is currently used as:
- QB dropbacks on the QB side;
- target opportunities on the receiver side;
- the complement used to construct team rushing opportunities.

If the documented semantics are correct, sacks and QB scrambles can remain in
the receiver target pool, while QB scrambles are absent from the complement
rush pool.

That is a football-partition question, not a model-fitting question.

## Phase A — code/provenance proof

Before any metric is interpreted, verify:

1. current `rules_pass_rate` / historical pass-rate authority is dropback rate;
2. `simulation_v2.pass_att` is the same state referred to by M89 as projected
   dropbacks;
3. receiver target allocation consumes that unconverted state;
4. QB pricing alone applies `pass_attempts_per_dropback`;
5. `rush_att = plays - pass_att` therefore represents non-dropback rushing
   plays, not all official rush attempts;
6. QB scramble metrics are available strictly pregame from completed prior
   weeks only;
7. zero sportsbook inputs enter any of the above.

If any premise is false, stop and document the actual semantics.

## Phase B — no-outcome current-slate quantification

Use the certified current 2026 Week-3 football universe only. Do not use Week-3
outcomes.

For each team, record:
- mean simulated plays;
- mean simulated dropbacks;
- strict-prior `pass_attempts_per_dropback`;
- implied official pass attempts:
  `dropbacks * pass_attempts_per_dropback`;
- implied non-attempt dropbacks:
  `dropbacks * (1 - pass_attempts_per_dropback)`;
- current receiver target-pool count;
- target-pool excess versus implied official attempts;
- current non-dropback rushing pool:
  `plays - dropbacks`.

For the primary QB, use only prior-week PBP:
- scramble rate per dropback;
- implied scramble count:
  `dropbacks * scramble_rate`.

Then report:
- implied sacks/non-scramble non-attempt dropbacks as the residual between total
  non-attempt dropbacks and implied scrambles;
- implied official rush opportunity if scrambles are restored:
  `current non-dropback rush pool + implied scrambles`.

No candidate projections are scored in this phase.

## Cross-market contradiction diagnostics

Quantify, team-by-team and pooled:

### Receiver side
- excess target-pool opportunities:
  current dropback pool minus implied official attempts;
- percentage inflation of target opportunities;
- correlation between target-pool inflation and the named-receiver-over-team
  incoherence found by Hierarchical Receiver Reconciliation V1;
- expected named target mass generated from non-attempt dropbacks under current
  entitlement shares.

### Rushing side
- implied missing scramble rush attempts;
- missing scramble attempts as a percentage of the current rush pool;
- QB share of current rushing allocation versus implied scramble volume;
- whether current team rush opportunity plus restored scrambles better matches
  the football partition identity.

## Frozen interpretation rule

This audit may establish a new candidate only if the semantic contradiction is
real and material.

It must not:
- tune a conversion factor;
- use target-game outcomes;
- use sportsbook data;
- change receiver share models;
- change RB/QB rush shares;
- carve out positions or players;
- alter M89/M90;
- alter C2;
- alter WR-R15 or TE-R5P.

The conversion authority is the already-promoted strict-prior
`pass_attempts_per_dropback`; it is not re-fit here.

## If the contradiction is confirmed

Freeze separate candidates before scoring:

### Candidate A — receiver opportunity semantics
Allocate receiver targets from official pass attempts rather than dropbacks,
using the existing promoted attempt-conversion authority.

All target shares, entitlement specialists, catch rates, YPT, residual-target
semantics and sportsbook independence remain unchanged.

### Candidate B — rushing partition semantics
Handled separately because scrambles must be restored to QB rushing without
double-counting QB designed-run/rush-share authority.

Do **not** combine A and B into one first experiment.

Candidate A should be tested first because its mechanical correction is more
direct and has fewer moving parts.

## Scientific discipline

Any historical test must use fold-safe / strictly prior attempt-conversion
authority.

2024-2025 are retrospective evidence, not prospective proof.

A historical pass still requires a separately frozen prospective 2026
confirmation before production promotion.
