# Player High-Resolution Role-State Source Audit — 2026-10-14

Status: **SOURCE / ANTI-RETEST AUDIT COMPLETE — DO NOT FAKE PLAYER-LEVEL STATE**

Parent scientific result:
`docs/research/PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY_V1_RESULT.md`

Question:

> After broad 11/12/OTHER player-conditioned target propensity failed, what
> genuinely higher-resolution named-player pregame football state can be tested
> without reverting to a position average, leaking target-game information, or
> inventing historical semantics?

## 1. Total routes run / route participation

Current disposition: **HISTORICAL SOURCE-PARITY BLOCKED**

Existing repo authority already established:
- historical PlayerForm only treats routes/routes_run as valid when a real total-routes field exists;
- nflverse weekly stats do not supply a clean historical total-routes-run field with current/live semantic parity;
- nflverse participation `route` is a route label for the primary/targeted receiver on a play;
- it is **not** total routes run and must not be counted as such.

Do not create a fake player route-participation history from target-route labels.

A live/current route-volume source has previously been identified, but historical weekly parity was not cleared. That supports prospective capture, not retrospective model fitting.

## 2. Player alignment: slot / wide / in-line / flex

Current disposition: **PARTIAL CURRENT SOURCE / HISTORICAL PARITY NOT CLEARED**

The production coverage source family has fields such as:
- `slot_pct`
- `wide_pct`
- alignment availability/source

Prior systems audits also found current SWR/slot alignment upstream.

However the repo's historical Stage-2 conclusion remains:
`HISTORICAL_SOURCE_UNAVAILABLE_PROSPECTIVE_ONLY`.

Therefore:
- current alignment is potentially useful for prospective player-state locks;
- it is not currently authorized as a historical walk-forward feature;
- missing historical alignment must not be reconstructed from position labels.

For TE, a validated historical in-line/slot/flex source is still absent.

## 3. nflverse participation formation/personnel

Current disposition: **AVAILABLE BUT BROAD PERSONNEL HYPOTHESIS NOW CLOSED**

Participation supplies on-field offensive player IDs/positions and formation/personnel context. It supported the clean Player Personnel-Conditioned Opportunity V1 run.

That exact broad construction failed:
`NO_ACTIONABLE_PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY`.

Do not retune 11/12/OTHER or create more broad package buckets from the same evidence after seeing the null.

## 4. FTN charting motion / tactical tags

Current disposition: **PLAY-LEVEL, NOT CURRENTLY PLAYER-ATTRIBUTED FOR THIS PURPOSE**

The repo has historical FTN charting coverage for fields including:
- `is_motion`
- `is_screen_pass`
- `is_rpo`
- `read_thrown`
- catchability/drop/contested tags

The prior M80/M81 family was audited/tested for QB tactical context.

For the present named-WR/TE opportunity question, the critical limitation is semantic:
`is_motion` is a play-level flag in the consumed source contract. It does not establish which on-field receiver moved.

Therefore a team/play motion rate is not a named-player motion feature.

Do not call a play-level motion flag "Player A motion rate" without a source that assigns the motion to Player A.

## 5. nflverse target-route label

Current disposition: **POTENTIALLY PLAYER-ATTRIBUTED TARGET ROLE, NOT ROUTE PARTICIPATION**

Because the participation route label describes the primary/targeted receiver's route, it can characterize the player's route **on targets**.

It cannot establish:
- total routes;
- route participation rate;
- route type on non-targets;
- target rate per route type.

A future candidate based only on prior targeted-route mix would need a distinct scientific hypothesis and anti-retest proof against prior WR target-depth/air-yard/NGS work. It is not automatically authorized by this audit.

## 6. NGS receiver tracking

Current disposition: **EXACT PRIOR TARGET-MODEL IMPLEMENTATION FAILED**

Prior WR R9-R11 source qualification and strict-prior NGS target model reached:
`WR_NGS_TARGET_MODEL_FAIL`.

The model slightly improved targets but materially worsened receiving-yard MAE. Do not rescue or repackage that exact NGS candidate.

Only a materially distinct information/interaction contract could reopen an NGS field.

## 7. Free WR-CB archive / projected alignment

Current disposition: **SEPARATE MATCHUP SOURCE LANE; NOT OBSERVED OFFENSIVE ROLE**

The repo has a free historical WR-CB archive research lane with projected pregame pairing/alignment snapshots.

Important semantics:
- projected pregame alignment is not observed route responsibility;
- editorial matchup grades are not model features;
- this is WR-scoped and does not solve TE role state;
- use belongs to the existing WR-CB source/science lane, not as a substitute for observed offensive alignment history.

Do not merge that lane into Player Personnel V1 after its result.

## 8. What remains legitimately open

The named-player opportunity persistence remains real for WR and TE, but the next
mean-relevant experiment needs a source that truly resolves one of:

1. **player-attributed alignment state** over time (WR slot/wide; TE in-line/slot/flex);
2. **actual player route participation / route family on all routes**, not targets only;
3. **player-attributed motion**;
4. another genuinely pregame named-player role/hierarchy state that is not just prior residual feedback.

Until historical parity is cleared, the correct action is:
- preserve current/live state prospectively where free/reliable;
- continue source discovery;
- do not fit a retrospective correction on fabricated semantics.

## 9. Separate confirmed player-level uncertainty state remains live

This source block applies to the unresolved **mean/opportunity explanation** only.

Still valid:
- `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
- `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`

Those should continue toward governed distribution integration / forward shadow independently.

## Boundary

- production changed: false
- sportsbook inputs introduced: 0
- OddsAPI calls: 0
- paid player-tracking source purchased: false
