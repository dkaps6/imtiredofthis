# Opportunity Partition Semantics V1 — Read-Only Audit Result

Date: 2026-09-25

Disposition: **RECEIVER OPPORTUNITY SEMANTICS DEFECT CONFIRMED**

This is a no-outcome structural result. It does not itself authorize production.

## Authority

- branch: `research-opportunity-partition-semantics-v1`
- authoritative run: `36141882752`
- job: `108093463911`
- head: `ecdffac45ab914f3e2eacd3784c93951ee2c09ed`
- artifact: `10867051736`
- digest: `sha256:5843d85f14a7e3a6644f74f5dbd24acb9867f5433e19659a097ea1209e2be616`
- current certified universe: 30 teams / 15 games
- iterations: 5000
- sportsbook inputs: 0
- target-game outcomes: 0
- production changed: false

## Source-semantics proof

All frozen premises were confirmed directly in current repo code:

1. historical `rules_pass_rate` is derived from `qb_dropback` share;
2. it therefore represents dropbacks / offensive plays;
3. historical MC explicitly converts that state with
   `pass_attempts_per_dropback` to obtain official pass attempts;
4. production QB pricing also applies the promoted attempt conversion;
5. canonical receiver target allocation consumes the unconverted simulated
   `pass_att` state;
6. canonical rushing opportunity is the complement
   `plays - pass_att`.

The repo itself documents that scrambles remain dropbacks while sacks are not
official pass attempts.

Therefore receiver target opportunity is currently sourced from a state whose
semantics include non-attempt dropbacks.

## Current-slate magnitude

Promoted strict-prior pass-attempt conversion:

- minimum: **0.7970**
- median: **0.8731**
- maximum: **0.9506**

So the current team dropback state materially exceeds official pass attempts.

Receiver target pool excess:

- mean: **4.086 opportunities/team**
- median: **4.094**
- p25: ~**3.189**
- p75: ~**5.169**
- p90: ~**5.537**

Inflation versus implied official pass attempts:

- minimum: **5.19%**
- median: **14.54%**
- p75: ~**18.86%**
- p90: ~**21.41%**
- maximum: **25.47%**

Because current explicit named-player entitlement is approximately 95% of the
pool, the expected named-player opportunity mass generated from non-attempt
dropbacks is:

- mean: **3.877 target opportunities/team**
- median: **3.884**
- p90: ~**5.256**

Those opportunities originate from plays that cannot produce a receiver target
if the promoted dropback/official-attempt semantics are accepted.

## Relationship to the earlier reconciliation finding

This defect is real, but it does **not** fully explain the earlier aggregate
QB/receiver incoherence.

Current-slate correlation between target-pool excess/inflation and canonical
named-receiver minus raw-QB receiving-yard gap was weak and negative:

- raw target-pool excess correlation: **-0.195**
- percentage inflation correlation: **-0.209**

Therefore:
- receiver opportunity semantics should be tested as its own correction;
- hierarchical reconciliation remains a separate research question;
- the two must not be conflated or combined before independent qualification.

## Rushing-side diagnostic

The current rushing complement excludes all dropbacks.

Using strict-prior primary-QB scramble rates:

- mean implied scramble attempts: **2.412/team**
- median: **2.292/team**
- median restored-scramble volume equals about **9.03%** of current rush pool.

However the current decomposition is not yet safe enough for a rushing
candidate:

- 3 teams produced negative
  `nonattempt_dropbacks - implied_scrambles` residual;
- current-starter prior sample sizes are heterogeneous, including very small
  samples;
- current QB simulated rush opportunity can already partially reflect scramble
  behavior indirectly through rush-share authority.

Therefore the audit does **not** authorize a rushing/scramble correction yet.

## Interpretation

The receiver-side defect is simpler and better identified:

`receiver target pool = simulated dropbacks`

while the promoted football semantics say:

`official pass attempts = simulated dropbacks * pass_attempts_per_dropback`

and only official pass attempts can create targets.

The existing promoted conversion is already football-only and strict-prior. No
new parameter needs to be fit.

## Next step

Freeze a separate historical receiver-opportunity candidate before scoring:

`RECEIVER_OFFICIAL_ATTEMPT_POOL_V1`

Initial scope:
- receiver MC opportunity only;
- convert dropback target pool to official-attempt target pool using the
  existing strict-prior `pass_attempts_per_dropback`;
- preserve all target-share / entitlement authority;
- preserve WR-R15 and TE-R5P;
- preserve catch rate and YPT;
- QB projections unchanged;
- rushing unchanged;
- receptions and receiving yards evaluated;
- dependent RB rush+receiving evaluated through current RB V2;
- no C2 routing requirement;
- no player/position carveout;
- no fitted conversion coefficient;
- no sportsbook input.

A separate rushing-partition study remains parked until scramble authority can
be made provenance-safe.
