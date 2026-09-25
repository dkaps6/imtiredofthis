# Opportunity Partition Semantics V1 — Read-Only Audit Result

Date: 2026-09-25

Disposition: **OPPORTUNITY_PARTITION_SEMANTICS_V1_CONFIRMED**

This is a no-outcome structural result. It does not authorize production changes.

## Authority

- branch: `research-opportunity-partition-semantics-v1`
- authoritative run: `36141882752`
- job: `108093463911`
- head: `ecdffac45ab914f3e2eacd3784c93951ee2c09ed`
- artifact: `10867051736`
- digest: `sha256:5843d85f14a7e3a6644f74f5dbd24acb9867f5433e19659a097ea1209e2be616`
- season/week: 2026 Week 3
- certified football teams: 30
- canonical games: 15
- sportsbook inputs: 0
- target-game outcomes: 0
- production changed: false

## Source-semantics proof

Every frozen premise passed:

1. historical `qb_dropback` is a dropback opportunity, not an official pass attempt;
2. historical code explicitly records QB scrambles as remaining inside dropbacks;
3. `rules_pass_rate` is documented and consumed as dropbacks / plays;
4. historical QB MC converts that state with `pass_attempts_per_dropback`;
5. production QB pricing also applies the promoted attempt conversion;
6. canonical receiver target allocation consumes the unconverted `pass_att` state;
7. canonical rushing opportunity is `plays - pass_att`.

Therefore the current simulator uses one state with two incompatible meanings:

- on the QB side it is a dropback state that must be converted to official attempts;
- on the receiver side it is treated directly as target-pool opportunity;
- on the rushing side all dropbacks are removed from the rushing complement, including QB scrambles.

## Receiver-side magnitude

Current strict-prior `pass_attempts_per_dropback`:

- min: `0.7970`
- median: `0.8731`
- max: `0.9506`

Thus the current receiver pool is materially larger than implied official pass attempts.

Across 30 certified teams:

- mean excess target-pool opportunities: **4.086 per team**
- median excess: **4.094**
- median target-pool inflation vs official attempts: **14.54%**
- 25th percentile inflation: **10.66%**
- 75th percentile inflation: **18.86%**
- maximum inflation: **25.47%**

Because the explicit named-player target entitlement is approximately 95% of the pool:

- mean named target mass generated from non-attempt dropbacks: **3.877**
- median: **3.884**
- maximum: **6.302**

This is not a subtle bookkeeping difference. The current receiver engine can generate roughly four named-player target opportunities per team-game from dropbacks that the promoted QB semantics classify as sacks/scrambles/non-attempts.

## Rushing-side magnitude

Using strict-prior primary-QB scramble rates:

- mean implied scramble attempts: **2.412 per team**
- median: **2.292**
- median restored-scramble volume equals **9.03%** of the current non-dropback rush pool.

The current primary-QB simulated rushing attempts have a median of `2.415`, near the implied scramble count, but this does not prove the rushing model is correct because QB rush share also contains designed-run / historical rushing information.

Three current teams produced a negative raw residual after subtracting a primary-QB scramble estimate from total non-attempt dropbacks. These were driven by limited or unusually high current QB scramble samples and are a reason **not** to combine a rushing repair with the receiver correction.

The rushing side therefore remains a separate research problem.

## Relationship to the reconciliation work

The target-pool inflation did not strongly explain cross-team variation in the prior raw-QB-vs-named-receiver yard gap:

- correlation using inflation percentage: approximately `-0.209`
- correlation using excess opportunities: approximately `-0.195`

That is important.

The semantics bug is real, but it is **not sufficient by itself** to explain the entire aggregate receiving-yard coherence problem. Reconciliation and opportunity partition may both matter.

## Interpretation

The receiver opportunity seam is now a legitimate historical candidate:

`RECEIVER_OFFICIAL_ATTEMPT_POOL_V1`

Mechanism:
- preserve simulated team dropbacks;
- convert receiver opportunity to official pass attempts using the already-promoted, strict-prior `pass_attempts_per_dropback` authority;
- allocate targets only from that official-attempt pool;
- preserve current target shares, entitlement specialists, catch-rate/YPT assumptions and residual target bucket;
- do not change QB mean/distribution authority;
- do not change rushing in the same experiment.

Before a player-level full-stack test, a simpler historical team-opportunity calibration should verify that the converted pool is directionally closer to realized official attempts and realized team targets than the existing dropback pool.

## Stopping rule

Do not rescue or combine with:

- hierarchical receiver reconciliation;
- QB C2 changes;
- WR/TE/RB carveouts;
- new target-share thresholds;
- rushing/scramble fixes;
- sportsbook-conditioned logic;
- target-game outcome fitting.

Receiver opportunity semantics must stand on its own first.
