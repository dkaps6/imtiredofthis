# GSIS RB Successor Lineup V1 — Frozen Prospective Plan

**STATUS: FROZEN BEFORE ANY ELIGIBLE FUTURE OUTCOME. RESEARCH ONLY.**

## Purpose

Test whether exact pregame GSIS offensive-lineup co-occurrence improves the
identity/concentration of rushing-opportunity transfer after a definitive RB/FB
vacancy beyond the already-frozen RB Vacancy Opportunity V1 snap-weight
successor rule.

This is a successor-weighting experiment only. It does not change team rushing
volume, vacated-share estimation, YPC, efficiency, sportsbook pricing, or any
production model.

## Why this is genuinely new

Already-established facts:
- production correctly removes definitive-unavailable RB/FBs before opportunity;
- RB Vacancy Opportunity V1 estimates the unavailable player's vacated share
  from strict-prior rush share and weights surviving successors by marginal
  strict-prior offense snap percentage;
- the unresolved structural problem is successor identity/concentration;
- GSIS Lineup Detail has passed the source-information gate for exact 11-player
  co-occurrence that cannot be reconstructed from marginal snaps/depth;
- the GSIS audit did not fit a predictive model or use target outcomes.

This V1 does **not** modify or rescue RB Vacancy Opportunity V1 using its exposed
Week-3 outcomes. Week-3 and any game completed before this freeze are forbidden
from fitting, selecting, or grading this mechanism.

## Eligible target boundary

A target team-game is eligible only if ALL of the following are locked before
kickoff:

1. canonical current-player availability identifies at least one RB/FB with
   `definitive_unavailable == 1` and `UNAVAILABLE_*`;
2. the existing RB Vacancy V1 strict-prior vacated-share estimate is available;
3. at least one surviving production-eligible RB/FB exists;
4. an immutable GSIS Lineup Detail snapshot was captured before kickoff and
   contains that offense's cumulative state only through already-completed
   games;
5. source identity bridge is exact and unambiguous;
6. the target game has not started.

Past games may contribute only as strict-prior source history. They can never
be counted as prospective candidate observations if no pregame lock existed.

## Frozen source fields

From GSIS Lineup Detail read only:
- exact offensive Lineup player set;
- Plays.

No gain, TD, turnover, EPA, grade, sportsbook, target-game outcome, or editorial
field may enter the candidate.

Current production role/availability sources are used only to:
- identify the unavailable RB/FB;
- identify surviving eligible RB/FB successors;
- bridge exact player identity.

Raw/private GSIS rows remain outside the public repository. Public artifacts may
contain only sanitized hashes, aggregate counts, and candidate summaries that
do not reveal access-controlled raw tables.

## Frozen successor score

For unavailable RB/FB `u`, surviving successor `j`, and the immutable
pregame lineup table for team `t`:

`gsis_absent_exposure_j = sum(Plays_l)`

over every exact offensive lineup `l` such that:
- `u` is NOT in lineup `l`;
- successor `j` IS in lineup `l`.

No minimum-play threshold is tuned in V1. Every positive-play source row is
used exactly as displayed after parser/source-quality validation.

Successor weight:

`w_gsis_j = gsis_absent_exposure_j / sum_k(gsis_absent_exposure_k)`

where `k` ranges only over surviving production-eligible RB/FB successors.

If the denominator is zero, the event is
`NO_GSIS_SUCCESSOR_EXPOSURE` and V1 abstains. Do not fall back to a tuned
mixture. The existing RB Vacancy V1 snap-weight candidate remains separately
auditable as its own frozen mechanism.

## Candidate transfer

The unavailable player's vacated rushing share `V_g` is copied unchanged
from the frozen RB Vacancy Opportunity V1 contract.

For each successor:

`transfer_gsis_j = V_g * w_gsis_j`

The candidate changes opportunity only and holds all existing YPC/efficiency
assumptions fixed.

No blend of GSIS weight and snap weight is permitted in V1.

## Three locked arms

For every eligible future event persist before kickoff:

1. **BASELINE** — current production;
2. **VACANCY_V1_SNAP** — exact already-frozen RB Vacancy Opportunity V1;
3. **GSIS_LINEUP_V1** — same vacated share, exact GSIS absent-lineup successor
   weight above.

The GSIS arm is compared with both BASELINE and VACANCY_V1_SNAP. Neither parent
may be refit after target outcomes.

## Pregame integrity gates

Before an event can be locked:
- GSIS snapshot capture time < kickoff;
- snapshot hash and manifest immutable;
- source rows contain no target-game observations;
- unavailable and successor identities disjoint;
- exact identity mapping only; ambiguity fails closed;
- all lineup play counts finite and nonnegative;
- GSIS successor weights nonnegative and sum to 1;
- GSIS transfers sum to frozen vacated share within 1e-10;
- no target-game carries/yards/snaps;
- no sportsbook input;
- candidate changes no YPC/efficiency field;
- production output remains untouched.

## Prospective support gate

Do not issue a scientific PASS/FAIL before:
- at least **6 distinct future NFL weeks** with eligible pregame locks;
- at least **10 qualifying vacancy team-games**;
- at least **20 scored successor player-games**.

If the season ends below any floor:
`INSUFFICIENT_PROSPECTIVE_SUPPORT_HOLD`.

No lowering of these floors is allowed after seeing outcomes.

## Frozen outcomes / metrics

Primary metric:
- successor rushing-attempt absolute error, paired at player-game level.

Secondary football metrics:
- successor rushing-yard absolute error with the pregame efficiency/YPC state
  held frozen;
- signed attempt bias;
- signed yard bias;
- team RB/FB carry/share conservation.

Primary candidate comparison:
`GSIS_LINEUP_V1` vs `VACANCY_V1_SNAP`.

Also report both against BASELINE.

Dependence-aware inference:
- cluster by target game;
- deterministic bootstrap seed 20261001;
- 10,000 cluster resamples;
- primary statistic = mean paired attempt-AE improvement
  (`VACANCY_V1_SNAP AE - GSIS_LINEUP_V1 AE`).

## Pass gate

`GSIS_RB_SUCCESSOR_LINEUP_V1_PASS` requires ALL:
1. prospective support floor met;
2. mean paired rushing-attempt AE improvement vs VACANCY_V1_SNAP > 0;
3. game-cluster bootstrap 95% CI lower bound for that improvement > 0;
4. rushing-yard MAE with frozen efficiency is non-worse vs VACANCY_V1_SNAP;
5. absolute attempt bias is non-worse;
6. conservation/integrity gates all pass.

Otherwise, with adequate support:
`GSIS_RB_SUCCESSOR_LINEUP_V1_FAIL`.

No ROI, sportsbook hit rate, subgroup, high-volume subset, side, threshold, blend,
or alternate co-occurrence formula may rescue a failed primary result.

## Production boundary

Even a PASS is research qualification only. Production promotion would require
a separate explicit integration plan and proof that the live/private GSIS
dependency is operationally acceptable.

No paid OddsAPI or paid external source is required by this research.
