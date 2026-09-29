# Week 4+ Legacy Coverage Heuristic Remove-Only Shadow V1 — Frozen Plan

Date frozen: 2026-09-29  
Status: **FROZEN BEFORE WEEK-4+ OUTCOMES — PROSPECTIVE SHADOW ONLY**  
Branch: `research-week3-postmortem-execution-v1`

Parent:
`docs/research/LEGACY_COVERAGE_HEURISTIC_MECHANICAL_MATERIALITY_V1_RESULT.md`

Parent disposition:
`LEGACY_COVERAGE_HEURISTIC_MECHANICALLY_MATERIAL_LOW_SELECTIVITY_UNVALIDATED`

## Question

Does removing the grandfathered static WR `coverage_penalty()` heuristic improve
future full-stack receiving forecasts when every other current production rule
is held fixed?

This is a remove-only qualification test of existing production logic, not a
search for new coverage coefficients.

## A/B definition

### A0 — CONTROL
Exact canonical production football stack.

### A1 — NO_COVERAGE_HEURISTIC
Exact A0 except:

```
coverage_penalty(ypt, target_share, ...) -> (ypt, target_share)
```

No other football value, threshold, feature, model, source or coefficient may
change.

A1 must still flow through:
1. explicit target entitlement;
2. M38 WR hierarchy;
3. TE-R5P;
4. WR-R15;
5. canonical joint simulation;
6. all currently promoted downstream distribution/mean mechanics applicable to
   the target week.

Sportsbook data remains downstream only and is not needed for primary science.

## Start boundary

The first eligible observation is the first canonical 2026 Week-4 production
football artifact created after this plan freeze.

Weeks 1-3 can never count toward confirmation.

## RNG prerequisite

Full-array A/B interpretation is permitted only after the specialist RNG
isolation repair reproduces its frozen research fingerprint, or after the
shadow proves equivalent common-random-number isolation by an independently
frozen invariant test.

Until then:
- deterministic pre-simulation means/entitlements may be captured;
- outcome-based A/B scoring must not be called production-qualification
  evidence if unrelated arrays can drift through shared RNG consumption.

## Immutable pregame capture

For every eligible future week preserve before kickoff:
- source production run / git SHA;
- A0 football universe and rule inputs;
- exact coverage source state;
- A0 explicit entitlement / TE-R5P / WR-R15 traces;
- A1 corresponding traces;
- A0/A1 player-market simulation means;
- exact changed-key / protected-key audit;
- capture timestamp and target kickoff times.

Never reconstruct the shadow after seeing outcomes.

## Scientific population

Primary football population:
- player-games with a finite production `rec_yards` projection and verified
  postgame actual;
- positions WR, TE, RB/FB;
- one player-game once, independent of sportsbook books/sides/lines.

Secondary count population:
- finite production `receptions` projection and verified actual.

VOID / DNP / unresolved identity handling must follow canonical settlement /
participation rules and may not be converted to zero ad hoc.

## Primary metric

Pooled receiving-yards paired absolute-error improvement:

`abs(A0_rec_yards - actual) - abs(A1_rec_yards - actual)`

Positive favors removing the heuristic.

Report:
- MAE A0 / A1;
- signed bias A0 / A1;
- p90 absolute error A0 / A1;
- game-cluster bootstrap of paired MAE improvement.

## Required position decomposition

Report the same receiving-yards metrics separately for:
- WR;
- TE;
- RB/FB.

These are guardrails, not independently tunable candidates.

## Secondary metric

Receptions:
- pooled MAE and bias A0/A1;
- by WR / TE / RB-FB;
- game-cluster paired bootstrap.

## Minimum support

Do not call PASS or FAIL until all are true:
- >=8 distinct eligible future NFL weeks beginning Week 4;
- >=1,200 pooled receiving-yards player-games;
- >=500 WR receiving-yards rows;
- >=250 TE receiving-yards rows;
- >=250 RB/FB receiving-yards rows;
- >=800 receptions player-games.

Before all floors:
`FORWARD_OBSERVATION_ONLY_INSUFFICIENT_SUPPORT`

## Frozen qualification gate

### `LEGACY_COVERAGE_HEURISTIC_REMOVAL_FORWARD_QUALIFIED`
Requires all:
1. pooled rec-yards MAE improves by >=1.0%;
2. game-cluster bootstrap 95% CI for pooled paired absolute-error improvement is
   entirely > 0;
3. pooled absolute signed bias does not worsen;
4. no eligible position with its support floor met worsens rec-yards MAE by
   >1.0%;
5. pooled receptions MAE does not worsen by >0.5%.

### `LEGACY_COVERAGE_HEURISTIC_REMOVAL_FORWARD_FAILED`
Once support floors are met, any qualification condition fails.

No intermediate weekly PASS/FAIL.

## If qualified

Qualification still does not mutate production automatically.

Next step is a separate integration certification proving:
- only the legacy coverage heuristic is removed;
- no source/identity/availability behavior changes;
- all non-receiving protected paths remain invariant;
- no sportsbook input moves upstream;
- exact forward A/B evidence is preserved.

## If failed

Close the remove-only lane.

Do not rescue with:
- new man/zone thresholds;
- new position thresholds;
- alternate 0.94/1.04 multipliers;
- matchup-only carveouts;
- WR1-only rules;
- Week-specific rules;
- outcome-fitted coverage coefficients.

## Production state

Production remains unchanged.
