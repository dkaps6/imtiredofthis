# RB-PD2 Forward / Shadow Confirmation V1 — Frozen Plan

**STATUS: FROZEN BEFORE ANY PROSPECTIVELY LOCKED 2026 CANDIDATE OUTCOME IS GRADED. RESEARCH SHADOW ONLY. NO PRODUCTION DECISION CHANGE.**

## 1. Authority and purpose

This is the separately required forward/shadow confirmation authorized by the historical qualification:

`RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`

from PR #562 / canonical run `35039152022`.

The historical candidate passed all 28 frozen gates, including:
- pooled CRPS improvement;
- targeted high-difficulty CRPS improvement;
- player-clustered and crossed player × game dependence-aware support;
- interval calibration protection;
- fixed football-threshold Brier protection;
- exact point-mean neutrality;
- four-season robustness.

That historical qualification did **not** authorize production. It authorized only a separately frozen prospective confirmation. This document freezes that confirmation before any future candidate result is graded.

This study asks one question only:

> On prospectively locked 2026 RB/FB rushing-yard player-games from the certified production pricing path, does the already-qualified, exactly mean-neutral player-difficulty width transform improve the predictive distribution out of sample?

It does not test or alter the RB rushing-yard mean.

## 2. Frozen candidate — no retuning

The candidate is inherited unchanged from the qualified historical experiment.

History:
- same player only;
- last 8 completed eligible prior games;
- minimum 4 prior games;
- raw difficulty statistic = mean absolute rushing-yard projection error over those prior games;
- exact rolling strictly-prior empirical percentile reference;
- minimum strictly-prior reference size = 100;
- all rows in a target week are scored before any row from that week enters the percentile reference.

Mapping:

`width_mult = 1.0 + 0.30 * clip((difficulty_score - 0.50) / 0.50, 0.0, 1.0)`

Consequences:
- difficulty <= 0.50 => 1.00x width;
- linear widening above the median;
- maximum width = 1.30x;
- no narrowing.

No coefficient, onset, history window, minimum prior-game count, percentile convention, subgroup, or transform may change after this plan is frozen.

## 3. Predictor-history lineage

### 3.1 2025 history

For 2025 RB/HB/FB rushing-yard history, use the documented production-route-equivalent calibrated generic ensemble parent.

The promoted `rush_yards` weights in `data/model_ensemble_weights.csv` are:
- MC = `0.5569542426070742`;
- ML = `0.4430457573929258`;
- State = `0.0`;
- fit scope = `all_2024_oos_frozen_for_2025`;
- promotion lineage = `RB_STACK1_RUN_33535308110_FOR_P3`.

These weights were fit on 2024 OOS data only and frozen for 2025. No 2025 outcome entered the weights.

2025 component predictions must be rebuilt with the leakage-safe historical pipeline. Realized 2025 rushing yards may then be compared with that production-route-equivalent football mean solely to form strictly-prior predictor history for 2026.

### 3.2 Completed 2026 weeks

A completed 2026 game may become predictor history for a later 2026 target only if its legitimate pregame football projection lineage is reconstructible without target/future information.

Already-played 2026 games:
- may enter later strictly-prior history after completion;
- may enter the rolling strictly-prior percentile reference after completion;
- **may not** be backfilled and counted as prospective confirmation observations.

The Week-1 P3 authority contract states that Week-1 P3 equals its STACK1 full-stack parent within tolerance. If Week 1 is used as prior history, this parity must be mechanically reasserted in the shadow preflight.

## 4. Rolling strictly-prior reference

Preserve the exact qualified `strict_prior_difficulty_scores()` semantics.

For target week W:
1. score every eligible row in W against only earlier eligible reference rows;
2. do not let any W row enter another W row's reference;
3. after all W rows are scored, completed eligible W rows may enter the reference for later weeks.

Hard chronology invariant for every locked target row:

`difficulty_reference_max_ord < target_ord`

where `target_ord = season * 100 + week`.

Violation is an integrity failure, not a scientific result.

## 5. Prospective confirmation population

The confirmation population is the existing certified production pricing-loop population, not a newly invented full-roster ensemble population.

Include every unique RB/HB/FB `rush_yards` player-game that:
- reaches the existing production pricing loop;
- receives a finite final football `target_mean`;
- has a valid baseline empirical draw array;
- has at least 4 valid strictly-prior same-player history games;
- has a valid difficulty score after the 100-row reference floor;
- is prospectively locked before kickoff.

Do **not** filter by:
- OVER vs UNDER;
- edge magnitude;
- published decision;
- sportsbook result;
- whether the production board ultimately recommends a bet.

One player-game counts once scientifically even if downstream pricing emits both side rows.

Sportsbook line/odds may determine whether the row exists in the live offer-bearing pricing population, but line/odds are forbidden from candidate construction and from the primary scientific metrics.

## 6. Baseline empirical distribution

The baseline must be the exact empirical production football distribution used by the certified Full Slate pricing path.

In `scripts/run_pricing_v2.py` the operative sequence is:
1. canonical Monte Carlo produces `base_outcomes`;
2. final football authority determines `target_mean`;
3. empirical draws are aligned:
   `adjusted_outcomes = base_outcomes * max(0, target_mean / mc_proj)`;
4. production fair probability is later computed from the empirical array.

The shadow must use the exact `adjusted_outcomes` array, not a Normal approximation and not only `model_sd`.

## 7. Shadow capture seam

Preferred implementation is an observational same-process capture hook at the exact point where `adjusted_outcomes` exists and before sportsbook probability comparison.

The hook must:
- be gated by an opt-in research environment flag;
- default OFF;
- copy the empirical array without mutating it;
- never replace, rescale, or write back to production `adjusted_outcomes`;
- write only a separate research shadow artifact;
- leave production fair probabilities, edges, decisions, workbook, and board unchanged.

Required no-op proofs before any live shadow lock:
1. hook-disabled path remains byte-identical to current priced output on a controlled fixture;
2. hook-enabled vs hook-disabled production `model_proj`, `model_sd`, fair probability, edge, and decision fields are identical on controlled fixtures;
3. shadow candidate output exists only in the separate research artifact;
4. production code path with flag OFF remains the default.

Deterministic replay may be used as an engineering diagnostic, but it is not accepted as direct proof of empirical array identity merely because mean, SD, and one CDF point match.

## 8. Frozen candidate distribution transform

For baseline empirical draws `x`:

1. `mu = mean(x)`;
2. `candidate_raw = max(0, mu + width_mult * (x - mu))` elementwise;
3. if `mean(candidate_raw) > 0`, set:
   `candidate = candidate_raw * (mu / mean(candidate_raw))`;
4. otherwise fail closed.

Hard invariants:
- same draw count;
- all finite;
- all nonnegative;
- candidate mean equals baseline mean within `1e-8`;
- no carry, YPC, allocation, receiving, role, or mean input changes;
- no sportsbook input enters the transform.

## 9. Pregame lock artifact

Before kickoff, persist one immutable scientific row per eligible player-game containing at minimum:

- season / week;
- event/game identity;
- team / opponent;
- stable player identity;
- position;
- market = `rush_yards`;
- lock timestamp;
- scheduled kickoff timestamp when available;
- production code SHA;
- workflow run / job identity;
- input artifact hashes sufficient to reproduce the football distribution;
- final football `target_mean`;
- baseline draw count;
- baseline draw digest or exact separately stored array reference;
- baseline mean / SD / q05 / q10 / q50 / q90 / q95;
- prior eligible game count;
- prior8 rushing-yard MAE;
- difficulty score;
- difficulty reference N;
- `difficulty_reference_max_ord`;
- width multiplier;
- candidate draw count;
- candidate draw digest or exact separately stored array reference;
- candidate mean / SD / q05 / q10 / q50 / q90 / q95;
- explicit flags:
  - sportsbook_inputs_used_in_candidate = false;
  - production_output_mutated = false;
  - outcome_present_at_lock = false.

Any realized target-game outcome in the pregame lock artifact is a fatal integrity failure.

## 10. Prospective start boundary

The first confirmation observation is the first **future** eligible player-game whose baseline and candidate distributions are both persisted before kickoff under this frozen protocol.

No already-played 2026 game can become a confirmation observation.

If a shadow artifact is generated after kickoff, that row is excluded from prospective confirmation and recorded as a lock failure.

## 11. Grading metrics

After games complete, join realized rushing yards to the immutable pregame locks.

Use the paired empirical baseline/candidate draw arrays.

Per player-game:
1. empirical CRPS;
2. central 80% interval coverage using q10-q90, endpoints inclusive;
3. central 90% interval coverage using q05-q95, endpoints inclusive;
4. empirical exceedance probabilities and Brier scores at fixed football thresholds:
   - >=50 rushing yards;
   - >=75 rushing yards;
   - >=100 rushing yards;
5. baseline and candidate draw means;
6. point absolute error only as a mean-neutrality audit.

Quantiles use `np.quantile(..., method="linear")`.

No sportsbook line is a primary scientific grading threshold.

## 12. Minimum support gate

A scientific PASS/FAIL disposition may be issued only after BOTH:
- at least **8 distinct prospectively locked NFL weeks**; and
- at least **400 unique eligible locked RB/HB/FB rushing-yard player-games**.

If the 2026 regular season ends before both floors are reached:

`NO_FORWARD_CONFIRMATION_INSUFFICIENT_SUPPORT`

This is a **HOLD / underpowered** disposition, not a scientific failure of the mechanism.

No sample floor may be lowered after outcomes are observed.

## 13. Frozen dependence-aware inference

### 13.1 Primary game-cluster bootstrap

Primary estimand:

`mean(baseline_crps - candidate_crps)`

Positive values favor the candidate.

Bootstrap:
- cluster unit = unique NFL game;
- sample games with replacement;
- every sampled game contributes all eligible locked RB rows from that game;
- paired baseline/candidate rows remain together;
- 10,000 valid replicates;
- deterministic seed = `42027`.

Primary statistical gate:
- observed pooled CRPS gain > 0; and
- 95% percentile bootstrap CI lower bound for the game-clustered CRPS gain > 0.

No minimum effect magnitude such as the historical +0.5% or +1.0% qualification bars is required in the forward confirmation. Those historical thresholds belonged to mechanism qualification and may not be retrofitted as a new prospective magnitude requirement.

### 13.2 Crossed player × game robustness

Preserve the historical Amendment-3 dependence check.

For each of 10,000 valid replicates, seed `42027`:
- independently sample unique players with replacement;
- independently sample unique games with replacement;
- convert both samples to multiplicities;
- weight each observed row by `player_multiplicity * game_multiplicity`;
- compute the weighted paired mean of `candidate_crps - baseline_crps`;
- redraw zero-weight replicates.

Hard additive robustness gate:

`P(weighted mean(candidate_crps - baseline_crps) < 0) >= 0.95`

This cannot rescue failure of the primary game-cluster CI gate.

A player-cluster-only bootstrap may also be reported descriptively for continuity with the historical qualification, but it is not a substitute for either required gate above.

## 14. Frozen calibration / tail guardrails

These are hard protection gates. They cannot rescue a failed CRPS primary gate.

### 14.1 Pooled interval calibration

Across all eligible locked rows:
- candidate 80% absolute coverage gap must be <= baseline 80% absolute coverage gap;
- candidate 90% absolute coverage gap must be <= baseline 90% absolute coverage gap;
- at least one of the two pooled coverage gaps must improve strictly.

### 14.2 High-difficulty targeted protection

Define the high-difficulty slice using the same global-Q75 rule as historical Amendment 3:
- compute the linear-interpolated 75th percentile of locked-row `difficulty_score`;
- include all rows with `difficulty_score >= threshold`;
- ties are included.

On that frozen slice:
- candidate mean CRPS must be strictly lower than baseline;
- candidate 80% coverage gap must be <= baseline;
- candidate 90% coverage gap must be <= baseline;
- at least one of the two interval gaps must improve strictly.

No +1.0% CRPS magnitude requirement is imposed prospectively.

### 14.3 Fixed football-threshold Brier protection

Across all eligible locked rows:
- Brier >=100 rushing yards must improve strictly;
- Brier >=50 must be non-worse;
- Brier >=75 must be non-worse.

These are inherited football-threshold calibration guards and are not sportsbook-based.

## 15. Mean-neutrality / production-isolation hard gates

All must pass:
- maximum rowwise candidate-vs-baseline mean difference <= `1e-8`;
- pooled point-MAE difference absolute value <= `1e-8`;
- zero target-game outcome values in pregame locks;
- zero same/future history violations;
- zero same-week percentile-reference contamination;
- zero sportsbook inputs used in candidate construction;
- zero candidate-driven changes to production fair probability, edge, decision, workbook, or board;
- production changed = false.

Failure of these gates is an integrity failure, not evidence against the scientific mechanism.

## 16. Frozen dispositions

### Integrity failure

If any lineage, chronology, lock-timing, array-capture, mean-neutrality, or production-isolation invariant fails:

`RB_PD2_FORWARD_SHADOW_INTEGRITY_FAILURE`

Do not interpret scientifically until the mechanical defect is repaired without changing candidate science.

### Insufficient support

If the regular season ends before >=8 locked weeks and >=400 eligible player-games:

`NO_FORWARD_CONFIRMATION_INSUFFICIENT_SUPPORT`

HOLD. Not PASS. Not scientific FAIL.

### Scientific confirmation

Only if support is sufficient and **every scientific hard gate** passes:

`RB_PD2_YARD_DIFFICULTY_WIDTH_FORWARD_CONFIRMED`

### Scientific non-confirmation

If support is sufficient, all integrity gates pass, but any scientific hard gate fails:

`NO_ACTIONABLE_RB_PD2_YARD_DIFFICULTY_WIDTH_FORWARD_CONFIRMATION`

This closes the exact production-shadow promotion case for this mapping. Do not rescue it by changing width cap, onset, history window, reference construction, sample floor, metric thresholds, dependence method, or subsets after exposure.

## 17. Production consequence

A forward confirmation PASS still does not silently mutate production.

It authorizes only a separate, explicit production-promotion review / PR.

Until such a promotion is separately approved:
- production uses the existing baseline distribution;
- shadow candidate probabilities are not published as the betting board;
- bankroll / staking / side selection is unchanged by this study.

## 18. Parallel RB mean-information lane

RB remains unresolved even if this width study confirms.

After the first future shadow artifact is successfully locked pregame, begin the separate RB mean-information source audit in parallel, before any forward-width outcome grading is available.

Frozen audit order:
1. routes-run / route-volume live + historical source readiness;
2. opponent-injury propagation into matchup context;
3. CLV capture architecture as a downstream diagnostic only, not a football-model feature.

This parallel lane may not alter the frozen width confirmation candidate or its evaluation population.

## 19. Explicit forbidden actions

- no backfilling played 2026 games as prospective confirmation;
- no retuning `0.30`, `0.50`, last-8, min-4, or reference floor;
- no alternative difficulty statistic;
- no sportsbook feature in candidate construction;
- no threshold/subgroup search after exposure;
- no P3 Weeks2-18 backport;
- no mean-model rescue inside this shadow study;
- no use of WR results to retune RB;
- no production board mutation from shadow outputs;
- no paid OddsAPI pull solely for this research layer.

## 20. Collaboration record

This plan incorporates the resolved GPT-5.6 / Claude review in Issue #535:
- Claude accepted the empirical-MC correction;
- Claude accepted rolling strict-prior reference semantics;
- GPT-5.6 verified 2025 frozen-weight lineage;
- both sides agreed on >=8 weeks / >=400 rows;
- both sides agreed on exact empirical-array capture;
- both sides agreed the forward CRPS gate should use sign + dependence-aware confidence rather than requiring the historical +1.218% magnitude;
- Claude corrected the WR disposition-catalog lineage at `research/disposition-catalog-v1@ffc16219`;
- no future 2026 width outcome was used to choose any rule in this document.
