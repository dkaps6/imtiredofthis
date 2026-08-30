# Migration 88 — Untouched 2023 Low-Chaos Regime Replication

## Status

`PREREGISTERED / CONFIRMATION ONLY`

M88 is the untouched-season confirmation of the two conditional pregame regimes isolated by M87. M88 does not fit a passing-yard correction and does not tune regime thresholds on 2023 outcomes.

## Discovery boundary

M87 used 2024 and 2025 outcomes to identify four stable pregame differentiators inside the 38 low-event-chaos 100+ yard full-stack misses. Those seasons are discovery only for M88.

M88 opens 2023 for the first time for this regime question. The current production-style walk-forward stack is reconstructed for 2023 Weeks 1-18 from 2022 + strictly-prior 2023 information at 2,000 Monte Carlo iterations. No 2023 outcome may be inspected before the regime definitions and replication gates below are frozen.

## M1-M87 anti-retest crosswalk

The raw ingredients are not novel by themselves:

- M56 and related defensive-context work tested generic defensive strength / matchup information.
- M67-M69 tested offensive intent, playcaller/opening structure and game-script mechanisms.
- M72 tested an aggregate explosive-weapon matchup bridge.
- M87 did not authorize another generic defense/YPA/pass-rate model.

The unresolved hypothesis is conditional regime failure of the current full stack: (1) a pass-funnel defense paired with a lower-deep-attempt offense may create sustained short/intermediate volume that the model underprojects, and (2) sufficiently strong recent defensive efficiency suppression may create ordinary-looking efficiency collapses that the model overprojects. M88 tests only those frozen conjunctive regimes and their directional enrichment.

## Frozen M87-derived regime thresholds

Thresholds are the midpoint between the M87 target mean and matched-control mean for each of the four preregistered stable differentiators. They are frozen before 2023 is scored.

### PASS_FUNNEL_SHORT_INTERMEDIATE_VOLUME

A 2023 QB-game is in the volume regime only when both are true:

- opponent defense prior 8-game pass rate faced `>= 0.6062143950065955`;
- target offense prior 8-game deep-attempt rate `<= 0.18505508535326465`.

Expected replication direction: positive attempt residual and enrichment for low-chaos, volume-dominant, 100+ yard **underprojections**.

### EFFICIENCY_SUPPRESSION

A 2023 QB-game is in the efficiency-suppression regime only when both are true:

- opponent defense prior 8-game success rate allowed `<= 0.42486131276490247`;
- opponent defense prior 8-game YPA allowed `<= 6.426889901131469`.

Expected replication direction: negative YPA residual and enrichment for low-chaos, efficiency-dominant, 100+ yard **overprojections**.

No alternative quantile, threshold, interaction, or direction may replace these rules after 2023 is observed.

## Historical feature contract

The four regime inputs use the same football definitions as M87 and the latest 8 prior regular-season team games, crossing into 2022 when needed. A feature is unavailable with fewer than 4 prior games. Target-game PBP never enters a pregame regime feature.

## 2023 full-stack cohort

M88 uses the M47/M75 stable-primary QB definition on the clean pre-market current-stack 2023 walk-forward output:

- projected pass-yards QB must be the realized team primary passer;
- QB must handle at least 80% of team official QB pass attempts.

The 2023 OOS ensemble uses the M82 2024 construction: expanding earlier-2023 OOS component weights, with MC fallback until at least 40 complete prior pass-yard rows are available. No later 2023 week trains an earlier-week ensemble.

## Frozen postgame confirmation labels

Postgame information is permitted only to score confirmation outcomes.

- `tail100`: absolute OOS-ensemble passing-yard error >=100.
- M86 component attribution: volume-dominant when absolute attempt contribution >=1.25x absolute YPA contribution; efficiency-dominant under the symmetric rule; otherwise mixed.
- M86 high-event-chaos markers are reproduced unchanged. A low-chaos event is a tail that does not trigger any frozen M86 chaos marker.
- Underprojection means ensemble projection minus actual passing yards <= -100.
- Overprojection means ensemble projection minus actual passing yards >= +100.

## Replication gates

Each regime is evaluated independently against all eligible non-regime stable-primary 2023 QB-games with complete regime history.

A regime is `REPLICATED_2023` only if every gate passes:

1. feature coverage among stable-primary 2023 games >=90%;
2. regime contains at least 15 QB-games;
3. at least 3 expected-direction low-chaos catastrophic events occur inside the regime;
4. expected-direction low-chaos catastrophic event rate is at least 1.50x the non-regime rate and at least +5 percentage points higher;
5. mean component residual is in the expected direction versus non-regime games (attempt residual higher for volume; YPA residual lower for efficiency suppression);
6. mean OOS-ensemble passing-yard error is in the expected direction versus non-regime games;
7. deterministic paired/bootstrap support for the expected component-residual difference is >=0.70.

Bootstrap configuration is frozen at 2,000 draws with seed 88. No threshold optimization is allowed.

## Decision contract

Possible per-regime outcomes:

- `REPLICATED_2023`
- `NOT_REPLICATED_2023`
- `INSUFFICIENT_2023_COVERAGE`

Overall M88 disposition:

- `M87_REGIME_REPLICATION_CONFIRMED` if at least one frozen regime replicates;
- `M87_REGIMES_NOT_REPLICATED` if neither replicates;
- `M88_SOURCE_OR_COHORT_FAILURE` if a clean 2023 current-stack cohort cannot be reconstructed.

Only a replicated regime may graduate to a later preregistered full-stack predictive correction test. M88 itself is not production actionable.

## Hard boundaries

- no sportsbook variables;
- no 2023 threshold tuning;
- no pass-yard correction model;
- no model selector;
- no new Ridge/HGB/XGB/neural architecture;
- no promotion from exploratory subgroups;
- target-game PBP is postgame scoring/forensics only.

`production_actionable = false`
