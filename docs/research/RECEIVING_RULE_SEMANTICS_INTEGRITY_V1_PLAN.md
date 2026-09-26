# Receiving Rule Semantics Integrity V1 — Frozen Plan

Date frozen: 2026-09-26
Status: **FROZEN CORRECTNESS-REPAIR ABLATION — NO RESULT READ YET**

Branch: `research-public-intent-week3-prospective-v1`

## Motivation

The systems-integrity audit identified two deterministic production semantic defects in the receiving-rule path.

### Defect A — middle-open unit mismatch

`rules_v2.matchup_multipliers()` treats:
- `coverage_zone_rate`
- `coverage_man_rate`
- `middle_open_rate`

as 0-1 rates and uses a frozen threshold:
`middle >= 0.50`.

The preserved Week-3 production `team_context_v3.csv` instead contains:
- man/zone as decimals, e.g. 0.230 / 0.697;
- middle-open as percentage points, e.g. 46.9 / 69.6.

The existing rule therefore interprets 46.9 as 4690%, making the middle-open gate true essentially universally. In the exact Week-3 rule diagnostics, `slot_target_mult=1.10` for all 161 WR rows.

### Defect B — slot alignment is dropped before slot labeling

`run_player_form_v2._assign_model_roles()` deliberately preserves Ourlads alignment in:
`alignment_position`,
then replaces:
- `position` with generic family `WR`;
- `role` with usage-ranked `model_role` such as WR1/WR2/etc.

`context_bridge.build_player_contexts()` forwards only normalized `position` and `role`; it does not forward `alignment_position`.

`simulation_rules._wr_role_labels()` identifies slot WRs only when:
- `p.position == "SWR"`, or
- `"SLOT" in p.role`.

Thus the current production path cannot identify a PlayerForm SWR as SLOT.

Preserved Week-3 evidence:
- PlayerForm WRs: 161
- PlayerForm `alignment_position == SWR`: **56**
- PlayerContext WR positions: all 161 are generic `WR`
- current rule labels: WR1=30, WR1_5=30, SLOT=**0**, unlabeled=101.

## Objective

Measure the predictive and structural effect of restoring the already-intended semantics, without inventing any new coefficient, threshold, feature, or rule.

## Frozen 2x2 cells

Only these four cells are allowed:

- **A0B0 — current production baseline**
  - current raw middle_open semantics;
  - current slot-alignment drop.

- **A1B0 — middle-unit repair only**
  - canonicalize `middle_open_rate` to 0-1 before the existing 0.50 gate;
  - slot alignment remains dropped.

- **A0B1 — slot-alignment repair only**
  - preserve PlayerForm `alignment_position` through PlayerContext and identify SWR as SLOT;
  - leave current middle-open unit semantics untouched.

- **A1B1 — combined semantic repair**
  - both exact repairs above.

No other cell or variant is allowed.

## Exact repairs

### A1 middle-open canonicalization

No threshold tuning.

Semantic contract:
- if finite middle_open value is in [0,1], use unchanged;
- if finite value is in (1,100], divide by 100;
- otherwise fail closed as invalid.

The downstream threshold remains exactly `0.50`.

Do not alter man/zone/light/heavy box semantics.

### B1 slot alignment carry

No role-ranking redesign.

Carry the already-produced `alignment_position` into `PlayerContext.features` and allow `_wr_role_labels()` to classify:
`alignment_position == SWR` as `SLOT`.

After removing identified slots, preserve the existing perimeter ranking logic:
- top perimeter usage share -> WR1
- second perimeter usage share -> WR1_5.

Do not modify M38, WR-R15, target entitlement, coverage penalty, or target multipliers.

## Stage 1 — no-outcome Week-3 structural A/B

Use immutable no-odds authority:
- run `36204768034`
- artifact `10892728623`
- source main `f7d2011b73950488ea209124ba895b92c401b2b1`

No Week-3 outcomes.

For each cell report:
- WR role-label counts;
- teams/players whose `rules_tgt_share` changes;
- TE rows whose `rules_tgt_share` changes;
- target-entitlement changes;
- max/median absolute target-share delta;
- exact non-receiving invariance;
- sportsbook inputs = 0.

Stage 1 cannot promote anything.

## Stage 2 — predeclared accuracy evidence

Before production integration, use leakage-safe pregame historical/preserved evidence that contains the required alignment and middle-open fields.

Evidence preference order:
1. preserved completed 2026 pregame Full Slate artifacts (Weeks 1-2) if exact fields exist;
2. historical replay only where pregame slot alignment provenance is honest;
3. if no valid past alignment authority exists, fail historical B1 scoring closed and keep B1 prospective.

No current Week-3 outcome may be used.

### Mean targets / receiving gates

On eligible completed pregame rows, compare each repaired cell to A0B0 for:
- target-share absolute error where actual targets/team targets are available;
- receptions MAE;
- receiving-yards MAE;
- WR subgroup;
- TE subgroup;
- slot WR subgroup for B1 cells where alignment is authoritative.

A semantic repair is not automatically promoted merely because mechanics are correct.

### Directional minimum bar

For a production-facing repair:
- no material pooled WR/TE MAE regression;
- targeted subgroup error must improve;
- p90 targeted-subgroup absolute error must be non-worse;
- no protected-market contamination;
- improvement must not depend on one game/team.

If evidence is too small, result remains prospective / unpromoted.

## Stage 3 — exact production-order integration only if qualified

Only if Stage 2 supports a repair:
- run the full production order;
- preserve sportsbook separation;
- preserve M38 / WR-R15 / TE-R5P / RB V2;
- verify current non-receiving outputs are exact;
- run stable Full Slate no-live-odds certification.

## No-go rules

Do not:
- tune the 0.50 middle-open threshold;
- choose a different percentage conversion after seeing results;
- redefine SWR/slot after scoring;
- rescue by position/team/coverage subgroup;
- change slot/TE multiplier magnitudes;
- retune ensemble weights;
- alter M38 / WR-R15 / TE-R5P;
- use Week-3 outcomes in the repair decision;
- use sportsbook lines/odds upstream.

## Possible dispositions

- `MIDDLE_OPEN_UNIT_REPAIR_QUALIFIED`
- `SLOT_ALIGNMENT_REPAIR_QUALIFIED`
- `COMBINED_RECEIVING_RULE_REPAIR_QUALIFIED`
- `SEMANTIC_REPAIR_MECHANICALLY_CORRECT_BUT_NOT_ACCURACY_QUALIFIED`
- `HISTORICAL_SLOT_AUTHORITY_UNAVAILABLE_PROSPECTIVE_ONLY`
- `FAILED_CLOSED`
