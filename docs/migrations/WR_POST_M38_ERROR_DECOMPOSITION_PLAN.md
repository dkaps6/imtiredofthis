# WR Post-M38 Error Decomposition — Frozen Plan

## Purpose

Resume WR research from the **promoted M38 parent**, not from the pre-M38 receiving diagnostics.

M31-M38 already established and promoted the receiving target-pool / WR hierarchy foundation. The next question is therefore:

> After M38 repaired WR hierarchy concentration, what now dominates WR receiving-yard error: target opportunity, catch conversion, ordinary/non-explosive efficiency, or explosive-play yardage?

This migration is **diagnostic only**. It does not fit a predictive model and cannot change production.

## Frozen parent

- Exact M38 merge commit: `b98518d97b3038f471aee9ae3201009b2c70bb29`.
- Exact integrated post-M38 2025 walk-forward: run `32485770487`, job `96781810815`, artifact `9449461710` (artifact now expired; run/log lineage remains authoritative).
- Expected integrated all-receiver checks from that run:
  - `rec_yards` MC: n=4647, MAE=17.099905, RMSE=25.196100, bias=-5.238641, corr=0.567946.
  - `receptions` MC: n=4647, MAE=1.328964, RMSE=1.858555, bias=-0.513425, corr=0.578792.
- The new workflow must reconstruct the parent from the exact M38 commit in an isolated checkout. If the historical rebuild materially disagrees with these frozen checks, fail before scientific interpretation.

## Population

- Season: 2025.
- Weeks: 1-18.
- WR positions only: `WR`, `LWR`, `RWR`, `SWR`.
- Canonical M38 MC-covered player-games only.
- Sportsbook inputs prohibited.

## Component representation

For each WR player-game, preserve the exact M38 MC receiving-yard mean as the empty-subset parent.

Four components are attributed:

1. `OPPORTUNITY` — target count.
2. `CONVERSION` — catch probability / receptions per target.
3. `NON_EXPLOSIVE_EFFICIENCY` — yards on receptions gaining <20 yards, per reception.
4. `EXPLOSIVE_YARDAGE` — yards on receptions gaining >=20 yards, per reception.

The parent mean is factorized exactly as:

`M38_MC_YARDS = PRED_TARGETS * PRED_CATCH_RATE * (PRED_NONEXP_YPR + PRED_EXP_YPR)`

where implied total YPR is chosen so the identity reproduces the exact M38 MC mean row-by-row.

### Truth components

- actual targets and receptions come from the canonical weekly player history;
- actual catch rate = receptions / targets when targets > 0;
- actual non-explosive and explosive receiving yards come from target-game PBP and must sum back to canonical receiving yards within tolerance;
- a reception is explosive at `yards_gained >= 20`.

### Pregame split robustness

The parent implied YPR must be split into non-explosive vs explosive portions under **two frozen strict-prior schemes** so attribution is not an artifact of one baseline split:

1. `LEAGUE_PRIOR`: league WR receiving-yard fraction generated on 20+ yard receptions using only games before the target week (2024 plus earlier 2025 weeks).
2. `PLAYER8_SHRUNK`: receiver's last 8 prior games, shrunk to the same league prior with fixed pseudo-sample = 20 receptions.

Both schemes preserve total parent implied YPR exactly. No target-week outcome may construct the parent split.

## Attribution

Use exact four-component Shapley attribution over all 16 correction subsets.

For each subset, components in the subset are replaced by target-game truth; all others remain at the frozen parent value. Score MAE against actual receiving yards.

Required identities under both prior schemes:

- empty subset reproduces M38 MC receiving yards row-by-row within `1e-9`;
- full subset reproduces actual receiving yards row-by-row within `1e-9`;
- Shapley contributions sum to parent WR-only MAE within `1e-9`;
- strict-prior explosive split coverage is reported and no target-game PBP enters the parent split.

## Required slices

Report Shapley contributions and parent MAE for:

- ALL_WR;
- M38 pregame WR1 / WR2 / WR3 / WR4+ rank;
- Week 1;
- Weeks 2-18;
- Weeks 13-18;
- actual 100+ receiving-yard games;
- parent underprediction >=25 yards;
- parent underprediction >=50 yards;
- parent overprediction >=25 yards;
- games with at least one 20+ yard reception;
- games with no 20+ yard reception.

Also report descriptive 20+ and 40+ reception frequency/yards by WR rank and by error-tail slice.

## Frozen interpretation gate

No predictive model is authorized by this run itself.

The next architecture is chosen as follows, requiring the same top component under both prior-split schemes:

- `OPPORTUNITY_DOMINANT` if opportunity is the top overall Shapley component and >=35% of overall MAE headroom.
- `CONVERSION_DOMINANT` if conversion is top overall and >=30%.
- `NON_EXPLOSIVE_EFFICIENCY_DOMINANT` if non-explosive efficiency is top overall and >=30%.
- `EXPLOSIVE_YARDAGE_DOMINANT` if explosive yardage is top overall and either >=30% overall **or** >=50% of the >=50-yard underprediction-tail headroom.
- otherwise `MIXED_WR_ERROR_COMPONENTS`.

No waivers and no threshold tuning after seeing results.

## Closed / do-not-repeat work

This run must not reopen without new evidence:

- M31 target-allocation plumbing trace;
- M32 broad target-pool pruning;
- M33 keyed MC coverage/plumbing reconciliation;
- pre-M38 generic receiving opportunity decomposition;
- generic WR hierarchy sharpening / another multiplier search around M36-M38;
- fake individual WR-CB assignment inferred from nflverse participation. Historical explicit assignment coverage was 0%.

## Market rule

Vegas/player props remain downstream only. This run contains no sportsbook comparison and no market variable.
