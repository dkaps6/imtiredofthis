# WR-R7 YPR-Mechanism-Conditioned Efficiency — Frozen Plan

## Why this is new
WR-ND6 tested player/defense explosive and yard-generation traits across the entire WR population and correctly returned `NO_ACTIONABLE_EXPLOSIVE_CEILING_SIGNAL`. That result remains a failure and is not reopened.

WR-R5 was learned later and supplied genuinely new information: among 133 qualifying WRs, 16 have recurring receiving-yard misses dominated specifically by the **YPR** component, while 72 are TARGETS-dominant and 44 are mixed.

WR-R7 therefore asks a different, mechanism-conditioned question:

**Do legitimate pregame player/defense yard-generation traits explain the YPR Shapley residual specifically for WRs whose individual historical miss mechanism is YPR-dominant?**

This is not a retry of the ND6 global gate and cannot alter the ND6 disposition. No production change is authorized.

## Frozen evidence
- WR-R5 run `34066489537`, artifact `9999098530`, artifact name `wr-r5-target-catch-ypr-individual-decomposition`.
- WR-ND6 run `34056970854`, artifact `9996440756`, digest `sha256:2382554e2c723589ae502a1bee7bb0df08e4f811cf4b83d1c8b7e380269edc81`, artifact name `wr-nd6-player-level-explosive-ceiling`.
- Expected WR-R5 full casebook: 2,130 rows.
- Expected qualifying YPR-dominant players: **16**.
- Sportsbook inputs prohibited.

## Conditioned population
Merge exact WR-R5 player mechanism labels onto exact ND6 player-games by player key. Primary analysis includes only player-games belonging to the 16 WR-R5 `YPR`-dominant qualifying players.

The target is the exact WR-R5 signed `ypr_component` Shapley contribution to receiving-yard residual. Secondary outcomes are total receiving-yard residual and ND6 underprojection tails.

## Frozen signal family
No new feature engineering and no interactions. Reuse exactly the eight strictly-prior ND6 traits:

Player side:
1. `PLAYER_EXP20_PER_TARGET_PRIOR8`
2. `PLAYER_EXP40_PER_TARGET_PRIOR8`
3. `PLAYER_YAC_PER_RECEPTION_PRIOR8`
4. `PLAYER_AIR_PER_TARGET_PRIOR8`

Defense side:
5. `DEF_EXP20_PER_ATT_ALLOWED_PRIOR8`
6. `DEF_EXP40_PER_ATT_ALLOWED_PRIOR8`
7. `DEF_YAC_PER_COMPLETION_ALLOWED_PRIOR8`
8. `DEF_AIR_PER_ATT_ALLOWED_PRIOR8`

No combinations, alternate history windows, transformations, or nearby thresholds.

## Frozen signal scoring
For each signal within the YPR-dominant population:
- N and coverage;
- Spearman versus signed `ypr_component`;
- Q4-minus-Q1 mean `ypr_component` gap;
- Q4-minus-Q1 total receiving-yard residual gap;
- under-25-yard residual tail enrichment in Q4 versus overall valid rows;
- W2-18 YPR-component gap;
- W13-18 YPR-component gap;
- per-player Spearman sign stability among YPR-dominant players with >=8 valid signal rows.

Quartiles are frozen from the full conditioned population for that signal.

## Candidate pass gate
A signal passes only if every condition holds:
1. conditioned population has >=120 valid rows for the signal;
2. coverage within conditioned population >=0.75;
3. Spearman with `ypr_component` >=0.10;
4. Q4-Q1 `ypr_component` gap >=4.0 yards;
5. Q4-Q1 total receiving-yard residual gap >=5.0 yards;
6. Q4 under-25 tail enrichment >=1.20x;
7. W2-18 `ypr_component` gap >0;
8. W13-18 `ypr_component` gap >0;
9. at least 10 players have >=8 valid rows and >=55% of those player-level Spearman values are positive.

No failed gate may be lowered after results.

## Family disposition
Because eight previously-known signals are being evaluated in a new subgroup, WR-R7 requires replication inside the family rather than accepting one isolated hit.

`WR_YPR_MECHANISM_CONDITIONED_EFFICIENCY_DISCOVERY_PASS` requires:
- at least **2** signals independently pass every candidate gate; and
- at least **1** passing signal is player-side.

Otherwise disposition is:
`NO_ACTIONABLE_WR_YPR_CONDITIONED_EFFICIENCY_SIGNAL`.

Even a discovery pass is diagnostic only. It would authorize a separately frozen full-stack simulation experiment affecting the YPR/upper-tail layer for the appropriate player mechanism class; it would not authorize a mean fudge factor.
