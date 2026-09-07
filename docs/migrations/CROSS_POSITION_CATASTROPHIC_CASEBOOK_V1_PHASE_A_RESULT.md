# Cross-Position Catastrophic Player Casebook V1 — Phase A Result

## Status

**PHASE_A_ARTIFACT_DECOMPOSITION_COMPLETE**

This is an artifact-level forensic decomposition using exact frozen model rows. It is not the final PBP-enriched casebook and it does not change production.

Frozen plan commit: `38b53b22a741f3e7754e2158eed7e08c1dcb5678`  
Evaluator commit: `afe8133ca3e1079d0cd2b55bb7309961fec25411`  
Workflow wiring commit: `ca2d8dad0758d4e217c7af44f1b40b8ec05fc167`

## Exact evidence used

### QB
M89 run `33331073376`, artifact `9737913528`, exact 2024-2025 football-synthesis trace (`884` QB-games).

### WR
C2 full-stack run `34142510405`, artifact `10026729563`, exact B0 paired-player rows. The B0 WR receiving rows preserve M38; the resulting `12396` WR-row receiving-yard MAE is `23.247439`, effectively reproducing the canonical M38 pooled `23.247553` result.

### TE
TE-R5 run `34132127351`, artifact `10022512461`, exact `3214` OOS TE player-games 2023-2025 using `candidate_rec_yards_r5` / `candidate_targets_r5`.

### RB
STACK3 run `33539468967`, artifact `9812993290`, `1393` 2025 RB rows. P3 is reconstructed from the frozen contract: Week 1 `arm_week1_stack`; Weeks 2-18 `arm_stack2_parent` (enriched opportunity × STACK1 efficiency).

Sportsbook variables were not used for cause classification.

## Catastrophic definitions frozen before inspection

- QB passing yards: absolute miss >= `100` yards.
- WR receiving yards: absolute miss >= `50` yards.
- TE receiving yards: absolute miss >= `40` yards.
- RB rushing yards: absolute miss >= `40` yards.

## Headline result

The worst misses are **not one universal problem**. The dominant causal layer changes by position and, critically, by whether a high-opportunity player smashes above the projection or busts below it.

### All catastrophic rows

| Position | Rows | Catastrophic rows | Rate | Dominant catastrophic error-mass split |
|---|---:|---:|---:|---|
| QB | 884 | 124 | 14.03% | team/pass opportunity `50.59%`; efficiency `48.14%`; other `1.27%` |
| WR | 12,396 | 1,408 | 11.36% | efficiency `41.48%`; player entitlement/share `28.51%`; WR/team pool `21.12%`; conversion `8.89%` |
| TE | 3,214 | 209 | 6.50% | efficiency `45.86%`; TE pool `29.77%`; player share `17.46%`; conversion `6.91%` |
| RB | 1,393 | 170 | 12.20% | rushing efficiency/YPC `62.64%`; carry opportunity `37.36%` |

This means the pass-catcher catastrophic tails are roughly half opportunity/allocation and half efficiency, while RB catastrophic rushing misses are more strongly efficiency-driven.

## High-opportunity players are the central tail problem

Catastrophic-miss rate rises sharply with baseline opportunity quartile:

| Position | Q1 catastrophic rate | Q4 catastrophic rate | Q4 MAE | Q4 p90 AE |
|---|---:|---:|---:|---:|
| QB | 14.93% | 17.65% | 58.71 | 113.50 |
| WR | 6.07% | 18.20% | 31.12 | 64.37 |
| TE | 1.12% | 14.68% | 22.40 | 44.19 |
| RB | 3.72% | 26.72% | 29.75 | 61.99 |

The effect is extreme for RB/WR/TE. The players the model already believes will receive the most opportunity are exactly where the most damaging tails remain.

## Directional asymmetry — most important finding

### QB Q4
39 catastrophic cases.

- Overprojected/bust: 25 cases, `63.88%` of Q4 catastrophic error mass.
- Underprojected/smash: 14 cases, `36.12%`.

When a high-projected QB **smashes above** us, `80.04%` of that underprojection error mass is opportunity/attempt driven. Mean predicted attempts were `29.25`; actual attempts averaged `46.79` (`+17.53`).

When a high-projected QB **busts below** us, `63.10%` of that overprojection error mass is efficiency/YPA driven. Mean predicted attempts were `28.57`; actual attempts `24.76`, but efficiency collapse is the dominant mechanism.

**Interpretation:** QB upside tail needs a better shared pass-volume/game-script state; QB downside tail needs a more realistic efficiency/suppression distribution.

### WR Q4
564 catastrophic cases.

- Underprojected/smash: 484 cases, `89.15%` of Q4 catastrophic error mass.
- Overprojected/bust: 80 cases, `10.85%`.

For Q4 WR **smash games**, dominant error mass:
- efficiency/YPR `42.10%`;
- team/WR target pool `40.34%`;
- player share `9.87%`;
- conversion `7.69%`.

Mean expected targets were `6.48`; actual targets averaged `11.22` (`+4.74`). Predicted YPR averaged `12.95`; actual YPR averaged `17.08`.

For Q4 WR **bust games**, `61.78%` of error mass is player entitlement/share. Mean expected targets `7.21`; actual targets only `3.89` (`-3.32`).

**Interpretation:** M38 hierarchy is not the primary cause of star-WR upside misses. The ceiling problem is shared/team opportunity plus explosive efficiency. But when an elite WR busts, player-share/teammate-role allocation is the dominant problem. This calls for asymmetric entitlement/distribution handling rather than one universal WR correction.

### TE Q4 after TE-R5
118 catastrophic cases.

- Underprojected/smash: 92 cases, `82.29%` of Q4 catastrophic error mass.
- Overprojected/bust: 26 cases, `17.71%`.

For Q4 TE **smash games**, dominant error mass:
- efficiency/YPR `42.81%`;
- team TE pool `42.17%`;
- conversion `9.34%`;
- individual TE share only `5.68%`.

Mean expected targets `5.82`; actual targets `9.97` (`+4.14`). Predicted YPR `10.58`; actual YPR `13.88`.

For Q4 TE **bust games**, team TE-pool error dominates `58.24%` of error mass.

TE-R5 also reduced 40+ yard misses materially: B0 had `281`; TE-R5 has `209`. It fixed `99` prior 40+ misses while introducing `27` new ones.

**Interpretation:** TE-R5 did what we wanted at individual entitlement. Among high-volume TEs, individual share is now a small residual tail mechanism. The next TE gains should come primarily from the finite **team TE pool** and efficiency/explosiveness—not another individual-share retune.

### RB Q4
93 catastrophic cases.

- Underprojected/smash: 58 cases, `70.87%` of Q4 catastrophic error mass.
- Overprojected/bust: 35 cases, `29.13%`.

For Q4 RB **smash games**, `80.70%` of error mass is rushing efficiency/YPC and only `19.30%` is carry opportunity. Mean predicted carries `15.91`; actual carries `20.33`, but predicted YPC averaged only `3.99` while actual YPC averaged `7.05`.

For Q4 RB **bust games**, the relationship reverses: carry opportunity accounts for `65.08%` of error mass. Mean predicted carries `17.29`; actual carries `10.00` (`-7.29`).

**Interpretation:** RB needs asymmetric architecture. Upside explosions are primarily an efficiency/explosive-run problem. Downside busts are primarily workload/game-script/room-allocation problems. Another single mean carry correction or detached tail overlay is unlikely to solve both.

## Season stability

### QB catastrophic error mass
- 2024: opportunity `52.3%`, efficiency `45.4%`.
- 2025: efficiency `51.4%`, opportunity `48.6%`.

The 50/50 opportunity-efficiency split is stable across both seasons.

### WR catastrophic error mass
Across every season 2020-2025, efficiency and opportunity/share remain the large mechanisms. Efficiency is the largest single bucket in every season except near-ties; player share and team pool together remain material every year.

Recent seasons:
- 2023: efficiency `52.4%`, share `20.7%`, team pool `16.6%`, conversion `10.4%`.
- 2024: efficiency `45.4%`, share `25.7%`, team pool `18.4%`, conversion `10.6%`.
- 2025: efficiency `37.1%`, share `32.5%`, team pool `24.3%`, conversion `6.1%`.

### TE catastrophic error mass
- 2023: efficiency `45.1%`, TE pool `36.5%`, share `10.2%`, conversion `8.2%`.
- 2024: efficiency `38.1%`, TE pool `31.0%`, share `22.5%`, conversion `8.3%`.
- 2025: efficiency `54.4%`, TE pool `23.0%`, share `18.1%`, conversion `4.4%`.

Pool + efficiency are persistent across all three OOS seasons.

## Largest individual examples

### QB
- 2024 W5 Kirk Cousins: actual `509`, projection `242.07`, miss `266.93`; actual attempts `58` vs `30.31` projected. Opportunity contribution is enormous.
- 2024 W13 Jameis Winston: actual `497`, projection `230.21`, miss `266.79`; attempts `58` vs `29.06` projected.
- 2025 W9 Joe Flacco: actual `470`, projection `229.58`, miss `240.42`; both attempts and YPA exploded.

### WR
- 2021 W17 Ja'Marr Chase: `266` actual vs `47.55`; miss `218.45`, with large team-pool, share, conversion and efficiency contributions.
- 2024 W13 Jerry Jeudy: `235` vs `37.94`; miss `197.06`, efficiency contribution `110.58` yards.
- 2023 W16 Amari Cooper: `265` vs `71.77`; miss `193.23`.

### TE
- 2025 W7 Oronde Gadsden II: `164` vs `28.39`; miss `135.61`, efficiency contribution `91.64` yards.
- 2024 W3 Dallas Goedert: `170` vs `35.18`; miss `134.82`, TE-pool contribution `41.00`, efficiency `68.91`.
- 2025 W15 Kyle Pitts: `166` vs `43.35`; miss `122.65`.

### RB
- 2025 W10 Jonathan Taylor: `244` vs `78.07`; miss `165.93`; opportunity contribution `84.69`, efficiency `81.24`.
- 2025 W12 Jahmyr Gibbs: `219` vs `61.34`; miss `157.66`; almost entirely efficiency (`159.46`).
- 2025 W5 Rico Dowdle: `206` vs `49.41`; miss `156.59`; opportunity `45.67`, efficiency `110.93`.
- 2025 W8 James Cook: `216` vs `66.06`; miss `149.94`; efficiency `144.48`.

## Repeated catastrophic players

The catastrophic tails repeatedly involve the players with the largest real ceilings, not only random fringe players.

Examples:
- WR: Justin Jefferson 33 threshold misses; Tyreek Hill 28; A.J. Brown 27; Davante Adams 25; Ja'Marr Chase 22.
- TE: Trey McBride 15; Travis Kelce 14; George Kittle 12; Brock Bowers 9.
- RB in 2025: Bijan Robinson 8; James Cook 8; Jahmyr Gibbs 7; Derrick Henry 7; Rico Dowdle 6.
- QB 2024-25: Baker Mayfield 8; Jalen Hurts 7; Matthew Stafford 6; Geno Smith 6; Josh Allen 6.

This fixed-threshold view naturally selects high-volume stars more often, so it must not be interpreted as a fixed-player correction. It does, however, reinforce the need for role-tier-specific tail distributions.

## Existing QB PBP casebook overlap

Of the current `124` M89/M90 100+ misses, `81` overlap the prior M89 forensic casebook built on the same 884-game cohort.

Within those 81:
- YAC-driven explosion: `27`;
- forced pass volume: `13`;
- unexpected low volume: `13`;
- sustained efficiency collapse: `11`;
- voluntary pass volume: `5`;
- sustained efficiency explosion: `4`;
- single explosive play: `3`;
- smaller turnover/protection/garbage-time groups.

`58/81` of those current overlapping catastrophes still remain 100+ misses even after removing the single largest completion; `23/81` fall below 100. This suggests the remaining M89 tail problem is materially structural, not merely one-play noise. The other 43 current catastrophic QB games require Phase-B PBP reconstruction before final classification.

## Phase-A decisions

1. **Do not discard C2/shared opportunity work.** QB upside catastrophes are strongly attempt-volume driven.
2. **Do not spend the next TE cycle retuning individual TE entitlement.** TE-R5 materially reduced tails and Q4 residual share error is small. Target TE pool + efficiency next.
3. **WR needs asymmetric treatment:** star-WR upside = team pool + efficiency; star-WR downside = individual entitlement/share.
4. **RB needs asymmetric treatment:** star-RB upside = efficiency/explosive rushing; star-RB downside = opportunity/workload.
5. **High-opportunity role tiers need wider/more football-realistic distributions.** A single symmetric residual distribution is structurally mismatched to the observed failure modes.
6. Phase B must add PBP longest-play/YAC/score-state/drive/in-game participation reconstruction so `EFFICIENCY` can be separated into predictable matchup/environment versus low-predictability explosive events.

## Final Phase-A disposition

`PHASE_A_ARTIFACT_DECOMPOSITION_COMPLETE`

No production promotion is authorized from Phase A alone.

`postgame_forensic_fields_used_for_prediction = false`

`sportsbook_features_used_for_cause_classification = false`
