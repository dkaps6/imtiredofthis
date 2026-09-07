# Joint Pass/Receiving Conservation V1 — Frozen Plan

## Purpose

Test the two mechanisms independently supported by the frozen Receiving Ecosystem Conservation Audit without discarding or retuning any promoted model:

1. leakage-safe WR/TE/RB+FB **position-group target-mass calibration** before within-position hierarchy; and
2. a **single completed-pass/receiving-yard process** in which receiver yardage plus an explicit residual receiver bucket equals QB passing yardage in every Monte Carlo iteration.

This is a research/backtest migration only. It does **not** replace M89/M90 QB mean production, M38 WR hierarchy production, RB P3 rushing production, or any sportsbook-facing output.

## Lineage

- Branch: `research-joint-pass-receiving-conservation-v1`
- Parent branch: `audit-receiving-ecosystem-conservation`
- Parent SHA: `dc120eb186bf0f63c465de15fea2f02e5cf1c14f`
- Receiving ecosystem audit run: `34075068419`
- Receiving ecosystem artifact: `10001852203`
- Audit disposition A: `POSITION_GROUP_MISALLOCATION_SUPPORTED`
- Audit disposition B: `MATERIAL_PASS_RECEIVING_CONSERVATION_GAP`
- Prior shared-volume disposition: `STRONG_SHARED_PASS_VOLUME_MECHANISM`
- Promoted QB mean anchor: `QB_PASS_SYNTHESIS_V1` (M89/M90)
- Promoted WR within-room hierarchy: M38 multipliers `1.40 / 1.14 / 0.91 / 0.78`
- RB rushing production: P3 unchanged
- Sportsbook inputs: **0**

## Why a frozen 2x2 experiment

Both mechanisms were authorized before this migration by separate frozen diagnostics. To avoid obscuring attribution, V1 predeclares exactly four architectures and no others:

- `B0_CURRENT`: exact current simulation.
- `C1_GROUP_ONLY`: group target-mass calibration only; current independent QB/receiver yard draws otherwise unchanged.
- `C2_CONSERVATION_ONLY`: current target allocation; new completed-pass/receiving-yard conservation process.
- `C3_JOINT`: C1 + C2 together.

No additional coefficient, window, interaction, threshold, or variant may be added after results are visible. C3 is not a post-result combination search; it is the predeclared factorial combination of the two independently supported mechanisms.

## Frozen historical scope

### Receiving markets
- Seasons: **2020-2025**, regular season only.
- 2020: Weeks 1-17.
- 2021-2025: Weeks 1-18.
- Exact leakage-safe pregame historical context.
- Positions reported independently: WR (`WR/LWR/RWR/SWR`), TE, RB, FB where sample permits; RB+FB for group-mass accounting.

### QB distribution comparison
- Seasons: **2024-2025** only, matching exact available promoted M89/M90 common-era evidence.
- M89/M90 `football_synthesis` remains the QB mean anchor.
- Candidate architecture may change the **shape** of the QB distribution but may not change its frozen promoted mean.

## Frozen target-mass candidate (C1/C3)

### Step 1 — preserve total modeled receiver mass

For each team-game, start from the exact current leakage-safe target shares after Bayesian/rules context and the promoted M38 within-WR sharpening. Let the current modeled pass-catcher mass be:

`M = sum(current shares for WR + TE + RB + FB)`

V1 preserves `M` exactly. Therefore this migration tests **position-group allocation**, not total pass volume. The existing residual target bucket remains the residual bucket.

### Step 2 — leakage-safe team group history

At each target-week cutoff, construct historical target counts using completed games strictly before the target game.

Groups:
- WR
- TE
- RB_FB

For each team, use its **most recent 8 completed regular-season games** available at the cutoff, crossing the season boundary when needed. Week 1 therefore uses only prior-season games.

The league prior at the same cutoff uses the **immediately prior regular season plus completed current-season games before the target week**. No future game is permitted.

### Step 3 — fixed empirical-Bayes shrinkage

For each group `g`:

`team_group_share_g = (team_group_targets_last8 + 105 * league_group_share_g) / (team_targets_last8 + 105)`

The **105 target pseudo-count** is frozen as a three-team-game equivalent at 35 targets/game. It is not tuned from V1 results.

Normalize WR/TE/RB_FB group shares across those three groups, then multiply by the preserved modeled receiver mass `M`.

### Step 4 — within-group allocation

- WR: retain the exact M38 sharpened player proportions inside the calibrated WR group mass.
- TE: retain current rules/Bayesian player proportions inside calibrated TE mass.
- RB/FB: retain current rules/Bayesian player proportions inside calibrated RB_FB mass.
- If a calibrated group has no eligible pregame player with positive current share, its unassignable mass remains in the residual bucket; it may not be assigned equally to backups or filled from target-game outcomes.

No WR, TE, RB, or FB player-level coefficient is tuned in this migration.

## Frozen completed-pass / receiving-yard candidate (C2/C3)

### Core accounting process

For each team and MC iteration:

1. Generate team plays/pass attempts from the same pregame context as B0.
2. Allocate finite targets to modeled players plus the existing residual bucket.
3. Generate catches from each player's frozen pregame catch-rate input.
4. Convert each modeled player's pregame YPT into an implied yards-per-reception value:
   `YPR = YPT / catch_rate`, with the same bounded/fallback semantics documented in the candidate code.
5. A player with zero simulated receptions receives exactly zero simulated receiving yards.
6. Generate a synthetic residual receiver bucket from residual targets using frozen neutral defaults `catch_rate = 0.64` and `YPT = 7.5` (the existing simulation fallback values), hence residual `YPR = 7.5 / 0.64`.
7. Sum modeled-player and residual receiving yards. This raw sum is the raw team passing-yard process.

### Preserve the promoted QB mean

For 2024-2025, apply one **constant pregame team-game scale factor** to every iteration so the candidate team passing-yard distribution mean equals the exact M89/M90 `football_synthesis` mean for that team-game.

Then:

`candidate QB pass yards = sum(candidate modeled receiver yards) + candidate residual receiver yards`

for every iteration.

The scale factor is constant within a team-game; it does not force each iteration toward a realized outcome and cannot use target-game results.

For 2020-2023 receiving evaluation, where the exact promoted M89/M90 common-era anchor is unavailable, preserve the exact B0 pregame QB MC mean as the team-game mean anchor. Those earlier seasons are used to judge receiver effects only; they are not claimed as M89 QB evaluation seasons.

### Distribution semantics

The independent B0 QB yardage noise is removed only in C2/C3. QB distribution shape emerges from the same target/catch/receiver-efficiency process as receiver outcomes. This is specifically intended to allow WR/TE/RB receiving composition and explosive/efficiency behavior to inform QB tails while preserving the promoted QB mean.

## Frozen evaluation outputs

### Team/group target allocation
For B0, C1, and C3, pooled + season-level:
- WR / TE / RB_FB target-share MAE, RMSE, bias, correlation;
- cross-position target residual correlations;
- modeled receiver mass and residual mass parity.

### Player receiving accuracy
For B0/C1/C2/C3, pooled + season-level, separately for WR/TE/RB:
- target MAE/RMSE/bias/correlation;
- receptions MAE/RMSE/bias/correlation;
- receiving-yard MAE/RMSE/bias/correlation.

RB additionally:
- rushing-yard parity check (must remain unchanged by receiving-only mechanisms);
- rush+receiving-yard MAE/RMSE/bias/correlation.

### QB 2024-2025
For B0/C2/C3:
- exact mean-anchor parity to `football_synthesis`;
- pass-yard MAE/RMSE/bias/correlation (mean projections);
- MC CRPS against actual QB passing yards;
- central 50%, 80%, and 90% interval empirical coverage and absolute calibration error;
- 100+ yard absolute miss count;
- distribution p10/p25/p50/p75/p90.

### Conservation
For C2/C3, iteration-level:
- max absolute `QB pass yards - (modeled receiver yards + residual receiver yards)`;
- mean/median/p90/p95 absolute gap;
- percentage > 0.01 / 1 / 10 yards.

### Integrity
- no target-week outcome in any pregame feature;
- no sportsbook input;
- exact current B0 reproduction before candidate interpretation;
- exact M38 within-WR ordering/multipliers preserved in C1/C3;
- exact RB rushing outputs unchanged in C1/C2/C3 except rush+receiving total changing only through receiving yards.

## Frozen gates

### Gate 0 — baseline parity
Scientific interpretation stops if B0 does not reproduce the current frozen receiving baseline within MC tolerance:
- 2025 all-receiver rec-yard MAE must be within **0.05 yards** of the established M38 reference `17.099904733366` when evaluated on the same all-receiver cohort semantics; and
- M89/M90 2024-2025 QB mean rows must align one-to-one with no duplicate team-weeks.

Any mismatch is mechanical/integrity work only.

### C1 group-mass gate
`GROUP_MASS_CANDIDATE_PASS` requires all:
1. pooled macro-average WR/TE/RB_FB target-share MAE improves by **>= 0.005** (0.5 percentage points) vs B0;
2. no individual group target-share MAE worsens by more than **0.0025** (0.25 percentage points);
3. macro target-share MAE improves in **>= 4 of 6 seasons**;
4. pooled 2024-2025 macro target-share MAE improves;
5. pooled WR/TE/RB player target MAE: no position worsens by more than **0.03 targets**;
6. pooled WR/TE/RB reception MAE: no position worsens by more than **0.03 receptions**;
7. pooled WR/TE/RB receiving-yard MAE: no position worsens by more than **0.50 yards**.

Otherwise: `GROUP_MASS_CANDIDATE_FAIL`.

### C2 conservation gate
`CONSERVATION_CANDIDATE_PASS` requires all:
1. iteration-level max accounting gap <= **1e-6 yards**;
2. 2024-2025 candidate QB mean differs from M89/M90 `football_synthesis` by <= **0.01 yards** in every aligned team-game;
3. pooled QB mean MAE differs from the M89/M90 mean-anchor MAE by <= **0.01 yards**;
4. mean paired QB CRPS improves by **>= 0.25 yards** vs B0;
5. paired bootstrap probability that QB CRPS improves is **>= 0.90**, using **10,000** team-game bootstrap resamples with fixed seed **5601**;
6. absolute 80% interval coverage error does not worsen by more than **0.02**;
7. pooled WR/TE/RB receiving-yard MAE: no position worsens by more than **0.50 yards**;
8. pooled macro-average WR/TE/RB receiving-yard MAE is no worse than B0;
9. zero-reception player-iterations with positive receiving yards = **0**.

Otherwise: `CONSERVATION_CANDIDATE_FAIL`.

### C3 joint gate
`JOINT_ARCHITECTURE_CANDIDATE_PASS` requires all:
1. C3 independently satisfies all C1 group-mass gates;
2. C3 independently satisfies all C2 conservation gates;
3. pooled macro-average WR/TE/RB receiving-yard MAE improves by **>= 0.25 yards** vs B0;
4. RB rush+receiving-yard MAE does not worsen by more than **0.50 yards**;
5. no single season/position receiving-yard MAE regresses by more than **1.50 yards**;
6. latest-era 2024-2025 pooled macro receiving-yard MAE improves vs B0.

Otherwise: `JOINT_ARCHITECTURE_CANDIDATE_FAIL`.

## Frozen disposition logic

- If C1 passes and C2 fails: retain group-mass mechanism as research-supported; conservation architecture is rejected in this form.
- If C2 passes and C1 fails: retain conservation mechanism as research-supported; group-mass formulation is rejected in this form.
- If both independently pass but C3 fails: **do not merge them automatically**; document the interaction failure and stage the mechanisms separately.
- If C3 passes: `JOINT_ARCHITECTURE_CANDIDATE_PASS`; this authorizes a separate production-integration/full-slate migration, not direct production promotion.
- If all fail: preserve B0 and the audit evidence; no threshold relaxation or nearby-window retry.

## Explicitly preserved work

Nothing in V1 invalidates or discards:
- M89/M90 QB football-only mean synthesis;
- M38 WR hierarchy result;
- WR-R1 2020-2025 replication backbone;
- WR ND/R-series failed and passed diagnostics;
- WR-R9 NGS partial-source evidence;
- RB P3 rushing production;
- the unresolved RB current-depth/carry-allocation audit;
- existing weather, defense, pressure, pace, PROE, injury, matchup, Bayesian, rules, and MC building blocks.

V1 is a reconciliation layer around those components. Any future replacement requires its own frozen, paired full-stack evidence.

## Future model compatibility

A conserved joint game simulation is intentionally designed to become the shared state engine for later:
- anytime/first/last touchdown scorer distributions;
- game moneyline win probabilities;
- game totals;
- spreads.

Those future markets are not tested here and cannot influence V1.

## Stopping rule

Run only B0/C1/C2/C3 exactly as frozen above. No additional window, pseudo-count, residual defaults, YPR transform, coefficient, or gate may be introduced after results are known.