# Migration 68 — QB Playcaller / Opening Script / Leverage

## Why M68 exists

M66 found extreme residual redundancy in the existing QB model library while a hindsight model oracle retained ~14.9 yards of MAE headroom. M67 then tested broad prior-game offensive intent, aggregate injuries, formation/personnel and continuity without clearing a frozen breakthrough gate. Its one replicated clue was recent Q1 dropback rate, consistent with planned opening behavior carrying more signal than full-game aggregate tendency.

M68 therefore moves toward information that more directly sets a specific week's offensive regime: the actual primary playcaller, opening-script behavior, and game-specific competitive leverage.

## Historical / leakage boundary

- 2024 is training; 2025 is the only prospective test season.
- 2023 is history context only.
- 2022 is not target-tested.
- No player-prop line is a model feature.
- Opening-script features use games strictly before the target week.
- Opening-script PBP is constructed directly from nflverse PBP and is independent of postseason participation data.
- Playcaller season-opening mappings are frozen from ESPN all-32-team inventories. Documented midseason handoffs are effective only beginning with their public effective week.
- Playcaller names are audit metadata only. Models receive numeric prior-history/change/tendency variables, not coach-name one-hot encoding.
- Simulated playoff leverage uses only results from weeks strictly before the target week and the known schedule. Future actual results and future sportsbook lines are never used.
- Approximate simulated playoff qualification is not described as exact NFL-tiebreaker probability.

## Frozen new families

1. `opening_script_live`
   - first 10 offensive plays DBR
   - first 15 offensive plays DBR
   - first drive DBR
   - first two drives DBR
   - first-15 early-down neutral DBR
   - first-15 shotgun rate
   - first-15 vs rest-of-game DBR delta
   - Q1 DBR
   - leakage-safe last1 / mean3 / mean8 / mean3-minus-mean8 history

2. `verified_playcaller_live`
   - caller change since previous team game
   - current caller prior games, current-team prior games and new-to-team state
   - caller recent first-15 / first-drive / first-two-drive / Q1 tendencies across prior games
   - caller-current-team recent tendencies

3. `playoff_leverage_live`
   - P(playoffs | forced current-game win), approximate
   - P(playoffs | forced current-game loss), approximate
   - leverage delta
   - entering playoff probability proxy
   - entering record / games remaining

4. `playcaller_plus_opening_live`
5. `all_m68_new_live`

Attribution controls:
6. `existing_only_control`
7. `existing_plus_m68_new`

## Frozen model architecture

Each family is tested for passing-yards residual, attempt residual and M64-neutral DBR residual with the same two models:

- Ridge alpha = 50
- HistGradientBoosting absolute error, max_iter 150, learning_rate .04, depth 2, min leaf 15, L2 5, seed 68

Fit 2024 only; test 2025 only. No hyperparameter search, feature-subset search, post-hoc thresholds or production winner selection.

## Frozen base gates

New-feature median coverage must be >= .75.

Passing requires all:
- residual correlation >= .20
- MAE gain >= 1.0 yard vs Raw
- correlation gain >= .03
- 100+ yard misses non-increase

DBR requires all:
- residual correlation >= .20
- DBR MAE gain >= .0075 vs M64 neutral
- DBR correlation gain >= .10

Attempts requires all:
- residual correlation >= .20
- attempt MAE gain >= .25
- attempt correlation gain >= .05
- 10+ attempt misses reduced >= 5%

## Frozen incremental attribution gate

An `existing_plus_m68_new` result cannot be credited to M68 merely for clearing a base gate. It must also beat the same-model `existing_only_control`:

- Passing: >= .50 additional MAE gain, >= .02 additional correlation, 100+ misses non-worse.
- DBR: >= .0025 additional MAE gain and >= .03 additional correlation.
- Attempts: >= .10 additional MAE gain, >= .03 additional correlation, 10+ misses non-worse.

Standalone new-only families do not need the incremental old-feature control because they contain no old feature universe.

## Interpretation

- Any standalone new family clears a base gate, or existing+new clears both base and incremental gates -> `m68_new_information_breakthrough_followup`
- No model gate, but strong replicated leverage feature-target pairs -> `m68_leverage_partial_signal`
- No model gate, but strong replicated opening/playcaller pairs -> `m68_opening_playcaller_partial_signal`
- Neither -> `seek_deeper_week_specific_information_or_randomness_transition`

M68 is diagnostic only. No result directly promotes production QB logic.
