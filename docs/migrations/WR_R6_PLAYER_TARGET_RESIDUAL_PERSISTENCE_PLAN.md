# WR-R6 Player Target Residual Persistence — Frozen Plan

## Question
WR-R5 showed that TARGETS are the dominant recurring receiving-yard error mechanism for 72 of 133 qualifying WRs. WR-R6 asks a narrower individual-player question: **does the model systematically under- or over-allocate target entitlement to the same player from one pregame state to the next, after removing team-level WR target mass?**

This is diagnostic-only discovery. No production projection changes are authorized by WR-R6. A positive 2025 result must be replicated on a leakage-safe multi-season target casebook before any full-stack candidate is allowed.

## Frozen evidence
- WR-R5 run `34066489537`, artifact `9999098530`.
- Input file: `wr_r5_mechanism_casebook.csv` (2,130 exact 2025 WR player-games).
- Exact upstream M38 mechanics remain unchanged.
- Sportsbook variables are prohibited.

## Leakage contract
For every target game, lagged signals may use only rows with an earlier week for that player. The current game and all future games are excluded. Signals require at least 3 prior player-games and use at most the previous 4.

## Player-specific signals
Primary:
- `PRIOR4_WR_SHARE_RESID`: mean of `(actual WR target share - projected WR target share)` over the previous up-to-4 player-games. Team WR target mass is recomputed within each historical game before the lag, so this isolates within-WR allocation rather than team passing volume.

Secondary:
- `PRIOR4_RAW_TARGET_RESID`: mean of `(actual targets - projected targets)` over the previous up-to-4 player-games.

No nearby windows, no alternate smoothing, no post-result combinations.

## Pregame role regimes
Current M38 WR role is available before the outcome. We predeclare:
- `STABLE_ROLE`: current `m38_wr_role` equals the player's immediately previous-game M38 role.
- `ROLE_CHANGE`: current role differs from immediately previous-game role.

Primary evaluation is `STABLE_ROLE`, because persistent individual allocation error should only be expected when the modeled role itself has not changed. `ROLE_CHANGE` is reported as a diagnostic contrast and cannot rescue a failed primary result.

## Outcomes
- current within-WR allocation residual: `actual_wr_share - pred_wr_share`
- current raw target residual: `actual_targets - pred_targets`

## Frozen scoring
For the primary `PRIOR4_WR_SHARE_RESID` signal on `STABLE_ROLE` rows, report:
- N and coverage;
- Pearson and Spearman versus current allocation residual;
- same-sign rate where both lagged and current residuals are non-zero;
- Q4-minus-Q1 current allocation-residual gap;
- Q4-minus-Q1 current raw-target-residual gap;
- W2-18 and W13-18 Spearman;
- WR1 / WR2 / WR3 role-slice Spearman.

Also report the same core correlations for `PRIOR4_RAW_TARGET_RESID` and the `ROLE_CHANGE` contrast.

## Frozen actionable gate
Disposition is `WR_PLAYER_TARGET_PERSISTENCE_DISCOVERY_PASS` only if every condition below is met for the primary signal:
1. lagged-signal coverage >= 0.65 of the 2,130 rows;
2. `STABLE_ROLE` N >= 700;
3. stable-role Spearman >= 0.10;
4. stable-role Q4-Q1 allocation-residual gap >= 0.025;
5. stable-role Q4-Q1 raw-target-residual gap >= 0.75 targets;
6. stable-role same-sign rate >= 0.58;
7. W2-18 Spearman > 0;
8. W13-18 Spearman > 0;
9. at least two of WR1, WR2, WR3 role-slice Spearman values are positive.

Otherwise disposition is `NO_ACTIONABLE_WR_PLAYER_TARGET_PERSISTENCE_2025`.

A discovery pass is **not** a model win and is **not** production authorization. It only authorizes a separately frozen 2020-2025 replication/full-stack candidate.
