# RB RECEIVING SHARE STATE COVERAGE V1 — FROZEN CONTRACT

Date: 2026-10-07
Branch: `research-rb-receiving-share-state-coverage-v1`

## Purpose

Audit whether the residual RB receiving target-share miss identified by
`PLAYER_OPPORTUNITY_VOLUME_VS_SHARE_DECOMPOSITION_V1` is already described by
strict-prior RB receiving-role state that exists in the repository but is not
generally consumed by the canonical RB target allocator.

This is coverage / consumption / mechanism-localization only.

No fitted candidate, threshold, router, or production promotion is authorized.

## Frozen parent evidence

The ACT-only opportunity decomposition showed:

- RB target model MAE: 1.386 targets
- actual team-volume diagnostic removes only 2.8%
- actual player-share diagnostic removes 76.5%
- target-share error vs player target-error Pearson: 0.941

Therefore the open mechanism is individual RB receiving share, not team
dropback volume.

## Canonical production path to audit

Current generic RB target path:

1. PlayerForm consensus target-share evidence
2. empirical-Bayes `bayes_tgt_share`
3. rules layer, including the coarse RB receiving matchup multiplier and any
   authorized injury redistribution
4. explicit M38 target entitlement
5. TE-R5P and WR-R15, both required to preserve non-target-position entitlement
6. joint allocation

The audit must verify this code path directly.

## Existing strict-prior RB receiving identity state

Use only the already-defined state family in
`scripts/modeling/rb_receiving_identity_runtime_v1.py`.

Predeclared room-share state fields:

- `prior_rb_room_share`
- `last8_rb_room_share`
- `prev_season_rb_room_share`
- `same_team_prior_rb_room_share`

Predeclared supporting target-state fields:

- `prior_targets_pg`
- `last8_targets_pg`
- `prev_season_targets_pg`
- `same_team_prior_targets_pg`
- `prior_target_share`
- `last8_target_share`
- `prev_season_target_share`

Do not add fields after seeing results.

## Population

- 2026 Weeks 1-4
- ACT-only historical availability-parity population
- RB / FB target rows only
- full football universe, not sportsbook conditioned
- parent model-side rows from artifact
  `player-opportunity-volume-vs-share-decomposition-v1-37694474836`

## Within-room normalization

To isolate player allocation from team-volume and total RB-pool size:

For each team-week:

- model RB room share =
  player predicted target probability / sum RB+FB predicted target probability
- actual RB room share =
  player actual targets / sum RB+FB actual targets

If actual RB room targets equal zero, room-share outcome is undefined and that
team-week is excluded from room-share accuracy scoring, while retained in
coverage counts.

## Required diagnostics

1. current model within-RB-room share MAE / bias
2. coverage of every predeclared strict-prior state field
3. Spearman correlation of each predeclared state field with:
   - realized RB room share
   - current model room-share residual
4. no-fit carry-forward proxy diagnostics for each of the four room-share state
   fields:
   - raw state value versus realized RB room share
   - MAE / bias
   - improvement or worsening versus current model room-share MAE
5. rank diagnostics:
   - whether model RB-room leader matches realized RB target-room leader
   - whether strict-prior room-state leader matches realized leader
6. role-transition diagnostics:
   - same-team history available vs unavailable
   - prior-season history available vs unavailable
7. code-consumption audit:
   - whether the generic canonical RB target path consumes each state field
   - whether Week-1 R26 consumes the richer identity state only inside its
     separately frozen, Week-1-specific receptions adapter
   - confirm R22 tail science does not change target/reception means
8. no sportsbook inputs

## Interpretation

Possible descriptive dispositions:

- `EXISTING_RB_RECEIVING_STATE_PRESENT_BUT_NOT_GENERICALLY_CONSUMED`
- `EXISTING_RB_RECEIVING_STATE_ALREADY_CONSUMED_NO_NEW_GAP`
- `RB_RECEIVING_STATE_COVERAGE_INSUFFICIENT`
- `RB_RECEIVING_STATE_AUDIT_INCONCLUSIVE`

If existing strict-prior room-share state clearly outperforms the current model
room allocation as a raw no-fit proxy, a separately frozen candidate may be
authorized later.

This audit itself cannot choose a blend coefficient, threshold, window, or
promotion rule.

## Protected boundaries

Do not reopen:

- R23-R27D RB receiving-yard mean family
- M96 retrospective RB router/threshold program
- R22 tail-distribution science
- Week-5 RB carry/snap allocation shadow
- WR/TE target-share trajectory rules
- generic position-level target-share retuning

No paid OddsAPI.
No sportsbook input.
No automatic production change.
