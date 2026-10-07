# PLAYER SHARE INPUT COVERAGE AND RESIDUAL AUDIT V1 — FROZEN CONTRACT

Date: 2026-10-07  
Branch: `research-player-share-input-coverage-residual-audit-v1`

## Purpose

Explain why the current ACT-only historical reconstruction underallocates focal
RB/WR/TE opportunity share before proposing any new player-share mechanism.

The preceding decomposition established:

- QB residual pass-attempt error is team-volume dominant; QB is out of scope here.
- RB carries, RB targets, WR targets and TE targets are player-share dominant.

This audit asks whether those skill-position share errors arise because:

1. player-specific strictly-prior evidence is missing;
2. existing Bayesian shrinkage pulls available player evidence too far toward
   the position population;
3. football-rule transforms move share away from the realized player role;
4. M38 / TE-R5P / WR-R15 redistribution compresses focal players;
5. or the residual persists after all existing player-level inputs are used.

This is an audit only. No coefficient, threshold, or new model is fit.

## Frozen population

- Season 2026, Weeks 1-4.
- Same explicit-ACT historical parity universe as
  `HISTORICAL_AVAILABILITY_PARITY_DIAGNOSTIC_V1`.
- RB/FB, WR and TE only.
- Parent realized opportunity/share labels from
  `PLAYER_OPPORTUNITY_VOLUME_VS_SHARE_DECOMPOSITION_V1`.
- Target Share Trajectory V1 remains ineligible in Weeks 1-4 and is not applied.
- Week-5 RB player-state room allocation remains prospective only and is not
  back-applied.

## Frozen pregame stage trace

Reconstruct deterministic share inputs using the exact historical cutoff and
existing protected stack.

### Common upstream stages

For each player:

1. `raw_history_share`
   - current-season strictly-prior average if at least one current-season game
     exists;
   - otherwise prior-season average;
   - exactly the historical PlayerForm consensus behavior.

2. `bayes_share`
   - existing `bayesian_v2` posterior;
   - record Bayesian evidence state, prior games, current games, effective N and
     posterior SD.

3. `rules_share`
   - existing football rule layer after any currently-authorized matchup/injury
     adjustment.

### RB carries

Record:

- raw `rush_share`;
- `bayes_rush_share`;
- `rules_rush_share`;
- final allocator carry probability after the existing top-five rushing pool,
  0.95 cap and residual bucket.

No Week-5 room shadow may enter.

### RB targets

Record:

- raw `tgt_share`;
- `bayes_tgt_share`;
- `rules_tgt_share`;
- explicit entitlement after the existing team-mass cap;
- final allocator target probability.

TE/WR specialist redistributions must not alter RB target entitlement.

### WR targets

Record:

- raw `tgt_share`;
- `bayes_tgt_share`;
- `rules_tgt_share`;
- post-M38 explicit entitlement;
- WR-R15 baseline entitlement;
- final WR-R15 entitlement;
- WR1 anchor flag;
- WR-R15 applied flag/route;
- WR-R15 residual score/delta where applicable;
- strictly-prior participation coverage already consumed:
  prior counts, prior1/prior3 same-team/any-team availability and snap evidence.

### TE targets

Record:

- raw `tgt_share`;
- `bayes_tgt_share`;
- `rules_tgt_share`;
- pre-TE-R5P explicit entitlement;
- final TE-R5P entitlement;
- TE-R5P applied flag;
- TE-R5P residual score/delta;
- strictly-prior participation coverage already consumed:
  prior counts, prior1/prior3 same-team/any-team availability and snap evidence.

## Outcome-side labels

Only after the full pregame trace is frozen, join the already-frozen parent:

- actual individual opportunity;
- actual team volume;
- actual player share;
- actual opportunity bin;
- linked yard/count error.

Realized share is never an upstream input.

## Required outputs

1. `player_share_stage_rows.csv`
2. `player_share_stage_error_summary.csv`
3. `player_share_evidence_coverage_summary.csv`
4. `player_share_specialist_effect_summary.csv`
5. `player_share_input_coverage_residual_summary.json`

## Required stage scoring

For each position/opportunity type and each applicable stage, report:

- rows;
- share MAE;
- share signed bias;
- share RMSE;
- predicted-vs-actual share correlation.

For WR/TE additionally report paired change in absolute share error:

- rules -> post-M38/pre-specialist;
- pre-specialist -> final specialist;
- rules -> final.

For RB report:

- raw -> Bayes;
- Bayes -> rules;
- rules -> final allocator.

## Required evidence-coverage diagnostics

Group without fitting by frozen pregame evidence states:

### Bayesian evidence
- `position_prior_only`
- `prior_only`
- `prior+current`

Also report exact `current_games` values 0,1,2,3 where observed in W1-4.

### WR/TE specialist participation evidence
Report:
- prior1 same-team available yes/no;
- prior3 same-team available yes/no;
- prior1 any-team available yes/no;
- prior3 any-team available yes/no.

For each evidence state report:
- rows;
- final share MAE/bias;
- actual-zero rate;
- active-player share MAE/bias;
- high-workload-bin share MAE/bias using already-frozen bins:
  - RB carries `15_PLUS`;
  - RB/WR/TE targets `09_PLUS`.

## Required focal-player diagnostic

For the frozen high-workload bins only, report each stage's mean predicted share,
mean actual share, bias and MAE.

The audit must answer:

1. Did the stack have player-specific prior/current evidence on the focal misses?
2. When evidence existed, did Bayesian shrinkage improve or worsen share error?
3. Did M38 / TE-R5P / WR-R15 improve or worsen focal-player share error?
4. Is the dominant RB carry-share failure upstream history/shrinkage or the final
   rushing allocator?
5. Does substantial final share error remain even with strong same-team
   participation history?

## Integrity gates

Fail closed if:

- ACT-only universe lineage is not exact;
- prediction features use target/future rows;
- sportsbook inputs enter;
- Week-5 RB allocation shadow enters W1-4;
- Target Share Trajectory is applied in W1-4;
- WR/TE specialist room/team conservation fails;
- WR-R15 changes the WR1 anchor;
- TE-R5P changes TE room mass;
- a stage is compared on different player identities without explicit pairing;
- outcome-side actual share is loaded before pregame stage trace freeze;
- any parameter is fit or threshold selected.

## Prohibited

- no production promotion;
- no Bayesian strength retune;
- no new share multiplier;
- no new RB router;
- no WR/TE specialist refit;
- no generic position calibration;
- no paid OddsAPI;
- no post-hoc rescue of closed science.

## Next-step rule

If focal misses mostly occur with missing player evidence, the next lane may
address evidence acquisition/availability.

If focal misses persist when player evidence is present and a specific existing
stage systematically worsens share error, that stage becomes the bounded next
research target.

If all existing stages help but substantial residual remains, a new player-state
share mechanism may be justified only under a separately frozen contract.
