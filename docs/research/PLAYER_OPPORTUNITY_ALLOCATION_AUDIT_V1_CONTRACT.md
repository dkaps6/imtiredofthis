# PLAYER OPPORTUNITY ALLOCATION AUDIT V1 — FROZEN CONTRACT

Date: 2026-10-07  
Branch: `research-player-opportunity-allocation-audit-v1`

## Purpose

Localize the all-player replay's individual-player workload compression **before efficiency**.

This is not a new position-level model and not a production promotion lane. The unit of analysis is one pregame player-game.

## Frozen population

- Season: 2026
- Weeks: 1-4
- Same leakage-safe pregame football universes as `ALL_PLAYER_ALL_POSITION_REPLAY_V1`
- Positions: QB, RB/FB, WR, TE
- Full football universe, not sportsbook-conditioned

## Opportunity quantities

For each individual player-game, audit:

- QB: predicted pass attempts vs realized pass attempts
- RB/FB: predicted carries vs realized carries
- RB/FB: predicted targets vs realized targets
- WR: predicted targets vs realized targets
- TE: predicted targets vs realized targets

Receiving target and rushing carry predictions must come from the **exact canonical joint simulation allocation seam** after the protected entitlement order:

`M38 -> TE-R5P -> WR-R15 -> explicit entitlement simulation`

The audit may wrap the canonical multinomial allocator to record the already-generated target/carry counts. The wrapper must call the original allocator exactly once and return its output unchanged. It may not alter RNG routing, probabilities, shares, totals, or simulation values.

QB predicted attempts must use the existing football-only `mc_expected_pass_attempts` authority from the historical MC trace.

## Actual opportunity authority

Use canonical leakage-safe player logs **only after predictions are frozen**.

For a player in the pregame universe with no weekly stat row, realized opportunity is zero for the audited opportunity type. This is an outcome-side grading rule only.

Realized opportunities are diagnostic labels and are never allowed upstream as a feature.

## Required outputs

1. `player_opportunity_allocation_rows.csv`
2. `player_opportunity_allocation_summary.csv`
3. `player_opportunity_zero_state_audit.csv`
4. `player_opportunity_audit_summary.json`

Each player row must include:
- season/week/event/team/opponent/player/player key/position
- opportunity type
- predicted opportunity mean
- realized opportunity count
- error / absolute error
- predicted team opportunity total where applicable
- final player allocation probability/share where applicable
- residual/unmodeled probability where applicable
- source lineage
- zero-opportunity label

## Required diagnostics

By position + opportunity type:
- rows
- MAE
- median AE
- signed bias
- RMSE
- correlation of prediction vs actual
- zero-opportunity rate
- mean prediction when actual opportunity = zero
- actual-zero separation diagnostics using the continuous prediction distribution
- fixed descriptive zero-state cutpoints at predicted opportunities <0.5 and <1.0; report both precision/recall pairs, select neither, and do not promote either cutpoint
- low/middle/high realized-opportunity bins as descriptive failure localization only

Also report the relationship between opportunity error and the corresponding yard/reception error from the frozen all-player replay where identities overlap.

## Conservation / integrity gates

The run must fail closed if:
- any target-week/future football outcome is used upstream;
- sportsbook fields enter projection/allocation;
- target or carry wrapper changes any canonical simulation result;
- target allocation exceeds simulated pass attempts;
- carry allocation exceeds simulated rush attempts;
- protected TE/WR room mass conservation fails;
- the Week-5 RB allocation shadow appears in W1-4;
- any outcome-selected threshold or fit is introduced.

## Interpretation gate

This audit may identify whether workload compression originates in:
1. player participation / role selection,
2. target allocation,
3. carry allocation,
4. QB starter/attempt allocation,
5. or downstream efficiency.

It may **not**:
- fit a correction,
- choose a cutoff,
- promote a new feature,
- reopen closed generic position mean/width research,
- weaken the target-share trajectory eligibility rule,
- retrofit the Week-5 RB allocation shadow to Weeks 1-4.

Any candidate fix must be frozen in a separate prospective or historical contract after this audit is complete.

## Market boundary

- No paid OddsAPI.
- No sportsbook line, price, side, edge, fair probability, or betting outcome upstream.
