# PRISM Stage-2 Player Hierarchical Mechanism Eligibility Signal V1 — Result

Date: 2026-10-18

Status: **QUALIFIED PARTIAL — RB RUSHING + RB RECEIVING QUALIFIED; TE RECEIVING CLOSED ON FROZEN TAIL-HARM GATE; NO PROJECTION CHANGE**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen-before-scoring contract:
`docs/research/PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1_FREEZE.md`

Final pre-score freeze:
`8a94c8cae1448520acf283e52c641d3bd631de3a`

Canonical execution:
- run `39099164620` — **SUCCESS**;
- exact head `36b34a1b392d4d6196b72116b0ef5b88273906bd`;
- job `117591687262` — **SUCCESS**;
- artifact `12125359950`;
- digest `sha256:efa919e2590e7409e1d989b59cafe293d6d4831dfb434635df8fb5d15143e95b`.

Overall disposition:

`PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_PARTIAL`

## 1. What was tested

For canonical-positive Stage-2 rows only, the baseline meta-prediction was the
strictly-prior **other-player current-role prior**.

The candidate added the named player's own strictly-prior positive-entitlement
Stage-2 benefit history:

`hierarchical_expected_benefit = (n * player_prior_mean + 3 * role_prior_mean) / (n + 3)`

for `n >= 3`.

The role prior:
- used the same Stage-2 cell;
- used current canonical entitlement-rank state;
- excluded the target player's own history;
- excluded same-week history;
- excluded structural canonical-zero rows.

No coefficient, threshold, window or shrinkage strength was fit after outcomes.

Target:
`delta_crps = canonical_baseline_crps - universal_stage2_crps`.

This asked whether the named player's prior OOS **mechanism utility** predicts
future utility beyond the role prior.

## 2. RB rushing — QUALIFIED

Disposition:

`RB_RUSHING_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED`

2024 development:
- 867 rows / 120 players;
- meta-AE improvement **+0.153555**;
- baseline meta-AE 0.612817 -> candidate 0.459262;
- RMSE 0.897808 -> 0.712050;
- Spearman **-0.0156 -> +0.4355**;
- equal-player mean improvement +0.153500;
- 77.89% of evaluable named players improved.

2025 independent confirmation:
- 1,154 rows / 137 players;
- meta-AE improvement **+0.099254**;
- baseline meta-AE 0.554661 -> candidate 0.455407;
- RMSE 0.818714 -> 0.737440;
- Spearman **-0.0136 -> +0.3826**;
- player-cluster CI low **+0.066250**;
- game-cluster CI low **+0.076779**;
- equal-player CI low **+0.064247**;
- equal-player mean improvement +0.098917;
- median named-player improvement +0.065768;
- **68.63%** of evaluable named players improved;
- p10 named-player improvement -0.095180;
- frozen tail floor -0.137618.

All frozen gates passed.

Interpretation:
even in the Stage-2 cell that was already broadly successful across named RBs,
the named player's own prior mechanism-utility history adds substantial
information about future mechanism utility.

## 3. RB receiving — QUALIFIED

Disposition:

`RB_RECEIVING_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED`

2024 development:
- 952 rows / 126 players;
- meta-AE improvement **+0.027357**;
- baseline meta-AE 0.097185 -> candidate 0.069828;
- RMSE 0.158042 -> 0.126536;
- Spearman **+0.2051 -> +0.4558**;
- equal-player mean improvement +0.026905;
- 83.50% of evaluable named players improved.

2025 independent confirmation:
- 1,278 rows / 142 players;
- meta-AE improvement **+0.019477**;
- baseline meta-AE 0.082225 -> candidate 0.062749;
- RMSE 0.130907 -> 0.113533;
- Spearman **+0.0877 -> +0.4842**;
- player-cluster CI low **+0.013832**;
- game-cluster CI low **+0.016312**;
- equal-player CI low **+0.012776**;
- equal-player mean improvement +0.018838;
- median named-player improvement +0.012802;
- **79.65%** of evaluable named players improved;
- p10 named-player improvement -0.007275;
- frozen tail floor -0.020649.

All frozen gates passed.

This is the clearest first confirmation of the user's new architecture:
the same Stage-2 RB receiving mechanism does **not** have equal future utility
for every named RB, and the player's own prior OOS mechanism history helps
predict that difference beyond his current role prior.

## 4. TE receiving — CLOSED

Disposition:

`TE_RECEIVING_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_CLOSED`

2024 development:
- 936 rows / 121 players;
- meta-AE improvement +0.011133;
- RMSE 0.113554 -> 0.093694;
- Spearman +0.3018 -> +0.5130.

2025 independent confirmation:
- 1,215 rows / 127 players;
- meta-AE improvement +0.006943;
- player-cluster CI low +0.001380;
- game-cluster CI low +0.004873;
- equal-player CI low +0.002017;
- median named-player improvement +0.007419;
- 75.89% of evaluable named players improved;
- p10 named-player improvement **-0.012228**;
- frozen tail floor **-0.010402**.

Every frozen gate passed **except**:

`confirmation_tail_harm_guard`.

The failure is binding.

Do not rescue TE with:
- a different pseudo-count;
- higher support floor;
- rank subsets;
- alternate history windows;
- positive-only players;
- player-name exclusions;
- alternate meta loss;
- soft/hard threshold changes.

The universal TE Stage-2 parent remains qualified.
Only this V1 player-specific TE mechanism-utility signal is closed.

## 5. Scientific meaning

PRISM now has direct historical evidence that mechanism authority can be a
**named-player state** rather than a permanent position-wide constant.

For RB rushing and RB receiving:

`role prior -> same-player prior mechanism evidence -> future mechanism utility`

is materially better than the role prior alone.

This does **not** prove the correct final projection integration.

It authorizes one separate bridge question:

> If a supported named RB has negative hierarchical expected utility for the
> otherwise-qualified Stage-2 active-role mechanism, can PRISM reduce that
> player's Stage-2 authority while preserving finite opportunity and improving
> actual player opportunity distributions?

## 6. Next boundary

Authorized next:
- RB rushing + RB receiving only;
- conservation-safe Stage-2 authority integration;
- low-support player falls back to universal Stage-2;
- TE remains universal Stage-2 in V1;
- QB remains closed;
- WR remains source-blocked.

No free per-player coefficients.

No projection production mutation without a separately successful bridge and a
separate deployment decision.

## 7. Integrity

Canonical run:
- projection values changed: **0**;
- Stage-2 formula changed: **false**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook inputs: **0**;
- OddsAPI calls: **0**;
- production changed: **false**.

Final disposition:

`PRISM_STAGE2_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_PARTIAL`.
