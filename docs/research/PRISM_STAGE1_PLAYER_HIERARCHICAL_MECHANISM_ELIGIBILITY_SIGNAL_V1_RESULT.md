# PRISM Stage-1 Player Hierarchical Mechanism Eligibility Signal V1 — Result

Date: 2026-10-18

Status: **QUALIFIED PARTIAL — RB + TE QUALIFIED; QB + WR CLOSED; META-SIGNAL ONLY**

Model: **PRISM — Player Role & Individual Situation Model**

Frozen-before-scoring contract:
`docs/research/PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1_FREEZE.md`

Freeze commit:
`9ec910700f7febf9e5b1552c0b2d41c837051287`

Canonical execution:
- run `39099930673` — **SUCCESS**;
- exact head `2e61f48ceace967c2533765eeb631c0f421617e9`;
- job `117593934819` — **SUCCESS**;
- artifact `12126251820`;
- digest `sha256:9ef8c1c7e07fd4be381a745b5eb96d5ed976809d88aff478e623a5d74794f519`.

Overall disposition:

`PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_PARTIAL`

## 1. Frozen signal

For each family, the baseline meta-prediction was the strictly-prior
**other-player family mean** Stage-1 Brier benefit.

The candidate added the named player's own strictly-prior same-family Stage-1
mechanism-benefit history:

`hierarchical_expected_benefit = (n * player_prior_mean + 3 * family_prior_mean) / (n + 3)`

for `n >= 3`.

Same-week outcomes were excluded.

The target player was excluded from his own family prior.

No Stage-1 probability mapping was changed.

## 2. QB — CLOSED

Disposition:

`QB_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_CLOSED`

2024:
- 620 rows / 71 players;
- meta-AE gain +0.012843;
- Spearman -0.0327 -> +0.3359.

2025:
- 781 rows / 76 players;
- meta-AE gain +0.009125;
- player-cluster CI low +0.001526;
- game-cluster CI low +0.005652;
- equal-player CI low **-0.001136**;
- player improved share 57.58%;
- median +0.002525;
- p10 -0.029984.

Frozen failed gates:
- 2025 equal-player CI lower bound >0;
- 2025 candidate RMSE improvement.

Binding disposition: **CLOSED**.

Do not rescue with different shrinkage, support, recent history or QB subsets.

## 3. RB — QUALIFIED

Disposition:

`RB_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED`

2024:
- 998 rows / 113 players;
- meta-AE gain **+0.011039**;
- Spearman +0.2871 -> +0.5155.

2025:
- 1,244 rows / 129 players;
- meta-AE gain **+0.013680**;
- player-cluster CI low **+0.009691**;
- game-cluster CI low **+0.010407**;
- equal-player CI low **+0.009438**;
- player improved share **87.74%**;
- median +0.012606;
- p10 -0.002629.

All frozen gates passed.

## 4. WR — CLOSED

Disposition:

`WR_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_CLOSED`

2024:
- 1,457 rows / 189 players;
- meta-AE gain +0.009631;
- baseline Spearman +0.4604;
- candidate Spearman **+0.4157**.

2025:
- 1,889 rows / 209 players;
- meta-AE gain +0.007746;
- player-cluster CI low +0.004886;
- game-cluster CI low +0.005135;
- equal-player CI low +0.005899;
- player improved share **92.86%**;
- median +0.009387;
- p10 +0.003881.

The single binding failure is the frozen **2024 development Spearman
improvement** gate.

This is deliberately CLOSED even though many player-level diagnostics are
strong.

Do not rescue with a different ranking metric, threshold, WR subset, history
window or shrinkage.

## 5. TE — QUALIFIED

Disposition:

`TE_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED`

2024:
- 944 rows / 114 players;
- meta-AE gain **+0.002988**;
- Spearman +0.3491 -> +0.6549.

2025:
- 1,207 rows / 119 players;
- meta-AE gain **+0.001956**;
- player-cluster CI low **+0.000819**;
- game-cluster CI low **+0.001279**;
- equal-player CI low **+0.000978**;
- player improved share **87.50%**;
- median +0.003527;
- p10 -0.001692.

All frozen gates passed.

## 6. Scientific meaning

Stage-1 player-identity information is globally qualified across all four
families, but **future authority for that player-identity rule is not**.

The new player-specific layer separates families:

- RB: prior same-player Stage-1 utility is stable enough to predict future
  utility beyond the family prior;
- TE: same;
- QB: not stable enough on equal-player / RMSE confirmation;
- WR: strong player diagnostics but failed the frozen development ranking gate.

This is the intended architecture:

`qualified population mechanism != automatic permanent authority for every named player`.

## 7. Next authorized action

Freeze one projection-level Stage-1 authority bridge for **RB + TE only**.

Direct V1 intervention:

- meta-support <3 -> keep qualified Stage-1 `p_candidate`;
- supported and expected Stage-1 benefit >0 -> keep `p_candidate`;
- supported and expected benefit <=0 -> revert to family `p_baseline`.

Compare against universal Stage-1 candidate.

QB and WR remain universal Stage-1 in V1.

No production mutation is authorized.

## 8. Integrity

Canonical signal run:
- participation probabilities changed: **0**;
- Stage-1 formula changed: **false**;
- coefficients fit: **0**;
- thresholds searched: **0**;
- 2026 outcomes used: **0**;
- Week-6 outcomes used: **0**;
- sportsbook/OddsAPI: **0**;
- production changed: **false**.

Final disposition:

`PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_QUALIFIED_PARTIAL`.
