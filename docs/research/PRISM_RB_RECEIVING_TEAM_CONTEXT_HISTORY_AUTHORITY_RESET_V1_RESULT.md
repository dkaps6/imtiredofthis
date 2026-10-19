# PRISM RB Receiving Team-Context History-Authority Reset V1 — Result

Date: 2026-10-19

Status: **PREDICTIVE CLOSED — RESEARCH ONLY**

## Canonical execution

- predictive freeze: `c4226d642ac5b7ac4f20260949c9eddd726c19b8`
- run: `39254999922` — SUCCESS
- exact head: `c174184353789e935905c3a400ff062b2a3f3223`
- artifact: `12188016034`
- digest: `sha256:caeb537661afd2916ecf03ba044faf9624131cae55ea942df016b5150b4f6855`

Disposition:

`PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_PREDICTIVE_CLOSED`

## Parent state preserved

Source parent:

`PRISM_RB_RECEIVING_TEAM_CONTEXT_HISTORY_AUTHORITY_RESET_V1_SOURCE_READY`

Current qualified baseline preserved:

`RB_RECEIVING_PLAYER_SPECIFIC_STAGE2_AUTHORITY_BRIDGE_QUALIFIED`

This result does not revoke the team-context boundary source or the existing qualified RB-receiving player-specific authority.

## Frozen intervention

On the first canonical post-team-change RB-receiving row, only when old-history-derived same-player authority was actually active:

- baseline = current qualified RB-receiving named-player architecture;
- candidate = reset the transitioned target RB's old-history-derived Stage-1/Stage-2 player authority to the frozen Stage-1 role prior;
- current-team entitlement `authority_q` unchanged;
- all teammate `p_effective` unchanged;
- finite room `N` unchanged;
- exact room enumeration/conservation unchanged.

No soft blend, lag cutoff, transition-kind carveout, subgroup rescue, fitting or threshold search.

## 2024 development

- actionable events / stable IDs: 22 / 22
- target CRPS gain: **-0.059970**
- target expected-count AE gain: **-0.055789**
- room allocation-error gain: **-0.000420**
- equal-player mean / median / p10 CRPS gain:
  **-0.059970 / -0.051931 / -0.193545**
- improved / tied / worsened:
  **31.82% / 0.00% / 68.18%**
- player-cluster 95% CI:
  **[-0.113764, -0.008835]**
- game-cluster 95% CI:
  **[-0.118940, -0.000950]**
- equal-player 95% CI:
  **[-0.113034, -0.009797]**

Development clearly rejected the reset.

## 2025 independent confirmation

- actionable events / stable IDs: 17 / 17
- target CRPS gain: **+0.011032**
- target expected-count AE gain: **+0.004037**
- room allocation-error gain: **-0.000668**
- equal-player mean / median / p10 CRPS gain:
  **+0.011032 / +0.070381 / -0.138202**
- improved / tied / worsened:
  **52.94% / 0.00% / 47.06%**
- player-cluster 95% CI:
  **[-0.056543, +0.078002]**
- game-cluster 95% CI:
  **[-0.048120, +0.080926]**
- equal-player 95% CI:
  **[-0.055220, +0.079124]**

The 2025 central target-player direction was positive, but uncertainty crossed zero and room allocation worsened.

## Failed frozen gates

- development equal-player mean positive;
- development improved share > worsened;
- development target-count AE nonworse;
- development target CRPS positive;
- development median player nonworse;
- development room allocation nonworse;
- 2025 player-cluster CI low > 0;
- 2025 game-cluster CI low > 0;
- 2025 equal-player CI low > 0;
- 2025 room allocation nonworse.

The frozen tail-harm floor passed; it does not rescue the mechanism.

## Interpretation

The team-context boundary is real and remains source-certified.

What failed is one exact intervention:

> automatically resetting an RB's established same-player receiving authorities to the role prior at the first canonical new-team receiving observation.

In 2024, the established named-player authority was materially better than the reset. 2025 did not provide independent confirmation for the opposite rule.

Therefore:
- do not encode a generic new-team RB-receiving history reset;
- do not weaken it into a soft reset after seeing these outcomes;
- do not carve out offseason/in-season or lag subsets post hoc;
- preserve the already-qualified RB-receiving named-player authority.

This is exactly the additive PRISM philosophy: a team-context boundary tells the model to ask an authority question. It does not force the answer to be "forget the player's history."

No production change.
No sportsbook/OddsAPI input.
No 2026 outcome use.
No yards/TD scoring.
