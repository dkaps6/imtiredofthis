# Pass-Catcher Efficiency-Difficulty Trust Selectivity V1 — Result

Date: 2026-10-15

Status: **COMPLETE / WR QUALIFIED / TE CLOSED / RESEARCH ONLY**

Frozen contract:
- `docs/research/PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_V1_FREEZE.md`

Canonical authority:
- workflow: `Research Pass Catcher Efficiency Difficulty Trust Selectivity V1`
- run: `38826418315` — **SUCCESS**
- job: `116734822024`
- exact scored head: `c15f70f3bacf9d0d7f301d334419841ad6d9d492`
- artifact: `12015303274`
- digest: `sha256:cbab5ac5b424861d7dfd1fc8e965b6ad579936eab5e53820019392e25f2b6cdf`

Overall disposition:
`PASS_CATCHER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_PARTIAL`

## Scientific question

This V1 did **not** retest raw YPT volatility and did not alter any point
projection or distribution.

It asked whether the named player's own strictly-prior **efficiency projection
difficulty** predicts reliability/fragility of the existing final receiving-yard
projection after the already-qualified opportunity-difficulty state is
controlled.

Primary frozen statistic:

`partial_rank_corr(efficiency_difficulty, final_abs_error | opportunity_difficulty)`

All variables were ranked within exact season/week before inference.

No sportsbook, OddsAPI, 2026 outcomes, predictive fit, mean change, width change,
or production mutation entered the experiment.

## WR — QUALIFIED

Disposition:
`WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Exact parent:
- `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`
- run `37634044900`
- artifact `11487359185`
- 3,224 rows / 211 WRs.

### 2023 development

- rows: **1,357**
- WRs: **160**
- primary partial rank association: **+0.16473**
- efficiency-mechanism partial association: **+0.18671**
- HIGH_EFFICIENCY_TRUST final abs error: **14.65 yd**
- LOW_EFFICIENCY_TRUST final abs error: **33.09 yd**
- overall: **23.46 yd**

### 2024 independent confirmation

- rows: **1,867**
- WRs: **183**
- primary partial rank association: **+0.16065**
- efficiency-mechanism partial association: **+0.11721**
- HIGH_EFFICIENCY_TRUST final abs error: **15.72 yd**
- LOW_EFFICIENCY_TRUST final abs error: **29.60 yd**
- overall: **23.09 yd**

Confirmation dependence-aware inference:
- player-cluster 95% CI: **[+0.11340, +0.20499]**
- team-week cluster 95% CI: **[+0.11521, +0.20595]**
- both P(positive) = **1.000**

### Pooled WR

- rows: **3,224**
- WRs: **211**
- primary partial rank association: **+0.16238**
- efficiency-mechanism partial association: **+0.14602**
- HIGH_EFFICIENCY_TRUST final abs error: **15.27 yd**
- LOW_EFFICIENCY_TRUST final abs error: **31.07 yd**
- overall: **23.25 yd**
- player-cluster 95% CI: **[+0.12437, +0.19757]**
- team-week cluster 95% CI: **[+0.12851, +0.19455]**

Every frozen WR gate passed.

Interpretation:
same-WR strictly-prior efficiency-projection difficulty contains incremental
named-player reliability information beyond opportunity difficulty. This is a
distinct second WR trust axis around the unchanged WR-R15 receiving-yard
projection.

It does **not** mean high trust = OVER or low trust = UNDER/fade.

## TE — exact V1 CLOSED

Disposition:
`NO_ACTIONABLE_TE_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_SELECTIVITY`

Exact parent:
- `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`
- run `37631011196`
- artifact `11486082927`
- 2,005 rows / 121 TEs.

### 2024 development

- primary partial rank association: **+0.09740**
- efficiency-mechanism partial association: **+0.04460**
- HIGH_EFFICIENCY_TRUST final abs error: **11.03 yd**
- LOW_EFFICIENCY_TRUST final abs error: **23.20 yd**

### 2025 independent confirmation

- primary partial rank association: **+0.03121**
- efficiency-mechanism partial association: **+0.05548**
- HIGH_EFFICIENCY_TRUST final abs error: **8.74 yd**
- LOW_EFFICIENCY_TRUST final abs error: **20.20 yd**

The direction remained favorable, but the frozen independent-confirmation
dependence-aware gates failed:
- 2025 player-cluster 95% CI:
  **[-0.03681, +0.09582]**
- 2025 team-week cluster 95% CI:
  **[-0.03273, +0.09362]**

Those were the only failed TE gates.

Do not rescue this exact TE construction through pooled seasons, different
windows, thresholds, TE subgroups, removal of opportunity control, or alternate
partial-correlation transforms.

TE opportunity trust remains separately qualified and Week-6 frozen.

## Integrity

WR max decomposition gap:
- `5.68e-14`

TE max decomposition gap:
- `4.26e-14`

For both positions:
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- 2026 outcomes read: **0**
- predictive models fit: **0**
- projection/distribution values changed: **0**
- production changed: **false**

## Follow-through

Because WR qualified, the only immediate prospective follow-through was to
freeze a separate Week-6 WR efficiency-trust shadow before target-week outcomes.

That lock is recorded in:
`docs/research/WR_PLAYER_EFFICIENCY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1_RESULT.md`.

Opportunity trust and efficiency trust remain separate axes. No composite rule
or production selector is authorized.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
