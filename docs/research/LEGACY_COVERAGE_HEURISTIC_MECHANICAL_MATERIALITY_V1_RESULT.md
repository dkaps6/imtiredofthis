# Legacy Coverage Heuristic Mechanical Materiality V1 — Result

Date: 2026-09-29  
Status: **COMPLETE — OUTCOME-FREE MECHANICAL AUDIT ONLY**  
Branch: `research-week3-postmortem-execution-v1`

## Authority

Exact preserved Week-3 paid production artifact:
- run `36293274478`
- artifact `10923570170`
- artifact name `run_36293274478`
- digest `sha256:5a3d4f64592c70553e66dd51bb3bff45263d2900f4d270e370353fa60ea1c480`
- production head `0982b62276303403e2ca58b16e6f4fc3e041f65d`

Audit implementation:
`scripts/research/audit_legacy_coverage_heuristic_mechanical_materiality_v1.py`

Week-3 outcomes used: **0**  
Sportsbook fields used as football-model inputs: **0**  
Production changes: **0**

## Question

Is the grandfathered pre-research `coverage_penalty()` rule mechanically large
enough in the current promoted receiving stack that it still deserves explicit
scientific qualification?

Current rule:

```python
if tough_shadow or heavy_man:
    ypt *= 0.94
    share *= 0.92
if heavy_zone:
    ypt *= 1.04
    share *= 1.06
```

The rule executes before explicit target entitlement, M38, TE-R5P, WR-R15 and
joint Monte Carlo.

This audit removes only those multipliers in a deterministic counterfactual.
It does not grade the counterfactual against outcomes.

## Reproduction gates

The audit reconstructs the exact frozen M38 / explicit-entitlement state from
the archived `rules_tgt_share` values before accepting any counterfactual.

- max M38/current-baseline reproduction gap: **1.94e-16**
- WR-R15 control residual reproduction max gap: **7.77e-16**
- WR-R15 control final-entitlement reproduction max gap: **1.80e-16**
- WR-R15 anchor identities changed under the counterfactual: **0**

Therefore the deterministic seams used for the materiality calculation reproduce
the frozen current state to numerical tolerance.

## Current Week-3 activation state

Frozen football universe:
- WR rows: **161**
- WR rows actually receiving a coverage adjustment with finite YPT: **131**
- fraction of WR rows affected: **81.4%**
- affected teams: **27 of 30**
- WR rows with a nonmissing `primary_cb`: **0**

The Week-3 heuristic is therefore not behaving primarily like a selective
WR-vs-CB matchup rule.

All 131 active rows were the generic `heavy_zone` branch.

The three teams not receiving the generic heavy-zone adjustment were CAR, HOU
and PHI. No Week-3 WR had a frozen primary-CB/shadow assignment.

Among WR identities that appeared in a priced receiving-yards market:
- priced WR identities: **81**
- affected by the coverage heuristic: **71**

## Direct WR effect

Remove-only counterfactual, after exact M38 + WR-R15 replay:

Affected WR final target entitlement:
- mean change: **-2.63%**
- median change: **-2.64%**

Affected WR YPT:
- exact change from removing the 1.04 zone multiplier: **-3.846%**

The deterministic receiving-mean kernel
`final_target_entitlement * rules_ypt` therefore changes:

- mean: **-6.37%**
- median: **-6.39%**
- range: **-12.38% to -3.81%**

This is mechanically material. It is not a rounding-level feature.

## Conservation spillover into other positions

Because explicit target opportunity is capped/conserved at the team level, the
generic WR zone boost also removes target mass from non-WRs.

On the 27 affected teams, removing the legacy coverage rule changes final
position target pools approximately:

- WR pool: **-2.30% mean** / **-2.30% median**
- TE pool: **+2.91% mean** / **+3.10% median**
- RB/FB pool: **+2.91% mean** / **+3.10% median**

The total modeled player target mass remains exactly conserved at 0.95 to
floating-point tolerance.

Therefore this old WR heuristic is also a TE/RB opportunity rule indirectly.

## Prior historical evidence already in repo

Do not mistake mechanical materiality for predictive value.

The previously executed 2025 whole-slate `no_coverage` ablation
(run `32316784561`) was near-neutral:
- rec_yards MAE delta vs full: approximately **-0.0395** when coverage was
  removed;
- receptions delta: approximately **-0.0022**;
- both were classified near-neutral.

That older test predates the full modern promoted receiving stack and is not
sufficient authority to remove the rule today, but it also provides no positive
historical validation for keeping the coefficients.

## Disposition

`LEGACY_COVERAGE_HEURISTIC_MECHANICALLY_MATERIAL_LOW_SELECTIVITY_UNVALIDATED`

Meaning:
1. the rule is definitely active and materially changes current football means;
2. its Week-3 action is dominated by a generic team-zone threshold rather than
   player matchup information;
3. it materially reallocates opportunity across WR vs TE/RB;
4. the closest historical ablation was null/near-neutral;
5. there is still no rigorous current-stack evidence authorizing removal.

## What is NOT authorized

- do not remove or change the production rule from this audit;
- do not retune the 0.50/0.60 thresholds;
- do not retune 0.94/0.92/1.04/1.06;
- do not use Week-3 realized outcomes to choose a replacement;
- do not create WR/TE/RB carveouts;
- do not infer that the rule caused Week-3 misses.

## Next action

Freeze a **prospective remove-only current-stack shadow** beginning Week 4+.

The shadow must:
- change only `coverage_penalty()` to identity multipliers;
- preserve all other football science and sportsbook separation;
- run through current M38 + TE-R5P + WR-R15 + joint simulation;
- capture cross-position consequences;
- be frozen before outcomes;
- remain shadow-only until a predeclared support/inference gate is met.

Because specialist RNG isolation is currently under repair, any full-array A/B
must use the repaired/frozen semantic RNG routing or otherwise prove that
unrelated arrays cannot drift from random-stream path changes.
