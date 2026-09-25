# One-Pass-State Integration V1 — Historical Full-Stack Result

Date: 2026-09-24

Disposition: **ONE_PASS_STATE_INTEGRATION_V1_FAILED_CLOSED**

This is a scientific failure under the frozen gates, not a mechanical failure.

## Authority

- branch: `research-one-pass-state-integration-v1`
- authoritative run: `36078172403`
- job: `107893965347`
- head: `dd7d2f98fdecabc29a51401e2e0f40d0eb12e0de`
- artifact: `10841142451`
- digest: `sha256:4892d126a4b30917b56e91d6299543e1155cfa4fee9795d1510edb11cf3517d7`
- parameters fit: `0`
- candidate variants scored: `1`
- sportsbook inputs used: `0`
- target-game outcomes used upstream: `0`
- production changed: **false**

## Selector provenance

The leakage-safe historical selector exactly reproduced preserved Phase-J 2025 authority:

- eligible 2025 rows: **440**
- max absolute selector-delta gap: **1.4210854715202004e-14**
- selected/unselected decision mismatches: **0**
- selected 2025 eligible rows: **412**

Historical Phase-C selector coverage remained intentionally fail-closed where the old research authority had no eligible QB row:

- 2024 football team-games: **544**
- 2024 selector-eligible: **444**
- 2024 selector-ineligible / canonical: **100**
- 2024 C2-selected: **294**
- 2025 football team-games: **544**
- 2025 selector-eligible: **440**
- 2025 selector-ineligible / canonical: **104**
- 2025 C2-selected: **412**

## Mechanical/integrity result

All major architecture guards passed:

- QB arrays baseline vs candidate bit-identical: **PASS**
- unselected-team receiver arrays bit-identical: **PASS**
- rush attempts bit-identical: **PASS**
- rush yards bit-identical: **PASS**
- target entitlement unchanged: **PASS**
- selected-team zero-reception + positive-yard violations: **0**
- selected-team QB/receiver conservation max gap: **2.2737367544323206e-13**
- RB Rush+Receiving Conservation V2 pathwise identity gap: **0**
- sportsbook inputs: **0**

Therefore the scientific result is interpretable.

## Receiving yards — positive mean signal

Pooled WR/TE/RB macro receiving-yard MAE:

`16.234770 -> 16.171014`

improvement:

**0.063756 yd**

The macro mean gate improved in both seasons:

- 2024: `16.477994 -> 16.433772`
- 2025: `15.995675 -> 15.912031`

Pooled by position:

- WR: `21.972342 -> 21.963744`
- TE: `15.593026 -> 15.574637`
- RB: `11.138943 -> 10.974662`

RB carried the strongest signal.

Among C2-selected receiver rows:
- candidate mean projection shift: approximately **-0.90 yd**
- candidate closer on receiving-yard rows: approximately **56.2%**
- RB candidate closer rate: **63.94%**
- TE candidate closer rate: **54.68%**
- WR candidate closer rate: **51.94%**

## Receiving yards — decisive protection failures

Pooled macro p90 absolute error worsened:

`34.326908 -> 34.636564`

High-entitlement Q4 worsened:

- MAE: `26.541481 -> 26.584176`
- p90 AE: `55.546741 -> 55.664723`
- bias became more negative: `-6.2301 -> -7.0968`

Entitlement-scope diagnostic among selected rows:

- Q1 low: mean absolute-error change approximately **-0.334 yd**
- Q2: approximately **-0.076 yd**
- Q3: approximately **-0.018 yd**
- Q4 high: approximately **+0.064 yd**

So the team-level C2 reconciliation helped low-volume receivers most and became slightly harmful at the highest-entitlement end.

Large Q4 regressions included repeated alpha-WR cases where the shared-state team adjustment materially lowered an already-strong player projection, including CeeDee Lamb and Justin Jefferson. These are diagnostics only; no carveout rescue is authorized.

## Receptions

Pooled macro receptions MAE was essentially flat but slightly worse:

`1.290311 -> 1.290509`

Season macro:
- 2024: `1.321120 -> 1.321142`
- 2025: `1.259955 -> 1.260327`

Pooled p90 AE improved slightly:

`2.765526 -> 2.763117`

The frozen mean-MAE protection gates therefore failed even though the magnitude was tiny.

## RB rush+receiving

RB combo mean accuracy improved in both seasons:

- 2024 MAE: `25.978597 -> 25.944454`
- 2025 MAE: `25.167190 -> 25.105659`
- pooled MAE: `25.573039 -> 25.525207`

But pooled p90 worsened:

`55.534860 -> 55.992912`

Therefore the dependent RB tail-protection gate failed.

## Frozen gate failures

Scientific gates that failed:

- pooled receiving-yard macro p90 nonworse
- high-entitlement Q4 receiving-yard MAE nonworse
- high-entitlement Q4 receiving-yard p90 nonworse
- pooled receptions macro MAE nonworse
- 2024 receptions macro MAE nonworse
- 2025 receptions macro MAE nonworse
- pooled RB rush+receiving p90 nonworse

All mechanical/integrity gates passed.

## Interpretation

The read-only structural audit was correct: current production contains materially divergent QB and receiver Monte Carlo states.

But **wholesale receiver replacement with the C2 completed-pass state is not stable enough for production**.

The most important scientific signal is directional:

1. team-level pass-state reconciliation improves average receiving-yard accuracy;
2. improvement is strongest for RB / low-entitlement receivers;
3. the same adjustment can over-correct high-entitlement alpha receivers;
4. receptions are already strong enough that wholesale replacement has essentially zero mean benefit;
5. a future mechanism should therefore not treat every receiver's existing marginal authority as equally uncertain.

This result does not authorize a WR/RB carveout, Q4 exemption, threshold search, or player-specific rescue of V1.

## Closure

`ONE_PASS_STATE_INTEGRATION_V1_FAILED_CLOSED`

No production change.

The next research lane, if pursued, must be a genuinely separate mechanism grounded in uncertainty / authority of player-level receiver projections rather than a post-hoc carveout of this candidate.
