# WR Post-M38 Error Decomposition — Canonical Result

## Status

**Completed successfully.** This document records the frozen post-M38 diagnostic result and the mechanical lineage required to reproduce it.

No production projection logic was changed. No sportsbook or market information was used upstream. No frozen scientific threshold, component definition, prior, slice, or routing gate was changed after seeing results.

## Canonical lineage

- Exact M38 parent: `b98518d97b3038f471aee9ae3201009b2c70bb29`
- Frozen plan: `520d94e843cd595ce23a3c90cb18b134822d4a65`
- Frozen Shapley implementation: `e1b3d18fc4a372e8efd5d617bd20adcf16e9643f`
- Workflow launch SHA: `f24ae9613e179afa1b2dd19b11b721a6576dc57d`
- Final mechanical-wrapper SHA: `5475933876d86baab6531464f7f1e6846c479865`
- Canonical successful run: `34044780465`
- Canonical successful job: `101517783210`
- Artifact: `9992822311`
- Artifact SHA256: `66c18223f8fc83a81ee0aed423488719912d9f038eec8c1f1bdf8b0f57c56239`
- Artifact size: 442,934 bytes, 6 files

## Mechanical failure lineage

The first run (`34042923907`, job `101512793467`) failed before scientific disposition because Isaiah Bond, CLE, 2025 W16 had **0 recorded targets but +21 receiving yards**. The frozen targets × catch-rate × YPR factorization cannot mathematically reproduce that provider/stat anomaly.

This is the same non-factorizable source anomaly already identified in WR-ND1. The repair therefore carried forward the ND1 integrity treatment rather than changing the experiment:

- exclude the affected WR row from the **target-game receiving-yards evaluation view only**;
- preserve the complete historical weekly-log/PBP universe for all strict-prior construction;
- do not alter M38 reconstruction, component definitions, priors, thresholds, slices, or routing gates.

An intermediate wrapper run (`34044383003`, job `101516704899`) correctly bypassed the PBP factorization error but removed the anomaly from the wrong side of the join, producing a missing-player-id integrity failure. The final wrapper at `5475933876d86baab6531464f7f1e6846c479865` scopes the exclusion to the matching M38 `rec_yards` evaluation prediction row while leaving historical logs untouched.

The successful run audited exactly **1 excluded evaluation row**: Isaiah Bond, CLE W16, 0 targets, 0 receptions, +21 receiving yards. `scientific_protocol_changed = false`.

## Exact M38 parent parity

The successful diagnostic reproduced the exact post-M38 parent reference:

- n = 4,647
- receiving-yards MC MAE = **17.099904733366**
- RMSE = 25.196099510686
- bias = -5.238640833495
- correlation = 0.567945850835

Scientific WR evaluation population: **2,130 player-games**.

## Frozen four-component decomposition

The diagnostic decomposes receiving-yard error into:

1. `OPPORTUNITY` — target count
2. `CONVERSION` — receptions per target
3. `NON_EXPLOSIVE_EFFICIENCY` — receiving yards from receptions <20 yards, per reception
4. `EXPLOSIVE_YARDAGE` — receiving yards from receptions >=20 yards, per reception

Two strict-prior explosive/non-explosive representations were frozen before results:

- `LEAGUE_PRIOR`
- `PLAYER8_SHRUNK`

### ALL_WR result

| Scheme | Component | Shapley MAE recovery | Share of parent headroom |
|---|---|---:|---:|
| LEAGUE_PRIOR | **OPPORTUNITY** | **9.203630** | **41.8035%** |
| LEAGUE_PRIOR | EXPLOSIVE_YARDAGE | 6.535401 | 29.6843% |
| LEAGUE_PRIOR | CONVERSION | 3.754778 | 17.0545% |
| LEAGUE_PRIOR | NON_EXPLOSIVE_EFFICIENCY | 2.522578 | 11.4577% |
| PLAYER8_SHRUNK | **OPPORTUNITY** | **9.214435** | **41.8526%** |
| PLAYER8_SHRUNK | EXPLOSIVE_YARDAGE | 6.544892 | 29.7274% |
| PLAYER8_SHRUNK | CONVERSION | 3.762618 | 17.0901% |
| PLAYER8_SHRUNK | NON_EXPLOSIVE_EFFICIENCY | 2.494443 | 11.3299% |

## Frozen disposition

**`OPPORTUNITY_DOMINANT`**

- Top component under both strict-prior schemes: `OPPORTUNITY`
- Minimum overall opportunity share: **41.8035%**
- Frozen opportunity gate required top component + >=35%; it passes cleanly.

Explosive yardage does **not** receive the global `EXPLOSIVE_YARDAGE_DOMINANT` disposition. Its overall share is 29.6843% / 29.7274%, and it is not the top overall component. The frozen threshold must not be changed retroactively.

## Critical tail finding

Although opportunity is the global leader, **explosive yardage is the dominant error source in the large positive ceiling misses**.

### LEAGUE_PRIOR

- `ACTUAL_100_PLUS` (n=130): EXPLOSIVE **36.1885 / 45.94%**, OPPORTUNITY 20.8384 / 26.45%
- `UNDER_25_PLUS` (n=472): EXPLOSIVE **21.1118 / 40.61%**, OPPORTUNITY 14.0802 / 27.08%
- `UNDER_50_PLUS` (n=191): EXPLOSIVE **33.7854 / 45.03%**, OPPORTUNITY 21.6434 / 28.84%

### PLAYER8_SHRUNK

- `ACTUAL_100_PLUS`: EXPLOSIVE **46.05%**
- `UNDER_25_PLUS`: EXPLOSIVE **40.77%**
- `UNDER_50_PLUS`: EXPLOSIVE **45.09%**

The frozen explosive special gate required top under both schemes plus either >=30% overall or >=50% of the 50+ underprediction-tail headroom. It therefore does not pass the formal global routing gate, despite being clearly important in the ceiling tail.

## Overprojection asymmetry

For `OVER_25_PLUS` (n=156), opportunity is dominant:

- OPPORTUNITY ~**51.1%**
- CONVERSION ~25.9%
- EXPLOSIVE_YARDAGE ~26.6–26.8%
- NON_EXPLOSIVE_EFFICIENCY is slightly negative (~-3.6% to -3.8%)

This supports asymmetric treatment: the mechanism behind large false-low receiving-yard games is not the same as the mechanism behind large false-high games.

## Role / phase stability

Opportunity remains the largest component for WR1, WR2, WR3, and WR4+ under both prior schemes.

Representative LEAGUE_PRIOR shares:

- WR1: opportunity 44.29%, explosive 29.70%
- WR2: opportunity 43.92%, explosive 28.67%
- WR3: opportunity 37.79%, explosive 31.50%
- WR4+: opportunity 38.55%, explosive 29.01%

Phase slices are similar:

- W1: opportunity 44.37%, explosive 32.34%
- W2–18: opportunity 41.63%, explosive 29.50%
- W13–18: opportunity 46.34%, explosive 28.61%

## Reconciliation with WR-ND1

WR-ND1 concluded `YARDS_PER_TARGET_DOMINANT` when opportunity was split into three separate components:

- TEAM_TARGET_VOLUME = 2.762802
- WR_TARGET_MASS = 2.042554
- WITHIN_WR_ALLOCATION = 7.651051
- YARDS_PER_TARGET = 9.566137

The combined ND1 opportunity-side attribution is **12.456407**, which is larger than YPT. Therefore the new four-way result is not a contradiction: this diagnostic collapses all target-count error into one `OPPORTUNITY` component, while ND1 split it into three pieces.

ND1 also established the key opportunity routing clue: **WITHIN_WR_ALLOCATION** is the largest individual opportunity component, and it dominates the 10+ target tier (22.597776 vs 13.221916 for YPT) plus the broader false-low lane.

## Recovered WR-ND2 result

The successful WR-ND2 run (`34001617221`, job `101401378925`) decomposed YPT into catch rate vs yards per reception.

- YARDS_PER_RECEPTION MAE recovery = **6.026**
- CATCH_RATE MAE recovery = **4.496**
- YPR was larger for WR1, WR2, and WR3
- YPR represented **57.27%** of the positive efficiency pair

The frozen dominance threshold was 60%, so the official ND2 disposition is:

**`MIXED_WR_EFFICIENCY_MECHANICS`**, with a clear YPR lean.

This is consistent with the present diagnostic: among the three efficiency-side pieces, explosive yardage is much larger than conversion or ordinary non-explosive efficiency.

## Scientific interpretation

The post-M38 evidence now supports **two distinct WR error lanes**:

### 1. Mean / opportunity lane

The canonical global route is opportunity. The highest-priority mean-projection research question is therefore **dynamic WR target entitlement / within-WR allocation**, not another generic hierarchy multiplier search.

The next research should ask whether strict-prior pregame information can identify when a receiver should materially deviate above or below the static M38 hierarchy allocation, especially in high-volume/10+ target games.

This must preserve the existing anti-duplication constraints:

- do not retune M38 multipliers;
- do not rerun M31–M32 generic target-pool pruning;
- do not disguise prior failed feature sets under a new algorithm;
- do not infer fake WR-CB assignments;
- do not use sportsbook information upstream.

### 2. Ceiling / distribution lane

Explosive yardage is not the global mean winner, but it is the strongest mechanism in actual 100+ yard games and 25+/50+ yard underprediction tails. It should be preserved as a **distribution-shape / ceiling-risk lane**, not discarded.

A later frozen diagnostic should separate explosive gains into football mechanics such as completed air yards / depth-driven gains versus yards after catch, with special attention to joint WR–QB tail behavior. That is the appropriate place to test whether a WR explosive-play mechanism explains large QB passing-yard overs.

## Routing decision

For the **next mean-research branch**, prioritize **dynamic within-WR target entitlement/opportunity** because the frozen global disposition is `OPPORTUNITY_DOMINANT` and ND1 already localizes the opportunity problem to within-WR allocation / high-volume entitlement.

Preserve explosive-yardage research as a separate, explicitly documented **tail/distribution lane**. Do not promote a predictive explosive correction from this diagnostic alone.
