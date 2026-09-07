# QB-R1 Player + Context Mechanism Router — Result

## Canonical evidence
- Branch: `research-qb-r1-player-context-mechanism-router`
- Scientific run: `34072312396`
- Job: `101591795688`
- Tested SHA: `2a4d38d8b743f6b21b2ab95d112767adc3f697a0`
- Artifact: `10000858730` (`qb-r1-player-context-mechanism-router`)
- Artifact SHA256: `983ea3ef55dcb00b4a131d118d7b557b722302e9e0265ee6cc24fafe8f95161b`
- Sportsbook inputs used: **false**
- Production changed: **false**

The earlier run `34070770339` was a mechanical source-contract failure before scientific execution. The M89 artifact was audited and the source contract was repaired before the successful run. The authoritative source became `m89_2024_2025_synthesis_trace.csv`; unavailable intended fields were dropped rather than substituted. The frozen target, alpha, split, thresholds, and disposition gates were unchanged.

## Cohort
- Total mechanism rows: **884**
- 2024 training rows: **444**
- 2025 untouched test rows: **440**
- Mechanism-state counts across 2024-2025:
  - `ATTEMPTS_DOMINANT`: **420**
  - `YPA_DOMINANT`: **352**
  - `MIXED`: **112**
- 2025 rows with raw prior-4 player mechanism signal: **394**

## OOS 2025 results

| View | N | Pearson | Spearman | Q4-Q1 actual mechanism-share gap | Q4-Q1 ATTEMPTS_DOM rate gap | W2-18 Spearman | W13-18 Spearman |
|---|---:|---:|---:|---:|---:|---:|---:|
| PLAYER_ONLY | 440 | -0.068888 | -0.064381 | -0.047776 | -0.118182 | -0.058296 | -0.112277 |
| CONTEXT_ONLY | 440 | 0.122379 | 0.132641 | 0.109220 | 0.200000 | 0.118907 | 0.062211 |
| COMBINED | 440 | 0.061860 | 0.073443 | 0.050333 | 0.109091 | 0.063687 | 0.000768 |

Raw prior-4 player mechanism share vs current-game mechanism share:
- Spearman: **-0.081559**

COMBINED minus CONTEXT_ONLY Spearman:
- **-0.059198**

## Frozen gate
COMBINED required all of:
1. N >= 300 — **PASS**
2. Spearman >= 0.15 — **FAIL**
3. Q4-Q1 actual mechanism-share gap >= 0.10 — **FAIL**
4. Q4-Q1 ATTEMPTS_DOM rate gap >= 0.12 — **FAIL**
5. W2-18 Spearman > 0 — **PASS**
6. W13-18 Spearman > 0 — **PASS**
7. Individual-value condition — **FAIL**

## Disposition
**`NO_ACTIONABLE_QB_PLAYER_CONTEXT_MECHANISM_ROUTER`**

## Interpretation
The player-specific recent attempt-vs-YPA error mechanism is not persistent enough to route the next game's QB uncertainty. In this sample it is mildly anti-predictive, and adding it to the legitimate football-context view materially worsened the context-only OOS ranking signal.

The context-only view did show real descriptive regime structure: 0.132641 Spearman, a 0.109220 Q4-Q1 mechanism-share gap, a 0.200000 ATTEMPTS_DOM rate gap, and positive W2-18/W13-18 direction. However it missed the preregistered 0.15 Spearman gate and therefore is not promotable and will not be rescued by a nearby threshold/window retry.

## Authorized next step
No QB mean change and no retry of this player-history router. QB work returns to the promoted M89/M90 production path: generate the independent 2026 Week-1 no-odds passing-yard distributions, freeze them, and only afterward attach sportsbook lines for downstream market comparison. A future distribution-router experiment requires genuinely new football context/source information.
