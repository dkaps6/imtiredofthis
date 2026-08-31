# M95O — Agreement-Gated Stable-Workhorse 20+ Tail Candidate Results

## Authoritative run

- Workflow: `M95O RB Agreement-Gated Tail Candidate`
- Run: **`33437157679`**
- Job: **`99636245739`**
- Tested SHA: **`d194a69a4d8939067b1c7d495de438c3062822eb`**
- Branch: `research-rb-m95o-agreement-gated-tail`
- Artifact: `migration-95o-rb-agreement-gated-tail`
- Artifact ID: **`9774831420`**
- Artifact SHA256: **`00bd31e6cb821edefc900a93d12b7e79a0f72056d8da49fa2b250138496e1cef`**
- Execution: success
- Disposition: **`RETAIN_M95O_AS_DIAGNOSTIC_DO_NOT_PROMOTE`**
- Feature search: `0`
- Coefficient search: `0`
- Sportsbook inputs: `0`
- Production change: `0`
- M94C central carries changed: `0`
- Stable-workhorse 25+ changed: `0`

Run #1 (`33436965152`) failed mechanically before model execution because the M95L artifact stored files under a nested `rb_m95l/` directory while the workflow used a root-only file check. The workflow was changed only to recursive/path-agnostic artifact discovery; no scientific logic changed. Run #2 above is authoritative.

## Precommitted design

M95O tested the conservative architecture motivated by M95N:

- M95F remains the stable-workhorse 20+ backbone.
- Frozen M95K (`k=4`, `feed_compact_env`, `C=.03`) provides the historical-feed reranking signal.
- Agreement thresholds are fixed from **2024 W13-15 pregame distributions only**, without using outcomes.
- Historical feed may change 20+ probability only in `aligned_high` or `aligned_low` games.
- `context_only` and `history_only` games remain exactly M95F.
- Eligible aligned rows use the frozen M95K ranking and are logit-mean-anchored to M95F probability mass.
- Stable 25+ remains exactly M95F.
- 2024 W16-18 is development/selection evidence, not independent confirmation.
- 2025 and opened 2023 W13-18 are retrospective research evidence, not pristine confirmation.

## Important 2024 clarification

2024 was not absent from the M95K lineage. It was the M95K development/selection year:

- M95K fit on 2024 W13-15 and selected on 2024 W16-18.
- Frozen architecture selected: `k=4`, `feed_compact_env`, `C=.03`.

On the 2024 W16-18 stable-workhorse selection sample (`n=34`):

- M95F 20+ AUC: `.619048`
- frozen M95K 20+ AUC: `.725275`
- M95F Brier: `.247201`
- frozen M95K Brier: `.225884`

Thus 2024 looked directionally more like 2025 than the sealed 2023 M95L sample. However, because M95K was selected using 2024, that year cannot be used as an independent vote that overrules the 2023 sealed failure.

## Workload regime / base-rate context

Stable-workhorse workload rates differ materially by season/window:

| Scope | n | 20+ rate | 25+ rate | Mean carries | Mean M95F p20 |
|---|---:|---:|---:|---:|---:|
| 2024 W16-18 development/selection | 34 | 38.24% | 17.65% | 16.47 | 26.43% |
| 2025 full research | 237 | 21.94% | 4.64% | 15.50 | 29.60% |
| 2025 W13-18 research | 85 | 28.24% | 7.06% | 15.74 | 29.43% |
| 2023 W13-18 opened | 73 | 32.88% | 13.70% | 16.97 | 16.19% |

This is a major result in its own right. 2023 is not simply the lone "high-workload" outlier: the 2024 selection window was even higher. The more important issue is that the relationship between actual workload base rate and the frozen M95F probability level shifts materially across years/windows.

In particular:

- 2023 W13-18 actual 20+ rate `32.88%` vs M95F mean probability `16.19%` — strongly under-bullish.
- 2024 W16-18 actual `38.24%` vs M95F `26.43%` — under-bullish.
- 2025 W13-18 actual `28.24%` vs M95F `29.43%` — approximately calibrated.
- 2025 full actual `21.94%` vs M95F `29.60%` — over-bullish.

Therefore the stable-workhorse population prior itself is nonstationary. A fixed absolute interpretation of context/history cannot be assumed to transfer cleanly across seasons.

## M95O probability results

### 2024 W16-18 — development/selection only

- M95F: AUC `.619048`, Brier `.247201`
- frozen M95K: AUC `.725275`, Brier `.225884`
- M95O gate: AUC `.659341`, Brier `.242236`

M95O improved on M95F but retained much less of M95K's development-set gain.

### 2025 full research

- M95F: AUC `.581185`, Brier `.186593`
- frozen M95K: AUC `.641164`, Brier `.171528`
- M95O: AUC `.589605`, Brier `.184791`

M95O retained only a small portion of the strong M95K 2025 improvement.

### 2025 W13-18 research

- M95F: AUC `.646858`, Brier `.194926`
- frozen M95K: AUC `.732923`, Brier `.186846`
- M95O: AUC `.635246`, Brier `.197812`

The agreement gate was worse than M95F in this same-window 2025 slice.

### 2023 W13-18 opened

- M95F: AUC `.727041`, Brier `.233221`
- frozen M95K: AUC `.545068`, Brier `.244446`
- M95O: AUC `.568027`, Brier `.243097`

M95O partially reduced the M95K damage but remained far below the M95F backbone. It did not solve the cross-season reversal.

## Why the fixed gate failed

The fixed 2024 reference thresholds did not create comparable player-game regimes across years.

Aligned share:

- 2024 W16-18: `73.53%`
- 2025 full: `60.34%`
- 2025 W13-18: `64.71%`
- 2023 W13-18: `80.82%`

But the composition changed dramatically. Under the fixed 2024 threshold, 2023 produced only **one `aligned_high` row** and 58 `aligned_low` rows, even though 31.03% of those aligned-low games actually reached 20+ carries.

That is evidence of a distribution-level shift rather than merely an incorrect player-specific expert choice. The current-context probability and historical-feed scales do not have stationary absolute meanings across seasons.

M95N's within-scope percentile audit could show stable high-vs-low agreement because each season/window was normalized to itself. M95O deliberately made the gate prospective by freezing a 2024 reference distribution. The resulting failure exposed that the **reference distribution itself moves**.

## Formal gates

- `gate_2023_no_material_regression = 0`
- `gate_2025_full_retains_value = 0`
- `gate_2025_late_nonnegative = 0`
- `stable_probability_mass_preserved = 0`
- `retrospective_research_pass = 0`

The mass flag requires care: full-population stable mass was preserved in 2024, 2025 full, and 2023, but the already-scored full-2025 candidate had a different mean when later sliced to W13-18 (`.294330 -> .307945`). The frozen gate correctly records failure under its strict per-evaluation-slice definition. Do not retroactively redefine this gate after seeing results. A future week-batch/local-mass design would have to be a new precommitted candidate.

## Scientific conclusion

M95O does **not** validate a simple agreement-gated mixture-of-experts architecture.

The stronger new finding is upstream:

> Stable-workhorse workload prevalence and probability calibration shift materially by season/window, so historical feed and current-context evidence must be interpreted relative to the current pregame workload regime rather than against one fixed cross-season scale.

This also answers the 2023-versus-2025 concern. 2023 may represent a different workload regime, but 2024 demonstrates that high-workload behavior is not unique to 2023. We cannot label 2023 a bad/weird year and discard it. The correct research question is whether the regime shift can itself be detected prospectively.

## Recommended next migration — M95P

**M95P — Dynamic Workload-Regime / Population-Prior Audit**

Primary question:

> Using only information available before each game, can we identify the current league/team stable-workhorse workload regime so that player history and current game context are normalized relative to the appropriate contemporaneous population prior?

Audit before fitting another candidate:

- prior-week league stable-workhorse 20+/25+ prevalence;
- prior-week/team lead-RB workload distribution;
- rolling team rush volume and lead-RB concentration;
- QB rush siphoning and RB-count/committee environment;
- league/team play volume and rush tendency;
- whether rolling season-to-date normalization makes 2023/2024/2025 player-game archetypes comparable;
- whether a pregame regime state explains the large year-to-year M95F calibration shifts.

No new promoted model should be fit until this audit demonstrates repeatable pregame structure. Any eventual candidate still requires a new untouched/prospective confirmation protocol.
