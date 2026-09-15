# WR Phase 4B Authority-Exact Opportunity Attribution V2 — Result

## Status

Canonical research diagnostic complete. No challenger model authorized. No production change. No sportsbook inputs. No paid Full Slate. No RB work.

This result follows the frozen Phase 4B plan and Amendment 1, after Claude returned `IMPLEMENTATION_REVIEW_PASS` in Issue #535 comment `5682650117`.

## Canonical execution

- Branch: `research-wr-phase4-authority-exact-opportunity-attribution-v1`
- Workflow head: `686849db8a5d28930505ef108259dfca9448b21d`
- Workflow: `.github/workflows/research-wr-phase4b-attribution-v2.yml`
- Run: `34987211287`
- Job: `104442281504`
- Conclusion: `SUCCESS`
- Artifact: `10404525877`
- Artifact name: `wr-phase4b-authority-attribution-v2`
- Artifact digest: `sha256:0dad0dbc91a4bd6aa3e3f45cbc62e8ed493ca7bf2c95daaa0c4028a54a0fd0d4`

Exact WR-R15 authority remained pinned to:
- run `34238301577`
- artifact `10061328722`
- digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`

The workflow re-ran both reviewed synthetic suites, exact authority verification, and the full source/identity preflight immediately before attribution. Every guard passed.

## Source and identity integrity

- WR-R15 target parity: `4,193 / 4,193` exact; max absolute target delta `0.0`.
- Layer 4 WR2+ feature identities: `5,320 / 5,321` resolved to GSIS; one ambiguous identity remains fail-closed.
- Layers 2/3 canonical identity domain: one team-game excluded because of that ambiguity (`2024 W15 NYJ`).
- Complete Layer 2/3 modeled rooms: `1,025` team-games.
- No fuzzy matching.
- No predictive temporal-rule change.
- Sportsbook inputs: `0`.

## Layer 1 — receiving-yard residual accounting

Exact symmetric decomposition over all `4,193` graded WR-R15 rows:

| Slice | n | Mean abs opportunity | Mean abs efficiency | Share opportunity > efficiency | Mean signed opportunity | Mean signed efficiency |
|---|---:|---:|---:|---:|---:|---:|
| Pooled | 4,193 | 16.2602 yd | 16.4267 yd | 44.96% | +7.9951 yd | -0.6367 yd |
| 2023 | 2,076 | 15.9853 | 16.9123 | 43.64% | +7.5928 | -0.4746 |
| 2024 | 2,117 | 16.5298 | 15.9505 | 46.24% | +8.3897 | -0.7955 |
| WR1 | 1,026 | 22.6663 | 22.4316 | 49.81% | +9.3739 | +0.6202 |
| WR2+ | 3,167 | 14.1849 | 14.4813 | 43.38% | +7.5485 | -1.0438 |
| Actual 100+ yards | 307 | 46.1145 | 38.1929 | 60.26% | +43.4304 | +34.9084 |
| Abs residual >=30 yd | 1,048 | 31.4871 | 27.6872 | 55.92% | +22.8325 | +12.2528 |

### Layer 1 interpretation

Pooled receiving-yard error is not opportunity-dominant. Mean absolute opportunity and efficiency contributions are essentially equal, with efficiency slightly larger overall and opportunity larger on only 44.96% of rows.

However, opportunity becomes materially more important in the tail: it dominates 60.26% of actual 100+ receiving-yard games and 55.92% of >=30-yard residual games. That preserves the earlier WR1-decomposition finding that opportunity matters substantially in big misses, while rejecting the idea that pooled WR receiving-yard weakness is primarily an opportunity-allocation problem.

This is accounting, not causality.

## Layer 2 — team target pool vs modeled WR-room share

Population: `1,025` complete static-identity modeled rooms.

Pooled:
- WR-room target MAE: `4.5204` targets/team-game.
- WR-room signed bias, predicted minus actual: `-0.6437` targets.
- Mean absolute team-pool component: `3.6810` targets.
- Mean absolute WR-room-share component: `3.3668` targets.
- Team-pool component larger in absolute magnitude on `53.85%` of team-games.

By season:
- 2023 room MAE `4.4743`; mean abs pool `3.5967`; mean abs room-share `3.2806`.
- 2024 room MAE `4.5661`; mean abs pool `3.7645`; mean abs room-share `3.4521`.

### Layer 2 interpretation

There is no overwhelming single upstream winner between team target-pool error and modeled WR-room-share error. Team target pool is modestly larger on average and in a small majority of games, but both are material and stable across seasons.

Off-model WR target mass on the complete-room cohort was exactly `0.0`, so this result is not being driven by raw-position WR targets omitted from the canonical modeled room.

## Layer 3 — M38 WR1 vs canonical WR2+ secondary pool

Pooled target diagnostics:
- WR1 MAE: `2.7620`, RMSE `3.6310`, bias predicted minus actual `-1.1426`.
- WR2+ secondary-pool MAE: `3.7142`, RMSE `4.6415`, bias predicted minus actual `+0.4989`.

2023:
- WR1 MAE `2.8083`.
- WR2+ pool MAE `3.8014`.

2024:
- WR1 MAE `2.7162`.
- WR2+ pool MAE `3.6279`.

### Layer 3 interpretation

The frozen tree's M38-specific condition is not met: WR1 target error is not materially worse than canonical secondary-pool error. The opposite ordering appears in both seasons. Phase 4B therefore does not isolate M38/WR1 as the structured opportunity failure.

## Layer 4 — R15 within-WR2+ allocation

Resolved population: `5,320 / 5,321` WR2+ rows.

| Slice | n | Baseline target MAE | R15 target MAE | Delta R15-baseline | 95% cluster-bootstrap CI | R15 toward actual |
|---|---:|---:|---:|---:|---:|---:|
| Pooled | 5,320 | 1.9748 | 1.8100 | **-0.1648** | [-0.1833, -0.1466] | 62.14% |
| 2023 | 2,680 | 1.9648 | 1.7982 | **-0.1665** | [-0.1919, -0.1410] | 62.57% |
| 2024 | 2,640 | 1.9850 | 1.8220 | **-0.1630** | [-0.1895, -0.1369] | 61.70% |

Additional secondary-share diagnostic:
- baseline share MAE: `0.160892`
- R15 share MAE: `0.145149`

Formal frozen disposition:

`R15_WR2PLUS_ALLOCATION_HEALTHY_OR_IMPROVED`

R15 improves pooled WR2+ target MAE by roughly 8.35%, with nearly identical gains in both seasons and a pooled 95% cluster-bootstrap interval entirely below zero. Secondary-share MAE also improves by roughly 9.78%.

### Layer 4 interpretation

R15 is affirmatively exonerated as the source of the structured receiving-yard weakness under the frozen diagnostic. Do not retune or replace R15 on the basis of Phase 4B.

## Program-level verdict

Phase 4B does **not** authorize a challenger model.

The evidence supports all of the following simultaneously:

1. Pooled receiving-yard error remains approximately half opportunity and half efficiency; opportunity is not the primary pooled error source.
2. Opportunity is more important in high-end and large-residual misses, so it remains a meaningful tail contributor.
3. At the team/room level, target-pool error and WR-room-share error are both material; neither overwhelmingly dominates.
4. M38 WR1 is not isolated as the structured opportunity failure.
5. R15 WR2+ allocation is healthy/improved, strongly and consistently across both seasons.

Therefore:
- preserve M38;
- preserve R15;
- do not perform an R15 challenger or post-hoc rescue;
- do not infer that opportunity is irrelevant just because it is not dominant pooled;
- do not resume closed R17-R20 target-quality fishing without a genuinely new mechanism/source;
- no production change from this audit.

The next research decision should be made only after independent result audit of this canonical artifact and should target a genuinely unresolved layer rather than revisiting R15.
