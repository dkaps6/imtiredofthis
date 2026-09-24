# TE Target-Quality Efficiency V1 — Source Audit Result

Date: 2026-09-24

Status: `TE_TARGET_QUALITY_NGS_SOURCE_SUPPORTS_FROZEN_PREDICTIVE_STUDY`

## Canonical execution

- branch: `research-te-target-quality-efficiency-v1`
- source-only plan: `docs/research/TE_TARGET_QUALITY_EFFICIENCY_V1_SOURCE_AUDIT_PLAN.md`
- run: `36000625120`
- job: `107636107826`
- artifact: `10808077908`
- artifact name: `te-target-quality-efficiency-v1-source-audit`
- artifact digest: `sha256:c1f2cfd76e4a27be44117c2b0a005d78f7578ec54f6da106a579cc961869ba05`

Integrity:
- predictive model fit: **false**
- prediction error scored: **false**
- sportsbook inputs used: **0**
- production changed: **false**

## Finding 1 — NGS receiving is unusually dense for TE

TE NGS player-week rows:

| season | rows | unique players | core tracking coverage |
|---|---:|---:|---:|
| 2020 | 338 | 60 | ~100% |
| 2021 | 348 | 62 | 99.7-100% |
| 2022 | 318 | 62 | 100% |
| 2023 | 339 | 60 | 100% |
| 2024 | 330 | 58 | 100% |
| 2025 | 360 | 62 | 100% |
| 2026 W1-W2 | 56 | 25 | 100% |

The audited player-week fields are:
- targets
- receptions
- avg separation
- avg cushion
- avg intended air yards
- avg expected YAC
- avg YAC above expectation
- percent share of intended air yards

This is enough historical coverage and current 2026 strict-prior coverage to justify a separately frozen player-level tracking-quality predictive study for Week 3+.

## Finding 2 — PBP target-quality is also strong and is the fallback / mechanism authority

TE target rows:

| season | TE targets | completed TE targets | air-yards coverage | xYAC coverage on completions |
|---|---:|---:|---:|---:|
| 2020 | 3,712 | 2,482 | 99.46% | 93.80% |
| 2021 | 3,837 | 2,617 | 99.24% | 95.15% |
| 2022 | 3,664 | 2,511 | 99.45% | 95.54% |
| 2023 | 3,752 | 2,664 | 99.63% | 96.32% |
| 2024 | 3,825 | 2,761 | 99.50% | 95.87% |
| 2025 | 3,992 | 2,867 | 99.40% | 95.47% |
| 2026 W1-W2 | 436 | 319 | 99.77% | 92.79% |

Target-context fields down, distance, shotgun, no-huddle and score differential are effectively complete. YAC itself is complete on TE catches.

## Scientific interpretation

This closes the data-availability question. The next TE mean-science lane does **not** need another raw YPT/catch-rate recency model and does **not** need historically unreconstructable WR-CB matchup assignments.

A genuinely new, timestamp-safe player-level information family exists:
- how deep the player's targets are;
- how separated/cushioned the player is;
- how much YAC the catch context is expected to create;
- whether the player has produced YAC above/below contextual expectation;
- share of intended air yards.

The preferred next study is NGS-first because it is already player-week aggregated, nearly complete, and available for 2026 Weeks 1-2. PBP target-level depth/xYAC remains a high-quality fallback and mechanism-check source.

## Disposition

`TE_TARGET_QUALITY_NGS_SOURCE_SUPPORTS_FROZEN_PREDICTIVE_STUDY`

This is source qualification only. It does not authorize production change. The predictive design must be frozen before any error relationship is inspected.
