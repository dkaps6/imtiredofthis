# TE Player Difficulty Trust / Selectivity V1 — Result

Date: 2026-10-15

Status: **COMPLETE / TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED / RESEARCH ONLY**

Frozen contract:
- `docs/research/TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_V1_FREEZE.md`

Canonical authority:
- workflow: `Research TE Player Difficulty Trust Selectivity V1`
- run: `38821176218` — SUCCESS
- job: `116717842485`
- exact scored head: `270f01266bff125cfaccb53a57b378674d919901`
- artifact: `12013570473`
- digest: `sha256:66c7b39f08b956dc1a96bd068272d1b97ddb5ddba22ef928a7d6e02c7f463530`

Parent:
- `TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_V1`
- run `38800926331`
- artifact `12003898778`
- parent disposition `NO_ACTIONABLE_TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS`

## Frozen disposition

`TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Every frozen gate passed.

## Question tested

Without changing the existing TE-R5P projection mean or distribution, does the named TE's own strictly-prior allocation difficulty identify when that player's canonical receiving-yard projection/distribution is reliable versus fragile?

Frozen trust score:
- within each season/week, average percentile rank of same-player strictly-prior `player_mae`;
- lower = higher trust;
- `HIGH_TRUST <= 0.25`;
- `LOW_TRUST >= 0.75`;
- no fitted coefficient or outcome-tuned threshold.

Primary reliability outcome:
`baseline_crps / baseline_sd`.

Secondary:
`abs(actual_rec_yards - baseline_mean) / baseline_sd`.

## Support

- pooled: 1,476 rows / 114 TEs / 411 games
- 2024: 717 rows / 92 TEs / 205 games
- 2025 independent confirmation: 759 rows / 103 TEs / 206 games
- HIGH_TRUST rows: 357 pooled
- LOW_TRUST rows: 385 pooled

## Primary rank association

2024:
- `+0.0537813`

2025:
- `+0.1153773`

Pooled:
- `+0.0854571`

Dependence-aware pooled:
- player-cluster 95% CI: `[+0.0289353,+0.1420485]`
- game-cluster 95% CI: `[+0.0344800,+0.1360742]`

2025 confirmation:
- player-cluster 95% CI: `[+0.0419128,+0.1853534]`
- game-cluster 95% CI: `[+0.0467862,+0.1828009]`

## Trust separation

2024 standardized CRPS:
- overall `0.93795`
- HIGH_TRUST `0.88816`
- LOW_TRUST `1.05625`

2025 standardized CRPS:
- overall `0.88375`
- HIGH_TRUST `0.67012`
- LOW_TRUST `1.09274`

Pooled standardized CRPS:
- overall `0.91008`
- HIGH_TRUST `0.77700`
- LOW_TRUST `1.07483`

Raw CRPS pooled:
- HIGH_TRUST `7.50287`
- LOW_TRUST `18.20888`

Absolute mean error pooled:
- HIGH_TRUST `10.5762 yd`
- LOW_TRUST `23.7186 yd`

2025 raw absolute mean error:
- HIGH_TRUST `8.7035 yd`
- LOW_TRUST `23.5957 yd`

The trust split is therefore not merely a standardized-score artifact.

## Incremental player information

Pooled median same-player difficulty relative to contemporaneous pooled TE difficulty:
- HIGH_TRUST: `0.4252x`
- LOW_TRUST: `1.8833x`

The score is calculated within each week, so it removes the week-wide pooled TE difficulty level by construction. The qualified signal is player-specific relative reliability.

## Integrity

- models fit: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes read: 0
- projection/distribution values changed: 0
- production changed: false
- failed frozen gates: none

## Scientific interpretation

This is a successful individual-player reliability result.

The same-player TE allocation-difficulty state should **not** be forced into another mean or width formula. The exact width mappings already failed their own gates.

Instead, the named TE's prior difficulty reliably identifies when the existing TE-R5P projection/distribution is comparatively trustworthy versus fragile.

That directly supports a research-only player trust/selectivity layer for live-board decision support while leaving the football projection untouched.

This is not production promotion because 2024/2025 are not untouched external seasons for this broader TE research program.

## Mandatory next step

Freeze a 2026 Week-6 pregame TE trust shadow using the exact same player-difficulty semantics and football-only authority **before any Week-6 outcomes are read**.

The Week-6 lock:
- may label/rank trust only;
- may not change the projection, distribution, sportsbook price, or selection signal;
- must use strictly-prior player history;
- must fail closed rather than substitute target-share volatility or another proxy if exact allocation-difficulty history cannot be reconstructed.

Prospective scoring after Week 6 settles is required before any production consideration.

Canonical model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
