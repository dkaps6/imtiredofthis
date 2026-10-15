# NFL HANDOFF — 2026-10-15 — TE TRUST SELECTIVITY QUALIFIED / WEEK-6 LOCK FRONTIER

GitHub is canonical. Research remains isolated from production.

Repository:
- `dkaps6/imtiredofthis`

Production:
- main `9b70b12e2fb4bc39f08556f244370068aad9c47b`
- Week-6 live board complete and parked
- final priced replay `38792341891` SUCCESS
- exactly one paid OddsAPI acquisition used
- do not return to PIT-TB unless user asks

Research:
- branch `research-individual-opportunity-roadmap-2026-10-08`
- draft PR #672
- Issue #535
- canonical status `PLAYER_INDIVIDUALIZATION_PARTIAL`

## New canonical result

`TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`

Authority:
- run `38821176218` SUCCESS
- job `116717842485`
- scored head `270f01266bff125cfaccb53a57b378674d919901`
- artifact `12013570473`
- digest `sha256:66c7b39f08b956dc1a96bd068272d1b97ddb5ddba22ef928a7d6e02c7f463530`
- result doc `docs/research/TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_V1_RESULT.md`

All frozen gates passed.

2025 independent confirmation:
- primary within-week trust association `+0.11538`
- player-cluster CI `[+0.04191,+0.18535]`
- game-cluster CI `[+0.04679,+0.18280]`
- HIGH_TRUST standardized CRPS `0.67012`
- LOW_TRUST standardized CRPS `1.09274`
- HIGH_TRUST raw abs error `8.70 yd`
- LOW_TRUST raw abs error `23.60 yd`

Pooled:
- player-cluster CI `[+0.02894,+0.14205]`
- game-cluster CI `[+0.03448,+0.13607]`
- raw CRPS HIGH `7.50` vs LOW `18.21`

No mean or distribution changed. No fitted model. No sportsbook. No OddsAPI. No 2026 outcomes.

## Interpretation

The named TE's own strictly-prior allocation difficulty is useful as a **trust/reliability ranking** for the existing TE-R5P projection.

Do not reinterpret this as authorization for:
- another TE width law;
- another joint allocation family;
- a mean correction;
- a TE1/TE2 subgroup;
- fitted thresholds.

The failed marginal-width and joint-Dirichlet mappings remain closed.

## Exact next action

Create the frozen Week-6 pregame TE trust shadow before Week-6 outcomes:

1. exact same `player_mae` semantics as the historical trust result;
2. trailing max 8 strictly-prior same-player allocation errors, min 4;
3. allocation error must use the promoted pregame TE-R5P target mean and pregame team target expectation against realized player target share;
4. current Week-6 TE cohort from football-only authority;
5. within-week percentile trust score;
6. HIGH_TRUST <= .25 / LOW_TRUST >= .75;
7. no sportsbook input and no Week-6 outcomes;
8. trust labels only; no projection/distribution mutation.

Preferred current-slate authority:
- no-odds Week-6 Full Slate run `38790425676`
- artifact `11999618082`
- digest `sha256:d7473705d4534c3172b58843272e7d95f5fbbd9aa910238d48916a8567cde1de`

Fail closed if exact prior allocation-error history cannot be reconstructed. Do **not** substitute raw target-share volatility.

After locking, return to the broader individual-player research frontier while Week 6 remains prospective.

No paid OddsAPI without explicit authorization.
No production mutation.
