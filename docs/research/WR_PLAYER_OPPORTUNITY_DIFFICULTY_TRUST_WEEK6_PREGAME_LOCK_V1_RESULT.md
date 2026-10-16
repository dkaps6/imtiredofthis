# WR Player Opportunity-Difficulty Trust — Week 6 Pregame Lock V1 — Result

Date: 2026-10-15

Status: **COMPLETE / WEEK6_WR_TRUST_PREGAME_LOCK_FROZEN / RESEARCH ONLY**

Frozen contract:
- `docs/research/WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1_FREEZE.md`

Parent trust authority:
- disposition `WR_PLAYER_OPPORTUNITY_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38823063793` SUCCESS
- job `116723999254`
- exact scored head `35a10d09ba92fd1218d37e812768fda5d2f4023e`
- artifact `12013153664`
- digest `sha256:12fd0de1dc3ebceee4d356669d59f30d96776d8c0800d1fb3ac6a293af457b7c`

Prospective Week-6 lock authority:
- run `38823520954` SUCCESS
- job `116725474887`
- exact head `638037c82b81e3e0f4d01ce37b1db24e5cc8144b`
- artifact `12012884262`
- digest `sha256:e5bc113e507ded651ecc29080c7fdbe4fd55182d1d039d0dd14057ada4b7bb38`
- immutable row CSV digest `sha256:81d9e950b96d7ac9d5699eba8490fc741a5aab2c938ef29287dc261534346b8f`

## Frozen Week-6 WR cohort

- current WR cohort: **152**
- supported WRs: **141**
- unsupported WRs: **11**
- HIGH_TRUST: **35**
- MIDDLE: **70**
- LOW_TRUST: **36**
- minimum qualifying prior games: **4**
- reconstructed 2026 W1-W5 WR history rows: **717**
- supported-cohort week-pool MAE: **16.1793567183 yards**

Historical semantic parity was exact:
- matched historical WR-R15 rows: **3,224**
- players: **211**
- max opportunity-error gap <= frozen `1e-9` tolerance.

## Prospective integrity

- 2026 history source max week: 5
- Week-6 outcomes read: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- models fit: 0
- projection/distribution values changed: 0
- production changed: false

The workflow explicitly failed closed if Week-6-or-later outcomes appeared. That boundary passed before the labels were frozen.

## Interpretation

This is a prospective **reliability label only** around the unchanged WR-R15 receiving-yard projection.

HIGH_TRUST / MIDDLE / LOW_TRUST do **not** mean:
- over / neutral / under;
- automatic bet / pass / fade;
- a mean correction;
- a width correction.

The labels may later be scored against settled Week-6 outcomes exactly as frozen. Do not alter the row set or thresholds after outcomes.

Canonical overall model status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.

No production mutation.
No sportsbook upstream.
No additional paid OddsAPI work.
