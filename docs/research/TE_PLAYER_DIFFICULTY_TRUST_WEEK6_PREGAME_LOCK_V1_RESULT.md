# TE Player Difficulty Trust — Week 6 Pregame Lock V1 — Result

Date: 2026-10-15

Status: **COMPLETE / WEEK6_TE_TRUST_PREGAME_LOCK_FROZEN / RESEARCH-ONLY PROSPECTIVE SHADOW**

Frozen contract:
- `docs/research/TE_PLAYER_DIFFICULTY_TRUST_WEEK6_PREGAME_LOCK_V1_FREEZE.md`

Canonical authority:
- workflow: `Research TE Player Difficulty Trust Week6 Pregame Lock V1`
- run: `38822346088` — SUCCESS
- job: `116721639322`
- exact head: `1edecb1b3f60089e95bfc4c100522baf60d560a9`
- artifact: `12013018536`
- digest: `sha256:98667675753bfb410c6ac6259dc097476a87e49ce29f5cfe71be6fe14a74e1f1`

Parent:
- `TE_PLAYER_DIFFICULTY_TRUST_SELECTIVITY_QUALIFIED`
- run `38821176218`
- artifact `12013570473`

## Frozen Week-6 lock

- current TE-R5P cohort: **105**
- supported: **103**
- unsupported: **2**
- HIGH_TRUST: **25**
- MIDDLE: **52**
- LOW_TRUST: **26**
- supported-cohort median player allocation MAE: **1.6769208511 targets-equivalent**

History used:
- 2024 exact parent rows: **512**
- 2025 exact parent rows: **597**
- 2026 W1-W5 strict-prior reconstructed rows: **500**
- max 2026 source week: **5**

Immutable Week-6 row digest:
- `sha256:fb5e2f83822a76adf3f09a721738237158bfddd9f95a3e63a97d3e4334d8ffa5`

## Semantic parity

The exact historical team-target bridge was independently reproduced before the Week-6 labels were emitted:

`pred_team_targets = clip(rules_plays_est,50,80) * clip(rules_pass_rate,.35,.75)`

against the frozen 2025 TE-R5P authority:

- matched rows: **1,113**
- max absolute gap: **7.5484458e-7**
- mean absolute gap: **1.1967007e-7**
- frozen tolerance: `1e-5`

PASS.

## Prospective boundary

The workflow explicitly failed closed if Week-6 outcomes were present.

Boundary checks:
- Week-6 outcomes read: **0**
- sportsbook inputs: **0**
- OddsAPI calls: **0**
- models fit: **0**
- projection/distribution values changed: **0**
- production changed: **false**
- exact Week-6 roster authority: **PASS**
- complete Week-5 stats/snaps: **PASS**

The trust labels are therefore legitimately prospective.

## Interpretation

This lock is not a bet selector and does not move any TE projection.

`HIGH_TRUST` means the named TE's own strictly-prior allocation history places his existing TE-R5P projection in the historically more reliable tail.

`LOW_TRUST` means the existing projection is historically more fragile for that named player.

It does **not** mean:
- automatically bet/fade the player;
- automatically take an under/over;
- alter the mean;
- alter distribution width;
- reopen the failed TE marginal-width or joint-Dirichlet laws.

The prospective Week-6 result must remain untouched until settlement.

## Notable locked examples

HIGH_TRUST includes:
- Davis Allen
- Andrew Ogletree
- Luke Farrell
- Cole Kmet
- Payne Durham

LOW_TRUST includes:
- Isaiah Likely
- Trey McBride
- Brock Bowers
- Dallas Goedert
- George Kittle
- Travis Kelce
- Mark Andrews
- Kyle Pitts
- Tyler Warren
- Cade Otton
- Dalton Kincaid

Two unsupported TEs:
- Teagan Quitoriano — one qualifying prior game
- Hayden Rucci — three qualifying prior games

These names are reported only to make the prospective lock auditable; they are not outcome claims.

## Next

Leave Week-6 trust frozen.

Continue historical individual-player research in a distinct lane. Do not use Week-6 outcomes until the prospective scoring stage.

Canonical overall status remains:
`PLAYER_INDIVIDUALIZATION_PARTIAL`.
