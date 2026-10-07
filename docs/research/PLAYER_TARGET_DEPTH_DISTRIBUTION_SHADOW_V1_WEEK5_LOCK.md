# Player Target Depth Distribution Shadow V1 — 2026 Week-5 Pregame Lock

Status: **FROZEN / SUCCESS / RESEARCH SHADOW ONLY**

Disposition: `WEEK5_PREGAME_DISTRIBUTION_LOCK_FROZEN`

## Canonical authority

- branch: `research-player-target-depth-distribution-shadow-v1`
- contract freeze commit: `39be8402f4d1bf936759ccedf6f162967b38f5b0`
- implementation head: `55d06850e483fb7a4e6baa754b146f2fddefe6e8`
- workflow run: `37677366697`
- job: `112984218995`
- workflow conclusion: **SUCCESS**
- artifact: `11507970080`
- artifact digest: `sha256:ecd7c8683d65bf9701353f0dbccb3ace59c992157bdf75b2b87191b08a8f314f`
- lock row digest: `sha256:1a5b6a914517310cdfb0fe5c92fcba7a77a28b6a988664c0f7dcc49cd2b8eb2a`

Parent trajectory universe:
- run `37654382316`
- artifact `11497153776`
- parent row digest `sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`

Parent target-depth science:
- run `37655282486`
- artifact `11497703984`
- disposition `PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY_CONFIRMED`

## Frozen transform

Exact position anchors:
- WR: `10.02786868561449`
- TE: `6.735519692444827`

No-fit scale:
`depth_distribution_scale = sqrt(prior8_target_depth_sd / position_anchor)`

Eligibility is the exact confirmed parent target-depth feature:
- latest up to 8 strictly-prior receiver target-games;
- at least 4 prior receiver target-games;
- at least 10 finite air-yard targets;
- population SD of finite `air_yards`.

Unsupported players use scale `1.0`.

The reusable draw transform in
`scripts/research/lock_player_target_depth_distribution_shadow_v1.py`
changes only deviations around the exact final receiving-yard mean, clips at zero,
and multiplicatively re-aligns back to that same exact mean.

## Week-5 locked universe

- players: **277**
- WR: **167**
- TE: **110**
- resolved receiver IDs: **256 / 277 = 92.42%**
- target-depth feature available: **210 / 277 = 75.81%**
  - WR: **133**
  - TE: **77**
- uncertainty-changed players: **210**

Frozen scale distribution:
- min: **0.6139186346**
- P10: **0.8455753213**
- P50: **1.0000000000**
- P90: **1.1643001044**
- max: **1.3910914742**

## Integrity

- same/future feature violations: **0**
- parameters fit: **0**
- sportsbook inputs used: **0**
- Week-5 outcomes read: **0**
- football means changed: **false**
- target entitlement changed: **false**
- team volume changed: **false**
- production changed: **false**

Workflow safeguards passed:
- frozen transform unit tests;
- exact parent row-digest verification;
- exact 277-row parent universe;
- exact 167 WR / 110 TE counts;
- finite positive scales;
- pregame/source boundary;
- mean-neutral invariant.

## Scientific interpretation

This closes the missing WR/TE player-specific distribution/uncertainty implementation layer for the current player-individualization phase.

It does **not** promote a new production distribution model. It freezes the exact player-specific uncertainty treatment that the later all-player/all-position replay may score.

TE-R5P Receiving-Yards Width V2 remains failed closed. This is not a retry:
- no global residual-width factor is fit;
- no residual history is used;
- the treatment variable is newly-confirmed, strictly-prior individual target-depth dispersion;
- receiving-yard means remain exact.

## Replay scope

For the eventual all-position replay:
- target-depth distribution may legally use strictly-prior historical receiver history for 2026 Weeks 1-4 under the same frozen feature definition;
- exact Target Share Trajectory V1 still cannot apply to 2026 Weeks 1-4 because its contract requires four prior same-season team games;
- Week-5 trajectory and target-depth locks remain outcome-blind prospective evidence.

No paid OddsAPI pull was made.
