# TE Player Mechanism Persistence V1 — Result

**STATUS: COMPLETE / TE_PLAYER_PERSISTENCE_MIXED_MECHANISM**

Frozen plan: `docs/research/TE_PLAYER_MECHANISM_PERSISTENCE_V1_PLAN.md`

Authority:
- run `37631011196` — SUCCESS
- source SHA `69618c13ed0eba2da1dd7d7032a94dc509de4ce9`
- artifact `11486082927`
- digest `sha256:1b208531a8cf34735760081b7a232c4cd164a551c325f518d33428a50d7da8ab`

Parent: `TE_PLAYER_ERROR_PERSISTENCE_DETECTED`.

## Disposition

`TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`

Exact decomposition identity:
- total error = opportunity error + efficiency error
- max numerical gap: `2.842170943040401e-14`

Panel:
- 2,005 rows / 121 TEs
- 2024: 978 rows / 99 players
- 2025: 1,027 rows / 108 players
- support PASS
- same/future violations 0

## Opportunity

O1 signed opportunity persistence PASS:
- 2024 rho 0.230130
- 2025 rho 0.266435
- pooled rho 0.249982
- player-cluster bootstrap P(positive) 1.000
- 95% CI [0.1831, 0.3116]

O2 opportunity difficulty PASS:
- 2024 rho 0.383882
- 2025 rho 0.401264
- pooled rho 0.393593
- bootstrap P(positive) 1.000
- 95% CI [0.3346, 0.4436]

Meaning: despite TE-R5P's individual entitlement layer, some TEs remain consistently harder to project for target opportunity, and miss direction persists.

## Efficiency

E1 signed efficiency persistence FAIL:
- 2024 rho -0.017734
- 2025 rho -0.098322
- pooled rho -0.056675
- bootstrap P(positive) 0.0042

E2 efficiency difficulty PASS:
- 2024 rho 0.180593
- 2025 rho 0.259241
- pooled rho 0.224049
- bootstrap P(positive) 1.000
- 95% CI [0.1616, 0.2821]

Meaning: some TEs are persistently harder to translate from targets into yards, but there is no stable high/low efficiency bias.

Secondary catch-rate signed persistence is negative rather than positive:
- pooled rho -0.163310
- bootstrap P(positive) 0.000

## Error mass

Pooled:
- mean absolute total error 16.1923 yards
- mean absolute opportunity component 12.4881
- mean absolute efficiency component 11.3605
- normalized opportunity mass 49.84%
- normalized efficiency mass 50.16%
- component signs oppose in 48.58% of rows

## Next scientific implication

Opportunity is the stronger actionable research direction. The next question is which genuinely player-specific pregame football state is missing when TE-R5P converts participation into target entitlement.

Efficiency should be treated as a player-specific difficulty/uncertainty question, not a simple directional YPT correction.

No production change is authorized. Do not feed prior residuals directly back into targets or YPT, and do not reopen the failed generic TE Width V2.

Models fit: 0. Sportsbook inputs: 0. 2026 outcomes: 0. Production changed: false.
