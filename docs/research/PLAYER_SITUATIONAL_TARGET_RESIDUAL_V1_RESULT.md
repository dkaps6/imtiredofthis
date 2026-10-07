# Player Situational Target Residual V1 — Result

**STATUS: COMPLETE / NO_ACTIONABLE_PLAYER_SITUATIONAL_TARGET_ROLE_SIGNAL**

Frozen plan:
`docs/research/PLAYER_SITUATIONAL_TARGET_RESIDUAL_V1_PLAN.md`

Certified authority:
- branch `research-player-situational-target-residual-v1`
- optimized run `37637017878` — SUCCESS
- source SHA `4cf9ff50cad362786debe5f63d3f8c210c8817fc`
- artifact `11489713078`
- digest `sha256:145ff317427e5ed40ec5857e92ff35fa099ed9a40c6e878bb3fb0d9abc6b368c`
- pre-optimization run `37636407989` also SUCCESS and produced identical scientific metrics

Final disposition:

`NO_ACTIONABLE_PLAYER_SITUATIONAL_TARGET_ROLE_SIGNAL`

## Integrity / support

Exact promoted authorities:
- WR: 4,193 M38/WR-R15 rows
- TE: 3,214 TE-R5P rows

Identity mapping coverage:
- WR: 100%
- TE: 100%

Historical diagnostic panel:
- 5,448 player-games
- 202 WR identities
- 119 TE identities
- target-game feature rows read: 0
- sportsbook inputs: 0
- 2026 outcomes: 0
- models fit: 0
- production changes: 0

Every context cleared the predeclared support floors.

## Third down — CLOSED

WR:
- 2023 rho: `+0.00034`
- 2024 rho: `+0.05099`
- pooled rho: `+0.02633`
- cluster P(expected negative): `0.1168`

TE:
- 2024 rho: `+0.04140`
- 2025 rho: `-0.01523`
- pooled rho: `+0.00128`
- cluster P(expected negative): `0.4782`

Combined pooled rho:
- `+0.01756`

No replication and the pooled direction is opposite the frozen hypothesis.

## Red zone — CLOSED

WR:
- 2023 rho: `-0.03230`
- 2024 rho: `-0.02096`
- pooled rho: `-0.02641`
- cluster P(expected negative): `0.8856`

TE:
- 2024 rho: `+0.01901`
- 2025 rho: `-0.00762`
- pooled rho: `+0.02118`
- cluster P(expected negative): `0.2136`

Combined pooled rho:
- `-0.00774`
- combined bootstrap does not support a meaningful negative signal.

WR direction was weakly compatible, but TE did not replicate and the effect missed the frozen materiality/inference gates. No WR-only rescue.

## Two minute — CLOSED

WR:
- 2023 rho: `-0.02436`
- 2024 rho: `+0.00477`
- pooled rho: `-0.00902`
- cluster P(expected negative): `0.6642`

TE:
- 2024 rho: `-0.00147`
- 2025 rho: `-0.02600`
- pooled rho: `+0.02073`
- cluster P(expected negative): `0.2760`

Combined pooled rho:
- `+0.00313`

No replication.

## Interpretation

The live source audit correctly showed that third-down, red-zone and two-minute target shares are real player-specific states and are not redundant with ordinary target share.

But they do **not** explain the persistent target/opportunity errors left behind by M38/WR-R15 and TE-R5P under the frozen cross-position test.

Therefore:
- do not integrate these context shares into entitlement;
- do not rescue red zone for WR-only;
- do not search alternate context windows or thresholds;
- do not reinterpret source nonredundancy as predictive value.

The individual-player hypothesis remains supported by the earlier persistence results; this specific explanatory feature family is simply not the missing mechanism.

The next distinct question is whether **directional recent target-share trajectory** is lost when the system compresses current-season player usage into season-to-date target share plus participation state.

No production change is authorized.
