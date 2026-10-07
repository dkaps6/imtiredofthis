# TE Player Error Persistence V1 — Result

**STATUS: COMPLETE / TE_PLAYER_ERROR_PERSISTENCE_DETECTED**

Frozen plan:
`docs/research/TE_PLAYER_ERROR_PERSISTENCE_V1_PLAN.md`

Authority:
- branch: `research-te-player-error-persistence-v1`
- run: `37630274063` — **SUCCESS**
- source SHA: `a44ec422a28fd60f9fac92fe88def17fe9e85599`
- result artifact: `11486201614`
- result artifact digest: `sha256:52b488b4e989ecd18cc061ad5f3697e7aa31ac8ddbb8e55f97d44e1cf0cd14a6`

Parent TE-R5P authority:
- run `34152797603`
- artifact `10029942404`
- digest `sha256:f9951441b748ef72514dbc81adf6fbe9cd023c9bf64ecb52a016840989ab4cdb`
- exact OOS casebook rows: 3,214
- exact TE-R5P projection: `candidate_rec_yards_r5p`

Final disposition:

`TE_PLAYER_ERROR_PERSISTENCE_DETECTED`

## Scoreable persistence panel

Strictly-prior same-player history:
- latest up to 8 prior TE-R5P OOS player-games
- minimum 4 prior games
- target seasons 2024 and 2025
- scoreable rows: **2,005**
- distinct TEs: **121**
- same/future history violations: **0**

Support:
- 2024: 978 rows / 99 players
- 2025: 1,027 rows / 108 players
- support gate: **PASS**

## A — Directional bias persistence: PASS

2024:
- Spearman(prior signed error, current signed error): **0.140433**
- sign agreement: **54.60%**

2025:
- Spearman: **0.133781**
- sign agreement: **55.70%**

Pooled:
- Spearman: **0.137799**
- sign agreement: **55.16%**

Player-cluster bootstrap:
- 5,000 reps
- P(Spearman > 0): **1.000**
- 95% CI: **[0.0740, 0.1964]**

Interpretation:
The direction in which TE-R5P tends to miss an individual TE is weak-to-moderately persistent and replicates by season.

## B — Individual difficulty persistence: PASS

2024:
- Spearman(prior mean absolute error, current absolute error): **0.290586**

2025:
- Spearman: **0.266578**

Pooled:
- Spearman: **0.279836**

Player-cluster bootstrap:
- P(Spearman > 0): **1.000**
- 95% CI: **[0.2193, 0.3327]**

This is the strongest of the three families.

Interpretation:
Some TEs are persistently easier or harder for the promoted TE-R5P receiving-yard system to project, even after TE-R5P's individualized entitlement layer is already active.

## C — Extreme-miss persistence: PASS

Target event:
- absolute TE-R5P receiving-yard error >= 30 yards.

2024:
- Spearman(prior miss30 rate, current miss30): **0.207254**
- target miss30 rate: 16.05%

2025:
- Spearman: **0.219453**
- target miss30 rate: 12.37%

Pooled:
- Spearman: **0.214532**
- target miss30 rate: 14.16%

Player-cluster bootstrap:
- P(Spearman > 0): **1.000**
- 95% CI: **[0.1458, 0.2748]**

Interpretation:
A TE's strictly-prior history of large TE-R5P misses contains real information about his chance of another large miss.

## Scientific meaning

This is a strong player-level result because the target projection is already the promoted individual-entitlement TE-R5P architecture.

Therefore the signal cannot be dismissed as merely:
- “TEs have different target shares,” or
- “the model only knows that the player is a TE.”

TE-R5P already individualizes entitlement using strict-prior player participation/continuity inside a finite TE room.

Yet after that layer, individual residual behavior still persists.

The correct interpretation is:

> TE-R5P captures meaningful individual opportunity, but it does not fully capture stable individual differences in how receiving-yard outcomes behave around that opportunity.

That remaining state may arise from persistent efficiency, role geometry, catch/yard translation, ceiling/variance behavior, or another player-level mechanism.

This result does **not** identify which mechanism is causal.

## Important non-authorization

This result does NOT reopen:
- TE-R5P Receiving-Yards Width V2, which is failed/closed;
- generic TE YPT additive adjustment;
- position-level TE matchup boosts;
- arbitrary same-player bias shrinkage.

Do not immediately add prior error to the projection.

The next proper step is a diagnostic decomposition of the detected same-player persistence into the already-observable TE-R5P mechanisms:
- target/entitlement residual;
- catch/reception translation;
- yard-per-target / yard-per-reception efficiency;
- large-miss / ceiling behavior.

Only if one mechanism itself replicates can a new candidate be frozen.

Research boundary:
- candidate models fit: 0
- sportsbook inputs: 0
- 2026 outcomes: 0
- production changed: false
