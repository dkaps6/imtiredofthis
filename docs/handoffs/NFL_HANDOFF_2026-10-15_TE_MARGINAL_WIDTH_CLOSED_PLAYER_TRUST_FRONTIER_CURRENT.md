# NFL HANDOFF — 2026-10-15 — TE MARGINAL WIDTH CLOSED / PLAYER TRUST FRONTIER

GitHub is canonical. This supersedes the 2026-10-14 TE allocation handoff for immediate research execution while preserving all prior frozen lineage.

Repository:
- `dkaps6/imtiredofthis`

Production main:
- `9b70b12e2fb4bc39f08556f244370068aad9c47b`
- Week 6 production/live board is complete and parked.
- No production model changes came from this research.

Active research:
- branch `research-individual-opportunity-roadmap-2026-10-08`
- draft PR #672
- continuity Issue #535

Canonical status:
`PLAYER_INDIVIDUALIZATION_PARTIAL`

## 1. Individual-player standard remains non-negotiable

This is an individual-player projection system.

Position-wide averages are guardrails, not the hypothesis grain.

Do not interpret a failed transmission law as proof that the underlying named-player signal is false.

## 2. Prior confirmed/closed state remains valid

WR targeted-route:
- named-WR YPT/catch matchup mechanism confirmed;
- exact full-stack mean transmission closed;
- do not rescue.

WR opportunity uncertainty:
- confirmed as a named-player phenomenon;
- exact allocation-only bridge closed;
- existing WR joint allocation lineage remains authority.

TE opportunity/allocation uncertainty:
- `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`;
- `TE_PLAYER_ALLOCATION_DIFFICULTY_CONFIRMED`;
- exact joint Dirichlet transmission closed.

RB:
- historical same-player yard difficulty qualification remains valid.

QB:
- no generic mean/YPA/bias reopening.

## 3. NEW result — TE marginal receiving-yard width

Frozen result:
- `docs/research/TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_V1_RESULT.md`

Canonical successful replay:
- run `38800926331` SUCCESS
- job `116649432539`
- exact head `e7961d5bf6abe9097a8fbf21d776fb1696a57bba`
- artifact `12003898778`
- digest `sha256:0c59e6e551e6ac025ac21e4098c5298a3a6ac3e16ecef9f7cd5e8d2bb0db73a4`

Disposition:
`NO_ACTIONABLE_TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS`

Support:
- 1,476 scored rows
- 114 TEs
- 2024: 717 rows / 92 TEs
- 2025: 759 rows / 103 TEs
- exact mean drift <= 1.42e-14
- no sportsbook / no OddsAPI / no 2026 outcomes / no production mutation.

### Named vs canonical is genuinely positive

2024:
- CRPS `12.74750 -> 12.61581`
- gain `+0.13168`

2025:
- CRPS `11.93887 -> 11.85061`
- gain `+0.08826`

Pooled named-vs-canonical:
- gain `+0.10936`
- player-cluster 95% CI `[+0.01851,+0.20876]`
- game-cluster 95% CI `[+0.02306,+0.19571]`

Coverage improved sharply.
High-difficulty Q4 improved.
2025 69 TEs >=6 games:
- median player gain `+0.02004`
- 55.07% positive.

Therefore the current canonical TE-R5P receiving-yard marginal is under-dispersed and the named-player state is scientifically informative.

### But named does not beat pooled TE allocation width

2024:
- pooled CRPS `12.51938`
- named CRPS `12.61581`
- named-vs-pooled `-0.09643`

2025:
- pooled CRPS `11.80102`
- named CRPS `11.85061`
- named-vs-pooled `-0.04959`

2025 named-vs-pooled clustered CIs cross zero and point negative.

Failed gates:
- 2024 named beats pooled CRPS
- 2025 named beats pooled CRPS
- 2025 named-vs-pooled player-cluster CI low > 0
- 2025 named-vs-pooled game-cluster CI low > 0

Exact mapping CLOSED:
`same-player trailing allocation MAE -> Normal target sigma -> pregame YPT -> independent rec-yard marginal overlay`.

Do not rescue with scaling, caps, slices, subgroups, alternate windows, alternate YPT conversion, alternate noise family, or post-result player filtering.

No Week 6 prospective lock is warranted for this failed width mapping.

## 4. What the result actually teaches

The signal is not fake.

Instead:
- canonical TE width is too narrow;
- a pooled TE allocation uncertainty control improves it;
- named-player width also improves it;
- but the exact named sigma magnitudes are too uneven to claim incremental distribution calibration over pooled.

This separates:
1. **signal existence** — confirmed;
2. **exact distribution transmission** — failed;
3. **possible player-level trust/selectivity value** — not yet tested.

Do not collapse those three statements.

## 5. Immediate legal frontier — PLAYER TRUST / SELECTIVITY

Do not force TE difficulty into another width law.

Freeze a separate research-only reliability question:

> Does strictly-prior named-player TE allocation difficulty predict which existing
> canonical TE-R5P player projections/distributions will be reliable vs fragile,
> after removing contemporaneous pooled TE difficulty?

This is a player-level **trust/selectivity** layer, not a projection transform.

Requirements:
- preserve canonical TE-R5P mean and distribution unchanged;
- no sportsbook input in the historical discovery test;
- no coefficient tuning from outcomes;
- 2024 development + 2025 independent confirmation;
- player-cluster and game-cluster inference;
- prove incremental information beyond the contemporaneous pooled TE state;
- evaluate canonical absolute mean error and canonical CRPS;
- pre-freeze exact ranking/group rules;
- no post-result TE subgroup rescue.

If qualified:
- freeze 2026 Week 6 named-TE trust labels from the already-generated pregame football slate **before reading Week 6 outcomes**;
- research shadow only;
- no production promotion without prospective scoring.

If failed:
- close named-TE trust/selectivity from this exact difficulty state and move to the next individual-player mechanism.

## 6. Week 6 production state — parked

Week 6 live-priced production is already complete:
- final replay run `38792341891` SUCCESS
- exact main head `9b70b12e2fb4bc39f08556f244370068aad9c47b`
- artifact `12000113891`
- one paid OddsAPI acquisition total;
- preserved replay used zero additional credits.

Do not return to PIT-TB/live tickets unless the user asks.

For prospective research locks, prefer football-only/no-odds Week 6 authority:
- no-odds run `38790425676`
- artifact `11999618082`
- digest `sha256:d7473705d4534c3172b58843272e7d95f5fbbd9aa910238d48916a8567cde1de`.

## 7. Restart rule

Read:
1. `AGENTS.md`
2. newest TOP only in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this handoff
4. `docs/research/TE_PLAYER_ALLOCATION_MARGINAL_REC_YARDS_V1_RESULT.md`
5. newest Issue #535 comments
6. live PR #672 / branch / exact-head Actions

Do not recursively reread old handoffs.

No paid OddsAPI without explicit authorization.
No production mutation from research.
