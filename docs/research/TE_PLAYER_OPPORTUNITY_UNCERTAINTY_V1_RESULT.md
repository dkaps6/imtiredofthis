# TE Player Opportunity Uncertainty V1 — Result

Date: 2026-10-14

Status: **COMPLETE / TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED / RESEARCH ONLY**

Frozen contract:
`docs/research/TE_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_FREEZE.md`

Authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- exact scored head: `32f11c67b3028d49b6d915085b3ec7c0d972181f`
- workflow: `Research TE Player Opportunity Uncertainty V1`
- run: `38594365393` — SUCCESS
- job: `115957146392`
- artifact: `11909006619`
- artifact digest: `sha256:76308043d022ebf57aa7728c330d8562051d31bc1ae5b65dddc33ebf225e82a3`

Parent:
- `TE_PLAYER_PERSISTENCE_MIXED_MECHANISM`
- run `37631011196`
- artifact `11486082927`
- digest `sha256:1b208531a8cf34735760081b7a232c4cd164a551c325f518d33428a50d7da8ab`

## Frozen disposition

`TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`

All frozen scientific and integrity gates passed.

## What was tested

The test compared two zero-mean opportunity-error distributions around the
unchanged TE-R5P receiving-yard mean.

Baseline:
- one contemporaneous pooled TE uncertainty width per target week;
- width derived from the median strictly-prior TE opportunity difficulty state.

Candidate:
- a separate uncertainty width for each named TE;
- width derived only from that TE's own strictly-prior
  `prior8_abs_opportunity_error_mean`.

No coefficient was fit. No projection mean changed.

This is therefore a direct player-vs-position-pool uncertainty test.

## Population

- 2,005 eligible TE player-games
- 121 distinct TEs
- 2024: 978 rows / 99 TEs
- 2025: 1,027 rows / 108 TEs

Source parity:
- exact expected rows: PASS
- exact expected players: PASS

## CRPS

2024:
- pooled TE baseline: `10.36052`
- named-player candidate: `9.95972`
- gain: `+0.40080`

2025:
- pooled TE baseline: `9.02549`
- named-player candidate: `8.78711`
- gain: `+0.23838`

Pooled:
- baseline: `9.67669`
- candidate: `9.35909`
- gain: `+0.31760` (~3.28% relative)

Both seasons independently improved.

## Player-clustered inference

5,000 paired player-cluster bootstrap replicates:
- observed CRPS gain: `+0.31760`
- 95% CI: `[+0.19727, +0.46175]`
- P(gain > 0): `1.000`

The uncertainty improvement is not being driven by treating repeated
player-games as independent rows.

## Coverage

Pooled central 80%:
- baseline coverage: `77.456%`
- candidate coverage: `80.000%`
- absolute nominal gap: `2.544 pp -> 0.000 pp`

Pooled central 90%:
- baseline coverage: `83.441%`
- candidate coverage: `86.234%`
- absolute nominal gap: `6.559 pp -> 3.766 pp`

Both frozen pooled coverage guards passed.

## High-difficulty TEs

Frozen Q4 threshold:
- prior opportunity-difficulty MAE >= `16.70293` yards-equivalent

Q4:
- 502 rows / 43 TEs
- CRPS: `15.05654 -> 14.64296`
- 80% coverage: `59.16% -> 84.66%`
- 90% coverage: `68.92% -> 90.24%`

This is the clearest practical signal:
the pooled TE uncertainty model is dramatically too narrow for the TEs whose
own history says they are difficult to project.

## Low-difficulty TEs

Q1:
- 502 rows / 73 TEs
- CRPS: `5.29365 -> 4.63944`
- 80% coverage gap improved materially
- 90% coverage gap worsened slightly

Low-difficulty behavior was descriptive only and not a hard gate.

## Integrity

- candidate models fit: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes read: 0
- projection mean changes: 0
- production changed: false

## Scientific interpretation

This is direct evidence that TE uncertainty should not be treated as purely
position-level after TE-R5P has produced the named player's mean.

The named player's own strictly-prior opportunity-projection difficulty
contains enough stable information to improve probabilistic forecast quality
over a contemporaneous pooled TE uncertainty assumption.

This does **not** authorize changing TE-R5P target or receiving-yard means.

It authorizes only a separately frozen integration/shadow study that maps this
same-player opportunity-difficulty state into the real TE target-allocation /
receiving-yard Monte Carlo distribution while preserving:
- TE-R5P mean entitlement;
- team target mass;
- receiving-yard mean;
- sportsbook separation;
- production isolation until prospective confirmation.

## Anti-rescue / boundaries

Do not:
- reinterpret this as a mean correction;
- combine it with the failed global TE Width V2 factor;
- retune the last-8/min-4 source;
- add TE1/TE2 carveouts after seeing this result;
- use sportsbook lines to choose width;
- use 2026 outcomes to fit the integration mapping.

The next legal step is a separately frozen TE-R5P player-specific opportunity
uncertainty integration/shadow candidate.
