# WR Player Opportunity Uncertainty V1 — Result

Date: 2026-10-14

Status: **COMPLETE / WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED / RESEARCH ONLY**

Frozen contract:
`docs/research/WR_PLAYER_OPPORTUNITY_UNCERTAINTY_V1_FREEZE.md`

Authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- exact scored head: `3f672b6e4ca675a36525a5efb83db8d0a20055c8`
- workflow: `Research WR Player Opportunity Uncertainty V1`
- run: `38595088817` — SUCCESS
- job: `115959621036`
- artifact: `11908832554`
- artifact digest: `sha256:32fc0e08fc06cdf388151773338f25ce1614cc6038044d054b12cba4c5367bf5`

Parent:
- `WR_PLAYER_PERSISTENCE_MIXED_MECHANISM`
- run `37634044900`
- artifact `11487359185`
- digest `sha256:13bb2df4d239f9f2832370520c3d31a08ed4f39c782766fbfb96295edc11c4e3`

## Frozen disposition

`WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`

Every frozen scientific and integrity gate passed.

## What was tested

The baseline and candidate are both zero-mean opportunity-error distributions
around the unchanged promoted WR receiving-yard architecture.

Baseline:
- contemporaneous pooled WR uncertainty for the target week;
- scale from the median strictly-prior WR opportunity-difficulty state.

Candidate:
- a separate uncertainty scale for each named WR;
- scale from only that WR's own strictly-prior opportunity-difficulty history.

No coefficient was fit. M38/WR-R15 target entitlement and receiving-yard means
were unchanged.

This directly tests whether named-player history improves uncertainty beyond a
position-pool assumption.

## Population

- 3,224 eligible WR player-games
- 211 distinct WRs
- 2023: 1,357 rows / 160 WRs
- 2024: 1,867 rows / 183 WRs

Source row/player parity: PASS.

## CRPS

2023:
- pooled WR baseline: `12.50878`
- named-player candidate: `12.35416`
- gain: `+0.15462`

2024:
- pooled WR baseline: `13.21114`
- named-player candidate: `12.99474`
- gain: `+0.21640`

Pooled:
- baseline: `12.91551`
- candidate: `12.72511`
- gain: `+0.19040` (~1.47% relative)

Both seasons independently improved.

## Player-clustered inference

5,000 paired player-cluster bootstrap replicates:
- observed CRPS gain: `+0.19040`
- 95% CI: `[+0.08180, +0.30846]`
- P(gain > 0): `0.9996`

The result therefore survives within-player dependence-aware inference.

## Coverage

Pooled central 80%:
- baseline coverage: `77.295%`
- candidate coverage: `78.412%`
- absolute nominal gap: `2.705 pp -> 1.588 pp`

Pooled central 90%:
- baseline coverage: `85.298%`
- candidate coverage: `85.453%`
- absolute nominal gap: `4.702 pp -> 4.547 pp`

Both frozen pooled coverage guards passed.

## High-difficulty WRs

Frozen Q4 threshold:
- prior opportunity-difficulty MAE >= `22.36996` yards-equivalent

Q4:
- 806 rows / 85 WRs
- CRPS: `17.87079 -> 17.61484`
- 80% coverage: `63.03% -> 88.46%`
- 90% coverage: `74.57% -> 93.18%`

The pooled WR uncertainty baseline is substantially too narrow for the
historically difficult named players.

## Role diagnostics

These were descriptive and could not rescue the pooled test.

WR1:
- 878 rows / 81 WRs
- CRPS: `16.95538 -> 16.77119`
- 80% coverage: `65.15% -> 79.38%`
- 90% coverage: `76.54% -> 86.56%`

WR2+:
- 2,346 rows / 197 WRs
- CRPS: `11.40357 -> 11.21085`
- 80% coverage becomes slightly less well calibrated;
- 90% coverage becomes less well calibrated;
- CRPS still improves.

Because role slices were diagnostic only, the confirmed disposition rests on
the frozen pooled/player-clustered gates, not a role rescue.

## Integrity

- candidate models fit: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes read: 0
- projection mean changes: 0
- production changed: false

## Scientific interpretation

WR independently replicates the player-first TE result.

After promoted M38/WR-R15 opportunity science has set the named player's mean,
the WR's own strictly-prior opportunity-projection difficulty still improves
probabilistic forecast quality over a contemporaneous pooled WR uncertainty
assumption.

Therefore remaining pass-catcher uncertainty is not adequately represented as
position-only noise.

This does **not** authorize a target or receiving-yard mean correction.

It authorizes only a separately frozen player-specific opportunity-uncertainty
integration/shadow design that preserves:
- M38/WR-R15 mean entitlement;
- team target mass;
- receiving-yard mean;
- sportsbook separation;
- production isolation until prospective confirmation.

## Cross-position implication

Together with:
- `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`; and
- historical `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`;

the repository now has repeated evidence across RB/WR/TE that named-player
history can improve uncertainty/distribution quality beyond shared positional
assumptions while leaving the football mean unchanged.

That is evidence for a player-specific reliability/distribution layer, not for
a new generic positional multiplier.

## Anti-rescue / boundaries

Do not:
- reinterpret this as a mean correction;
- reopen WR-R3 combined calibration or direct bias shrink;
- reopen target-depth width;
- fit a new width coefficient from this exposed result;
- rescue using role-specific thresholds;
- inject sportsbook lines into width;
- use 2026 outcomes to fit the integration mapping.

The next legal step is a separately frozen WR/TE player-specific opportunity
uncertainty integration/shadow design, with RB kept on its already-qualified
separate lineage.
