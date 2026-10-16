# Player Personnel-Conditioned Opportunity V1 — Result

Date: 2026-10-14

Status: **COMPLETE / NO_ACTIONABLE_PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY / RESEARCH ONLY**

Frozen contract:
`docs/research/PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY_V1_FREEZE.md`

Authority:
- branch: `research-individual-opportunity-roadmap-2026-10-08`
- freeze parent: `91f23daa237991c7083c1c91b858cb6ad099df92`
- freeze commit: `f5eed57c01d349cb3110f46ea63b3b678c8e3c56`
- exact green scored head: `5afaefb2dd250cd6c0c5a9ea5cac473c5dd8439c`
- workflow: `Research Player Personnel-Conditioned Opportunity V1`
- run: `38627780518` — SUCCESS
- job: `116071243946`
- artifact: `11924472635`
- artifact digest: `sha256:3522e4fb14d6a0fd31c9b5733fe719887bbabeada55c037303e9285282500219`

Mechanical paper trail:
- initial run `38627625678` failed before science because Polars -> pandas invoked a pyarrow / NumPy ABI mismatch;
- the only repair was the dataframe bridge (`to_dicts()` before `to_pandas()`);
- no hypothesis, feature, bucket, history window, support gate, scoring gate, or threshold changed.

## Frozen disposition

`NO_ACTIONABLE_PLAYER_PERSONNEL_CONDITIONED_OPPORTUNITY`

Both WR and TE had adequate support and clean source integrity. Neither position
cleared the frozen player-first scientific gates.

This closes the exact `11 / 12 / OTHER` named-player personnel-conditioned
target-propensity construction. Do not rescue it by changing personnel buckets,
history windows, role subsets, or target seasons.

## What was tested

The hypothesis grain was the named WR/TE.

For each target player-game, the diagnostic reconstructed strictly-prior
play-level participation and asked:

1. when this exact player was on the field on an offensive dropback, how often
   was he targeted in `11`, `12`, and `OTHER` personnel?;
2. what personnel mix had his team used in its latest strictly-prior games?;
3. did weighting that player's own personnel-specific target rates by the team's
   strictly-prior personnel mix explain the next promoted-authority opportunity
   miss?

The target-game personnel mix was never used.

No position-average target-rate coefficient was fit.
No prior residual was fed back into the projection.
No sportsbook information was used.

## Source / integrity result

Parent rows:
- WR: 3,224
- TE: 2,005
- total: 5,229

Identity:
- mapped parent rows: 5,229 / 5,229
- identity coverage: **100%**
- ambiguous name/team/season identity keys: **0**

Historical play state:
- PBP rows loaded: 147,928
- participation rows loaded: 137,271
- joined offensive dropback plays: 60,815
- expanded WR/TE on-field dropback-player rows: 238,723
- player-game-personnel rows: 27,074
- team-game-personnel rows: 4,445
- malformed parallel player/position lists skipped: **0**
- same/future history violations: **0**

Final eligible player-games:
- total: 4,220
- WR: 2,715
- TE: 1,505

Integrity therefore passed. This is a scientific null, not a source-limited
result.

## WR result

Support: **PASS**

2023:
- rows: 1,287
- players: 154
- Spearman(`personnel_lift`, realized opportunity gain): **-0.00106**
- Q4-Q1 realized opportunity gain: **-0.8824**

2024:
- rows: 1,428
- players: 162
- Spearman: **+0.04513**
- Q4-Q1 realized opportunity gain: **+3.4481**

Pooled:
- rows: 2,715
- players: 203
- Spearman: **+0.02356**
- Q4-Q1 realized opportunity gain: **+1.4209**

Player-cluster bootstrap:
- 5,000 replicates
- observed rho: `+0.02356`
- 95% CI: **[-0.01996, +0.06960]**
- P(rho > 0): `0.8464`

Interpretation:
2024 was directionally favorable, but 2023 was null/slightly negative and the
player-cluster confidence interval crossed zero. The frozen contract required
the expected direction in both seasons and dependence-aware support.

Do not rescue the 2024 slice.

## TE result

Support: **PASS**

2024:
- rows: 739
- players: 89
- Spearman: **-0.03858**
- Q4-Q1 realized opportunity gain: **-2.1834**

2025:
- rows: 766
- players: 94
- Spearman: **-0.04061**
- Q4-Q1 realized opportunity gain: **-2.0308**

Pooled:
- rows: 1,505
- players: 109
- Spearman: **-0.03950**
- Q4-Q1 realized opportunity gain: **-2.0043**

Player-cluster bootstrap:
- 5,000 replicates
- observed rho: `-0.03950`
- 95% CI: **[-0.09566, +0.01675]**
- P(rho > 0): `0.0876`

Interpretation:
The exact personnel-conditioned target-propensity hypothesis is not supported
for TE and points slightly opposite the frozen expected direction in both
seasons.

## Scientific interpretation

The previously confirmed named-player opportunity persistence for WR and TE is
real, but this broad personnel-family construction does **not** explain it.

Therefore:
- do not feed prior residuals directly back into means;
- do not retune `11 / 12 / OTHER` buckets;
- do not carve out 2024 WR after seeing the result;
- do not call personnel generally irrelevant.

What is closed is narrower:

> A player's own strictly-prior target rate by broad offensive personnel family,
> weighted by the team's recent strictly-prior personnel mix, is not a stable
> explanation of his next promoted-authority opportunity residual.

The remaining individual-mean frontier must use genuinely new named-player
pregame football state, not another transform of this result. Higher-resolution
player role/alignment/route state is scientifically distinct if a valid
historical and prospective source contract can be established.

## Separate player-first findings that remain alive

This null does not affect the already-confirmed player-specific uncertainty
evidence:

- `TE_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
  - pooled CRPS improvement ~3.28%;
- `WR_PLAYER_OPPORTUNITY_UNCERTAINTY_CONFIRMED`
  - pooled CRPS improvement ~1.47%;
- `RB_YARD_DIFFICULTY_MC_WIDTH_QUALIFIED`
  - pooled CRPS improvement ~1.22%.

Those are separate distribution/uncertainty findings and remain valid.

## Integrity boundary

- candidate models fit: 0
- sportsbook inputs: 0
- OddsAPI calls: 0
- 2026 outcomes read: 0
- target-game personnel used: 0
- prior residual fed to projection: 0
- production changed: false
