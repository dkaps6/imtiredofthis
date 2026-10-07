# HISTORICAL AVAILABILITY PARITY DIAGNOSTIC V1 — RESULT

Date: 2026-10-07  
Branch: `research-historical-availability-parity-diagnostic-v1`  
Successful head: `ba882b910a744c9d94420a5ce842b1da763feb61`  
GitHub Actions run: `37687979574` — **SUCCESS**  
Artifact: `11511927864`  
Artifact digest: `sha256:6bcbf63f8dfc5b2b0362e16fbd612653d6f813e5c2d9aec79e9d49c98cb0303b`

## Disposition

`HISTORICAL_AVAILABILITY_MISMATCH_EXPLAINS_PART_OF_COMPRESSION_RESIDUAL_PLAYER_ALLOCATION_REMAINS`

The historical `ACT+INA` universe mismatch is materially important for player
identity and does consume meaningful modeled opportunity mass. It is **not**
sufficient to explain the remaining active-player workload compression.

Therefore:

1. do not treat baseline replay inactive-player failures as current production
   failures — live availability is already promoted;
2. do not reopen availability science;
3. do continue player-specific opportunity research, but only on the residual
   explicit-active population and only after decomposing team-volume error from
   player-share error.

No production change is authorized.

## Mechanical result

Run `37687979574` passed:

- frozen diagnostic tests;
- baseline leakage-safe W1-4 input build;
- exact ACT-only universe derivation from the same nflverse weekly roster
  `status` field;
- baseline opportunity audit;
- ACT-only opportunity audit;
- exact paired comparison;
- diagnostic certification;
- strict repository audit.

Boundary:
- parameters fit: **0**
- automatic promotion: **false**
- sportsbook inputs upstream: **false**
- paid OddsAPI: **false**
- outcome fields used to select ACT-only universe: **false**

Rows:
- baseline opportunity rows: **2,308**
- baseline ACT rows: **2,100**
- baseline INA rows: **208**
- ACT-only rows: **2,106**
- paired exact ACT rows: **2,100**
- ACT-only new identity rows: **6**
- baseline dropped identity rows: **208**

## Opportunity mass assigned to explicit inactive rows

| Position | Opportunity | INA rows | INA modeled opportunity | Share of modeled player opportunity |
|---|---|---:|---:|---:|
| QB | pass attempts | 6 | 151.72 | **4.16%** |
| RB | carries | 35 | 164.08 | **6.36%** |
| RB | targets | 35 | 43.99 | **6.74%** |
| TE | targets | 53 | 117.24 | **10.35%** |
| WR | targets | 79 | 208.77 | **9.12%** |

This is a real historical replay reconstruction defect. Those opportunities
should not be interpreted as representative of current live production, whose
promoted availability stack excludes definitive unavailable players before
opportunity allocation.

## QB identity correction

Removing explicit `INA` rows changed the selected QB in **6 team-weeks**:

- 2026 W1 ATL: Michael Penix Jr. -> Cooper Rush
- 2026 W1 MIA: Brady Cook -> Malik Willis
- 2026 W3 ATL: Cooper Rush -> Michael Penix Jr.
- 2026 W3 CHI: Caleb Williams -> Tyson Bagent
- 2026 W3 WAS: Jayden Daniels -> Marcus Mariota
- 2026 W4 TB: Baker Mayfield -> Jalon Daniels

On these six changed identities:

- baseline mean pass-attempt MAE: **25.29**
- ACT-only mean pass-attempt MAE: **7.49**
- improvement: **17.80 attempts**

Five of the six ACT-only replacement identities recorded nonzero pass attempts.
The CHI replacement also recorded zero attempts.

This confirms that availability/identity selection explains a major subset of
the catastrophic QB zero-state errors. It does not solve QB attempt calibration
among already-correct active identities.

## Paired ACT-player result

The exact same 2,100 ACT opportunity rows were compared before vs after inactive
rows were removed from team allocation.

### RB carries

- paired rows: 436
- baseline MAE: **4.316**
- ACT-only MAE: **4.299**
- MAE improvement: **+0.017**
- active-row MAE: **4.574 -> 4.447** (+0.127)
- active bias: **-3.052 -> -2.734**
- mean predicted carries: **5.540 -> 5.831**
- 15+ carry bias: **-10.287 -> -9.860**

Availability dilution explains part of the underallocation, but a large
high-volume residual remains.

### RB targets

- baseline MAE: **1.375**
- ACT-only MAE: **1.386** (overall slightly worse)
- active-row MAE: **1.539 -> 1.464** (+0.075)
- active bias: **-1.277 -> -1.108**
- mean predicted targets: **1.395 -> 1.549**
- 9+ target bias: **-8.252 -> -8.088**

The active-player bias improves, but high-target backs remain severely
underallocated.

### TE targets

- baseline MAE: **1.751**
- ACT-only MAE: **1.796** (overall slightly worse)
- active-row MAE: **1.713 -> 1.671** (+0.042)
- active bias: **-0.881 -> -0.621**
- mean predicted targets: **2.323 -> 2.553**
- 9+ target bias: **-7.019 -> -6.630**

Again, availability removes some dilution but does not explain focal-TE misses.

### WR targets

- baseline MAE: **2.051**
- ACT-only MAE: **2.062** (overall slightly worse)
- active-row MAE: **2.098 -> 2.056** (+0.042)
- active bias: **-1.106 -> -0.756**
- mean predicted targets: **3.111 -> 3.417**
- 9+ target bias: **-5.324 -> -4.764**

High-volume WRs remain meaningfully underallocated after explicit inactive rows
are removed.

### Paired ACT QB rows

For the 122 QB rows whose selected identity was already ACT in baseline,
pass-attempt predictions are unchanged. This is expected: removing other roster
players does not change the already-frozen QB expected-pass-attempt authority.

The six identity-switch cases above must be evaluated separately and show the
large availability benefit.

## What this means

Availability was a real historical replay-parity blocker, especially for:

- wrong inactive QB identity;
- 6-10% of modeled RB/WR/TE opportunity mass being assigned to explicit inactive
  players.

But the residual pattern survives:

- active RB focal carry workloads remain too low;
- active RB target leaders remain too low;
- active TE target leaders remain too low;
- active WR target leaders remain too low.

Removing `INA` rows shifts opportunity toward active players, but only modestly
reduces active-player MAE. Therefore it would be incorrect to stop here and say
"availability already solved the player problem."

It would also be incorrect to jump directly into another target/carry correction
without determining whether the remaining miss comes from:

1. team opportunity volume; or
2. each player's share of that team opportunity.

## Next authorized diagnostic

`PLAYER_OPPORTUNITY_VOLUME_VS_SHARE_DECOMPOSITION_V1`

Use the ACT-only historical parity universe as the reconstruction basis.

For each individual player-game:

### QB
- predicted team official pass attempts;
- predicted QB attempt share;
- realized team official pass attempts;
- realized QB attempt share.

### RB carries
- predicted team rush attempts;
- predicted player carry probability;
- realized team rush attempts;
- realized player carry share.

### RB / WR / TE targets
- predicted team official pass attempts;
- predicted player target probability per pass attempt;
- realized team official pass attempts;
- realized player targets / realized team official pass attempts.

Compute without fitting:

- baseline predicted opportunity:
  `pred_team_volume * pred_player_share`
- oracle-team-volume diagnostic:
  `actual_team_volume * pred_player_share`
- oracle-player-share diagnostic:
  `pred_team_volume * actual_player_share`

Compare each to actual individual opportunity.

Interpretation:
- if oracle team volume removes most error, team-volume modeling is dominant;
- if oracle player share removes most error, individual share allocation is
  dominant;
- if both matter, quantify both and continue only on the larger unresolved
  mechanism.

This is diagnostic only. Realized shares/volumes are outcome-side oracles and may
never become production inputs.
