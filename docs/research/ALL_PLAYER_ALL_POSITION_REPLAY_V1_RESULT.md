# ALL PLAYER / ALL POSITION REPLAY V1 — RESULT

Date: 2026-10-07  
Branch: `research-all-player-all-position-replay-v1`  
Exact successful head: `5e6d15cdebbb47d88217eb1f3437c2a506d56051`  
GitHub Actions run: `37683439543` — **SUCCESS**  
Artifact: `11509468659`  
Artifact digest: `sha256:abb5bb7ba85d13306c0f64f83dd7c1e9840847c3ad308b18d859724121b8a711`

## Disposition

`ALL_PLAYER_REPLAY_VALID__TARGET_DEPTH_FULL_TRANSFORM_NOT_CONFIRMED__PLAYER_OPPORTUNITY_ALLOCATION_NEXT`

This replay is a valid diagnostic of the current protected individual-player stack across 2026 Weeks 1-4. It does **not** authorize any automatic production promotion.

The new WR/TE target-depth distribution transform does **not** confirm on pooled current-season W1-4 evidence. Do not rescue it post hoc by selecting a favorable threshold or subgroup.

The dominant new player-level failure signal is instead **opportunity / participation allocation**: the model compresses individual workload toward the middle, assigning too much production to players with little or no realized involvement while under-projecting players who become true high-volume focal points.

## Frozen replay boundary

- Season: 2026
- Weeks: 1-4
- Primary unit: individual player-game-market
- Universe: full leakage-safe pregame football roster universe, not sportsbook-only
- Positions: QB, RB, WR, TE
- Paid OddsAPI: **not used**
- Sportsbook fields upstream of football projection: **false**
- Week-5 RB room-allocation shadow applied retrospectively: **0 rows**
- Target Share Trajectory V1 eligible in W1-4: **0 rows**
- Automatic promotion: **false**

Protected production ordering was preserved:
- QB promoted M89/M90 synthesis
- Week-1 RB P3 only for exact frozen player identities from the certified 107-player parent
- non-Week1 RB rush+rec conservation V2
- M38 -> TE-R5P -> WR-R15 before explicit WR/TE simulation
- WR/TE target-depth distribution layer mean-neutral by construction

All replay contract tests, replay-boundary certification, and strict repository audit passed.

## Replay coverage

- Point rows: **4,488**
- Unique player-weeks: **1,837**
  - QB: 128
  - RB: 471
  - TE: 490
  - WR: 748
- QB synthesis rows: 128
- Week-1 RB P3 rows affected: 210
- RB rush+rec V2 rows: 352
- WR/TE rec-yards distribution rows: 1,238

### Point diagnostics

| Position | Market | Rows | MAE | Median AE | Bias | RMSE |
|---|---:|---:|---:|---:|---:|---:|
| QB | pass_yards | 128 | 73.995 | 53.275 | +28.374 | 97.963 |
| RB | rec_yards | 471 | 10.459 | 8.389 | +2.409 | 13.558 |
| RB | receptions | 471 | 1.182 | 1.026 | +0.229 | 1.551 |
| RB | rush_rec_yards | 471 | 24.368 | 20.719 | +3.758 | 31.591 |
| RB | rush_yards | 471 | 19.110 | 14.634 | +1.755 | 26.244 |
| TE | rec_yards | 490 | 15.615 | 13.786 | +5.350 | 20.041 |
| TE | receptions | 490 | 1.427 | 1.350 | +0.530 | 1.790 |
| WR | rec_yards | 748 | 22.299 | 17.186 | +1.536 | 29.735 |
| WR | receptions | 748 | 1.514 | 1.294 | +0.231 | 1.959 |

These are diagnostic values for the full roster universe, not sportsbook-board accuracy rates.

## Individual participation / workload signal

### Pregame-universe player-weeks with no weekly stat row

| Position | Player-weeks | No-stat player-weeks | Rate |
|---|---:|---:|---:|
| QB | 128 | 10 | 7.81% |
| RB | 471 | 142 | 30.15% |
| TE | 490 | 222 | 45.31% |
| WR | 748 | 236 | 31.55% |
| **Total** | **1,837** | **610** | **33.21%** |

The replay intentionally scores the complete pregame player universe. This exposes a player-specific participation problem that a sportsbook-only sample hides.

Mean projection on those verified-zero/no-stat rows:
- QB pass yards: **224.1**
- RB rush yards: **16.5**
- RB receiving yards: **9.3**
- WR receiving yards: **19.9**
- TE receiving yards: **14.9**

This is not evidence to remove those players from the universe after outcomes. It is evidence that the model needs a pregame player-level participation / role probability before assigning full workload.

### Active-row workload compression

Among rows with an NFLVerse weekly stat record, projection error is strongly ordered by realized opportunity count. The following correlations are **retrospective diagnostics only**; realized opportunity is not an authorized pregame feature.

Spearman correlation of `projection error = projection - actual` versus realized opportunity:
- QB pass_yards: **-0.586**
- RB rec_yards: **-0.611**
- RB receptions: **-0.809**
- RB rush_rec_yards: **-0.683**
- RB rush_yards: **-0.663**
- TE rec_yards: **-0.550**
- TE receptions: **-0.664**
- WR rec_yards: **-0.592**
- WR receptions: **-0.698**

The sign is consistent across every scored market: low-workload players tend to be overprojected and high-workload players tend to be underprojected.

Examples of the compression:
- WR rec_yards, 1-2 realized targets: bias **+10.88 yards**
- WR rec_yards, 9+ realized targets: bias **-45.16 yards**
- TE rec_yards, 1-2 realized targets: bias **+8.06 yards**
- TE rec_yards, 9+ realized targets: bias **-40.34 yards**
- RB rush_yards, 1-3 realized carries: bias **+13.19 yards**
- RB rush_yards, 15+ realized carries: bias **-35.81 yards**
- QB pass_yards, <=20 realized attempts: bias **+110.00 yards**
- QB pass_yards, 41+ realized attempts: bias **-83.30 yards**

This is the clearest cross-position player-level finding from the replay.

## WR/TE target-depth distribution result

Feature availability:
- 859 / 1,238 rows = **69.39%**
- exact mean-invariance max gap: **4.26e-14** (passes <=1e-10)

Pooled empirical CRPS:
- baseline: **13.43749**
- shadow: **13.50947**
- baseline - shadow: **-0.07198**
- relative pooled change: approximately **0.54% worse**

Feature-available rows only:
- baseline CRPS: **15.11523**
- shadow CRPS: **15.21897**
- approximately **0.69% worse**

Feature-available + weekly-stat rows only:
- baseline CRPS: **16.37459**
- shadow CRPS: **16.40012**
- approximately **0.16% worse**

Therefore:

`TARGET_DEPTH_FULL_SYMMETRIC_DISTRIBUTION_TRANSFORM_NOT_CONFIRMED_W1_W4`

The historical Target Depth Dispersion V1 finding remains intact as a player-specific difficulty signal. What failed here is the frozen symmetric uncertainty transform.

### Descriptive asymmetry only — NOT a promotion rule

The frozen transform narrows distributions when depth dispersion is below the positional anchor and widens them when above it.

Descriptively:
- low-scale narrowing was harmful;
- high-scale widening was favorable.

Feature-available quartile CRPS change (baseline - shadow):
- Q1 low scale: **-0.7658**
- Q2: **-0.1723**
- Q3: **+0.0704**
- Q4 high scale: **+0.4496**

Do **not** convert this outcome-seen pattern into a threshold such as “only apply when scale > 1.” Any one-sided widening hypothesis must be separately pre-registered and tested without using these same outcomes to select its rule.

## Preserved live-board overlap

For a single canonical downstream comparison, use `weeks1_4_full_graded.csv` from the preserved Week-4 postmortem artifact. It contains **1,565 unique player-week-market rows** with no duplicate football identity.

Cross-checks:
- all 1,163 overlapping rows with the earlier W1-3 cumulative graded file have identical line/model/actual values;
- all 402 W4 rows match `week4_selected_full_graded.csv` exactly;
- all 746 W1-2 rows in the committed graded file match the cumulative W1-4 file exactly.

On those 1,565 canonical live-board rows:
- reconstructed replay MAE: **18.419**
- archived live model MAE: **18.267**
- replay is **+0.152 MAE worse** overall
- replay is closer than the archived model on **48.88%** of rows

Small descriptive improvements exist in RB rec_yards, RB receptions and RB rush_rec_yards, but there is no broad current-stack improvement on the preserved sportsbook overlap. This is consistent with the fact that the newly frozen trajectory and RB room-allocation shadows are not legally back-applicable to W1-4, and target-depth is mean-neutral.

## Scientific interpretation

The replay does **not** support a claim that the remaining problem is a generic positional mean.

The stronger finding is the opposite: the current stack still lacks enough **individual player workload separation**.

The current player projections are too compressed:
1. full-roster players with little/no game involvement still receive material workload;
2. low-opportunity active players remain too high;
3. true focal/high-volume players remain too low.

That pattern appears in pass attempts, carries, targets, yardage, and reception output across positions.

## Next authorized diagnostic

Next work should stay player-specific and trace the model's own pregame opportunity allocation before efficiency:

`PLAYER_OPPORTUNITY_ALLOCATION_AUDIT_V1`

Required comparison by individual player-game:
- QB predicted pass attempts vs actual attempts
- RB predicted carries vs actual carries
- RB predicted targets vs actual targets
- WR predicted targets vs actual targets
- TE predicted targets vs actual targets
- predicted zero/near-zero involvement versus realized zero involvement
- team-volume conservation preserved
- no sportsbook inputs
- no target-week/future leakage
- no new threshold or production promotion during the audit

The purpose is to determine whether the compression is created at:
- pregame participation / role state,
- team opportunity allocation,
- or downstream efficiency.

Do not open a new generic position-level calibration lane before this player opportunity audit is resolved.

## Follow-up RB receiving-room impact resolution

After this global replay identified cross-position workload compression, the RB
receiving allocation lane was isolated and tested separately using the completed
2026 Weeks 1-4 outcomes.

Authority:
- impact run: `37703522415` — SUCCESS
- impact artifact: `11518826839`
- Week-5 prospective lock run: `37705464974` — SUCCESS
- Week-5 lock artifact: `11518604971`

Frozen no-fit mechanism:
- state: strict-prior `prior_rb_room_share`
- preserve exact total RB receiving-room target entitlement
- redistribute only within the RB/FB room
- no coefficient fit
- no threshold search
- no sportsbook inputs
- rushing held exactly unchanged

### Weeks 1-4 retrospective individual-player impact

Across 436 RB/FB player-games:

- target MAE: **1.3863 -> 1.2810** (**7.60% better**)
- receptions MAE: **1.1906 -> 1.1451** (**3.83% better**)
- receiving-yards MAE: **10.7223 -> 10.4121** (**2.89% better**)
- rush+receiving-yards MAE: **24.0169 -> 23.6176** (**1.66% better**)
- rush-yards projections: **exactly unchanged**

The target and receiving-yard MAE improved in **all four completed weeks**.

Player-by-player closer counts:
- targets: candidate 242 / baseline 193 / tie 1
- receptions: candidate 247 / baseline 189
- receiving yards: candidate 251 / baseline 185
- rush+receiving yards: candidate 248 / baseline 188

High-workload player-games with 6+ realized targets improved more:
- target MAE: **10.52% better**
- receptions MAE: **6.06% better**
- receiving-yards MAE: **5.12% better**

Disposition:

`RB_RECEIVING_SHARE_RETROSPECTIVE_IMPACT_CONFIRMED__PROSPECTIVE_VALIDATION_FROZEN`

This is real individual-player projection improvement, not only an intermediate
share metric.

### Week-5 prospective lock

The exact same rule is frozen prospectively before Week-5 outcomes:

- exact frozen RB identities: **98**
- RB rooms: **30**
- strict-prior receiving-history coverage: **98 / 98**
- identity authority: frozen Week-5 RB player-state lock
- history bridge: exact GSIS identity
- parameters fit: **0**
- Week-5 outcomes read: **0**
- sportsbook inputs: **0**
- production changed: **false**
- canonical row digest:
  `sha256:4c5d325a552673d64c91e21b812dd05f5b15adb28c479640e382d4b4baf4ffd7`

The Week-5 lock is confirmation evidence only and does not by itself authorize
production promotion.

## Updated position interpretation

The all-player replay's cross-position workload-compression finding remains the
main organizing result.

However, RB is no longer merely an unresolved example of that problem:
the receiving-room component now has a concrete, no-fit player-level mechanism
that improves completed current-season projections and is frozen prospectively.

For WR/TE:
- the full symmetric target-depth distribution transform remains unconfirmed;
- target-share trajectory remains ineligible in W1-4 under its exact contract;
- opportunity allocation / participation remains the unresolved player-level
  mechanism family.

For QB:
- the replay still indicates participation / attempt-volume compression is more
  important to audit next than opening another generic positional mean lane.

