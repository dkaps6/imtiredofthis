# PLAYER LANDSCAPE TRANSMISSION AUDIT V1 — RESULT

Date: 2026-10-07  
Branch: `research-player-landscape-transmission-audit-v1`  
Successful head: `35820953a255ce0f8413c2c468501df7f267fa96`  
GitHub Actions run: `37717016519` — **SUCCESS**  
Artifact: `11524296325`  
Artifact digest: `sha256:e043133d4b9a48ce472dde8810d4d77d636f4ce2fbcee14708b23fe0ea35f6a0`

## Disposition

`INDIVIDUAL_PLAYER_CORE_CONFIRMED__LANDSCAPE_TRANSMISSION_INCOMPLETE`

The NFL stack is genuinely individualized at important layers. It is **not**
merely assigning position-average projections to named players.

However, the stronger standard is not yet satisfied everywhere:

> the final projection should coherently transmit the specific player's role,
> recent workload, room hierarchy, team environment, individual efficiency,
> opponent defense, player-role matchup, injuries/vacancies, and uncertainty.

The audit proves that meaningful parts of that chain are already player-specific,
but also identifies concrete football information that is available upstream and
does not currently reach generic RB/WR/TE projections.

No production change is authorized by this audit.

## Mechanical validity

Successful run `37716318460` passed:

- player-landscape governance tests;
- leakage-safe Week-5 football input construction;
- full dynamic player-landscape trace;
- audit certification;
- strict repository audit;
- evidence upload.

Boundary:

- season: **2026**
- target week: **5**
- pregame player universe: **467 players**
- Week-5 outcomes used: **false**
- sportsbook inputs used: **false**
- paid OddsAPI used: **false**
- parameters fit by audit: **0**
- threshold searches: **0**
- production changed: **false**

The trace ran through the promoted receiving-entitlement order:

`M38 -> TE-R5P -> WR-R15 -> explicit simulation`

and then through existing ensemble / QB / RB point authorities.

TE-R5P disposition:
`TE_R5P_FULL_SLATE_ENTITLEMENT_READY`

WR-R15 disposition:
`WR_R15_FULL_SLATE_ENTITLEMENT_READY`

## Scope

Static inventory:
- **51** concrete player-landscape feature/mechanism families.

Expanded position-market transmission matrix:
- **227** rows.

Required individual player markets:
- QB pass yards
- RB rush yards
- RB receiving yards
- RB receptions
- RB rush+receiving yards
- WR receiving yards
- WR receptions
- TE receiving yards
- TE receptions

All **9 / 9** required position-market rows have:
- individualized usage = **true**
- individualized efficiency = **true**
- opponent context materially consumed = **true**

This is the direct answer to the core architecture question:

> Yes, the current stack contains real player-specific usage, real
> player-specific efficiency, and real opponent context in every required
> player market.

But that does **not** mean every relevant landscape signal reaches the final
projection.

## Transmission totals

Across 227 expanded position-market feature rows:

Actively consumed:
- direct player: **45**
- indirect player/history: **39**
- opponent context: **32**
- team context: **18**
- player specialist: **9**

Total actively consumed:
**143 / 227 = 63.0%**

Non-current / incomplete rows:
- available but not consumed: **35**
- available but dropped: **5**
- prospective-only frozen: **17**
- source-parity blocked: **17**
- tested and closed: **8**
- not available: **2**

Total gap / blocked / prospective / closed:
**84 / 227 = 37.0%**

This 37% is not “37% of model quality missing.” It is an architecture inventory
count. Some gaps are duplicates across markets; some are already scientifically
closed; some are source blocked.

## Position-market summary

| Position / market | Feature rows | Consumed | Gap / non-current |
|---|---:|---:|---:|
| QB pass yards | 14 | **13** | **1** |
| RB rec yards | 26 | 16 | 10 |
| RB receptions | 24 | 16 | 8 |
| RB rush+rec yards | 24 | 13 | 11 |
| RB rush yards | 24 | 13 | 11 |
| TE rec yards | 30 | 17 | 13 |
| TE receptions | 25 | 17 | 8 |
| WR rec yards | 33 | 19 | 14 |
| WR receptions | 27 | 19 | 8 |

QB is materially more complete under the richer M89/M90 specialist architecture.

The larger transmission gaps are concentrated in generic RB/WR/TE football
logic.

## What IS individualized today

### Identity / availability
Player rows carry:
- named player identity;
- team;
- opponent;
- current pregame roster membership;
- role information where production consumes it.

Simulation distributions are keyed to individual player identity, not only to a
position bucket.

### Individual recent usage
Production consumes player-specific:
- target share;
- rush share;
- QB pass-attempt share / QB opportunity state;
- lagged same-player box-score state through ML;
- lagged same-player outcome regime through State.

The exact receiving entitlement stack further creates player-specific
allocation:
- M38;
- TE-R5P;
- WR-R15.

### Individual efficiency
Production consumes player-specific:
- YPT;
- catch/receptions-per-target rate;
- RB YPC;
- QB passing efficiency / YPA state.

### Room hierarchy
Production has real player-room competition through:
- M38 WR hierarchy;
- TE-R5P;
- WR-R15;
- alpha-receiver vacancy redistribution.

Additional player-state room mechanisms are frozen prospectively:
- RB carry/snap room allocation;
- RB receiving-room share;
- WR/TE target-share trajectory.

### Team environment
Production consumes:
- expected plays / pace;
- generic pass/rush split;
- richer QB team-attempt environment through M89/M90.

### Opponent context
Production does consume opponent context, including:
- pressure;
- coverage man/zone/middle state where available;
- RB light/heavy box state;
- player-role × coverage target multipliers.

Therefore it would be incorrect to describe the model as ignoring opponents.

## What is NOT fully transmitted

### 1. Individual route participation / YPRR

The PlayerForm schema supports:
- route rate;
- YPRR.

But the certified Week-5 historical-availability-parity reconstruction had:

- player route-rate coverage: **0 / 467**
- player YPRR coverage: **0 / 467**

Therefore these fields are **SOURCE_PARITY_BLOCKED**, not merely
available-but-unused.

Existing RB source-readiness work independently reached:

`LIVE_RB_ROUTE_VOLUME_CONFIRMED_HISTORICAL_WEEKLY_PARITY_NOT_CLEARED`

Public live route-volume sources exist, but canonical nflverse weekly history
does not supply total routes run with the required historical/live semantic
parity. Targets or primary-receiver route labels may not be relabeled as routes.

No route-rate/YPRR model candidate is authorized until that source gate clears.

### 2. Team game-environment state

Generic RB/WR/TE game script still uses the canonical `0.57` pass-share path.

Available:
- offensive PROE;
- richer lead / neutral / trail probabilities.

But the generic skill simulation does not fully transmit these into pass/rush
volume.

This architecture gap is real.

However:
- `FMT-INT-WR-TRUE-PROE-V1` already failed its frozen historical integration
  gate;
- no simple “replace 0.57 with PROE” rescue is authorized.

### 3. Richer opponent defense

Available but not generically consumed:
- defensive pass EPA;
- defensive rush EPA;
- explosive-play rate allowed.

Available upstream but dropped before canonical TeamContext:
- position-specific WR/TE/RB YPT allowed;
- outside / slot YPT allowed;
- defensive yards-before-contact / stuff-rate fields.

This confirms the user's intuition that more football matchup information exists
than reaches generic skill-position projections.

But multiple anti-retest boundaries apply:
- M95A/M95B generic RB role × run-defense integration is closed;
- RB opponent pass-rate-faced V1 integration failed;
- TE pass-success V1 integration failed the two-season replication gate;
- public position-YPT source parity / simple additive candidates have closed or
  blocked histories;
- arbitrary defense-vs-position multipliers remain prohibited.

### 4. Individual matchup assignment

WR role × generic coverage state is consumed.

Exact free reproducible full-slate WR-CB assignment remains unavailable.

The retired static coverage penalty must not be restored.

### 5. Defender injuries

Opponent-defender injury context remains source-parity blocked.

Do not interpret blank reports as healthy or use an unqualified source merely to
make the architecture look complete.

### 6. Distribution state

Player-specific right-tail / uncertainty work remains separate from point means.

Protected results:
- Right-Tail Asymmetry replicated historically for rush yards, rec yards,
  receptions, rush+rec yards;
- target-depth dispersion is a real player difficulty signal;
- the frozen universal symmetric target-depth transform did **not** improve
  W1-4 pooled CRPS and remains unpromoted.

## Dynamic context provenance

The dynamic trace used:

`HISTORICAL_AVAILABILITY_PARITY_WEEK5_RECONSTRUCTION`

It intentionally used zero target-week outcomes and zero sportsbook inputs.

It did **not** include supplemental live-only team coverage / box source
artifacts. This distinction matters.

Non-null coverage across the 467-player parity universe:

- expected plays: **467 / 467**
- offensive PROE: **467 / 467**
- offensive success rate: **467 / 467**
- opponent defensive success rate: **467 / 467**
- opponent pressure: **467 / 467**
- opponent defensive pass EPA: **467 / 467**
- opponent defensive rush EPA: **467 / 467**
- opponent explosive-play allowed: **467 / 467**
- player target share: **421 / 467**
- player rush share: **421 / 467**
- player YPT / catch rate: **357 / 467**
- player YPC: **262 / 467**
- player route rate: **0 / 467**
- player YPRR: **0 / 467**
- supplemental man / zone / middle-open / light-box / heavy-box fields:
  **0 / 467 in this parity reconstruction**

The static production matrix still records code paths that consume coverage/box
fields when live production sources populate them. The parity trace does not
pretend those supplemental live sources were present.

## Dynamic Week-5 trace

The successful audit selected **12 deterministic named players**:
- LOW, MEDIAN, HIGH projected workload for each of QB / RB / WR / TE;
- selection used pregame projected workload only;
- zero outcome-based selection.

The trace produced **27 market rows** and followed:

`identity -> availability -> player state -> room state -> team environment ->
opponent environment -> opportunity -> efficiency -> final mean -> distribution`

Representative high-workload examples from the frozen pregame trace:

### QB
Tyler Shough — NO vs MIN:
- expected pass attempts: ~35.26
- final pass-yards mean: ~271.24
- authority: M89/M90 QB synthesis

### RB
Jahmyr Gibbs — DET vs ARI:
- expected carries: ~12.88
- expected targets: ~3.42
- final rush yards: ~53.87
- final rec yards: ~25.04
- final receptions: ~3.79
- final rush+rec yards: ~78.91

### WR
Chris Olave — NO vs MIN:
- expected targets: ~8.56
- final rec yards: ~74.61
- final receptions: ~6.29

### TE
Brock Bowers — LV vs NE:
- expected targets: ~5.99
- final rec yards: ~47.87
- final receptions: ~4.28

These examples prove that current output is materially player-specific.

They are **not recommendations or bets** and were not selected from sportsbook
information.

## Relationship to completed W1-4 player replay

The landscape audit is consistent with the completed all-player replay.

That replay found:
- low-workload players systematically overprojected;
- high-workload focal players systematically underprojected.

The opportunity decomposition then localized:
- RB/WR/TE: individual share is the dominant opportunity problem;
- QB: team pass volume is the dominant opportunity problem.

The RB receiving-room mechanism subsequently proved that correcting an actual
player-room share seam improves final individual projections:
- targets MAE: **7.60% better**
- receptions MAE: **3.83% better**
- rec-yards MAE: **2.89% better**
- rush+rec-yards MAE: **1.66% better**

Therefore this audit is not merely architectural. We already have one direct
example where better individual landscape transmission produced better realized
player projections.

## Scientific interpretation

The correct description of the current model is:

> **Individual-player core + material positional shrinkage + promoted player
> specialists + partial team/opponent transmission.**

It is not:
> “a position model with names attached.”

It is also not yet:
> “a complete player-specific football landscape model.”

The biggest remaining problem is not a lack of player identity. It is that some
available player/team/opponent facts still fail to reach the final generic
RB/WR/TE opportunity/efficiency/distribution seams, while several obvious
transmission formulas have already failed historical qualification.

## Next-action rule

Do not dump all unused features into the model.

For each candidate missing seam:

1. prove it is genuinely new, not a renamed closed family;
2. prove the input is pregame and source-parity valid;
3. localize whether it belongs to opportunity, efficiency, mean, or
   distribution;
4. freeze the exact mechanism before scoring;
5. test on historical/current-season player outcomes;
6. fail closed if it does not improve individual projections.

This preserves the user's end-state requirement while protecting the existing
validated model science.

No automatic production promotion.
No paid OddsAPI.
