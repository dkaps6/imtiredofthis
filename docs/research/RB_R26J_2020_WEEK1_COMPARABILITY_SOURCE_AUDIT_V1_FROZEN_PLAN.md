# RB R26J — 2020 Week-1 Comparability / Offseason-Regime Source Audit V1 Frozen Plan

Status: FROZEN BEFORE SOURCE EXECUTION
Date: 2026-09-09
Production authority: `main@f8417f55b04ce0e19baf260e9d532765034c47f1`
Governance: `docs/research/RESEARCH_COMPONENT_PRESERVATION_DOCTRINE.md`

## Purpose

R26/R26E/R26I show a strong Week-1 RB receiving signal in 2021-2025 while 2020 is consistently harmful. R26F-R26H identified useful transition states but did not explain the 2020 regime sufficiently. R26I remained strong pooled but worsened 2020 to +10.05% reception MAE versus baseline.

R26J must NOT create another router or child prediction. It asks a prior question:

> Is 2020 Week 1 structurally/source-wise different from 2021-2025 in pregame RB-room transition state strongly enough to justify a later comparability/mechanism study?

This is source/model-state audit only. No target-game outcomes, errors, sportsbook data, or child predictions may be selected into R26J.

## Immutable parent evidence

R26:
- run `34356222339`
- artifact `10106271075`
- digest `sha256:607fca6e11c301ecb2a3bf74e3dfea8ae415bb33cf3c150a6d89eaedada2809e`

R26C:
- run `34361409319`
- artifact `10108036449`
- digest `sha256:1c7ee705179d91b7e3e53c39b57783314bd87876df6c784d5d1930ee5cf79156`

R26I result is context only and may not provide outcome/error columns to this audit.

## Population

Historical seasons: 2020-2025.
Week: 1 only.
Rooms: canonical R26 vacancy-active RB/FB team-weeks.

R26J compares each season's pregame room/source state. Primary contrast is 2020 versus the 2021-2025 distribution.

## Strictly prohibited columns / information

R26J may not select or derive from:
- `actual_targets`
- `actual_receptions`
- any actual receiving yards
- any baseline/candidate absolute error
- any R26/R26I error delta
- sportsbook line/odds/price
- target-game participation
- target-game outcomes
- same-week historical depth.

R26 predictions may be read only for pregame transition/model-state columns explicitly listed below. The audit code must fail closed if any `actual_*` column is selected into the working frame.

## Frozen source/model-state dimensions

### A. Room continuity / turnover
At each Week-1 vacancy team-week:
- current RB-room player count;
- prior RB-room player count;
- number of exits;
- number of entrants;
- `exits == entrants` rate;
- absolute room-size change;
- number and share of current-room same-team continuations;
- number and share of current-room entrants.

### B. Entrant composition
Using only R26 strict-prior flags:
- number/share `new_to_team_veteran`;
- number/share `no_prior_nfl_roster`;
- rooms with both entrant classes;
- rooms with neither resolved entrant class.

### C. Vacated receiving significance
Using R26C strict-prior exited-player state:
- exited-player prior-history coverage;
- number of exited players with positive prior history;
- max exited prior targets/game;
- sum exited prior targets/game;
- max exited prior RB-room target share;
- sum exited prior RB-room target share;
- max/sum last-8 targets/game;
- count of meaningful exits under the already-frozen R26D rule:
  `prior_targets_pg > 1 OR prior_rb_room_share >= 0.25`;
- multiple-meaningful-exit room rate.

Do not change this threshold.

### D. Returning-room projected structure
Using only pregame R26 model-state fields:
- baseline within-RB-room share concentration (HHI);
- baseline top-player room share;
- number of incumbents;
- R9 reliability value for the season;
- absolute magnitude of R26's **pregame allocation shift** versus baseline, measured as room-level L1 distance between candidate and baseline room shares.

This is allowed because it uses no target-game outcome. It asks whether R26 was structurally making larger/different adjustments in the 2020 source regime.

### E. Source quality / missingness
By season:
- vacancy-room reconstruction coverage versus R26;
- exited-player prior receiving-history coverage;
- lagged-depth availability coverage (diagnostic only; never used as same-week feature);
- entrant-state resolution coverage;
- duplicate/key integrity;
- canonical ACT/INA status contract inherited from R26C.

## Frozen season-comparability summaries

For every numeric dimension report:
- season mean;
- median;
- p25/p75;
- room count;
- 2020 value;
- 2021-2025 mean and range.

For every binary/rate dimension report season rate.

## Frozen definition of a 2020 structural distinction

R26J may label a dimension `2020_STRUCTURALLY_DISTINCT` only if at least one predeclared condition holds:

1. **Range separation:** 2020 season-level mean/rate lies outside the min-max range of all 2021-2025 season-level values; OR
2. **Large relative shift:** for a non-near-zero positive metric, 2020 differs from the 2021-2025 mean by at least 25%; OR
3. **Large absolute rate shift:** for a rate in [0,1], 2020 differs from the 2021-2025 mean by at least 0.15 absolute; OR
4. **Source integrity shift:** 2020 coverage/missingness differs from 2021-2025 mean by at least 0.10 absolute.

These thresholds are frozen before source results.

No one dimension by itself may authorize excluding 2020.

## Audit-level disposition

`2020_SOURCE_REGIME_DISTINCT_FOR_MECHANISM_FOLLOWUP` only if:
- all source integrity/leakage gates pass; AND
- at least 3 independent predeclared dimensions across at least 2 of sections A-D meet a structural-distinction rule.

Otherwise:
`2020_SOURCE_REGIME_NOT_DISTINCT_ENOUGH_FOR_EXEMPTION`.

Either disposition authorizes NO shadow, NO production, and NO exclusion of 2020. A `DISTINCT` result only authorizes a separately frozen mechanism/comparability outcome diagnostic.

## Source integrity gates

1. parent artifact digests exact;
2. Week-1 vacancy population only;
3. zero selected `actual_*` fields in working source frame;
4. target-game outcome/participation use zero;
5. sportsbook use zero;
6. same-week historical depth false;
7. canonical ACT/INA source contract preserved;
8. room keys unique where required;
9. R26C vacancy reconstruction >=99%;
10. protected production files unchanged.

## Component preservation

Regardless of result preserve:
- original R26 Week-1 pooled/2021-2025 gains;
- R9 mechanics;
- fixed RB target-pool conservation;
- non-vacancy baseline exactness;
- receiving-yard means and R22;
- R26E/R26G/R26I failed dispositions;
- 2020 negative evidence.

## Prohibited conclusions

R26J may NOT conclude:
- "drop 2020";
- "2020 was COVID, therefore ignore it";
- any new threshold/router;
- any Week-1 shadow authorization.

Those require later frozen evidence.
