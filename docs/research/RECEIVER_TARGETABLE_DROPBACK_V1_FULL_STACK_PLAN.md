# Receiver Targetable-Dropback V1 — Player Full-Stack Plan

Date: 2026-09-25

Status: **FROZEN BEFORE PLAYER-LEVEL OUTCOME SCORING**

Parent team-level evidence:

- 2022-2023 temporal qualification:
  `RECEIVER_TARGETABLE_DROPBACK_V1_TEAM_CALIBRATION_SUPPORTED`
- 2024-2025 unchanged confirmation:
  `RECEIVER_TARGETABLE_DROPBACK_V1_2024_2025_CONFIRMED`

The exact cumulative-count targetable-dropback mechanism improved team-target MAE
independently in 2022, 2023, 2024 and 2025.

This plan tests whether that team-opportunity correction survives the current
player projection stack. It does not authorize production.

## 1. Exact frozen opportunity candidate

For target season S, week W, team T:

Eligible strict-prior history contains:
- all regular-season games from S-1 for T;
- completed regular-season games from S with week < W.

Targetable-dropback rate:

`R_T = sum(history team targets) / sum(history team dropbacks)`

League fallback is permitted only if team prior dropbacks are zero, using the
same frozen pooled strict-prior count ratio.

The candidate introduces no fitted coefficient, recency weight, shrinkage
constant, minimum-games threshold or route.

## 2. Player-level simulation mechanism

The existing team pass-opportunity state remains unchanged:

`dropbacks = current canonical simulated pass state`

The candidate changes only receiver target opportunity.

Current explicit player target entitlement after M38 / TE-R5P / WR-R15 is
`p_i`.

Baseline modeled player target probabilities are the current `p_i`.

Candidate modeled player target probabilities are:

`p_i* = R_T * p_i`

The existing multinomial residual bucket therefore becomes:

`1 - R_T * sum(p_i)`

This residual exactly combines:
- non-targetable dropbacks;
- the existing unmodeled receiver residual share.

This is mathematically equivalent for named-player targets to first thinning
dropbacks into targetable/non-targetable opportunities and then allocating
targetable opportunities with the unchanged entitlement shares.

No player entitlement is re-ranked or re-estimated.

## 3. Candidate receiver generation

For candidate receiver arrays:

- use the exact baseline simulated team dropback array;
- use the exact baseline pass-efficiency shock;
- use the scaled target probabilities above;
- use the current canonical multinomial allocation;
- use unchanged canonical catch-rate logic;
- use unchanged canonical YPT logic;
- use unchanged canonical volatility logic;
- use a dedicated deterministic research RNG keyed by season/week/game/team and
  candidate version.

Only these arrays may change:
- `receptions`
- `rec_yards`
- derived `rush_rec_yards`

Candidate `rush_rec_yards` must equal:
`baseline rush_yards + candidate rec_yards`

The candidate is installed onto a copy of the baseline simulation state.
Therefore all other baseline arrays remain bit-identical.

## 4. Explicitly unchanged

The candidate must not change:

- projected plays;
- fixed 0.57 team dropback partition;
- team dropback arrays;
- team pass-efficiency arrays;
- M38;
- explicit target entitlement;
- TE-R5P;
- WR-R15;
- catch rate;
- YPT;
- receiver volatility;
- QB M89/M90;
- QB C2;
- QB pass-yards arrays;
- rush attempts;
- rush yards;
- rushing shares;
- YPC;
- ATD;
- ML component predictions;
- State component predictions;
- ensemble weights;
- RB Rush+Receiving Conservation V2 formula/scope;
- sportsbook inputs.

The known dynamic team-dropback/pass-rate problem is a separate science lane and
must not be combined into this first player-level test.

## 5. Historical full-stack authority

Evaluate:
- 2024 regular season;
- 2025 regular season;
- pooled.

Use the same leakage-safe current-stack historical reconstruction already used
by recent receiver integration work:

- M38 explicit entitlement;
- TE-R5P fold-safe OOS authority for 2024 and 2025;
- WR-R15 fold-safe OOS authority where available under the existing historical
  replay contract;
- current generic MC/ML/State ensemble weights;
- current RB Rush+Receiving Conservation V2;
- no sportsbook.

Do not use an in-sample WR-R15 fit merely to make 2025 look more like 2026
production. Preserve the established fold-safe historical authority.

Monte Carlo:
- 5000 draws;
- one candidate;
- fixed deterministic seeds;
- no post-result seed selection.

## 6. Required mechanical/integrity gates

All must pass:

1. reconstructed explicit-entitlement baseline matches the canonical historical
   receiver baseline exactly;
2. targetable-rate provenance is strict-prior for every changed team;
3. league fallback rate <=1%;
4. every targetable rate is finite and in [0,1];
5. target entitlement values are unchanged;
6. TE-R5P conservation gates pass;
7. WR-R15 historical conservation gates pass where applied;
8. QB pass-yards arrays are bit-identical baseline vs candidate;
9. rush-att arrays are bit-identical;
10. rush-yards arrays are bit-identical;
11. ATD arrays are bit-identical;
12. candidate rush+receiving raw identity equals baseline rush yards plus
    candidate receiving yards within `1e-10`;
13. RB V2 pathwise identity gap <= `1e-10`;
14. sportsbook inputs = 0;
15. target-game outcomes used upstream = 0;
16. parameters fit = 0;
17. candidate variants scored = 1.

Any mechanical failure stops interpretation.

## 7. Primary scored cohorts

Positions:
- WR
- TE
- RB

Markets:
- receptions
- receiving yards

Protected dependent market:
- RB rush+receiving yards

Also score high-entitlement Q4 separately using the final pregame explicit target
entitlement.

## 8. Frozen metrics

For each season, pooled, position and relevant cohort report:

- n;
- MAE;
- RMSE;
- bias;
- absolute bias;
- correlation;
- median AE;
- p75 AE;
- p90 AE;
- changed-row candidate closer rate.

Receiving yards also report:
- 20+ miss rate;
- 30+ miss rate;
- 40+ miss rate.

## 9. Frozen scientific gates

`RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_QUALIFIED` requires all:

### Receptions
1. pooled WR/TE/RB macro MAE strictly improves;
2. 2024 macro MAE is nonworse;
3. 2025 macro MAE is nonworse;
4. pooled WR MAE is nonworse;
5. pooled TE MAE is nonworse;
6. pooled RB MAE is nonworse;
7. pooled macro p90 AE is nonworse;
8. pooled macro absolute bias is nonworse;
9. changed-row pooled candidate-closer rate >50%.

### Receiving yards
10. pooled WR/TE/RB macro MAE strictly improves;
11. 2024 macro MAE is nonworse;
12. 2025 macro MAE is nonworse;
13. pooled WR MAE is nonworse;
14. pooled TE MAE is nonworse;
15. pooled RB MAE is nonworse;
16. pooled macro p90 AE is nonworse;
17. pooled macro absolute bias is nonworse;
18. high-entitlement Q4 MAE is nonworse;
19. high-entitlement Q4 p90 AE is nonworse;
20. pooled macro 40+ miss rate is nonworse;
21. changed-row pooled candidate-closer rate >50%.

### RB rush+receiving protection
22. 2024 RB rush+receiving MAE is nonworse;
23. 2025 RB rush+receiving MAE is nonworse;
24. pooled RB rush+receiving p90 AE is nonworse.

### Integrity
25. every mechanical/integrity gate passes.

Any failed frozen scientific gate:

`RECEIVER_TARGETABLE_DROPBACK_V1_FULL_STACK_FAILED_CLOSED`

No rescue is authorized.

## 10. No-rescue rule

After results are visible, do not try:

- position-specific targetable rates;
- player-specific targetable rates;
- Q4 exemptions;
- WR/TE/RB carveouts;
- alternate history windows;
- recency weighting;
- rate shrinkage;
- rate caps/floors selected from outcomes;
- dynamic 0.57 replacement;
- C2 routing;
- hierarchical reconciliation;
- catch-rate/YPT retuning;
- sportsbook-conditioned routing;
- combined scramble/rushing repair;
- 2026 outcome fitting.

Any such idea is a new hypothesis requiring a separately frozen validation plan.

## 11. Promotion rule

Historical full-stack qualification still does not authorize production.

If qualified:
1. freeze a prospective 2026 shadow-capture contract;
2. generate the candidate before game outcomes;
3. preserve candidate projections/artifacts;
4. score once outcomes are available;
5. only then consider separately frozen production integration.

No paid OddsAPI pull is authorized by this plan.
