# QB-Receiver Pair Efficiency V1 — Frozen Scientific Contract

**STATUS: FROZEN BEFORE RESULT. NO PRODUCTION CHANGE.**

Parent source authority:
- `QB_RECEIVER_PAIR_STATE_SOURCE_READY`
- run `37626968980`
- artifact `11483954236`
- digest `sha256:414c24ceb6406227c46654ddb8652b89feae85e3aa62ad2d58327e4c4430a065`

## Purpose

Test one player-centric football hypothesis:

> For an established receiver, strictly-prior yards-per-target with the current pregame passer proxy contains incremental information beyond the same receiver's strictly-prior yards-per-target with all passers.

This isolates the passer x receiver relationship.

It does **not** alter target entitlement, team pass volume, QB mean, opponent matchup, or production.

## Protected science

Do not reopen or alter:
- M38 / WR-R15 entitlement;
- TE-R5P entitlement;
- QB M89/M90;
- C1/C3 shared state closures;
- M72 team-level receiving-weapon work;
- M75 / R7 / R3 closed receiver-efficiency/residual families;
- current simulation/distribution authorities.

The novelty is exact pair identity, not another receiver-only trait.

## Data

Free public nflreadpy/nflverse regular-season PBP only.

Seasons:
- history begins 2022;
- score target seasons 2023, 2024, 2025.

No 2026 outcome is used.

## Pregame passer proxy

For each target team-week, use only plays strictly before the target week.

Define the current passer proxy as:
- passer with the most official pass attempts over that team's latest **3 completed games**;
- ties break deterministically by passer ID.

This proxy is deliberately pregame-safe and does not use the target-game starter/result.

Target weeks:
- Weeks 5-18 only.

## Receiver cohort

Primary cohort:
- WR and TE only.

Target-game row requires:
- player receives at least one target in the target game;
- stable receiver ID;
- valid pregame passer proxy;
- at least 10 strictly-prior receiver targets across all passers;
- at least 5 strictly-prior targets with the exact passer proxy.

Those support floors are frozen before results and are not routing rules for production.

RB receiving pairs may be reported diagnostically but cannot determine PASS/FAIL.

## History window

For both control and pair challenger:
- use the latest **8 completed receiver-games** strictly before the target week.

Control:
`receiver_ypt_prior8 = receiver prior receiving yards / receiver prior targets`

Challenger:
`pair_ypt_prior8 = same receiver + proxy passer prior receiving yards / pair prior targets`

The pair history is restricted to pair target events that fall inside the receiver's same latest-eight-game history horizon. This prevents the challenger from using a longer history than the control.

No shrinkage constant, fitted coefficient, clipping, threshold search, or post-hoc weighting.

## Target

For each scoreable target player-game:

`actual_ypt = target-game receiving yards / target-game targets`

This is an efficiency test only.

For diagnostic translation:
- multiply each pregame YPT estimate by the **actual target-game target count**;
- compare resulting receiving-yard translation to actual receiving yards.

That translation diagnostic intentionally holds opportunity fixed to actual targets. It does not represent a deployable projection and may not be confused with production accuracy.

## Frozen metrics

Per season and pooled:
- rows;
- distinct players;
- distinct games;
- control YPT MAE / RMSE / signed bias;
- pair YPT MAE / RMSE / signed bias;
- Spearman and Pearson vs actual YPT;
- actual-target-held-fixed receiving-yard MAE;
- 20+ yard efficiency miss count where abs YPT error >= 20;
- pair support targets;
- fraction of receiver targets historically delivered by proxy passer.

Report separately:
- WR;
- TE;
- pooled WR+TE.

Paired uncertainty:
- game-cluster bootstrap;
- 10,000 replicates;
- seed `20261007`;
- statistic = control absolute YPT error - pair absolute YPT error.

## PASS gate

`QB_RECEIVER_PAIR_EFFICIENCY_SIGNAL_CONFIRMED` requires all:

1. target 2023 support >= 300 rows / 100 games;
2. target 2024 support >= 300 rows / 100 games;
3. target 2025 support >= 300 rows / 100 games;
4. pooled pair YPT MAE < control YPT MAE;
5. pair YPT MAE improves in at least 2 of 3 seasons;
6. pooled pair RMSE <= control RMSE;
7. pooled game-cluster bootstrap P(MAE improve) >= 0.80;
8. pooled WR YPT MAE improves;
9. pooled TE YPT MAE is non-worse;
10. pooled actual-target-held-fixed receiving-yard MAE improves;
11. 20+ YPT miss count does not increase;
12. zero target/future leakage;
13. sportsbook inputs = 0;
14. 2026 outcomes read = 0;
15. production changed = false.

Otherwise:

`QB_RECEIVER_PAIR_EFFICIENCY_SIGNAL_CLOSED`

## Anti-rescue

After results are exposed, do not rescue with:
- 4-game / 6-game / season-long history windows;
- different 3-game QB proxy;
- actual target-game starter;
- >=10 or >=15 pair-target carveouts;
- WR-only or TE-only rescue;
- pair catch-rate/YAC/air-yard blends;
- fitted shrinkage;
- player-name exceptions;
- QB-change-only subgroups.

A distinct mechanism requires a new frozen plan.

## If confirmed

A PASS would authorize a separate integration candidate against the current production receiving-yard baseline.

It would **not** authorize direct production replacement.

That later integration would have to preserve:
- WR-R15 / TE-R5P opportunity;
- QB M89/M90;
- team pass volume;
- conservation;
- distribution science.

Models fit by this contract: **0**  
Sportsbook inputs: **0**  
2026 outcomes: **0**  
Production mutations: **0**
