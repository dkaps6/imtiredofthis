# WR-QB Shared Tail Signal V1 — Frozen Plan

**Status:** FROZEN BEFORE RESULT INSPECTION  
**Production actionable:** false  
**Sportsbook inputs:** forbidden  
**Branch:** `research-wr-qb-shared-tail-signal-v1`

## Question

Does the already-promoted, pregame QB C2 distribution contain incremental information about a true model-WR1 receiving-yard right-tail miss after M38 opportunity is already fixed?

This is **not** a rerun of M72/M75/M84, C1/C3, WR-R3, WR-R7/R11, or broad C2 receiver propagation.

The exact remaining hypothesis is narrower:

> A QB's pregame C2 upper-tail width may identify team-games in which the M38 WR1 has elevated receiving-yard right-tail risk, even when the WR1 point mean/entitlement is left unchanged.

No target-game QB or WR outcome may enter any feature. No sportsbook line/odds may enter any feature or cohort rule.

## Canonical preserved inputs

Use only preserved outputs from the already-completed Joint Pass/Receiving Conservation V1 scientific run:

- run `34081764151`
- artifact `10004223287`
- digest `sha256:753aa191e6c80a059918553d8567499c9cdbce82b12a26b4d5b19b21225764ac`
- QB casebook: exact 884 M89/M90 QB-games, 2024=444 and 2025=440
- paired player casebook: preserved football-only receiver projections/outcomes

The earlier WR1 Yardage Decomposition V1 is context only, not a feature source:

- run `34858963515`
- artifact `10353787250`
- digest `sha256:3e6fb21956e0f2681a732379412b6eff0c3e17dd561d7358e8995e279fb42f5d`

## Anti-retest boundary

Already closed / not reusable as new science:

- M72 aggregate explosive-weapon x defense matchup — failed.
- M75 NGS separation/cushion/aDOT/YACOE and secondary aggregates — tested.
- M84 receiver-defender responsibility source — blocked under free historical source contract.
- WR-R7/R11 generic YPR/NGS mechanisms — failed.
- WR-R3 combined residual calibration — tested; `NO_ACTIONABLE_WR_R3_COMBINED_CALIBRATION`.
- C1 group target-mass calibration — failed player-level protection.
- C3 broad joint combination — failed player-level protection.
- Broad C2 receiver propagation — already tested in run `34081764151`; macro receiver MAE improved slightly, but this plan must separately report true-WR1 b0 vs C2 mean results and may not call broad C2 propagation a new candidate.
- Team-level coverage — prior ablation near-neutral.
- Player-level WR/CB historical assignment — not reconstructable without violating the no-fake-assignment stop rule.

## Cohort

Seasons: 2024 and 2025 only.

True model WR1 definition for each `(season, week, event_id, team)`:

1. restrict preserved paired-player casebook to `position == WR`;
2. rank descending by `b0_target_probability`;
3. deterministic tie break ascending `player_clean_key`;
4. keep rank 1 only.

Expected cohort: exactly one WR1 per team-game, expected 544 WR1 rows in 2024 and 544 in 2025. Any duplicate/missing QB join fails closed.

Join QB casebook on `(season, week, event_id, team)`.

## Pregame primary signal

Primary scalar feature, fixed before result inspection:

`qb_c2_upper90 = c2_p90 - c2_mean`

This is a pregame C2 QB distribution shape quantity. It does not use actual QB yards, actual WR yards, or sportsbook data.

Secondary descriptive-only quantities (cannot rescue a failed primary signal):

- `qb_c2_upper95 = c2_p95 - c2_mean`
- `qb_c2_i80 = c2_p90 - c2_p10`
- `qb_c2_right_skew = (c2_p90 - c2_mean) - (c2_mean - c2_p10)`
- `qb_c2_tail_expansion80 = (c2_p90 - c2_p10) - (b0_p90 - b0_p10)`

No feature search, subset search, threshold sweep, model zoo, or post-result feature substitution is permitted.

## Outcome definitions — scoring only

These fields are never inputs:

- WR residual: `actual_rec_yards - b0_rec_yards`
- `tail100`: actual WR receiving yards >= 100
- `cat_under40`: actual WR receiving yards - b0_rec_yards >= 40
- `cat_under50`: actual WR receiving yards - b0_rec_yards >= 50

The study tests signal existence only. It does not change WR mean, target entitlement, or production distribution.

## Development / holdout protocol

2024 is development only.

From 2024 only:

- freeze the 75th-percentile threshold of `qb_c2_upper90`;
- freeze quartile cut points for `b0_rec_yards` used only for the opportunity-conditional diagnostic;
- compute the 2024 mean/SD of `qb_c2_upper90` for standardized descriptive regression.

Apply those exact frozen 2024 thresholds/scaling to 2025. Do not recompute thresholds from 2025 or pooled data.

The **scientific disposition is determined by 2025 only**. 2024 is reported for mechanism coherence but cannot carry a failed 2025 result.

## Mandatory outputs

1. cohort and identity audit;
2. broad-C2 anti-retest scoreboard for true WR1: `b0_rec_yards` vs `c2_rec_yards` MAE/RMSE/bias/p90 AE and 100+ actual-game miss behavior;
3. primary `qb_c2_upper90` signal summary for 2024 and blind 2025;
4. frozen-2024 top-quartile vs rest comparison in 2025 for WR residual, tail100, cat_under40, cat_under50;
5. Spearman correlation of `qb_c2_upper90` with WR residual by season;
6. bootstrap probability that 2025 top-quartile residual mean exceeds the rest (10,000 draws; fixed seed 5601);
7. opportunity-conditional check: apply 2024 `b0_rec_yards` quartile boundaries to 2025 and report top-tail-state residual gap inside each baseline-opportunity quartile;
8. secondary descriptive feature table, clearly marked non-gating;
9. exact pass/fail gate JSON and immutable result markdown.

## Frozen scientific gates

Disposition can be `WR_QB_SHARED_TAIL_SIGNAL_SUPPORTED` only if **all** are true on blind 2025:

1. identity/parity gates all pass; sportsbook inputs used = 0;
2. primary Spearman(`qb_c2_upper90`, WR residual) >= `0.08`;
3. frozen-2024 top-quartile state has mean WR residual at least `+5.0 yards` above the remaining 75%;
4. bootstrap P(top-quartile residual mean > rest) >= `0.90`;
5. `tail100` rate ratio, top-quartile / rest, >= `1.30`;
6. `cat_under40` rate ratio, top-quartile / rest, >= `1.20`;
7. opportunity-conditional direction is nonnegative in at least `3 of 4` frozen baseline-WR-projection quartiles;
8. 2024 mechanism direction is coherent: Spearman > 0 and top-quartile residual gap > 0. This is a coherence guard only, not a substitute for 2025.

If any required gate fails, disposition is `NO_ACTIONABLE_WR_QB_SHARED_TAIL_SIGNAL`.

Secondary features may not rescue a failed primary feature.

## Interpretation boundary

A PASS authorizes only a separately frozen **mean-neutral WR receiving-yard distribution/tail candidate**. It does not authorize changing M38/R15 entitlement or WR point means.

A FAIL closes this exact C2-QB-tail-to-WR-tail lane. Do not retune the percentile, thresholds, feature formula, or subgroup after result inspection.

## Stop rule / next project priority

After this WR lane is scientifically dispositioned, move immediately to the post-Week-1 RB program. No additional generic WR mean hunting is authorized by this plan.
