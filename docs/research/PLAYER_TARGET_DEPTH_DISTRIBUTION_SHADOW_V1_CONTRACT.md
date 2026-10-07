# Player Target Depth Distribution Shadow V1 — Frozen Contract

**STATUS: FROZEN BEFORE SCORING / RESEARCH SHADOW ONLY / NO PRODUCTION CHANGE**

This contract converts the already-confirmed individual receiver target-depth-dispersion signal into a strictly mean-neutral WR/TE receiving-yard distribution shadow.

Parent scientific authority:
- Player Target Depth Dispersion V1 run `37655282486`
- artifact `11497703984`
- digest `sha256:39110b5d8d344b5beb51f5eab02ba931ffddae41932f8d7dcf5ca5da66824f55`
- disposition `PLAYER_TARGET_DEPTH_DISPERSION_DIFFICULTY_CONFIRMED`

The parent proved that strictly-prior individual target-depth SD is positively associated with absolute receiving-yard efficiency-component error for both WR and TE. Signed efficiency persistence failed in the preceding mechanism studies. Therefore this contract may alter **uncertainty only** and may not alter any receiving-yard/YPT mean.

## Frozen player-state feature

Use the exact parent definition:
- individual receiver;
- regular-season target events only;
- target event = pass attempt, not sack, not two-point attempt, finite receiver ID;
- strictly before the target player-game;
- latest up to 8 completed receiver target-games;
- at least 4 prior receiver target-games;
- at least 10 finite air-yard targets;
- population SD (`ddof=0`) of finite `air_yards`.

No target-game play may enter the feature.

For the 2026 Week-5 pregame lock, source history may include prior regular-season games from 2022-2025 and 2026 Weeks 1-4 only.

## Frozen position anchors

Use the exact pooled medians from the confirmed parent artifact:
- WR target-depth-SD anchor = `10.02786868561449`
- TE target-depth-SD anchor = `6.735519692444827`

These anchors are descriptive constants frozen from the already-completed parent diagnostic. They are not refit in this shadow.

## Frozen no-fit uncertainty transform

For an eligible WR/TE player:

`depth_scale = sqrt(prior8_target_depth_sd / position_anchor)`

No coefficient is fit.
No threshold, subgroup, cap, floor, window, or alternate functional form may be searched after scoring.

If the player does not satisfy the exact feature-support contract, use:
`depth_scale = 1.0`

The scale is applied only to the player's receiving-yard outcome distribution.

## Mean-neutral draw transform

Given:
- the exact final football receiving-yard mean `mu`;
- the baseline empirical receiving-yard draw array after the existing production-order entitlement stack;
- the frozen `depth_scale`.

The shadow must:

1. mean-align the baseline non-negative draw array to `mu` using the existing non-negative multiplicative mean-alignment semantics;
2. spread/contract only deviations around `mu`:
   `candidate_preclip = mu + depth_scale * (baseline_draw - mu)`;
3. clip candidate draws at zero;
4. multiplicatively mean-align the clipped candidate back to the **same exact `mu`**.

Required invariant:
- baseline mean = candidate mean = exact final football mean to <= `1e-10`.

A scale of exactly 1.0 must be an exact no-op after baseline mean alignment.

## Protected science / invariants

This shadow may not change:
- target entitlement;
- target counts/opportunity arrays;
- team pass attempts;
- team target mass;
- M38 WR1 anchor;
- WR-R15 WR2+ entitlement;
- TE-R5P entitlement;
- QB M89/M90/C2 science;
- YPT or catch-rate point means;
- opponent/team environment mean adjustments;
- RB science;
- sportsbook separation.

Sportsbook lines/odds are forbidden upstream inputs.
No paid OddsAPI acquisition is authorized.
No production routing is changed by this research contract.

## Anti-retest boundary

This is **not** TE-R5P Receiving-Yards Width V2.

TE Width V2 fit a global season-level residual-SD multiplier and failed closed. That family remains closed.

This shadow is scientifically distinct because:
- its only treatment variable is newly-confirmed, individual-player, strictly-prior football state: target-depth dispersion;
- it fits no residual-width coefficient;
- it does not use player residual history as a feature;
- it does not rescue the failed global TE width factor;
- it is cross-position WR/TE player-specific rather than a global TE widening rule.

Also remain closed:
- WR-R3 residual-width calibration;
- WR R7 signed-YPR traits;
- M72 / M75;
- raw QB-receiver pair YPT.

## Week-5 prospective lock

The first implementation must create an immutable 2026 Week-5 pregame table using the already-frozen Player Target Share Trajectory Week-5 WR/TE universe:
- parent run `37654382316`;
- parent artifact `11497153776`;
- canonical parent row digest `sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`.

The lock records, for every parent WR/TE row:
- stable receiver identity when resolvable from strictly-prior public identity data;
- feature availability;
- prior receiver-game count;
- prior finite air-yard target count;
- target-depth SD;
- frozen position anchor;
- frozen depth scale.

No Week-5 outcome may be read.
No sportsbook input may be read.
No football mean or entitlement is changed by the lock itself.

## Replay boundary

The later all-player/all-position replay may evaluate this exact frozen transform:
- 2026 Weeks 1-4 may use strictly-prior historical target-depth history, including 2025, under the same feature definition;
- exact Target Share Trajectory V1 remains ineligible in Weeks 1-4 because it requires four prior same-season team games;
- historical Week-5+ can separately integration-test the frozen trajectory transformation.

Do not weaken either contract to manufacture early-season coverage.

## Distribution scoring for the eventual replay

Score the paired baseline vs shadow distributions on identical eligible rows.

Primary:
- empirical CRPS, lower is better.

Required calibration diagnostics:
- central 50% interval coverage;
- central 80% interval coverage;
- central 90% interval coverage;
- interval widths;
- realized absolute receiving-yard error by depth-scale quantile.

Point-mean MAE must be invariant by construction and is an integrity check, not a treatment target.

No production promotion is authorized by this contract or by the Week-5 pregame lock.
