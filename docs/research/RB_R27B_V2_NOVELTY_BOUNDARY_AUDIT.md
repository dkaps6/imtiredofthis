# RB R27B V2 — NOVELTY BOUNDARY AUDIT

**Status:** PRE-CANDIDATE / PRE-PLAN SOURCE-AUDIT STAGE

## Purpose

This document defines what is and is not scientifically novel for the next RB receiving-yard mean study after R27. It exists specifically to prevent reinvention of production `bayes_ypt` / `rules_ypt`, R23's failed shrunk-YPR candidate, R24's production-efficiency decomposition, R27's exact-R26 opportunity translation, or R19/R22 tail science.

## Exact parent evidence

- production authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- R27 valid run: `34423546037`
- R27 job: `102703879430`
- R27 head: `b7cfe5b2c450765208672fc6e2b017d0cb0c0dde`
- R27 artifact: `10132290573`
- R27 digest: `sha256:cd5c9e26efa44c47ef3374e4deefe9038dc504c155799b48548ad64746e10e45`
- R27 disposition: `R27_R26_OPPORTUNITY_TRANSLATION_MIXED_OR_FAIL`
- R27 gates: 24/27 PASS
- R27 result-record parent commit: `886c8432ff811882e006d84a61385862e3be7839`

## What is already modeled or already tested

### Production empirical-Bayes efficiency

`scripts/modeling/bayesian_v2.py` already constructs RB `bayes_ypt` from a position-family prior plus prior/current player evidence. `scripts/modeling/simulation_rules.py` already converts that football efficiency state into `rules_ypt` using the existing matchup/pass-efficiency multiplier. A study whose core novelty is generic player YPT persistence or generic YPT shrinkage is therefore not new.

### R23 strict-prior efficiency

R23 Candidate 3 already tested improved opportunity/receptions plus a strict-prior shrunk YPR component using frozen 6-game recent and 16-game stabilizing histories. It improved targets/receptions but failed receiving-yard robustness, including worse RB1 and p90 performance. R23 remains a preserved scientific null and may not be recreated under a different migration name.

### R24 decomposition

R24 removed R23's failed new efficiency component and paired improved opportunity with unchanged production YPT. It was explicitly designed to determine whether the opportunity improvement survived when efficiency was held to production authority.

### R27 exact-R26 decomposition

R27 repeated the key decomposition with the exact R26 vacancy-gated opportunity mechanism that later qualified for receptions. It materially improved pooled vacancy receiving-yard MAE and Week 1 but failed p90, 2023 and vacancy-RB1 stability. R27 therefore isolates a remaining efficiency/context problem without authorizing another generic persistence model.

### Earlier receiving diagnostics

`scripts/backtest/decompose_receiving_error.py` already diagnosed receiving error as opportunity, catch-conversion and YPT components. Merely documenting YPT error is not a novel hypothesis.

### R19/R22 tail science

R19/R22 classify/shape receiving-yard residual tails around an upstream mean and preserve that mean. R19 uses baseline opportunity/mean, prior RB-room share, R8/R9 identity state and frozen YPT. These authorities do not constitute a point-mean model of target depth, YAC style, screen usage, QB checkdown environment or RB-specific opponent receiving vulnerability.

## Excluded as primary V2 novelty

The V2 candidate may not claim novelty from:

- career-to-date YPT;
- current-season YPT;
- trailing-window YPT;
- career/trailing YPR;
- generic catch-rate persistence;
- generic empirical-Bayes YPT/YPR shrinkage;
- the existing `production_ypt`, `bayes_ypt`, `rules_ypt` state itself;
- R8/R9 opportunity identity features;
- R19/R22 tail probabilities or residual pools;
- any sportsbook line/price/market probability.

Existing production YPT may remain the baseline whose residual is being explained. Role/vacancy flags may remain structural controls. Neither counts as new information.

## Candidate information frontier authorized for source audit

Only the following football information families are eligible to become V2 primary features if historical source coverage is proven before the frozen candidate plan:

1. RB target depth / air-yards-per-target;
2. RB YAC-per-reception / YAC style;
3. RB screen or behind-line-of-scrimmage target rate;
4. RB explosive receiving-play propensity;
5. team/QB RB checkdown tendency using football-only pass attempts and RB targets;
6. team RB target-shape/YAC environment that is not simply a restatement of player raw YPT;
7. opponent RB-specific receiving vulnerability: target depth, YAC, catch rate, explosive allowance and related football-only context.

## Source-audit rule

Before a V2 frozen candidate plan is written, the repository must verify 2019-2025 historical schema/identity availability for the proposed PBP-derived families. This audit may inspect source coverage only. It may not fit a prediction model, create candidate projections, score model-vs-actual receiving-yard error, choose features by performance, or tune thresholds.

Source audit authority:

- script commit: `25c337ba957e366d803c4f741d803dae82329cdb`
- workflow commit: `fd347b8f743c3829049a8052fcd0bc5d1ba72222`
- workflow: `RB R27B V2 Novel Efficiency Source Audit`

Any feature unavailable or identity-unsafe across the intended historical window may be removed before the frozen candidate plan, based only on source/schema evidence.

## Production boundary

No production file, R26 mechanism, R22 authority, R26Q seal or sportsbook routing is changed by this audit. V2 remains research-only until a later separately frozen qualification path succeeds.