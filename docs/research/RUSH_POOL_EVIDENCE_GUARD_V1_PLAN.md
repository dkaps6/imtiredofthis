# Rush Pool Evidence Guard V1 — Frozen Research Plan

Status: **FROZEN BEFORE HISTORICAL OUTCOME SCORING**

Research-only branch: `research-rush-pool-evidence-guard-v1`

Parent production main at freeze:
`e5f630d7ece7762a71ae238c210408fb9d4fb3cb`

## Why this exists

The current canonical rushing simulator already has a finite opportunity pool:

1. compute player-level `rules_rush_share`;
2. retain the five largest shares with `_top_n_shares(..., 5)`;
3. normalize those retained players to at most 95% of team rush attempts;
4. preserve a 5% residual bucket.

Therefore raw team rush-share sums greater than 1 are **not** themselves a defect.

The new structural concern is narrower.

The Bayesian layer assigns a position-group rushing prior when a roster player has neither player-specific prior history nor current-season evidence. The resulting state is:

`bayes_evidence_state == "position_prior_only"`

Because the canonical top-five selector ranks only the numeric share, those synthetic fallback priors can occupy a finite top-five slot ahead of a teammate who has real player-specific strict-prior/current rushing evidence.

This can happen without any sportsbook input or target-game outcome. It is visible directly in preserved pregame Week-2 production inputs.

This candidate asks one football/provenance question:

> From Week 2 onward, should a synthetic position-only fallback be allowed to displace an evidenced rusher from the finite five-player carry pool?

## Frozen candidate

Version:

`RUSH_POOL_EVIDENCE_GUARD_V1`

Scope:

- regular-season Week >= 2 only;
- rushing opportunity / carry-pool selection only;
- all roster positions remain eligible;
- existing top-five pool size remains exactly five;
- existing raw `rules_rush_share` values remain unchanged;
- existing 95% player-mass cap and residual bucket remain unchanged;
- existing team rush-attempt process remains unchanged;
- existing YPC / rushing efficiency remains unchanged.

### Baseline

Exact current production selector:

1. sanitize/clip current `rules_rush_share`;
2. retain the five highest positive player shares;
3. normalize through the existing allocator.

### Candidate selector

For positive-share players only:

1. split players into:
   - **evidenced**: `bayes_evidence_state != "position_prior_only"`;
   - **fallback-only**: `bayes_evidence_state == "position_prior_only"`;
2. select evidenced players first, ranked by the exact existing raw `rules_rush_share`, up to five;
3. if fewer than five positive-share evidenced players exist, fill remaining slots with fallback-only players, also ranked by the exact existing raw `rules_rush_share`;
4. apply the exact existing 95% cap / normalization / residual semantics.

No player share is increased because of depth chart, role label, name, position, injury narrative, market line, or outcome.

A fallback-only player is **not banned**. The candidate only prevents a synthetic group prior from displacing player-specific evidence when enough evidenced positive-share rushers exist.

## Explicit no-op / protected scope

Do not change:

- Week 1;
- top-five pool size;
- Bayesian prior strengths;
- Bayesian group means/defaults;
- player form;
- current/prior history construction;
- depth charts;
- vacancy logic;
- P3;
- M94C/M95/M96 families;
- YPC;
- rush-efficiency distribution;
- target shares;
- receptions;
- receiving yards;
- QB passing;
- team pass/rush tendency;
- sportsbook inputs;
- fair-probability translator;
- bet-selection layer.

This is **not** a new RB residual router, M96 variant, depth-rank remap, threshold search, or position-weight search.

## Historical design

Use timestamp-safe canonical pregame construction independently for:

- 2024 Weeks 2-18, using 2023 as prior season;
- 2025 Weeks 2-18, using 2024 as prior season.

The formula is frozen before either historical outcome set is scored.

No fit parameters exist.

No candidate variants are allowed.

## Primary scoring population

For each season, score player-games where either:

- actual rush attempts > 0, or
- baseline projected rush attempts > 0, or
- candidate projected rush attempts > 0.

Report separately:

- ALL positions;
- RB/FB/HB family;
- QB;
- other positions.

Primary metric is rush-attempt point-projection error.

## Frozen integrity gates

All must pass:

1. zero sportsbook fields used;
2. zero target-game outcomes used in any pregame feature/selector;
3. Week 1 candidate delta = exactly zero;
4. raw `rules_rush_share` values are unchanged;
5. player pool size remains <= 5;
6. candidate player probability sum remains <= 0.95 and uses the same residual semantics;
7. team rush-attempt mean is unchanged;
8. no target/reception/receiving/pass input is changed;
9. only slot membership/normalization can differ;
10. historical evidence states are generated from strict-prior/current-to-cutoff information only.

## Frozen scientific qualification gates

`RUSH_POOL_EVIDENCE_GUARD_V1_QUALIFIED` requires ALL:

1. ALL-position rush-attempt MAE strictly improves in 2024;
2. ALL-position rush-attempt MAE strictly improves in 2025;
3. RB/FB/HB rush-attempt MAE strictly improves in 2024;
4. RB/FB/HB rush-attempt MAE strictly improves in 2025;
5. pooled 2024-2025 ALL-position rush-attempt MAE improves by at least **0.02 attempts**;
6. ALL-position p90 absolute rush-attempt error is non-worse in both seasons;
7. RB/FB/HB p90 absolute rush-attempt error is non-worse in both seasons;
8. QB rush-attempt MAE is non-worse in both seasons;
9. on rows whose projected carry mean changes, candidate is closer to actual more often than baseline in both seasons;
10. the count of evidenced positive-share players omitted from the finite top-five pool strictly decreases in both seasons;
11. all integrity gates pass.

If any qualification gate fails:

`RUSH_POOL_EVIDENCE_GUARD_V1_FAILED_CLOSED`

No rescue.

## If it qualifies

Qualification does **not** authorize production.

Freeze a separate integration plan and then prove, on the exact current production stack:

- carry projection improvement survives canonical MC;
- rush-yard means/tails do not regress;
- RB rush+receiving does not regress;
- QB rushing does not regress;
- Week 1 remains untouched;
- RB V2 combined-yard production identity remains intact;
- receiving/pass markets are bitwise or numerically unchanged as applicable;
- Full Slate lineage and fail-closed audit behavior are correct.

Only after those checks may production promotion be considered.

## Stopping rule

After historical scores are visible, do **not**:

- search another evidence definition;
- search pool sizes;
- add minimum-share thresholds;
- add position-specific priorities;
- add depth-chart gating;
- add rookie exceptions;
- add QB exceptions;
- add injury-conditioned exceptions;
- retune Bayesian priors;
- fit to 2026 Week-1/Week-2 outcomes.

If this exact invariant fails, close it and move to a genuinely different model-improvement hypothesis.
