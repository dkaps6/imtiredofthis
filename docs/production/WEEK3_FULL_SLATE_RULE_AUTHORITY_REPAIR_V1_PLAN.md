# Week 3 Full Slate Football Rule Authority Repair V1 — Frozen Plan

Date frozen: 2026-09-26
Branch: `fix-week3-full-slate-rule-authority-v1`
Parent: PR #644 head `f06d3c8807b554d7ef0288b98cef799fe43e1fe2`

Status: **FROZEN PRODUCTION-CORRECTNESS REPAIR — NO SCIENCE CHANGE**

## Failure authority

Farthest Week-3 live Full Slate run:
- run `36276421905`
- head `753a36a8ca77feb16373eb7c2439efed05e9781f`
- artifact `10917037948`
- digest `sha256:738422833937cd40ddbd7dd85ebf01741d11aec079eb681a7f8cf87818058f90`

The availability-aware 30-team seam passed. Pricing then failed at:

`full-universe football assumption drift for rules_tgt_share: max_abs_diff=0.014064361123645175`

The prior 32-team diagnosis is therefore superseded for this run.

## Root cause

LAR WR Puka Nacua:
- exists in certified PlayerForm / ModelContext;
- had no sportsbook-shaped metrics row in the failed live run;
- is the WR1 alpha seen by the injury redistribution rule;
- raw PlayerContext target share = approximately 0.309002;
- full empirical-Bayes target share = approximately 0.264775.

`simulation_rules.apply_rules_to_metrics()` currently builds its Bayesian target-share override map only from the metrics rows passed into the function.

Therefore:
- sportsbook-shaped pricing path omitted Puka from the Bayesian override map and used the raw PlayerContext fallback;
- full-roster football path included Puka and used his Bayesian posterior;
- the injury rule redistributed different vacancy mass to LAR successors.

This changed exactly the rule recipients in the observed failure while the football coefficients themselves were unchanged.

## Frozen repair

1. Build/load the full leakage-safe Bayesian baseline once in `run_pricing_v2.price()`.
2. Pass that same full baseline to:
   - `apply_bayesian_to_metrics()`, as before semantically;
   - `apply_rules_to_metrics()` as a new optional football-authority argument.
3. When the optional full Bayesian baseline is supplied, injury redistribution must source `bayes_tgt_share` from that complete football baseline rather than reconstructing the map from sportsbook-shaped metrics rows.
4. When no full baseline is supplied, preserve the existing fallback behavior exactly for historical/tests/other callers.

No change to:
- injury status definitions;
- 50% alpha vacancy fraction;
- 60/30/10 recipient weights;
- matchup multipliers;
- M38 / TE-R5P / WR-R15;
- ensemble weights;
- ML / State;
- QB synthesis;
- RB P3 or RB Rush+Receiving V2;
- player availability;
- sportsbook lines or odds;
- partial-slate team certification.

## Frozen validation

All required:

1. Focused unit test reproduces a missing sportsbook alpha:
   - alpha exists in PlayerContext/full Bayesian baseline;
   - alpha absent from metrics rows;
   - partial-metrics rule result with full baseline equals full-metrics rule result for all surviving recipients.
2. Backward-compatibility test:
   - no optional baseline => current legacy behavior remains callable.
3. Exact preserved Week-3 artifact replay:
   - no new OddsAPI acquisition;
   - reuse run `36276421905` artifact;
   - previously failing football-assumption equality gate passes;
   - exact current certified stack continues through pricing.
4. Non-scope protection:
   - no change to football coefficients or model weights;
   - no sportsbook input enters Bayesian/rule football authority;
   - no Week-3 outcomes used.
5. Repo CI green.

If the replay reaches a new independent blocker, stop and diagnose it separately; do not broaden this repair.
