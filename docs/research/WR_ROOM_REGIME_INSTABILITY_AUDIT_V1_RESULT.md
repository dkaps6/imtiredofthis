# WR Room Regime-Instability Audit V1 — Result

Status: **COMPLETE — STRUCTURAL HYPOTHESIS WARRANTED, NO CANDIDATE SCORED**

Parent model disposition remains unchanged:

`RECEIVER_ROOM_TARGETS_PER_PLAY_V1_2024_2025_FAILED_CLOSED`

No integration is authorized from this audit.

## Provenance

Diagnostic branch:
`research-wr-room-regime-instability-audit-v1`

Frozen plan:
`docs/research/WR_ROOM_REGIME_INSTABILITY_AUDIT_V1_PLAN.md`

Run:
`36175902960`

Job:
`108206313033`

Run head:
`d2de6f78a31a2c5d2641933e8e4f1bbf5bd76019`

Artifact:
`10882621716`

Artifact digest:
`sha256:50b5584b19872290ec9f06f4225fffa8ef9a50fff7a0e24ce957e6f5b1316402`

Artifact:
`wr-room-regime-instability-audit-v1`

Frozen scored inputs were taken directly from the already-authoritative Receiver Room Targets-Per-Play V1 discovery and confirmation artifacts:

- 2022-2023 discovery run `36172644864`;
- 2024-2025 confirmation run `36174077739`.

The failed candidate was not recomputed or changed.

Audit controls:

- candidate variants scored: **0**
- parameters fit: **0**
- sportsbook inputs used: **0**
- production mutations: **0**

## Frozen WR result reproduction

The audit reproduced the authoritative WR room score exactly:

| season | baseline MAE | candidate MAE | mean candidate harm | harmful teams | helpful teams |
|---|---:|---:|---:|---:|---:|
| 2022 | 4.761250 | 4.498155 | -0.263094 | 11 | 21 |
| 2023 | 4.538430 | 4.467731 | -0.070700 | 13 | 19 |
| 2024 | 4.561046 | 4.724467 | +0.163422 | 18 | 14 |
| 2025 | 4.510085 | 4.440251 | -0.069834 | 15 | 17 |

Here `candidate harm = abs(candidate - actual) - abs(baseline - actual)`.
Positive values mean the failed room-history candidate made the row worse.

## Main finding

**2024 is not well explained as a uniform league-wide WR-room regime shift.**

There is some league-wide instability — 18 of 32 teams were net harmful — but the direction of failure is strongly associated with a more specific structural state:

**continuity of the offense's primary QB and verified primary play-caller.**

The evidence is materially stronger for QB/play-caller continuity than for generic WR roster turnover.

### 1. Prior-primary-QB continuity separates the candidate in the same direction across all four seasons

At the team-game level:

| season | prior primary QB NOT on pregame roster | prior primary QB on pregame roster |
|---|---:|---:|
| 2022 mean harm | -0.055642 | **-0.335341** |
| 2023 mean harm | +0.019532 | **-0.126557** |
| 2024 mean harm | **+0.555071** | -0.079028 |
| 2025 mean harm | **+0.148745** | -0.198937 |

The exact magnitude changes by season, but the structural contrast has the same sign in all four seasons: the prior-season room-history candidate is worse when the prior primary QB is no longer on the target-week pregame roster.

The effect is especially large in the failed 2024 season.

A stricter current-season transition descriptor points the same way in 2024:

- strict-prior primary QB unchanged: mean harm **-0.200309**;
- strict-prior primary QB changed: mean harm **+0.638741**.

That stricter descriptor is weaker in 2025, so it is not sufficient by itself.

### 2. Verified play-caller continuity independently replicates in both seasons with complete cross-season source coverage

The repo's already-frozen verified play-caller source has complete prior-season comparison coverage for 2024 and 2025.

Team-game mean harm:

| season | play-caller unchanged vs prior-season end | play-caller changed |
|---|---:|---:|
| 2024 | **-0.110279** | +0.427241 |
| 2025 | **-0.331351** | +0.233192 |

The changed-minus-stable harm contrast is:

- 2024: **+0.537520 targets**
- 2025: **+0.564543 targets**

This is a genuine two-season replication in direction and approximate magnitude.

The source is not complete for a 2022 -> 2023 comparison, so no earlier-season claim is made for this feature.

### 3. The strongest state is the joint QB + play-caller structural break

Using only pregame information:

#### 2024

- prior primary QB absent **and** play-caller changed:
  - n = 133 team-games
  - mean harm = **+0.776751**
  - candidate closer rate = **41.35%**
- prior primary QB retained **and** play-caller unchanged:
  - n = 192
  - mean harm = **-0.216622**
  - candidate closer rate = **54.69%**

At the season-opening team classification level, the six 2024 teams with both a Week-1 QB break and a play-caller break produced **+89.900 total excess absolute error** across the season.

The entire 2024 WR candidate's net excess absolute error was **+88.901**.

That does not mean every double-transition team failed — PIT and NE were counterexamples — but the cohort is large enough to explain essentially the full net league failure after offsetting gains elsewhere.

#### 2025

The same joint direction replicated:

- prior primary QB absent + play-caller changed:
  - n = 133
  - mean harm = **+0.263535**
- prior primary QB retained + play-caller unchanged:
  - n = 223
  - mean harm = **-0.411438**

At the season-opening team level, the seven double-transition teams produced **+33.957 total harm** even though the overall 2025 WR candidate improved by **-37.990 total harm**.

In other words, the same structural transition cohort remained harmful inside a season where the aggregate candidate succeeded.

That is the most important replication in this audit.

### 4. The transition state also corresponds to larger realized WR-room regime drift

This is outcome-only mechanism evidence, not deployable input.

Absolute prior-season -> full-current-season WR targets-per-play drift:

#### 2024

- prior primary QB absent: **0.073878**
- prior primary QB retained: **0.036058**
- play-caller changed: **0.060668**
- play-caller stable: **0.038816**

#### 2025

- prior primary QB absent: **0.052511**
- prior primary QB retained: **0.031737**
- play-caller changed: **0.050152**
- play-caller stable: **0.028955**

Thus the pregame transition markers are not merely labels attached to error. In both confirmation-era seasons they identify teams whose realized WR-room allocation changed more from the prior season.

This realized-drift evidence is diagnostic only and cannot itself be used by a future model.

### 5. Generic WR-room roster turnover does not explain the failure

The leakage-safe fraction of prior-season WR target mass retained on the target-week pregame roster has almost no monotonic relationship with candidate harm:

- 2024 Spearman vs harm: **+0.021607**
- 2025 Spearman vs harm: **-0.014430**
- pooled: **+0.016636**

Prior top-WR roster presence is also directionally inconsistent across seasons.

Therefore the audit does **not** support a generic "new WR room means reset history" rule.

This is important because it keeps the result distinct from the already-closed Active-Roster Receiver Room State V1 lane.

### 6. The failed 2024 state shows slow adaptation, but no recency/window rescue is authorized

2024 mean candidate harm by early week:

- Week 1: **+0.686610**
- Week 2: **+0.859870**
- Week 3: **+0.666521**
- Week 4: **+0.191585**

Late 2024 becomes mixed and Weeks 17-18 are negative.

2025 also begins with a harmful Week 1 (**+0.514037**) but finishes with candidate gains in Weeks 15-18.

This is consistent with prior-season structural history becoming stale around offense-regime transitions and current-season evidence eventually becoming more informative.

However, this audit did **not** search a crossover week, window length, recency weight, shrinkage factor, or blend. None is authorized.

### 7. 2024 harm is concentrated enough to reject a pure league-regime explanation, but not confined to only a few teams

2024:

- 18 harmful teams;
- 14 helpful teams;
- top eight harmful teams account for **75.43%** of total positive harm.

The highest-harm teams were ATL, CHI, NYG, LV, NYJ, WAS, LAR and CLE.

The top three — ATL, CHI and NYG — each combined a season-opening primary-QB transition and verified play-caller transition and each realized a large positive WR targets-per-play drift.

But several transition teams were counterexamples, and several stable teams were harmful.

So the correct interpretation is **probabilistic structural instability**, not a deterministic router.

## Scientific disposition

**STRUCTURAL HYPOTHESIS WARRANTED.**

The evidence is strong enough to justify a separate frozen hypothesis because:

1. prior-primary-QB continuity has a same-direction relationship with room-history harm across 2022, 2023, 2024 and 2025;
2. verified play-caller continuity independently replicates across both seasons with complete source coverage, 2024 and 2025;
3. the joint QB + play-caller transition state is harmful in both 2024 and 2025 while the joint stable state improves;
4. transition states exhibit materially larger realized WR-room drift in both 2024 and 2025;
5. generic WR roster turnover does not provide the same explanation.

This does **not** rehabilitate Receiver Room Targets-Per-Play V1.

That exact candidate remains failed closed and may not be rescued by excluding 2024, WR-only routing, hand-picked windows, recency/shrinkage, fixed57 blending, bias offsets, specialist-order changes or 2026 outcome fitting.

## Next research action

Freeze a separate architecture hypothesis around an **offensive regime boundary** before any scoring.

The hypothesis must be structural rather than a retrospective threshold:

> prior-season team room history should not automatically be treated as belonging to the same offensive regime when the pregame offense has crossed a hard identity boundary such as a primary-QB change and/or verified primary play-caller change.

Any future candidate must:

- be frozen before scoring;
- use only pregame-identifiable regime boundaries;
- avoid fitted thresholds/windows/weights;
- not route the already-failed WR candidate around 2024;
- preserve TE/RB-FB safeguards;
- use no sportsbook information;
- use no 2026 outcomes for fitting;
- remain separate from the closed Active-Roster Receiver Room State and generic recency/shrinkage lanes.

Because the diagnostic evidence itself used 2022-2025 outcomes, those seasons are not an untouched confirmation set for a newly invented architecture. The next plan must explicitly solve the clean-validation problem before claiming support.
