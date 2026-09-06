# WR-ND6 — Player-Level Explosive Ceiling Diagnostic

## Status

Frozen research protocol. Diagnostic only. No production mean change is authorized by this document.

## Lineage

- Exact M38 parent remains `b98518d97b3038f471aee9ae3201009b2c70bb29`.
- M38 WR hierarchy multipliers remain frozen at `(1.40, 1.14, 0.91, 0.78)`.
- WR post-M38 four-component decomposition disposition: `OPPORTUNITY_DOMINANT`, but explosive yardage was the clear second family at ~29.7% overall positive attribution and ~40-46% in ceiling/large-underprediction tails.
- WR-ND1 showed YPT was the largest individual component while combined opportunity remained larger overall.
- WR-ND2 disposition: `MIXED_WR_EFFICIENCY_MECHANICS` with a YPR lean.
- WR-ND3 disposition: `NO_ACTIONABLE_DYNAMIC_ENTITLEMENT_SIGNAL`.
- WR-ND5 disposition: `NO_ACTIONABLE_SNAP_DEPTH_ENTITLEMENT_SIGNAL`.
- ND5 result commit: `e24287b0e26a705bda1035d18ea4a417493c6ede`.

The simple target-history, vacancy, snap-level, snap-acceleration, and depth-rank entitlement families have therefore failed the frozen actionable gates. ND6 moves to the next independently supported error family rather than retuning those signals.

## Research question

Can strictly pregame **player-level explosive receiving traits** or **opponent explosive vulnerability** identify WR receiving-yard upper-tail outcomes and large M38 underprojections?

ND6 is intentionally a ceiling / distribution diagnostic, not a mean retune. A tail phenomenon should not be forced into the M38 mean without separate evidence.

## Anti-duplication relative to QB M72

M72 tested an **aggregate offense receiving-weapon x defense bridge against QB passing-yard residuals** and failed.

ND6 is materially different:

- unit of analysis is the individual WR player-game;
- outcomes are WR receiving-yard residual / WR upper-tail events;
- player explosive history is preserved at player level rather than aggregated into a team weapon score;
- ND6 does not claim a QB effect;
- no WR-to-QB bridge may be tested here.

Do not recreate M72's aggregate weapon matchup score and call it a WR feature.

## Exact parent reconstruction

Every canonical ND6 run must separately rebuild exact M38 and verify:

- receiving-yard rows = `4647`
- receiving-yard MC MAE = `17.099904733366`
- WR evaluation rows = `2130` after exclusion of the known Isaiah Bond factorization anomaly
- target reconstruction MAE approximately `2.076010432546`

Any drift is an integrity failure.

## Leakage boundary

For target game G, candidate features may use only completed games strictly before G.

Allowed upstream:

- nflverse play-by-play from completed prior games;
- prior player targets, completed receptions, receiving yards, air yards, YAC;
- prior opponent pass attempts/completions and explosive receiving yards allowed.

Prohibited upstream:

- target-game receiving outcomes;
- realized target-game explosive catches;
- future games;
- sportsbook/player-prop or game-market inputs;
- postseason-released 2025 participation data;
- inferred WR-CB responsibility;
- current-result-driven window tuning.

## Frozen history windows

Use a maximum of the player's / defense's **last 8 eligible prior games**, crossing the prior season where necessary. This is fixed before results and mirrors the stable prior-history horizon already used in prior explosive research. No 4/6/10/12-game window search is permitted after results.

## Frozen player-level candidates

### A. `PLAYER_EXP20_PER_TARGET_PRIOR8`

Prior-8 completed gains of at least 20 receiving yards divided by prior-8 targets.

### B. `PLAYER_EXP40_PER_TARGET_PRIOR8`

Prior-8 completed gains of at least 40 receiving yards divided by prior-8 targets.

### C. `PLAYER_YAC_PER_RECEPTION_PRIOR8`

Prior-8 yards after catch divided by prior-8 completed receptions.

### D. `PLAYER_AIR_PER_TARGET_PRIOR8`

Prior-8 targeted air yards divided by prior-8 targets.

These are player-level ceiling traits. They are not combined in ND6.

## Frozen defense candidates

### E. `DEF_EXP20_PER_ATT_ALLOWED_PRIOR8`

Opponent prior-8 completions gaining at least 20 yards divided by pass attempts faced.

### F. `DEF_EXP40_PER_ATT_ALLOWED_PRIOR8`

Opponent prior-8 completions gaining at least 40 yards divided by pass attempts faced.

### G. `DEF_YAC_PER_COMPLETION_ALLOWED_PRIOR8`

Opponent prior-8 YAC allowed divided by completions allowed.

### H. `DEF_AIR_PER_ATT_ALLOWED_PRIOR8`

Opponent prior-8 targeted air yards faced divided by pass attempts faced.

## No interactions in ND6

Do not multiply player and defense candidates after results.

A later interaction branch is authorized only if at least one player-side candidate and at least one defense-side candidate independently pass the frozen ND6 gate.

## Outcomes

Continuous descriptive outcome:

`rec_yards_residual = actual receiving yards - M38 receiving-yard projection`

Frozen upper-tail events carried forward from the post-M38 decomposition:

1. `UNDER25`: actual receiving yards minus M38 projection >= 25 yards.
2. `UNDER50`: actual receiving yards minus M38 projection >= 50 yards.
3. `ACTUAL100`: actual receiving yards >= 100 yards.

These outcomes are evaluation-only. They may not feed features.

## Signal scoring

For every candidate report:

1. coverage;
2. Spearman correlation with continuous receiving-yard residual;
3. top-quartile vs bottom-quartile residual gap;
4. `UNDER25` tail enrichment;
5. `UNDER50` tail enrichment;
6. `ACTUAL100` tail enrichment;
7. W2-18 residual gap;
8. W13-18 residual gap;
9. WR1 / WR2 / WR3 residual gaps under the frozen M38 role classification.

High/low is top vs bottom quartile among available values. Quartile cut points are computed once over the canonical evaluation population and reported.

## Frozen actionable ceiling gate

A candidate advances only if ALL are true:

- coverage >= `0.75`;
- Spearman with receiving-yard residual >= `+0.08`;
- high-vs-low receiving-yard residual gap >= `+4.0` yards;
- `UNDER25` enrichment >= `1.25x`;
- at least one of `UNDER50` or `ACTUAL100` enrichment >= `1.25x`;
- W2-18 residual gap > `0`;
- W13-18 residual gap > `0`;
- positive residual gap in at least 2 of WR1/WR2/WR3.

No gate may be lowered after results are visible.

## Frozen disposition

- exactly one player-side signal passes and no defense signal passes: `<SIGNAL>_PLAYER_CEILING_SIGNAL`
- exactly one defense-side signal passes and no player signal passes: `<SIGNAL>_DEFENSE_CEILING_SIGNAL`
- one or more player-side AND one or more defense-side signals pass: `PLAYER_AND_DEFENSE_CEILING_SIGNALS`
- multiple pass on only one side: `MULTIPLE_<PLAYER_OR_DEFENSE>_CEILING_SIGNALS`
- none pass: `NO_ACTIONABLE_EXPLOSIVE_CEILING_SIGNAL`

A pass authorizes only a later frozen distribution/integration test. It does not authorize changing M38 mean projection logic.

## Separate future QB bridge

If ND6 eventually produces a validated **player-level pregame WR ceiling signal**, that signal may later be aggregated into receiver upper-tail mass and tested against QB passing-yard upper-tail misses. That later test must remain separate and must not reuse realized WR explosions upstream.
