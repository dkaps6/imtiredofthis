# One-Pass-State Integration V1 — Frozen Candidate Plan

Date: 2026-09-24

Status: **FROZEN BEFORE OUTCOME SCORING**

Parent structural audit:
- branch: `research-shared-pass-state-coherence-v1`
- result commit: `7cb01ea009c0d19fc2298f24c59b9902f8f8780a`
- authoritative audit run: `36073919850`
- artifact: `10839371645`
- disposition: `SHARED_PASS_STATE_COHERENCE_V1_STRUCTURAL_DIVERGENCE_CONFIRMED`

This candidate is research-only. It does not authorize production changes.

## 1. Structural problem being tested

Current production can use two different Monte Carlo passing-game realizations on the same team-game:

1. a C2 completed-pass / receiving realization to generate the selected QB's pass-yard distribution; and
2. a separate canonical receiving realization for WR/TE/RB receptions and receiving yards.

The read-only audit confirmed material divergence in the current 2026 Week-3 stack:
- median selected-team correlation between final C2 QB pass yards and canonical modeled receiving total: `0.5575`;
- median canonical-vs-C2-shadow player receiving-yard array correlation: `0.0399`;
- median player p90 absolute draw gap: `30.84 yd`;
- high-entitlement Q4 median p90 gap: `49.53 yd`;
- canonical median player rate of zero receptions with positive receiving yards: `16.42%`;
- exact C2 completed-pass process: `0%`.

No sportsbook inputs or target-game outcomes were used to discover that contradiction.

## 2. Exact candidate

Candidate name:
`ONE_PASS_STATE_INTEGRATION_V1`

The candidate changes **only** receiver outputs on team-games where the already-frozen production C2 selector chooses `C2_SELECTED`.

For those selected team-games:

1. keep the exact current production team pass-attempt state;
2. keep the exact current M38 + WR-R15 + TE-R5P target entitlement;
3. keep the exact current C2 target allocation and completed-pass process;
4. keep the exact current C2 residual receiving bucket;
5. keep the exact current C2 team-game scale factor that anchors the QB distribution to the existing production QB mean;
6. keep the final selected QB `pass_yards` array **bit-identical** to current production;
7. replace only the modeled pass-catcher:
   - `receptions`
   - `rec_yards`
   with the exact arrays generated inside that same C2 realization;
8. for RB rows, rebuild `rush_rec_yards` from unchanged rushing + candidate receiving before the current RB Rush+Receiving Conservation V2 downstream seam is applied;
9. leave all unselected-QB team-games on the exact current canonical receiver state.

No player/position carveout is allowed. The only route is the existing frozen QB C2 selector.

## 3. Explicitly unchanged

The candidate must not change:
- M89/M90 / QB_PASS_SYNTHESIS_V1 mean authority;
- the existing C2 selector or any selector threshold;
- selected/unselected QB identity;
- any QB `pass_yards` array;
- plays, pass rate, pass attempts or game pace state;
- M38;
- WR-R15;
- TE-R5P;
- target entitlement shares;
- target pool / residual target mass;
- rush attempts;
- rush yards;
- rushing share;
- YPC;
- RB Rush+Receiving Conservation V2 formula or scope;
- ATD;
- sportsbook inputs;
- production code.

C1 and C3 remain closed and are not used.

## 4. Historical evaluation design

This is a **current-stack requalification**, not a claim of pristine blind discovery.

The exact receiver mechanism has older historical evidence, but the current production stack now includes:
- WR-R15;
- TE-R5P;
- current C2 selector;
- current ensemble ordering;
- RB Rush+Receiving Conservation V2.

Therefore the historical question is whether the exact one-pass-state repair survives the **current stack**.

Use leakage-safe historical reconstruction with the same fold-safe specialist authority already used by the most recent current-stack full-stack research:
- 2024 regular season;
- 2025 regular season;
- strictly prior football inputs only;
- no target-game outcome in features;
- no sportsbook inputs;
- fixed candidate;
- no parameter fitting;
- no candidate variants.

Report 2024 and 2025 separately and pooled.

Because 2024-2025 outcomes have been exposed elsewhere in the project, historical qualification alone must **not** be described as prospective proof. A passing historical candidate remains eligible for a separate current-2026 shadow/prospective confirmation step before production promotion.

## 5. Required mechanical/invariance gates

All must pass:

1. current production baseline is reproduced under the historical harness;
2. exact current C2 selector is used with zero changes;
3. selected QB team identities are identical baseline vs candidate;
4. every QB `pass_yards` array is bit-identical baseline vs candidate;
5. every unselected-team receiver array is bit-identical baseline vs candidate;
6. target entitlement trace is identical baseline vs candidate;
7. team pass attempts and shared pass-efficiency state are identical;
8. every `rush_att` array is bit-identical;
9. every standalone `rush_yards` array is bit-identical;
10. candidate has zero player-iterations with zero receptions and positive receiving yards;
11. on selected teams, QB passing yards = candidate modeled receiver yards + candidate residual receiving yards within `1e-10` per draw;
12. RB rush+receiving pathwise identity remains exact after the current V2 downstream seam;
13. sportsbook inputs = 0;
14. target-game outcomes used upstream = 0;
15. parameters fit = 0;
16. variants scored = 1.

Any failure above is mechanical/integrity failure, not science.

## 6. Frozen scientific scorecard

Primary receiver positions:
- WR
- TE
- RB

Primary markets:
- receptions
- receiving yards

Dependent protected market:
- RB rush+receiving yards

Report for each season, each position and pooled:
- n;
- MAE;
- RMSE;
- bias;
- correlation;
- median AE;
- p75 AE;
- p90 AE;
- 20+ / 30+ / 40+ receiving-yard miss rates;
- high-entitlement Q4 separately.

Also report:
- changed-row candidate closer rate;
- player-level mean shift distribution;
- position-level mean shift;
- selected-team vs unselected-team parity;
- RB rush+receiving MAE/p90/tail misses.

## 7. Frozen scientific gates

`ONE_PASS_STATE_INTEGRATION_V1_QUALIFIED` requires **all**:

### Receiving yards
1. pooled WR/TE/RB macro receiving-yard MAE improves;
2. 2024 WR/TE/RB macro receiving-yard MAE is nonworse;
3. 2025 WR/TE/RB macro receiving-yard MAE is nonworse;
4. no individual position pooled receiving-yard MAE worsens by more than `0.35 yd`;
5. pooled macro receiving-yard p90 is nonworse;
6. high-entitlement Q4 receiving-yard MAE is nonworse;
7. high-entitlement Q4 receiving-yard p90 is nonworse;
8. pooled macro 40+ yard miss rate does not worsen by more than `0.25 percentage points`.

### Receptions
9. pooled WR/TE/RB macro receptions MAE is nonworse;
10. 2024 macro receptions MAE is nonworse;
11. 2025 macro receptions MAE is nonworse;
12. no individual position pooled receptions MAE worsens by more than `0.05 receptions`;
13. pooled macro receptions p90 AE is nonworse.

### RB dependent market
14. RB rush+receiving MAE is nonworse in 2024;
15. RB rush+receiving MAE is nonworse in 2025;
16. RB rush+receiving p90 is nonworse pooled.

### Architecture protection
17. all mechanical/invariance gates pass.

If any frozen scientific gate fails:
`ONE_PASS_STATE_INTEGRATION_V1_FAILED_CLOSED`

No rescue is authorized.

## 8. No-rescue rule

After results are visible, do not try:
- WR-only / TE-only / RB-only routing;
- player-level routing;
- entitlement thresholds;
- target-share thresholds;
- changing the C2 selector;
- changing residual catch rate or residual YPT;
- changing YPR clipping;
- player-specific scaling;
- position-specific scaling;
- alternate team scale formulas;
- sportsbook-conditioned routing;
- 2026 outcome fitting;
- C1/C3 resurrection.

A failure closes this exact candidate.

## 9. Promotion rule

Historical qualification does **not** directly authorize production.

If qualified:
1. freeze a separate 2026 current-stack shadow/prospective confirmation contract;
2. preserve pregame candidate outputs before outcomes;
3. score once outcomes exist;
4. only then consider a separately frozen production-integration certification.

No paid OddsAPI pull is authorized by this plan.
