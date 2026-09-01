# M96B — RB Modular Joint Workload × Efficiency Synthesis — Results

## Authoritative run

- Workflow: `M96B RB Modular Joint Synthesis`
- Run: **`33461369073`**
- Job: **`99711988023`**
- Tested SHA: **`a3018bf828bf0c78b09a2e0b8a6cd1af60b25f40`**
- Artifact: **`9783267179`**
- Artifact name: `migration-96b-rb-modular-joint-synthesis`
- Artifact SHA256: **`81f25e134a34a6b2d8b28195bb7f804f6ce54b5823bead5a2e4b717ee544718b`**
- Execution: **SUCCESS**
- Disposition: **`M96B_MODULAR_SYNTHESIS_COMPLETE`**
- Sportsbook inputs: `0`
- Feature search: `0`
- Weight search: `0`
- Production change: `0`

The protocol was frozen first in `docs/migrations/M96B_PLAN.md`. The five allowed components were C (M94C central point), W (M95F workload tail), V (M95I vacancy evidence), E (M95C environment mean-efficiency residual), and X (M95D incremental upside residual). No post-result weight or feature-combination search was allowed.

## Source / join integrity

Primary 2025 intersection:

- M95D 2025 OOS rows: `1290`
- joined M94C + M95D + M95F rows: `1274`
- coverage vs M95D: **`98.7597%`** — PASS vs frozen `97%` gate

2024 W13-18 temporal calibration intersection:

- M95F holdout rows: `479`
- common M95D/M95F rows: `449`
- coverage vs M95F holdout: `93.7370%`

No truth-parity failure was observed on shared rows.

## Capability ledger result

### C — M94C central point

**RETAIN.**

On the exact 2025 M96B intersection:

- rushing-yard MAE: **`21.8440`**
- RMSE: `30.5811`
- bias: `+1.0954`
- correlation: `.5853`

C remained the selected point anchor.

### E — transplanted M95C environment residual onto M94C point

**REJECT as an additive point module.**

Frozen direct construction:

`CE = M94C candidate_rush_yards + (M95C-environment prediction - role-baseline prediction)`

2025 all-RB:

- C MAE `21.8440`
- C+E MAE `21.9095`
- gain `-0.0654`
- C bias `+1.0954`
- C+E bias `+1.3486`

The worst ordinary-slice MAE regression was only `0.4274` yards, so E did not catastrophically damage the middle, but it failed the primary requirement that all-RB MAE improve.

Important nuance: the frozen M95C environment model remains a real signal inside its original model family. On the common M95D universe it improved its own role baseline slightly in both years:

- 2024 W13-18 common: `22.2302 -> 22.2168`
- 2025 common: `21.9897 -> 21.9514`

M96B therefore did **not** prove environment information useless. It proved that the old M95C residual is not plug-compatible as a simple additive correction to the stronger M94C point output. This motivates fitting any next efficiency correction directly against the M94C yard residual instead of transplanting a delta learned around another baseline.

### W — M95F upper workload distribution

**Directional positive, but REJECT under the frozen retention gate.**

Using equal rank fusion of the point-tail score with M95F p90/p95 workload information:

75+ rushing yards, full 2025:

- B AUC `.799739`
- B+W AUC `.802295` — gain `+.002556`
- Brier `.112127 -> .111325` — gain `+.000803`

100+ rushing yards, full 2025:

- B AUC `.799035`
- B+W AUC `.799324` — gain `+.000288`
- Brier `.063428 -> .063343` — gain `+.000085`

W improved all four full-season numbers directionally but failed the precommitted materiality threshold (`+.005` AUC or `+.001` Brier). The late-2025 comparable window did not show a damaging reversal, but the gain was too small to retain W as a yard-tail module from M96B.

This does **not** erase M95F's validated job as a carry/workload distribution. It means simply adding its upper-workload rank to rushing-yard tail ranking did not provide enough incremental value over M94C's yard score.

### X — isolated incremental M95D upside residual

**REJECT in additive/rank-residual form.**

M96B deliberately isolated:

`X_delta = full_environment_matchup - M95C_environment`

and fused that residual rank with the base tail score. This was strongly destructive:

75+ full-2025:

- B AUC `.799739`
- B+X `.726578`
- AUC change `-.073161`
- Brier worsened `.112127 -> .120215`

100+ full-2025:

- B AUC `.799035`
- B+X `.720655`
- AUC change `-.078381`

The `B+W+X` combination was also not preferred.

Important interpretation: M95D's full matchup model had shown better 100+ discrimination than its own environment-only control in both 2024 and 2025. M96B proves that this value is **not separable as a simple additive residual rank on top of M94C**. X behaves more like an interacting/conditional expert than an independent additive module.

### V — M95I vacancy module

**RETAIN AS DIAGNOSTIC ONLY, not promoted.**

Within 105 2025 vacancy rows, the frozen M96B V fusion improved against the predeclared non-V `B+W+X` comparison arm:

75+:

- events `17`
- AUC `.63035 -> .67213`
- gain `+.04178`

100+:

- events `9`
- AUC `.73264 -> .75752`
- gain `+.02488`

Caveat: the frozen comparison parent was `B+W+X`, which itself failed globally. Therefore M96B does **not** establish that V beats the best M94C-only yard-tail baseline. V remains a promising vacancy-specific signal requiring a direct-baseline and prospective confirmation test before any promotion.

## What M96B teaches about the modular / puzzle approach

The modular philosophy survives, but **simple addition is not the same as modularity**.

M96B found three distinct cases:

1. **Compatible but redundant:** E has real environment information, but transplanting its old residual onto M94C does not improve the stronger point anchor.
2. **Helpful but too small:** W improves rushing-yard tail metrics directionally but not enough to pass the materiality gate.
3. **Interactive rather than additive:** X's full model has tail signal in its native architecture, but isolating and adding its residual destroys ranking.
4. **Conditional regime signal:** V remains promising specifically for vacancy/transition rows, but is not yet proven against the best yard-tail baseline.

This is exactly why future research must keep a capability ledger and test modules by responsibility rather than treating every positive feature as a universal coefficient.

## Point-workload implications

M96B does not reopen generic carry-tail search. C/M94C remains the central workload and point-yard anchor.

M96A already established that substantial rushing-yard error remains recoverable from efficiency. M96B now shows the existing M95C efficiency delta cannot simply be transplanted onto M94C. Therefore the next justified point-model experiment is a **dedicated M94C-anchored efficiency residual model**.

## NEXT MIGRATION — M96C

Name: **M96C — M94C-Anchored RB Efficiency Residual Synthesis**

Primary question:

> Can leakage-safe player/offense/defense rushing-efficiency information predict the residual rushing-yard error left after M94C's frozen opportunity/yard center, without changing carries and without damaging ordinary workload regimes?

Required direction:

- Freeze M94C carries and central rushing-yard prediction.
- Train directly on `actual_rush_yards - M94C_rush_yards` (or an algebraically equivalent efficiency residual), rather than transplanting an M95C residual learned around a different baseline.
- Use predeclared feature blocks from the already-audited M95B/C/D lineage:
  - offensive/blocking/environment mean-efficiency block;
  - player-created efficiency block;
  - opponent run-efficiency/resistance block;
  - explosive/upside block only for tails unless it earns point value.
- Run **block ablations and compatible combinations**, with no broad feature soup or after-result weight search.
- Every retained block must show incremental value and non-degradation relative to its parent combination.
- Preserve C/M94C point workload; no carry adjustment in M96C.
- Continue separate 75+/100+ tail reporting.
- Keep V parked as vacancy diagnostic; test it directly against the best yard-tail baseline in a later precommitted conditional-expert step rather than declaring it promoted from M96B.
- Use strict temporal evaluation. 2024/2025 are development/validation evidence, not pristine confirmation.
- Any survivor remains research-only until prospective/untouched 2026 confirmation.

Stopping principle: M96C is not permission to restart an unlimited efficiency alphabet. If predeclared efficiency blocks cannot improve the M94C-anchored residual without cross-regime damage, retain the conservative point layer and move to prospective confirmation / conditional tail routing rather than repeatedly tuning history.
