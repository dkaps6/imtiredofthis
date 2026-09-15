# NFL HANDOFF — 2026-09-14 — WR RECEIVING YARDS CURRENT

Repo: `dkaps6/imtiredofthis`

**ACTIVE PRIORITY:** continue WR receiving-yards research. Do not move to RB until the user changes priority.

**COLLABORATION:** GPT-5.6 and Claude must continue through GitHub Issue #535, independently audit/falsify each other's work, and preserve anti-retest discipline.

Read `AGENTS.md`, root `CURRENT_NFL_RESEARCH_HANDOFF.md`, then this file. GitHub is canonical over chat memory.

## Current authority / repo state
Preserve production science while researching:
- QB mean: M89/M90 / `QB_PASS_SYNTHESIS_V1`
- QB distribution: `C2_QB_MEAN_NEUTRAL_DISTRIBUTION_V1`
- WR: M38 WR1 hierarchy + `WR_R15_PRODUCTION_MODEL_V1` for WR2+
- TE: `TE_R5P_PRODUCTION_MODEL_V1`
- RB Week-1 authorities remain P3/R26/R22.

Before these handoff commits, `main` was `c7bc9bc3bc335d531a981f533d7b4206389c6f53` (PR #599 receiving ensemble weights).

PR #600 remains OPEN/mergeable, head `1167f9fdadde452deb84d3891097865da2f163d5`. Canonical authority-exact run `34843204550`, artifact `10346639168`, digest `sha256:e0041386c7f6600c6e8a8781d0d91d7603c4d1fde75d9aa3d0f0055aaa746225`.
Football parity before market comparison:
- QB 884: MAE `57.638995 -> 55.060118`
- WR 4193 OOS: rec-yard MAE `22.856618 -> 22.526358`; target MAE `2.128499 -> 2.043886`
- TE 3214: rec-yard MAE `16.267971 -> 15.850608`; target MAE `1.682530 -> 1.618358`
- WR/TE #549 replay integrity PASS.

## WR-R15 is not a failure
Authority run `34238301577`, artifact `10061328722`, digest `sha256:8df31b5e136621d959272daf0422dfc665593cd0da2eb0892b4aa69c1417f3ce`.
OOS n=4193 (2023=2076, 2024=2117; 2025 confirmation forbidden):
- target MAE `2.128499 -> 2.043886`
- rec-yard MAE `22.856618 -> 22.526358`
- p90 rec-yard AE `51.558659 -> 50.311730`
- secondary WR rec-yard `20.086576 -> 19.655474`
- WR1 rec-yard `31.407029 -> 31.388044`
All frozen gates passed, zero sportsbook input. WR-R15 preserves the M38 WR1 anchor and primarily improves WR2+ entitlement.

## Receptions are comparatively healthy; yards are the bottleneck
From #600 `CURRENT_PRODUCTION_ORDER`:
- WR receptions n=1504: 827-677, `54.9867%` directional, model MAE `1.675848` vs market line MAE `1.640957`.
- strict model WR1 receptions n=493: 265-228, `53.7525%` directional.

WR receiving yards:
- all WR n=1575: 769-806, `48.8254%` directional, model MAE `25.017501` vs line MAE `23.863810`.
- strict model WR1 n=494: 237-257, `47.9757%`, model MAE `29.177472` vs line MAE `27.825911`.

Interpretation: opportunity/receptions are not perfect, but they are materially healthier than receiving-yard translation. The active research problem is yardage efficiency/ceiling, not another generic target-share rebuild.

## Frozen WR1 yardage decomposition V1
Branch `research-wr1-yardage-decomposition-v1`, head `25d904f7d8abe9c2af6698c1dee5568f812eac51`.
Run `34858963515` SUCCESS; artifact `10353787250`; digest `sha256:3e6fb21956e0f2681a732379412b6eff0c3e17dd561d7358e8995e279fb42f5d`.

Identity correction: literal historical `role==WR1` can label multiple lane receivers on one team-game. Artifact audit: 2117 authority rows, 516 true model WR1s, 1073 literal WR1 rows, 403 team-games with 2+ literal WR1 labels, max 3. Future WR1 studies must use the actual M38/R15 hierarchy (`wr_rank==1`).

True-WR1 494-row result:
- authority OOS: 253-241 = `51.2146%`, MAE `29.123760`
- current production order: 237-257 = `47.9757%`, MAE `29.177472`
- 98 rows changed side; authority won 57 of those vs 41 current-production wins, net -16 directional wins.
This is diagnostic only; do not revert #599 weights based on a downstream line comparison.

Error decomposition on current WR1 rows:
- bias `-9.256 yd`, MAE `29.177`
- absolute-error shares: target `40.61%`, catch translation `27.34%`, YPR `32.04%`
- collapsed target-vs-YPT: `50.25% / 49.75%`
- on directional losses: target/catch/YPR `42.60% / 25.26% / 32.14%`; target-vs-YPT `51.50% / 48.50%`.

Tail/efficiency failure:
- actual 100+ WR1 games n=78: only `25.64%` correct direction, bias/MAE `-67.43/67.43 yd`
- actual YPT top quartile n=124: `31.45%` correct direction, bias `-39.17 yd`, MAE `42.70`
- WR1 UNDER losses n=170: average underprojection `-40.98 yd`; 34.12% became 100+ yard games.

Not WR1-only: an opportunity-matched WR2+ control (494 rows) was also weak. Therefore the yardage translation problem is broader than WR1, while WR1 has the stronger negative-bias/ceiling concern.

## Shared QB-C2 -> WR1 tail lane is CLOSED
Frozen plan `b2fbf9a844b37e780f48bcb68e57503f3c8122e9`; cohort clarification `b72617d372ec6272f5f1daae4ae524ef3ed11829`. Canonical cohort = exact 884 C2 team-games (444 in 2024, 440 in 2025).

Canonical 2025 holdout:
- Spearman `0.0887`
- residual gap `+3.12 yd` vs +5 required FAIL
- bootstrap `75.58%` vs 90% FAIL
- 100+ rate ratio `1.237x` vs 1.30 FAIL
- catastrophic 40+ miss ratio `1.013x` vs 1.20 FAIL
- opportunity quartiles `2/4` vs 3/4 FAIL
- 2024 directional coherence negative FAIL.
Broad C2 propagation did not solve the WR1 ceiling either; p90 error worsened and the original paired-player casebook underprojected every one of its 140 actual 100+ WR1 games in both B0 and C2.

Claude independent sensitivity:
- trace run `34865304105`, artifact `10356647408`, digest `sha256:17aca6a561543ad3240cacf4bd7639d277e77cd138196d456585ef131365ffd1`
- evaluation run `34873351292`, artifact `10358739434`, digest `sha256:c366c7d419b1d3eb266538776b1c5ac39d9e0357d9e244885a7899d2bee5a158`
Claude graded 849 rows after an actual-usage join and independently reached `NO_ACTIONABLE_WR_QB_SHARED_TAIL_SIGNAL`. Treat it as sensitivity; canonical authority remains the preregistered 884-row result. GPT independently audited Claude's artifacts. Substantive FAIL agrees.

Do not rescue this lane with upper95, right-skew or threshold tuning.

## Closed adjacent WR work — anti-retest rules
Do not rerun under new names:
- M72 explosive-weapon x defense signal
- M75 simple separation/cushion/aDOT/YACOE/secondary-quality lane
- M84 player-level WR-CB responsibility without honest historical assignments
- R7 persistent explosive/YAC/air-yard player traits
- R9-R11 NGS source lane
- R3 combined residual-persistence calibration
- C1 shared QB/receiver target-mass adjustment
- C3 broad joint QB/receiver combination
- ND3 dynamic/vacated entitlement.
Do not blindly retune M38/R15 target shares.

## What is still OPEN
WR research is NOT over. User explicitly wants receiving yards pursued before RB.
Active question:
> Can football-only WR yardage-per-opportunity / efficiency / ceiling accuracy be improved without destabilizing the healthier reception/opportunity architecture?

Before any new result run, GPT and Claude must each build an anti-retest + feature-availability view. Only genuinely new, leakage-safe mechanisms may advance. Potential problem space, not yet authorized candidates: conditional YPT/YPR translation; leakage-safe route/depth/air-yard role interactions materially different from M75/R7; interaction/regime structure rather than another simple persistent trait; QB delivery/pass-environment interactions materially different from failed C2-tail; WR-specific uncertainty only from replicated pregame difficulty; richer coverage/alignment only with honest historical responsibility data.

**Recommended immediate next step:** build a WR receiving-yard efficiency anti-retest + feature-availability matrix over exact authority identities. Map each candidate feature family to prior test/verdict/source availability, identify genuinely untested interactions, then freeze ONE narrow hypothesis, dev/holdout split, cohort, metrics and gates before results. Primary scientific target remains football receiving-yard accuracy/tail behavior; downstream market comparison is diagnostic only.

## Claude collaboration
Use Issue #535 as the control room. Next chat must tell Claude the user overrode the earlier RB pivot: **WR receiving yards active; RB parked.** Claude should independently propose/falsify the next mechanism, not merely agree with GPT. Require exact runs/artifacts/cohorts, not verbal agreement.

## RB Weeks 2-18 — PARKED checkpoint
Branch `research-rb-pd2-yard-difficulty-mc-width-v1`, head `3e3da9ec7216ec836b4d593a3b5ac32f524442f0`.
Run `34876877949` ended before candidate evaluation:
- helper tests PASS
- 2022 reconstruction matches to floating noise through W07, then W08 MC-mean mismatch `1.0232351709`
- 2023 W01-W02 match to floating noise, then W03 mismatch `2.7762633539`
- evaluate-frozen-candidate SKIPPED.
No scientific RB-width result has been exposed. Preserve this exact state for later; do not continue while WR is active.

## Production
Week-1 Full Slate repair is already green. Use `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`. Do not trigger paid Full Slate for WR research.

## Next-chat checklist
1. Verify current `main` and PR #600.
2. Read latest Issue #535 GPT/Claude posts.
3. Reconfirm to Claude: WR receiving yards active, RB parked.
4. Preserve M38/R15/#599; receptions comparatively healthy, yardage translation active problem.
5. Perform anti-retest/feature-availability audit before any new candidate.
6. Have Claude independently do the same and challenge the design.
7. Freeze one genuinely new leakage-safe WR-yardage hypothesis before results.
8. Keep football accuracy primary and preserve every failed lane.
