# RB Final -> WR Research Handoff — 2026-09-03

## Authoritative RB checkpoint

The RB retrospective research loop is now frozen at the final qualification package.

- Current branch: `research-rb-final-qualification`
- Final qualification run: `33719205615`
- Final qualification job: `100534683157`
- Tested SHA: `0584a87e24e011cbd791be89b9fd4caa967349fa`
- Artifact: `9879617371` (`rb-final-qualification`)
- Results document: `docs/migrations/RB_FINAL_QUALIFICATION_RESULTS.md`
- STACK6 stop document: `docs/migrations/RB_STACK6_FINAL_STOPPING_EVIDENCE.md`
- Frozen protocol: `docs/migrations/RB_FINAL_QUALIFICATION_PLAN.md`

## RB model status

P3 / STACK3 remains the **RB research champion / shadow candidate**:

- W1: full-stack carry/yard projection.
- W2–18: STACK2 enriched within-RB opportunity/allocation, with full-stack implied rushing efficiency for yards.
- 2025 all-RB carries: MAE `3.357494`, RMSE `4.500432`, corr `0.753643`.
- 2025 all-RB rushing yards: MAE `19.949524`, RMSE `28.866519`, corr `0.631266`.
- Versus M94C, P3 improves all-RB yard MAE by `1.081626` and carry MAE by `0.053509`.
- P3 improves yard and carry MAE for RB1, RB2, and depth-3+ secondary backs.

Formal football status:

**`FOOTBALL_RESEARCH_CHAMPION_NOT_PRODUCTION_QUALIFIED`**

Reason: the precommitted actual-outcome tail guard failed. P3 materially underprojects ceiling games and is worse than M94C on both actual >=20-carry and actual >=100-rushing-yard slices. No waiver is authorized.

The available pregame M95F/M95I state slices still favor P3 over M94C, so there is no justified simple pregame route back to M94C. M95T's stop on retrospective carry-tail retuning remains binding.

## Market status

Exact 2025 downstream market universe: `899` rows.

- P3 rushing-yard MAE: `24.315798`
- Vegas consensus MAE: `23.701891`
- gap: Vegas better by `0.613907` yards
- P3 closer: `433 / 899 = 48.1646%`
- Vegas closer: `466 / 899 = 51.8354%`
- P3-vs-line directional sign accuracy: `53.3927%` (retrospective diagnostic only)

Formal market status:

**`AGGREGATE_VEGAS_NOT_CLEARED`**

Do not derive a betting threshold from exposed disagreement bins. 2026 contemporaneous market comparisons should be treated as forward/shadow evidence.

## RB permanent research rules from this checkpoint

1. Do not reopen STACK6 team-rush context slicing without a genuinely new timestamp-safe pregame information source.
2. Do not retune STACK6Q on 2025.
3. Do not start a new retrospective carry-tail candidate family after M95T.
4. Preserve P3 as the shadow/research point parent.
5. Preserve M95F workload-tail and M95I vacancy/tail signals as diagnostics/distribution-state inputs, not ad-hoc point routers.
6. The unresolved RB point-model issue is specifically **pregame ceiling calibration / high-end outcome compression**.
7. Gather true forward 2026 evidence before reopening that issue.
8. Sportsbook data remains downstream only.

## Primary research transition

Primary retrospective research now moves to **WR** while RB runs in shadow/forward evaluation.

The WR program should not begin by blindly fitting another large feature model. Start with the same successful forensic pattern used for RB:

1. Freeze the current production/full-stack WR projection path and exact historical universe.
2. Build an oracle decomposition of receiving-yard error into:
   - team pass attempts/dropbacks,
   - player target opportunity/share,
   - catch/reception conversion,
   - yards per reception / explosive gain efficiency.
3. Separate WR1 / WR2 / slot / secondary populations using only timestamp-safe pregame role/depth information.
4. Establish whether large WR misses are opportunity, catch-conversion, ordinary efficiency, or explosive-play driven before creating new models.
5. Explicitly test the parked QB/receiver hypothesis only after the receiver-side explosive mechanism is measured: elite explosive WR/TE matchups may explain some huge QB passing-yard overs; broadly poor receiver matchups may suppress passing volume/yardage.
6. Reuse existing QB, PBP, depth, target-share, coverage, pressure, explosive and opponent-defense source work where temporal contracts remain valid rather than rediscovering it.
7. Keep Vegas downstream and use exact-market benchmarks only after football projections are frozen.

TE follows the WR opportunity architecture where valid, with TE-specific route/blocking/red-zone work separated rather than assumed identical.
