# Current Player Availability — Full Slate Integration V1 Frozen Plan

Status: `FROZEN BEFORE FULL_SLATE PRODUCTION-CANDIDATE WIRING`

## Parent authority

- Protected production-code authority: `bb76ba9eabb08e2f0875a9af49301c3877f4141f`
- Confirmed roster/late-week gap result: `810c344a437d411707185317033acc7004f1c7db`
- Frozen operational fix plan: `b2206e7ad693148623447bcf9a3ad6b594033500`
- Canonical T-75 timing freeze: `905f1bbe55d51676587d941295d357a1d2c31e9b`
- Timing concurrency resolution: `910e707159bfd98f6d62f0c493d0e0fb30ab1881`
- Locked current-player availability core: `596f084750a8c8e3c36ca09d737dc2edb808dc09`
- Clean semantic availability fixture run: `34436970099`, job `102743973821`
- Live source smoke: `34437032282`, job `102744156238`, artifact `10136545256`, digest `sha256:decb703afe4769befba790d8b1adceb0accb5a49eec4f5fd8f5d2adc6c7eb75a`
- First valid T-75 timing fixture run: `34437715931`, job `102746163583`, head `092fee088f3402d6313bc02b2f8cc05d1f3f54f9`

This plan changes current-slate eligibility/role inputs only. It does not authorize any model coefficient, trained artifact, historical backtest, or sportsbook-derived role change.

## Frozen integration principle

**Availability is resolved before opportunity.**

Definitively unavailable players must be removed from the eligible current-player football universe before PlayerForm/current-role construction and before any promoted target/rush/pass opportunity allocator runs. Existing qualified component allocators then operate on the reconciled eligible universe.

There is no generic post-projection `OUT × 0.50` repair for definitively unavailable players.

## Frozen Full Slate build order

For the requested `(season, week, as-of)`:

1. build schedule with authoritative kickoff timestamps;
2. build timestamped Ourlads depth/status sidecar;
3. build weekly injury source/status;
4. acquire official NFL inactive sections;
5. run T-75 per-game timing certification;
6. build `current_player_availability.csv` using only source evidence valid at the requested as-of;
7. materialize a production-compatible **reconciled active-role artifact** containing only `definitive_unavailable == 0` rows and using `role_after_availability` where the role family is ordinal;
8. run current PlayerForm/model-context builders from that reconciled active-role artifact;
9. run promoted position components and explicit entitlement stack;
10. withhold any game whose timing certification is not production-eligible;
11. sportsbook matching/pricing remains downstream and cannot add a withheld/unavailable player back to the football universe.

## Frozen reconciled-role artifact

Create a versioned active-role artifact, e.g. `data/roles_ourlads_active_v1.csv`, with the production-compatible columns required by downstream role consumers plus:
- availability_authority
- final_availability_state
- source_asof_utc / availability_generated_at_utc
- original raw role/depth identifiers retained where useful for audit

Rules:
- include only players with `definitive_unavailable == 0`;
- never delete the unavailable row from `current_player_availability.csv`; only exclude it from the active-role artifact;
- QB/RB/FB/TE ordinal roles use `role_after_availability` from the locked resolver;
- WR alignment (`WR-L`, `WR-R`, `SLOT`, etc.) remains the surviving player's existing alignment role; do not invent a new WR alignment after removal;
- no duplicate `(team, player_clean_key)` active identity;
- every active role must trace back to exactly one availability row.

During the candidate workflow, downstream commands may consume this active artifact through an explicit CLI/input override or a staged candidate-only replacement path. Do not silently change the legacy raw Ourlads source artifact contract before verification.

## Frozen position/component seams

### QB

- Definitively unavailable QB has zero current eligibility and cannot be the active starter.
- The highest remaining eligible QB depth rank becomes `QB1` before current QB opportunity/mean construction.
- M89/M90 model artifacts, coefficients and C2 selector are unchanged.
- There must be exactly one active `QB1` per scheduled team that has at least one eligible QB.
- No sportsbook QB listing can define the starter.

### RB

- Definitively unavailable RB/FB is excluded before `scripts/run_rb_week1_no_odds.py` constructs the current RB universe.
- Remaining eligible backs are deterministically re-ranked; promoted P3 and R26 consume the reconciled current RB universe.
- P3 model/coefficient mechanics and R26 qualified opportunity/reception mechanics are unchanged.
- R22 remains tail authority and may only receive positive distribution mass for eligible players with positive upstream receiving mean/opportunity.
- An unavailable prior RB1 must have zero carries/targets/receptions/rush yards/receiving yards/distribution output; the eligible successor may inherit lead-role status through the existing role/opportunity construction, not through a new arbitrary percentage transfer.

### WR / TE receiving opportunity

- Remove definitively unavailable players before the full-roster football universe and explicit target entitlement are built.
- Existing production target-entitlement stack remains the conservation authority:
  - M38 establishes finite team target entitlement on the eligible universe;
  - TE-R5P redistributes only inside the conserved TE room;
  - WR-R15 redistributes only inside the conserved WR2+ room while preserving the eligible M38 WR1 anchor;
  - non-position room mass invariants remain enforced.
- Do not use legacy `simulation_rules` 50% retention for a definitive unavailable player.
- QUESTIONABLE/DOUBTFUL remain eligible and may continue through existing uncertainty mechanics.

### Full simulation / pricing

- Definitively unavailable players must have no simulation arrays / positive means in current football outputs.
- A downstream sportsbook offer for an unavailable or withheld-game player must remain unmatched/withheld; it may not resurrect the player.
- Timing certification is applied by game. Later games outside the T-75 requirement window remain eligible as `NOT_YET_REQUIRED`; only affected imminent games fail closed.

## Frozen implementation scope

Candidate integration may add:
- reconciled active-role artifact builder/adapter;
- explicit role-input override plumbing where needed;
- availability/timing validation and withholding logic;
- current-slate invariant audits;
- Full Slate workflow ordering changes required to build availability before current opportunity.

Candidate integration may NOT change:
- trained model JSON/NPZ/pickle artifacts;
- M89/M90/C2 scientific parameters;
- M38, WR-R15, TE-R5P scientific parameters;
- P3 scientific parameters;
- R26 scientific parameters;
- R22 scientific parameters/tail pools;
- historical training/backtest data or result records;
- sportsbook inputs into football eligibility/role/opportunity.

## Frozen candidate validation gates

All are required before promotion:

1. locked availability core blobs match `CURRENT_PLAYER_AVAILABILITY_V1_IMPLEMENTATION_LOCK.md`.
2. protected trained model/artifact hashes unchanged.
3. schedule contains exact requested slate and parseable kickoff timestamps.
4. timestamped Ourlads depth/status covers all scheduled teams.
5. every active-role row has availability provenance.
6. no definitive unavailable player is present in active-role artifact.
7. every unavailable row remains present in audit availability artifact.
8. no duplicate active team/player identity.
9. QB active roles have at most one `QB1` per team; every team with eligible QB depth has exactly one.
10. RB/TE/QB ordinal roles are gap-free after removals.
11. QUESTIONABLE/DOUBTFUL players are not removed solely for that designation.
12. official-inactive absence is used only from complete validated sections.
13. T-75 `REQUIRED_MISSING_FAIL_CLOSED` games are withheld; `NOT_YET_REQUIRED` games are not withheld solely for missing official lists.
14. official snapshot timestamps used for certification are strictly pre-kickoff.
15. PlayerForm/current contexts contain no definitive unavailable player as active opportunity row.
16. RB P3 output contains no definitive unavailable player with positive mean/opportunity.
17. R26 receiving opportunity/receptions contain no definitive unavailable player with positive target/reception value.
18. R22 receiving-yard distribution contains no definitive unavailable player with positive distribution/mean.
19. QB production has no definitive unavailable starter/positive pass opportunity.
20. WR/TE promoted target entitlement contains no definitive unavailable player.
21. M38/TE-R5P/WR-R15 entitlement conservation invariants still pass on the eligible universe.
22. team target entitlement is conserved according to the existing qualified explicit-entitlement contract; no new injury percentage allocation introduced.
23. promoted model version identifiers remain exactly the protected production versions.
24. sportsbook inputs used to define availability/role/opportunity = 0.
25. downstream odds matching cannot add players not in the eligible football universe.
26. candidate no-odds Full Slate completes successfully.
27. static production-readiness audit includes availability source/role/timing wiring and passes.
28. fixture-injected RB1 OUT candidate run demonstrates old RB1 zero and eligible successor lead-role/current-opportunity construction.
29. fixture-injected QB1 inactive candidate run demonstrates old QB1 zero and exactly one successor QB1.
30. fixture-injected WR/TE unavailable candidate run demonstrates zero removed-player entitlement and existing position-room conservation invariants.
31. current real-source candidate build produces explicit counts of unavailable/uncertain/unknown/withheld games without silently coercing unknown to healthy.
32. no historical research result/artifact changed.
33. no unintended diff to scientific model code outside explicit current-availability input seams.
34. if live pricing is executed, it succeeds only on production-eligible games and sports books remain downstream.
35. a result record captures branch/head/run/job/artifact/digest, all gate values, withheld games, source timestamps and exact production disposition.

## Required result dispositions

- `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`
- `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_FAIL_NO_PROMOTION`
- `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_MECHANICAL_FAILURE_NO_DECISION`

No partial pass or post-result exception may be promoted.

## Promotion boundary

Implement on a new dedicated candidate integration branch based on protected production/current main lineage plus the locked availability core. Lock exact implementation blobs before the first candidate Full Slate run. The first valid 35-gate result is immutable. Production promotion is permitted only on the PASS disposition followed by an exact post-promotion Full Slate verification.
