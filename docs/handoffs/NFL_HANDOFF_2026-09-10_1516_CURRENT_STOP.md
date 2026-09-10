# NFL HANDOFF — 2026-09-10 15:16 ET — CURRENT STOP

**Repository:** `dkaps6/imtiredofthis`  
**Read first with:** root `CURRENT_NFL_RESEARCH_HANDOFF.md`  
**GitHub is canonical; chat memory is secondary.**

## Why this handoff exists

The prior chat hit its context limit while we were doing a diagnostic reconstruction of the already-played New England–Seattle Week 1 game using the **current promoted production stack** plus preserved pre-kickoff sportsbook lines. This file records the exact stop point so the next chat can resume without re-auditing or guessing.

## Production status — DO NOT CONFUSE WITH THE DIAGNOSTIC

Production availability/current-role plumbing is already **promoted and complete** on main. The automatic master workbook publisher is also promoted and complete. The protected scientific/model stack remains:

- QB mean: M89/M90
- QB distribution: mean-neutral C2
- WR: M38 WR1 + WR-R15 WR2+
- TE: TE-R5P
- RB rushing: P3
- RB receptions: R26
- RB receiving-yard mean: existing production YPT/mean path
- RB receiving-yard tails: R22 using frozen R19 assets, exactly mean-preserving
- current availability/current roles: promoted availability-first production plumbing
- sportsbook: downstream only

The root handoff currently records operational production authority `3079d8ab0512c5a1304662609e3e880d6846292f` and protected scientific/model authority `bb76ba9eabb08e2f0875a9af49301c3877f4141f`.

### If the user runs Full Slate right now

**Yes:** it uses the latest promoted production stack that has actually earned integration, including the availability/current-role production work and automatic master betting workbook. Failed/research-only candidates remain correctly excluded. Because NE–SEA has already kicked off, the current production Full Slate should treat that game as `KICKED_OFF_LOCKED` and omit it from current betting/output eligibility. That is expected behavior, not an odds/data failure.

## NE–SEA controlled pregame reconstruction — diagnostic only

Goal: answer what the **current promoted stack** would have projected for NE–SEA under a defensible pregame state, without backfilling the actual game result into the model and without refetching sportsbook odds.

Diagnostic branch:
- `audit-ne-sea-pregame-current-stack-v1`

Frozen plan:
- `docs/audits/NE_SEA_PREGAME_CURRENT_STACK_COUNTERFACTUAL_V1_FROZEN_PLAN.md`

Protected production parent used for the diagnostic:
- `3e01e2bf3d6307f33562c18bea7e25f686757608`

Preserved historical sportsbook source:
- source Run `34152868136`
- source Artifact `10030344451`
- source digest `sha256:c19bd303a0eb7ca58a3484117e28b5e5144459b74459bd1032970873cae6d035`
- no odds refetch in the diagnostic

The diagnostic intentionally does **not** run target-game PBP enrichment and explicitly verifies strict-prior PlayerForm history so the already-played NE–SEA outcome cannot leak into the counterfactual.

### Diagnostic Run 1 — preserved mechanical/no-decision failure

- Run `34508864138`
- Job `102977690436`
- Head `229fb89af6a90c291c09a8b0b2ee46d48d69bafb`
- Artifact `10164979620`
- artifact digest `sha256:82639e7464e845ff4450188060d4b4ef0bf1dce6092257298bc389aa1645bd54`

Failure cause: the diagnostic used `ASOF_UTC=2026-09-09T23:00:00Z`, but the authoritative schedule had NE–SEA kickoff at `2026-09-09T20:20:00Z`. The availability timing system correctly computed `minutes_to_kickoff=-160` and returned `KICKED_OFF_LOCKED`.

**Classification:** mechanical timestamp error / no projection decision. Production timing logic behaved correctly. No model science or NE–SEA counterfactual projection was produced.

Frozen repair record:
- `docs/audits/NE_SEA_PREGAME_CURRENT_STACK_COUNTERFACTUAL_V1_RUN1_TIMESTAMP_MECHANICAL_REPAIR.md`
- repair-doc commit `15d2e54c615a28f60d210eea7907c9068b4685ef`

A documentation-only push caused a duplicate old-clock failure Run `34509347795`; preserve it as the same stale-clock mechanical failure, not science.

### Corrected authoritative rerun — current stop

- Run `34509425408`
- Job `102979570511`
- Head `3beb00ee518b8076873a5f54dffed51ee2009e5e`
- Artifact `10165290158`
- artifact digest `sha256:41c0e81b489f67d4b57bb98a249865e019fa644c04b4cf891b51734203e068c5`
- corrected pregame clock: `ASOF_UTC=2026-09-09T19:00:00Z`, exactly 80 minutes before authoritative kickoff

The corrected run passed the frozen pregame personnel/state assertions exactly:
- NE–SEA production eligible at the fixed pregame timestamp
- `Rhamondre Stevenson` = `RB1`, eligible for opportunity
- `TreVeyon Henderson` = definitive unavailable, excluded from opportunity
- `Sam Darnold` = `QB1`, eligible
- `A.J. Brown` = eligible
- sportsbook inputs used for availability = `0`

It then passed the strict-prior current production football rebuild:
- PlayerForm strict-prior Week 1 history check passed
- 32-team PlayerForm coverage passed with 467 active rows
- current-role RB P3 no-odds build passed
- QB distribution state context rebuilt from 2019–2025 only
- availability eligible-team seams passed
- no target-game PBP was intentionally run

Important NE–SEA current P3 values produced **before the later diagnostic plumbing failure**:
- Rhamondre Stevenson P3 rushing mean: `30.790770` yards
- Jadarian Price P3 rushing mean: `18.159696` yards

These are valid outputs of the strict-prior football build, but the complete cross-market NE–SEA counterfactual pricing board was **not** completed, so do not treat those two numbers as a completed diagnostic disposition.

The preserved Sept. 7 sportsbook artifact was then staged successfully **after** football generation:
- 162 side rows
- 22 players
- teams NE and SEA only
- markets: anytime TD, pass TDs, pass yards, reception yards, receptions, rush+rec yards, rush yards
- odds refetched: false

### Exact current blocker

The corrected run failed in the pricing step **before `run_pricing_with_full_roster_universe_v3.py` executed**.

`run_metrics_context.py` successfully produced 103 metrics rows. Then `scripts/metrics_ready.py` failed with:

`RuntimeError: Required artifact missing or empty: data/opponent_map_from_props.csv`

This happened because the special diagnostic manually staged preserved `outputs/props_raw.csv` rows but did not run the ordinary opponent-map-from-props builder expected by `metrics_ready.py`.

**Current classification:** diagnostic plumbing/staging failure, not football/model science. Do not change production timing, availability, P3, R26, R22, M89/M90, C2, WR-R15 or TE-R5P because of this failure.

### Exact next move for NE–SEA

1. Inspect the normal Full Slate ordering for the exact command that generates `data/opponent_map_from_props.csv` from the already-staged historical NE–SEA offers.
2. Freeze the smallest diagnostic-only Run-3 repair that adds only that missing normal staging step after the preserved sportsbook rows are installed.
3. Hash/lineage lock the repair before executing.
4. Rerun the diagnostic from the same protected parent and same `19:00Z` pregame clock.
5. Preserve Run `34509425408` exactly as the current mechanical/no-complete-decision evidence.
6. Only after a complete diagnostic succeeds should the NE–SEA current-stack projections be compared with the actual game and the older Sept. 7/8 sheet.
7. Do **not** use actual NE–SEA target-game PBP or results as pregame features.

## Next research phase after this short diagnostic

Per the current root handoff, the next active model-development lane is **QB opportunity / efficiency**, not another RB receiving-mean retry.

Before any new QB candidate is frozen:
- audit all prior QB migrations, result records and current production code;
- inventory what was already tested for attempts/dropbacks/pass rate, YPA/efficiency, sacks/pressure, scrambles, explosive passing, receiver interactions, game environment and tails;
- decompose the remaining production error with strict-prior football information;
- explicitly avoid reinventing already-promoted or failed QB work;
- source/schema-audit any genuinely new information before candidate design;
- freeze question, mechanism, data boundary, walk-forward protocol, baseline/candidate and gates before first scientific result;
- sportsbook remains downstream benchmark only.

The prior RB receiving-yard mean lane is closed/no-integration after R27D. Do not restart R27B/R27C/R27D generically. R26 receptions and R22 tails remain protected. R26Q/R26S Week 1 prospective grading remains sealed and should be run only through the exact frozen evaluator when the full authoritative Week 1 outcome scope required by that evaluator is available.

## User operating principles to preserve

- Build an elite football model that predicts real football outcomes; Vegas is opponent/benchmark, not teacher.
- One authoritative production projection per player/market.
- GitHub canonical, chats secondary.
- Preserve first valid scientific result and every failed/mechanical run.
- Never silently mutate production.
- Separate plumbing/integrity failure from scientific failure.
- Freeze hypotheses, gates and lineage before scientific evaluation.
- Current/live status questions require checking GitHub first.
- Tell the user when an avenue is weak or repeats prior work.

## Resume instruction

The next chat should read root `CURRENT_NFL_RESEARCH_HANDOFF.md` **and this file** first, verify live GitHub `main` and Run `34509425408`, then resume at the missing `opponent_map_from_props.csv` diagnostic staging seam. Finish that narrowly, record the result, and move immediately to the QB opportunity/efficiency anti-reinvention audit.