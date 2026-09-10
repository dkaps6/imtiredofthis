# NE–SEA Pregame Current-Stack Counterfactual V1 — Frozen Diagnostic Plan

## Purpose
Reconstruct what the CURRENT PROMOTED production football stack would have projected for the already-played 2026 Week 1 NE–SEA game using only information that was available pre-kickoff, then compare that diagnostic output with the preserved Sep. 7 sportsbook snapshot and the actual game. This is diagnostic evidence only and cannot modify/promote production science.

## Frozen authorities
- Parent/main head at freeze: `3e01e2bf3d6307f33562c18bea7e25f686757608`.
- Protected scientific/model authority remains `bb76ba9eabb08e2f0875a9af49301c3877f4141f`.
- Current promoted stack: QB M89/M90 + C2; WR M38 + WR-R15; TE TE-R5P; RB P3 + R26 + R22; availability/current roles; sportsbook downstream only.
- Preserved paid sportsbook source Run `34152868136`, Artifact `10030344451`, digest `sha256:c19bd303a0eb7ca58a3484117e28b5e5144459b74459bd1032970873cae6d035`, fetched `2026-09-07T18:45:55.831184+00:00`.
- Current postgame operational Full Slate input/evidence Run `34506301083`, Artifact `10164043236`, digest `sha256:c454076a2cd9121e4e252d569e36c4337331b92e8d8cd8f7fb2a8e50ff518de5`.

## Frozen pregame clock
Use diagnostic `asof_utc = 2026-09-09T23:00:00Z`, 80 minutes before the scheduled NE–SEA kickoff at 2026-09-10T00:20:00Z. This keeps the game pre-kickoff and outside the frozen T-75 mandatory-official-inactives window.

## Pregame personnel interpretation
The reconstruction must reflect the documented pregame state:
- NE Rhamondre Stevenson is current RB1.
- NE TreVeyon Henderson is unavailable and must receive no opportunity.
- NE Corey Kiner is the next active halfback behind Stevenson.
- SEA Sam Darnold is QB1 pregame; his later injury is NOT knowable and must not enter the reconstruction.
- NE A.J. Brown is active pregame; his later injury is NOT knowable and must not enter the reconstruction.

The run must prove these states from the current availability/current-role artifacts produced under the frozen `asof_utc`. If any required state is not reproduced, fail closed.

## Strict information boundary
- No target-game outcome, box score, postgame player stats, or target-game PBP may enter feature generation.
- Do NOT run current-season PBP enrichment.
- Week-1 PlayerForm/TeamForm must remain strict-prior under the existing certified wrappers.
- No new sportsbook fetch. Reuse the exact Sep. 7 pregame sportsbook offers only after football/current-role generation.
- Sportsbook lines cannot define player role, availability, carries, targets, receptions, or passing opportunity.

## Diagnostic execution
1. Build current Ourlads roles, Week-1 schedule, weather, and weekly injury inputs under current code.
2. Run current availability/current-role production prep with the frozen pregame `asof_utc`.
3. Assert NE and SEA are production-eligible in the reconstructed game universe and assert the personnel states above.
4. Rebuild strict-prior TeamForm, M89/M90 QB context, coverage, PlayerForm, model-context bridge, Bayesian/ML/State/rules/ensemble diagnostics, RB P3, and QB C2 using the exact current promoted code. Do not build target-game PBP enrichment.
5. Download the immutable Sep. 7 paid artifact and stage only NE–SEA sportsbook offers into the current metrics/pricing path. No Odds API call.
6. Run the current production metrics/pricing stack so the current promoted model is evaluated at the same historical sportsbook thresholds.
7. Emit a diagnostic CSV containing current-stack NE–SEA projections, preserved lines/odds, and lineage. This output is research/audit-only and must never overwrite production authority.

## Required integrity checks
- Parent code ref is exact frozen branch ancestry.
- zero sportsbook inputs to football/current-role generation.
- no 2026 Week-1 game logs in published PlayerForm strict-prior history.
- no target-game PBP enrichment.
- NE/SEA are eligible under diagnostic pregame clock.
- Stevenson active RB1; Henderson definitive unavailable / opportunity-ineligible; Darnold active QB1; A.J. Brown active pregame.
- current pricing path uses promoted production adapters/owners exactly as implemented; no scientific parameter changes.
- source sportsbook artifact/run/digest recorded exactly.

## Interpretation rules
- This is a counterfactual audit because the final availability-first production stack was promoted after the game; it is NOT an actual timestamped production forecast from Sep. 9.
- Darnold/A.J. Brown in-game injury outcomes remain injury-distorted and should be separately flagged in accuracy analysis.
- No result from this diagnostic can authorize model promotion, retuning, or a post-hoc router.
- After this audit, return immediately to the QB opportunity/efficiency anti-reinvention research lane in `CURRENT_NFL_RESEARCH_HANDOFF.md`.
