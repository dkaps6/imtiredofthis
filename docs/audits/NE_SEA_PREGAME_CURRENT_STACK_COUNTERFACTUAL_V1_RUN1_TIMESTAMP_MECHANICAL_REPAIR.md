# NE–SEA Pregame Current-Stack Counterfactual V1 — Run1 Timestamp Mechanical Repair

## Authority and status
This document records and freezes the first execution failure of the diagnostic plan before any repair is applied. It is an evidence-preserving mechanical repair record only; it does not alter football/model science, production timing semantics, sportsbook ordering, or any result gate.

Original frozen plan commit: `25ac9828670194d8b47b4d64846ce52f584c8783`.
Original implementation head: `229fb89af6a90c291c09a8b0b2ee46d48d69bafb`.

## Preserved Run1 failure
- Run: `34508864138`
- Job: `102977690436`
- Evidence artifact: `10164979620`
- Artifact digest: `sha256:82639e7464e845ff4450188060d4b4ef0bf1dce6092257298bc389aa1645bd54`
- Scientific/projection result: NONE
- Pricing result: NONE
- Disposition: `NE_SEA_COUNTERFACTUAL_RUN1_MECHANICAL_KICKOFF_TIMESTAMP_ERROR_NO_DECISION`

## Exact failure
The original plan/workflow used diagnostic `asof_utc=2026-09-09T23:00:00Z` under the mistaken assumption that NE–SEA kickoff was `2026-09-10T00:20:00Z`.

The repository's authoritative 2026 Week-1 schedule instead resolves NE–SEA as:
- game_id `2026_1_NE_SEA`
- kickoff_utc `2026-09-09T20:20:00+00:00`

Therefore the production availability timing validator correctly computed:
- `asof_utc = 2026-09-09T23:00:00+00:00`
- `minutes_to_kickoff = -160.0`
- `certification_state = KICKED_OFF_LOCKED`
- `production_eligible = false`

The run then failed the diagnostic assertion requiring NE–SEA to remain pregame eligible. All football-stack rebuild, source-artifact staging, current pricing, and scoring steps were skipped. No scientific/model conclusion exists from Run1.

## Frozen minimum repair
Use the repository-authoritative kickoff timestamp and set diagnostic clock to exactly 80 minutes before kickoff:

`ASOF_UTC = 2026-09-09T19:00:00Z`

The only authorized workflow changes are:
1. replace the incorrect `23:00:00Z` diagnostic clock with `19:00:00Z` everywhere in the diagnostic workflow/result metadata;
2. allow this exact repair-record file in the diagnostic branch protected-boundary check.

No production file, football feature, model parameter, scientific gate, player-role rule, sportsbook line, or target-game outcome is authorized to change.

## Expected repaired behavior
At `19:00Z`, the repo-authoritative `20:20Z` kickoff is 80 minutes away. Under the already-frozen T-75 availability contract this should be pre-kickoff and outside the mandatory official-inactives window (`NOT_YET_REQUIRED`). The diagnostic must still independently assert the requested reconstructed personnel state before any model calculation:
- Rhamondre Stevenson active/current RB1;
- TreVeyon Henderson definitive unavailable and opportunity-ineligible;
- Sam Darnold active/current QB1 pregame;
- A.J. Brown active/eligible pregame;
- NE and SEA included in the diagnostic eligible-team universe.

If those assertions do not hold, stop with no scientific/projection decision. This is a controlled current-stack counterfactual, not a claim that the final production stack actually ran at that historical timestamp.
