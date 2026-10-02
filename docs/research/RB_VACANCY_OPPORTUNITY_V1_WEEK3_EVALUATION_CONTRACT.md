# RB Vacancy Opportunity V1 — Week 3 Prospective Evaluation Contract

Date frozen: 2026-09-25 local / 2026-09-26 UTC  
Branch: `research-rb-vacancy-opportunity-v1`  
Status: **FROZEN BEFORE TARGET-GAME OUTCOMES**

## Purpose

This contract governs the first legitimate prospective RB Vacancy Opportunity V1 cohort.

It does **not** authorize production promotion from Week 3. The parent plan explicitly forbids promotion from two live weeks alone, and this first qualifying cohort contains only two vacancy-event teams. The Week-3 result will be recorded once as prospective evidence and the same frozen V1 formula will continue to accumulate future qualifying events.

## Immutable pregame authorities

### Source Full Slate

- run: `36204768034`
- source main SHA: `f7d2011b73950488ea209124ba895b92c401b2b1`
- artifact: `10892728623`
- digest: `sha256:f52b36fb7a9c929fadca473bd8303823fb9f63bd42c884341cd6c3b52c26ed67`
- live sportsbook acquisition: **disabled**
- target: 2026 Week 3

### Frozen V1 vacancy state

- run: `36205201005`
- research head: `c34ba8b761f042aca66bcd028435b2ca9b270900`
- artifact: `10893677385`
- digest: `sha256:69b55421be779cb474eea0735bfe98490d8c7d5a8e0847e96f1f65d1c5e9e732`
- qualifying teams: **DEN, PIT**
- unavailable RBs: **Jonah Coleman (DEN), Rico Dowdle (PIT)**
- direct transfer rows: **4**
- exclusions: **0**
- target-game outcomes attached: **0**
- sportsbook inputs used: **0**

Frozen transfer shares:

| Team | Successor | Strict-prior weight input | Successor weight | Transfer rush share |
|---|---|---:|---:|---:|
| DEN | JK Dobbins | 0.34 offense pct | 0.40 | 0.142857142857 |
| DEN | RJ Harvey | 0.51 offense pct | 0.60 | 0.214285714286 |
| PIT | Riley Nowakowski | 0.03 offense pct | 0.040540540541 | 0.012338425382 |
| PIT | Jaylen Warren | 0.71 offense pct | 0.959459459459 | 0.292009400705 |

The unavailable-player vacated shares remain:
- DEN / Jonah Coleman: **0.357142857143**
- PIT / Rico Dowdle: **0.304347826087**

## Exact production baseline/candidate seam

For Week 3 RB rushing, production uses the generic calibrated **MC + ML + State** ensemble. The promoted RB P3 rushing synthesis is Week-1-only.

The pregame projection lock therefore must:

1. reconstruct the exact sportsbook-independent full-roster football simulation from the frozen Full Slate artifact and exact source SHA;
2. use production Monte Carlo settings: **25,000 iterations, seed 42**;
3. freeze current production ML and State components from the same pregame artifact;
4. freeze the promoted market-specific ensemble weights from the same pregame artifact;
5. create the V1 candidate by changing **only** `rules_rush_share` for the four direct transfer recipients by the already-frozen transfer amounts;
6. leave `rules_ypc`, every efficiency field, game script, ML projection, State projection, ensemble weights and all other football inputs unchanged;
7. rerun the same canonical finite-volume simulation;
8. apply the existing production ensemble without new fitting.

Because the canonical simulator caps/normalizes finite team rushing allocation, the grading cohort is **all production-eligible RB/FBs on DEN and PIT**, not only the four direct recipients. Any teammate movement caused by finite-volume normalization is part of the candidate's real production effect and must not be hidden.

The unavailable backs themselves are already absent from the production-eligible simulation universe under the canonical availability resolver. V1 does not add a second zeroing rule.

## Outcome attachment contract

Outcomes may be attached only after the relevant target games are final.

Canonical outcome source:
- repository `player_game_logs.csv` generated after the games under the normal completed-game pipeline.

Required actuals:
- `rushes`
- `rush_yards`

Identity joins must be exact canonical team/player identity. No fuzzy matching.

For a locked active RB/FB with no player-game row after the team game is certified final, actual rush attempts/yards may be set to zero only after explicitly verifying that the team game is present/final and there is no ambiguous identity row.

No target-game snaps, sportsbook lines, closing lines or betting outcomes may enter the football comparison.

## Frozen Week-3 metrics

Report separately for:
1. all locked active RB/FB player-games on the two vacancy-event teams;
2. the four direct transfer recipients;
3. each vacancy-event team;
4. predeclared high-volume slices using the project's existing M96A opportunity regimes:
   - actual carries **20+**
   - actual carries **25+**

For both `rush_att` and `rush_yards`, report:
- baseline MAE;
- V1 candidate MAE;
- candidate minus baseline MAE delta;
- baseline signed bias `actual - projection`;
- candidate signed bias;
- absolute-bias delta;
- count of player-games candidate closer / baseline closer / tie.

Also report:
- actual team RB/FB carries and yards;
- baseline and candidate projected active-room carry totals;
- frozen raw/effective simulation allocation probabilities;
- candidate effect on nonrecipient active RB/FBs;
- all conservation/invariant checks.

No market scoring is part of this evaluation.

## Frozen observational classification

This first two-team cohort is **not promotion-eligible regardless of result**.

For descriptive continuity only:

- `WEEK3_OBSERVATIONAL_DIRECTIONALLY_SUPPORTIVE`:
  candidate pooled MAE improves for both rush attempts and rush yards on the full locked RB/FB cohort, with absolute bias non-worse for both and all mechanical invariants passing.

- `WEEK3_OBSERVATIONAL_MIXED`:
  mechanical invariants pass, but the two primary MAEs or their bias directions disagree.

- `WEEK3_OBSERVATIONAL_ADVERSE`:
  mechanical invariants pass and candidate pooled MAE is worse for both rush attempts and rush yards.

- `WEEK3_EVALUATION_INVALID`:
  provenance, identity, no-leakage, frozen-formula or conservation requirements fail.

The high-volume slices and direct-recipient subset are diagnostic reports only and cannot overturn the full-cohort classification.

## Anti-rescue rule

After Week-3 outcomes are visible, do **not**:
- change definitive-unavailable status semantics;
- include DOUBTFUL/QUESTIONABLE;
- change successor weights;
- blend snap share with another weight;
- add depth-chart multipliers;
- change the transfer coefficient;
- change YPC/efficiency;
- alter ensemble weights;
- select only successful recipients/teams;
- redefine the high-volume cutoff;
- use sportsbook information upstream;
- refit against the exposed Week-3 outcomes.

Record the result once and continue accumulating future events under the same V1.
