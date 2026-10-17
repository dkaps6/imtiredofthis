# PRISM Week-4/5 Player-First Forensic Replay V1 — Certified Result

Date: 2026-10-17

Status: **ALL-ROW REPLAY COMPLETE / NAMED-PLAYER ATTRIBUTION COMPLETE / MATCHED CONTROLS COMPLETE / NO PROMOTION**

Model:
**PRISM — Player Role & Individual Situation Model**

Disposition:

`PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_EXECUTED__NO_PROMOTION`

Frozen protocol:

`docs/research/PRISM_WEEK4_5_PLAYER_FIRST_FORENSIC_REPLAY_V1_PROTOCOL.md`

## 1. Governing scientific object

The named player in his actual pregame football situation remains the object:

`PLAYER IDENTITY/HISTORY -> CURRENT ROLE/STATE/TRAJECTORY -> TEAMMATE COMPETITION -> FINITE OPPORTUNITY ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> TEAM/GAME ENVIRONMENT -> PLAYER-CONDITIONED OPPONENT MATCHUP -> INDIVIDUAL EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

Position-average MAE/RMSE is reported only after the player-level work and is not the optimization target.

## 2. Certified execution

Canonical green replay:

- run `39056621963` — **SUCCESS**;
- exact head `12e3370908ccf135243f1c5ad6749b8fb6e15aec`;
- artifact `12109349708`;
- digest `sha256:7dd250813f04bfddfed95dc0df9d16547d6c07aa91eb7de26ac7f84bf43220dd`;
- strict repository audit — **PASS**.

Output population:

- **839** eligible player-market replay rows;
- Week 4: **447**;
- Week 5: **392**;
- the Week-5 source scope contained 396 rows; **4 DNP_INACTIVE rows were explicitly excluded rather than assigned fabricated outcomes**;
- Week-4 exact final-error component identity: **375** rows;
- Week-5 exact final-error component identity: the already-frozen **40** casebook rows;
- Week-5 direct-market rows with reconstructed realized opportunity: **354**;
- Week-5 rows with >=4 prior same-market current-season games: **323**;
- Week-4 WR dual-reliability score manufactured: **false**.

The maximum exact final-error identity gap was **1.631e-11**.

## 3. Important architecture correction — reference decomposition is not always the final PRISM mean

The preserved Week-4 component decomposition contains an opportunity x efficiency reference mean. It is not always numerically identical to the final Week-4 PRISM projection.

Therefore the replay does **not** pretend that old opportunity/efficiency components alone explain final-model error.

For rows with exact component authority, the replay uses:

`actual - final_model = opportunity_component + efficiency_component + downstream_transmission_component`

where:

`downstream_transmission_component = component_reconstructed_mean - final_model_projection`.

This keeps the old component evidence intact while attributing the **actual final PRISM projection**.

This matters for the player-first standard: if upstream football state was sensible but a later layer diluted or altered it, that transmission is visible rather than being misclassified as a player-opportunity or efficiency error.

No reviewed top-80 case was classified as *downstream-transmission-dominant* under the frozen 1.5x component-dominance rule. Downstream transmission still contributed materially in individual rows and remains an architecture trace, not a newly promoted mechanism.

## 4. Detailed named-player casebook

The deterministic casebook contains:

- Week-4 largest final-model misses: **40**;
- frozen Week-5 largest final-model misses: **40**;
- total: **80**;
- matched correctly projected controls: **80**.

Final player-case attribution:

| Primary attribution | Cases |
|---|---:|
| Opportunity quantity / allocation | 36 |
| Efficiency | 25 |
| Mixed | 14 |
| Source blocked / insufficient exact component identity | 5 |

The five source-blocked component rows are Week-4 rush+receiving combination markets for which the preserved artifact does not supply a legal exact final component identity. They are retained rather than reverse-engineered from the outcome.

## 5. Strongest repeated mechanism — named-player allocation, not position average

Restrict to the **22 opportunity-dominant direct skill-player cases**:
- rushing yards;
- receiving yards;
- receptions.

Results:

- **21 / 22** realized more opportunity than PRISM projected;
- Week 4: 13 / 14 underpredicted opportunity;
- Week 5: 8 / 8 underpredicted opportunity.

Then decompose the opportunity-count miss into:
1. parent team-opportunity quantity;
2. named-player allocation/share.

For **21 / 22** cases, the absolute named-player allocation component was larger than the parent team-volume component.

The lone Week-4 exception was Bijan Robinson, whose count miss was more parent-volume-driven.

This is the clearest repeated Week-4/5 forensic mechanism:

> Large direct-skill opportunity misses were usually not primarily “the offense ran/passed more than expected.” They were usually **PRISM allocating too little or too much of finite team opportunity to this named player**.

That is a player-projection finding, not an RB/WR/TE average-error finding.

## 6. Pregame role evidence often pointed toward the allocation miss — but raw recent share remains CLOSED

Among the 22 opportunity-dominant direct-skill cases:

- Week 4: **12 / 14** had the last strictly-prior player team share above the Week-4 component allocation;
- Week 5: **6 / 8** had the preserved current-role share above the frozen rules share;
- total with an above-projection/rules role signal: **18 / 22**.

Matched correctly projected controls showed the same directional signal in **12 / 22** cases.

Magnitude was more separated than simple sign:
- Week-4 opportunity-dominant cases: median pregame role gap **+6.46 percentage points**;
- matched Week-4 controls: **+2.10 points**;
- Week-5 opportunity-dominant cases: median current-minus-rules role gap **+6.94 points**;
- matched Week-5 controls: approximately **-0.10 points**.

These are selected-tail forensic diagnostics, not an independently validated selector.

They do **not** reopen raw recent-share replacement:
- Opportunity Authority Priority remains CLOSED;
- PlayerForm-vs-Bayes disagreement-as-uncertainty remains CLOSED;
- no “use recent share when the gap is large” threshold is authorized.

The correct implication is narrower:

> PRISM repeatedly possessed faster named-player role evidence in important misses, but it still lacks a validated state-arbitration rule for deciding when that evidence represents a real role regime versus noise.

## 7. Named-player examples

### Kenneth Walker III — Week 4 rushing

- final projection: 62.26 rush yards;
- actual: 177;
- projected carries: **8.09**;
- actual carries: **22**;
- strictly-prior opportunity mean: **21.67**;
- last prior game: **18** carries;
- last prior team-rush share: **75.0%**;
- named-player allocation count contribution: **+13.28 carries**;
- parent-volume contribution: **+0.63**.

This is a direct example of a known player role being compressed far more than team volume explains.

Matched correct control:
Jeremiyah Love, same week/position/market with a similar projected opportunity count.

### Chuba Hubbard — Week 4 rushing

- projected carries: **9.51**;
- actual: **20**;
- last prior game: **19** carries;
- last prior team-rush share: **82.61%**;
- allocation count contribution: **+8.79**;
- parent-volume contribution: **+1.70**.

Again the main miss is named-player allocation.

### Jahmyr Gibbs — Week 5 rushing

- projected carries: **12.94**;
- actual: **25**;
- preserved current rush-share state: **83.64%**;
- frozen casebook rules rush share: **58.41%**;
- realized team-rush share: **78.13%**;
- allocation count contribution: **+9.16**;
- parent-volume contribution: **+2.90**.

The raw-current-share replacement family remains closed. The case instead shows why player-state arbitration matters.

Matched correct control:
Jonathan Taylor, with a similar pregame projected carry count.

### Braelon Allen — Week 5 rushing

- projected carries: **6.92**;
- actual: **24**;
- allocation contribution: **+15.24**;
- parent contribution: **+1.84**.

This is the largest allocation-count component among the frozen Week-5 direct opportunity-dominant cases.

Exact target-week teammate/availability reconstruction is not preserved at sufficient parity, so the replay does not invent the room-change explanation.

### Aaron Jones — Week 5 rushing

- projected carries: **10.91**;
- actual: **25**;
- current rush-share state: **63.97%**;
- frozen rules rush share: **48.62%**;
- realized team-rush share: **69.44%**;
- allocation contribution: **+8.58**;
- parent contribution: **+5.51**.

Both parent volume and allocation were too low, but named-player allocation was still the larger count component.

### Malik Nabers — Week 5 receiving

- projected targets: **6.63**;
- actual: **20**;
- allocation contribution: **+9.92 targets**;
- parent contribution: **+3.45**;
- current target-share state was above the frozen rules state.

Week 4 for the same player was efficiency-dominant in the final-error identity; Week 5 was opportunity-dominant. A permanent player boost is therefore not supported.

### Puka Nacua — Week 5 receiving

- projected targets: **7.13**;
- actual: **24**;
- allocation contribution: **+13.23**;
- parent contribution: **+3.64**.

The allocation seam dominates despite a large team-volume increase.

### Brock Bowers — Week 5 receiving

Frozen casebook:
- upstream rules target share: **27.77%**;
- final target probability: **20.63%**;
- realized target share: **27.78%**.

The upstream role estimate was essentially exact while a later production path attenuated it by about **7.14 percentage points**.

Bowers is therefore a clean architecture example of:
`UPSTREAM_PLAYER_ROLE_READ_CORRECT__DOWNSTREAM_TRANSMISSION_LOST_INFORMATION`.

However, his final yard error is mixed rather than downstream-dominant because receiving efficiency also mattered. This is an architecture trace, not permission to remove the downstream layer wholesale.

### CeeDee Lamb — state reversal control against permanent correction

Week 4:
- receiving-yards miss was opportunity-dominant and high-side.

Week 5:
- receiving yards: 81.04 projected -> 9 actual;
- targets: 8.26 projected -> 5 actual;
- final state: mixed opportunity/efficiency;
- current role signal still sat above the frozen rules state.

This is decisive evidence against “known underprojected star -> permanent upward adjustment.”

## 8. Same player + same market is not a stable failure label

Nine named player + market pairs appeared in both Week-4 and Week-5 top-40 sets.

Final-error component state changed in **7 / 9**:

| Player | Market | Week 4 | Week 5 |
|---|---|---|---|
| Joe Burrow | pass yards | opportunity | mixed |
| CeeDee Lamb | receiving yards | opportunity | mixed |
| Malik Willis | pass yards | mixed | efficiency |
| Matthew Stafford | pass yards | opportunity | opportunity |
| Dak Prescott | pass yards | opportunity | opportunity |
| Ollie Gordon II | rush yards | mixed | opportunity |
| Aaron Jones | rush+rec yards | insufficient exact component evidence | opportunity |
| Malik Nabers | receiving yards | efficiency | opportunity |
| Jacoby Brissett | pass yards | efficiency | opportunity |

This strengthens the governing PRISM principle:

> player identity is the historical playbook; the current football state determines which mechanism is active this week.

Reliability can tell PRISM how much to trust a projection. It cannot substitute for state identification.

## 9. Efficiency findings remain player-specific and do not reopen generic tail science

There were **25** efficiency-dominant top-80 cases.

Examples include:
- Jordan Love Week-5 passing;
- Drake Maye Week-5 passing;
- Bucky Irving Week-5 rushing;
- Jayden Daniels Week-5 rushing;
- Jaylen Warren Week-5 rushing;
- Christian Watson Week-5 receiving.

Do not infer one generic efficiency correction.

Protected closures remain:
- generic efficiency-tail stability CLOSED;
- generic QB efficiency feature hunt frozen/closed;
- TE efficiency trust CLOSED;
- generic defense-vs-position CLOSED.

Where named-player reliability has already qualified, it remains a trust layer rather than a directional mean fix.

## 10. Opponent / matchup boundary

The replay preserves available opponent state where the Week-5 player-state artifact supplies it.

It does **not** label a generic opponent metric as the cause of a miss.

Exact player-conditioned matchup authority at the required role/opportunity-subtype grain remains source-blocked or unavailable for the reviewed cases.

Therefore:
- no generic DvP;
- no “bad defense -> boost player” rule;
- no usage x generic matchup rescue.

Future opponent work must remain downstream of player role and must be player-conditioned.

## 11. Teammate competition / availability boundary

The current historical evidence does not preserve exact row-level target-week teammate availability for all reviewed Week-4/5 players at the required parity.

The replay therefore does not manufacture injury or vacancy narratives.

Room Competition State V1 remains:

`PRISM_ROOM_COMPETITION_STATE_SOURCE_BLOCKED`.

Do not:
- lower its source gates;
- infer inactivity from target-game snaps;
- use sportsbook listings as availability;
- revive vacancy-transfer logic.

This is a real source gap because several of the largest allocation misses are exactly the cases where finite teammate competition is the football question.

## 12. What the matched controls say

Each case has a correctly projected control selected from the same:
- week;
- position family;
- exact market;

with the closest comparable pregame projected opportunity count where available, or closest model mean for combination markets, among the better half of the corresponding error distribution.

Examples:
- Kenneth Walker III -> Jeremiyah Love;
- Jahmyr Gibbs -> Jonathan Taylor;
- Braelon Allen -> Kendre Miller;
- Malik Nabers -> Nico Collins;
- Puka Nacua -> Rome Odunze;
- Brock Bowers -> Tucker Kraft;
- Jordan Love -> Jalen Hurts.

The controls prevent a simple “this signal appeared in a miss, therefore it caused the miss” story.

In particular, an above-projection/rules role signal also appeared in 12/22 matched controls for the 22 opportunity-dominant direct-skill cases.

Therefore sign alone is not a valid selector.

## 13. Repeated mechanisms after player-level attribution

Ranked by current forensic importance:

### 1. Named-player finite allocation compression
**Strongest repeated seam.**

Evidence:
- 21/22 opportunity-dominant direct-skill cases had the allocation-count component larger than parent volume;
- 21/22 underpredicted realized opportunity;
- repeated across Weeks 4 and 5 and across RB/WR/TE direct markets.

This does not authorize a raw-share replacement. It localizes the missing decision to current player-role/state arbitration and finite allocation.

### 2. Current-role evidence exists but needs regime arbitration
**Supported as a research frontier, not a rule.**

Role evidence was above the model/rules in 18/22 opportunity-dominant direct-skill cases, but the signal also appeared in correct controls.

The scientific question is:
> when does the current player state represent a structural role regime rather than noise?

### 3. Player mechanism can change week to week
**Strong repeated evidence.**

7/9 repeated player+market extreme cases changed final component state.

Static player corrections are invalid.

### 4. Upstream football knowledge can be attenuated downstream
**Architecture seam; not dominant population mechanism yet.**

Brock Bowers is the clearest exact example. Several other Week-5 skill cases also show rules-share -> final-probability attenuation, but the upstream share itself was still wrong in many of them.

### 5. Exact teammate competition and player-conditioned matchup remain source gaps
**Important but source-blocked.**

Do not fill these gaps with generic proxies.

## 14. Separately testable hypotheses

The replay generates hypotheses; it does not promote them.

Priority order:

1. **Current role-state arbitration source audit**  
   Determine whether a genuinely new, strictly-pregame regime-transition object can distinguish structural role change from short-term noise without reproducing raw recent share, existing target trajectory, RB carry/snap, Stage-1 participation, or Stage-2 active-set allocation.

2. **Qualified Stage-1/Stage-2 retrospective transmission check**  
   Replay the already-frozen independent Stage-1/Stage-2 mechanism on preserved Week-5 pregame state. This is legal because the mechanism was qualified independently of Week-5 outcomes and is not retuned from the replay.

3. **Finite-room source improvement**  
   Search only for a higher-parity historical/live teammate-competition source that can clear the already-frozen gates. Do not lower the gates.

4. **Player-conditioned matchup source audit**  
   Only if a source can represent the player's actual role/opportunity subtype with historical/live parity. Generic opponent metrics remain closed.

## 15. Aggregate QA — intentionally last

Aggregate MAE/RMSE exists in the certified artifact for downstream QA.

It is not the scientific conclusion.

The important result is not “WR MAE is X” or “RB MAE is Y.”

The important result is:

> Across the largest direct skill-player opportunity failures, PRISM usually missed **how the finite team opportunity was allocated to the named player**, and the same player could still fail for a different mechanism one week later.

That is exactly the individual-player problem PRISM is intended to solve.

## 16. Integrity

Certified replay:
- Week-6 outcomes read: **0**;
- sportsbook refetch: **false**;
- sportsbook predictor upstream: **false**;
- OddsAPI calls: **0**;
- coefficients fit: **0**;
- thresholds fit: **0**;
- production changed: **false**;
- Week-4 exact WR dual score manufactured: **false**.

Known Week-4/5 outcomes were used only for forensic scoring and attribution after the frozen pregame authorities were fixed.

No mechanism in this result is promoted.

