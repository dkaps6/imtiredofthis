# PRISM 2026 Player-First Replay — Week 5 Repeat-Player State Sheets V1

Date: 2026-10-17

Status: **PLAYER-STATE FORENSICS ACTIVE / SIX PRIORITY CASES ADJUDICATED TO CURRENT EVIDENCE BOUNDARY**

Parents:
- `docs/research/PRISM_2026_PLAYER_FIRST_REPLAY_W4_W5_V1_PROTOCOL.md`;
- `docs/research/PRISM_2026_PLAYER_FIRST_REPLAY_W4_W5_V1_INITIAL_EVIDENCE.md`;
- `docs/research/PRISM_2026_PLAYER_FIRST_REPLAY_W5_OPPORTUNITY_DECOMPOSITION_V1.md`.

This document is diagnostic only. It does not fit a correction or modify production.

## Governing standard

For every named player:

`IDENTITY/HISTORY -> CURRENT ROLE -> TEAMMATE COMPETITION -> PARENT OPPORTUNITY -> PLAYER ALLOCATION -> OPPORTUNITY TYPE/QUALITY -> OPPONENT -> EFFICIENCY -> RELIABILITY -> MARKET`.

The point is to determine **which link failed for this player**.

---

## 1. Jahmyr Gibbs — repeat rushing miss

### Prior identity / history signal

Gibbs already appeared in the earlier extreme-miss set:
- Week 1 rush yards: `59.47 -> 156`;
- Week 1 rush+rec: `80.50 -> 186`.

A Week-3 live diagnostic had also already identified Gibbs as an example of the
generic RB population-Bayes path pulling an RB1 rushing share downward before
the final top-five rushing selector.

That old direct RB1/Bayes repair family is closed and may not be reopened.

### Strictly-prior Week-5 room state

Prospective Week-5 RB-room lock:
- Gibbs recent RB-room carry share: **87.93%**;
- Sione Vaki: **12.07%**;
- Gibbs recent offensive snap fraction: **77.85%**;
- Vaki: **21.17%**.

Realized Week-5 RB-room carry share:
- Gibbs **80.65%**;
- Vaki **19.35%**.

Interpretation:
the **within-RB-room hierarchy was already known reasonably well pregame** and
remained Gibbs-dominant.

### Production path

- predicted team rush attempts: **26.14**;
- production `rules_rush_share`: **58.41%**;
- final rush probability: **49.50%**;
- predicted carries: **12.94**.

Actual:
- team rushes **32**;
- Gibbs carries **25**;
- team-rush share **78.13%**.

Opportunity-gap decomposition:
- parent quantity contribution: **+2.90 carries**;
- named-player allocation contribution: **+9.16 carries**.

### Current adjudication

`KNOWN_DOMINANT_ROOM_ROLE__NOT_FULLY_TRANSMITTED_TO_TEAM_RUSH_ALLOCATION`

This is not permission to revive the closed generic RB1/Bayes replacement.

It is a direct target for the newer PRISM architecture:
- active-role state;
- exact teammate active set;
- conditional finite allocation;
- same named player.

The qualified Stage-2 RB rushing mechanism is scientifically distinct and is
therefore the correct mechanism to replay against this case, if exact Week-5
source reconstruction can be achieved.

---

## 2. Aaron Jones — repeat rushing miss

### Earlier failure

Week 2:
- rush yards `39.40 -> 105`;
- rush+rec `35.75 -> 105`.

### Strictly-prior Week-5 room state

Prospective Week-5 RB-room lock:
- Aaron recent RB-room carry share: **86.96%**;
- Demond Claiborne: **8.70%**;
- DeeJay Dallas: **4.35%**.

Recent RB-room state was therefore already highly concentrated.

Realized Week-5 RB-room carries:
- Aaron Jones **25**;
- other scored RB-room rushes **0**;
- realized scored RB-room share **100%**.

### Production path

- predicted team rushes **23.91**;
- rules rush share **48.62%**;
- final rush probability **45.61%**;
- predicted carries **10.91**.

Actual:
- team rushes **36**;
- Aaron carries **25**;
- team-rush share **69.44%**.

Opportunity-gap decomposition:
- parent quantity: **+5.51**;
- player allocation: **+8.58**.

### Current adjudication

`KNOWN_DOMINANT_ROOM_ROLE__PARENT_AND_PLAYER_ALLOCATION_BOTH_TOO_LOW`

The case contains two distinct errors:
1. Minnesota ran more than PRISM's parent quantity expected;
2. Aaron received a larger share of those team rushes than the final player
   allocation expected.

Do not collapse those into one RB mean adjustment.

---

## 3. Ollie Gordon II — repeat rushing miss

### Earlier failures

- Week 3: `13.62 -> 41`;
- Week 4: `28.46 -> 100`.

Week 5:
- `33.47 -> 88`.

This is a three-week repeated named-player miss sequence.

### Strictly-prior Week-5 room state

Prior RB-room carry shares:
- Ollie Gordon II **49.15%**;
- De'Von Achane **38.98%**;
- Jaylen Wright **11.86%**.

Actual Week-5 scored RB-room share:
- Gordon **75%**;
- Wright **25%**;
- Achane **0%**.

### Production path

- predicted team rushes **22.20**;
- final Gordon rush probability **25.96%**;
- predicted carries **5.76**.

Actual:
- team rushes **34**;
- Gordon carries **18**;
- team-rush share **52.94%**.

Opportunity gap:
- **+12.24 carries**.

Decomposition:
- parent quantity **+3.06**;
- player allocation **+9.17**.

### Current adjudication

`TEAMMATE_COMPETITION_STATE_CHANGE_RECONSTRUCTION_REQUIRED`

Unlike Gibbs, the prior room hierarchy did **not** simply persist.
A major competitor went from ~39% of prior RB-room carries to zero realized
rushes.

Before calling this a model failure mode, PRISM must determine:
- whether Achane was known unavailable/inactive pregame;
- whether his target-week role was limited pregame;
- whether the zero-rush outcome was not knowable.

If pregame availability already removed/reduced Achane, this becomes a strong
`STATE_CHANGE_NOT_CAPTURED` case.

If Achane was fully expected to play normally, this may instead be target-game
variance / game-plan realization.

No inference is permitted until that pregame state is recovered.

---

## 4. Braelon Allen — repeat rushing miss

### Earlier failure

Week 4 rush yards:
- `21.39 -> 59`.

Week 5:
- `36.78 -> 116`.

### Production / realized opportunity

- predicted team rushes **25.28**;
- final Allen rush probability **27.37%**;
- predicted carries **6.92**.

Actual:
- team rushes **32**;
- Allen carries **24**;
- team-rush share **75%**.

Gap:
- **+17.08 carries**.

Decomposition:
- parent quantity **+1.84**;
- player allocation **+15.24**.

This is the largest allocation contribution in the current priority queue.

### Source seam

The Week-5 prospective RB-room lock contains no NYJ room row.

Therefore the replay currently lacks an equivalent frozen recent-room snapshot.

### Current adjudication

`REPEAT_ALLOCATION_MISS__STRICT_PRIOR_ROOM_SOURCE_RECONSTRUCTION_REQUIRED`

Do not infer a committee change from the target result.

Recover the Week-5 NYJ pregame room/availability state first.

---

## 5. Malik Nabers — repeat target-allocation miss

### Earlier failure

Week 4 receiving yards:
- `44.37 -> 112`.

Week 5:
- `52.71 -> 121`.

Week-5 casebook:
`OPPORTUNITY_DOMINANT`.

### Strictly-prior trajectory state

Frozen Week-5 WR trajectory row:
- recent-2 target share **26.00%**;
- earlier share **21.31%**;
- trajectory delta **+4.69 percentage points**;
- baseline entitlement target share **17.88%**;
- WR anchor present;
- trajectory shadow entitlement delta **0**.

Production:
- rules target share **23.43%**;
- final target probability **19.39%**;
- predicted targets **6.63**.

Actual:
- team targets **52**;
- Nabers targets **20**;
- realized share **38.46%**.

Opportunity gap:
- **+13.37 targets**.

Decomposition:
- parent quantity **+3.45**;
- named-player allocation **+9.92**.

### Current adjudication

`UPWARD_ROLE_TRAJECTORY_PRESENT__CONCENTRATION_STILL_UNDERTRANSMITTED`

Important boundary:
the broad frozen WR trajectory candidate was flat/slightly worse in Week 5 and
is not promoted.

Therefore the lesson is **not** to turn the recent-2 trajectory formula on for
all WRs.

The player-first question is why this specific WR's room/teammate/current-role
state supported much more concentration than the final allocation recognized.

WR Stage-2 remains source-blocked under the current exact historical OOS
contract, so this case also belongs in the WR source frontier rather than being
silently approximated.

---

## 6. Brock Bowers — repeat TE receiving miss / downstream transmission case

### Earlier identity

Week 3:
- receiving yards `39.35 -> 116`.

Week-4 pregame evidence already documented:
- TE-R5P entitlement share **19.17%**;
- entitlement delta **+0.84 percentage points**;
- prior-week offensive participation **79%**;
- Friday knee limitation, but no game designation.

The Week-4 model still treated Bowers as a strong receiving-yards UNDER
disagreement versus market.

### Strictly-prior Week-5 role state

Week-5 trajectory row:
- recent-2 target share **30.77%**;
- baseline entitlement **16.78%**;
- trajectory shadow **19.36%**.

Production Week 5:
- rules target share **27.77%**;
- final target probability **20.63%**;
- predicted targets **6.76**.

Actual:
- team targets **36**;
- Bowers targets **10**;
- realized target share **27.78%**.

The **rules target share and realized target share agree to ~0.01 percentage
points**, but the final projection does not preserve that role share.

Relative rules-share attenuation:
approximately **25.7%**.

Opportunity gap:
- parent quantity **+0.66 target**;
- player allocation/transmission **+2.57 targets**.

### Current adjudication

`UPSTREAM_PLAYER_ROLE_READ_CORRECT__DOWNSTREAM_TRANSMISSION_DEFECT`

This is currently the cleanest Dobbins-style PRISM architecture example in the
Week-5 replay:

> the model knew who the player was and approximately how large his role was,
> but that football knowledge was diluted before the final market projection.

This case should be traced through the exact target-probability bridge before
any new TE mean mechanism is proposed.

TE Stage-2 conditional allocation is already historically QUALIFIED, making
that current mechanism especially relevant to this replay.

---

## 7. Why these six cases cannot share one fix

The six repeat identities split into materially different football mechanisms:

- **Gibbs:** room hierarchy essentially right; team-level player rush allocation
  undertransmitted.
- **Aaron Jones:** dominant room role plus parent team-rush underprojection and
  player allocation underprojection.
- **Ollie Gordon:** teammate competition changed; pregame availability state must
  be reconstructed.
- **Braelon Allen:** huge allocation miss, but frozen prior-room source is missing.
- **Malik Nabers:** upward WR trajectory existed but final concentration remained
  too low; WR Stage-2 source remains blocked.
- **Brock Bowers:** upstream role estimate was essentially right, but downstream
  target-probability transmission diluted it.

This is precisely why PRISM cannot be repaired by a single positional MAE tweak.

## 8. Current next actions

1. Trace the exact production target-probability bridge for Brock Bowers.
2. Reconstruct target-week teammate/availability state for Ollie Gordon.
3. Recover the NYJ Week-5 strict-prior RB-room source for Braelon Allen if
   canonical artifacts permit.
4. Compare the later-qualified PRISM Stage-2 RB/TE conditional allocation
   mechanics against these replay cases **without retuning the qualified
   mechanism from their outcomes**.
5. Keep Malik Nabers in the WR source-blocked lane; do not fake Stage-2 WR
   authority.

## 9. Integrity

- paid OddsAPI calls: **0**;
- sportsbook refetches: **0**;
- Week-6 outcomes used: **0**;
- coefficients fit: **0**;
- thresholds fit: **0**;
- production changes: **0**.

Disposition:

`PRISM_W5_REPEAT_PLAYER_STATE_SHEETS_V1__MECHANISMS_SEPARATED`.
