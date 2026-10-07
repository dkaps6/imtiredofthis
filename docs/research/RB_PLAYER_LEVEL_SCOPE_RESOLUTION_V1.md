# RB Player-Level Scope Resolution V1

Status: **FROZEN SCOPE DECISION / NO NEW RB CANDIDATE / RESEARCH ONLY**

Disposition: `RB_PLAYER_LEVEL_SCOPE_BUTTONED_UP_BASELINE_PLUS_PROSPECTIVE_ALLOCATION`

## Question

Before the requested all-player/all-position replay, does RB require another player-level research candidate, or is the scientifically valid state:

- existing production RB stack in its already-authorized scopes; plus
- the already-frozen prospective Week-5 RB room-allocation shadow;
- with no retrospective M96 reopening and no receiving-mean/width retry?

## Evidence reviewed

### Existing production / historical authority

The current production architecture already contains individual RB history through PlayerForm and protected RB specialist routes:
- Week-1 RB P3 rushing scope;
- Week-1 R22 receiving-yard tail scope;
- Week-1 R26 opportunity/receptions scope;
- non-Week-1 RB Rush+Receiving Conservation V2 where authorized;
- existing PlayerForm individual history elsewhere.

The Week-5 live player-state audit makes the remaining architecture gap explicit. Its production-consumption label for current RB rows is:
`RB_PLAYERFORM_HISTORY_CONSUMED_NO_WEEK5_ROOM_SPECIALIST`.

The same frozen live source contains strictly-prior individual RB:
- prior games;
- targets;
- rushes;
- rushing yards;
- receiving yards;
- last-1 / last-3 rushes;
- last-1 / last-3 targets;
- latest / last-3 snap state.

Therefore raw individual RB target/carry history is **not a new missing source**. The missing layer is room-level player allocation.

### Prospective RB room-allocation shadow

Authority:
- branch `research-rb-player-state-allocation-shadow-v1`
- contract freeze `39074a743eff4e5cb47dcc58c24e9b68c5c48ef1`
- Week-5 lock run `37560824479` SUCCESS
- artifact `11456916566`
- row digest `sha256:66a428f0c19ee1dc356db117fa8e39092204896356461358667224826826e90a`

Frozen treatment:
- 50% recent RB-room carry share;
- 50% recent RB-room offensive snap fraction.

Lock:
- 30 scheduled team rooms;
- 98 RB/HB/FB identities;
- all 30 rooms receive a non-zero allocation change;
- zero sportsbook inputs;
- zero Week-5 outcomes;
- zero fitted coefficients;
- production unchanged.

This is the exact player-level RB workload layer that is new, legal, and already outcome-blind.

## Binding anti-retest closures

### M96

M96E remains a terminal stop on the retrospective RB rushing router/threshold/feature program:
`M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP` /
`AUTONOMOUS_RB_RESEARCH_STOP`.

Do not:
- replay the Week-5 carry/snap allocation shadow retrospectively on Weeks 1-4 to fit or select it;
- search alternate carry/snap weights;
- search new top-N selectors, rush-share thresholds, or routing rules;
- use Weeks 1-4 outcomes to choose an RB workload formula.

### RB receiving-yard mean

R23 through R27D closed the conventional historical receiving-efficiency mean family:
- YPR;
- YPT;
- YAC;
- xYAC;
- YACOE;
- related strict-prior transformations.

No new RB receiving-yard mean study is authorized absent genuinely new pregame football information.

### RB distribution / width

RB-PD2 found historical player difficulty persistence, but the follow-on PD3/PD4/PD5 width-style designs failed their frozen gates.

Do not create another residual/MC-width calibration merely to make the all-position matrix symmetric.

## Why no RB target-share-trajectory extension is opened now

The confirmed WR/TE Target Share Trajectory V1 result does **not** automatically authorize an RB version.

An RB extension would currently be:
- a new transformation of already-consumed target history, not a genuinely new football source;
- proposed only after the WR/TE result was visible;
- outside the exact frozen WR/TE authority cohorts;
- an additional retrospective candidate created mainly to make every position look structurally identical.

That is not required to answer the current player-individualization question.

This decision does **not** declare that RB target trajectory can never contain information. It declares that there is insufficient justification to open a new retrospective RB candidate **before the all-position replay**.

A future RB opportunity study would require its own genuinely justified pregame mechanism and frozen contract, not inheritance from WR/TE by analogy.

## Frozen phase-end RB state

RB is considered **buttoned up for the current player-individualization phase** under this exact scope:

### 2026 Weeks 1-4 replay
Use the already-authorized production RB stack only.

Do **not** back-apply the Week-5 prospective carry/snap shadow to Weeks 1-4.

### 2026 Week 5+ prospective state
Preserve the frozen RB room-allocation shadow as a prospective research layer.

Do not treat it as a result or production promotion until its own frozen prospective sample requirement is satisfied:
- at least 4 future locks;
- at least 80 team-games;
- at least 200 RB player-games.

### Efficiency / distribution
No new RB mean or width treatment is added in this phase.

## Position matrix after this decision

- QB: buttoned up; do not reopen generic mean.
- RB: buttoned up **by scope**, using production baseline + prospective allocation shadow only where legally available.
- WR: opportunity layer buttoned up; target-share trajectory frozen prospectively; player target-depth distribution shadow frozen.
- TE: same general state as WR.

This is sufficient to begin building the all-player/all-position replay without fabricating an RB retrospective promotion test.

## All-position replay implications

The replay must preserve legal timing differences rather than force every position to use every new feature:

- QB: use current protected QB science.
- RB Weeks 1-4: production baseline only.
- RB Week 5+: retain the prospective allocation shadow as a separately labeled shadow; do not score it before outcomes exist.
- WR/TE target-share trajectory: unavailable for 2026 Weeks 1-4 under its exact four-prior-same-season-game contract.
- WR/TE target-depth distribution: may use strictly-prior historical receiver history for Weeks 1-4 under the frozen target-depth feature definition.

No paid OddsAPI pull is authorized or required for this scope decision.
