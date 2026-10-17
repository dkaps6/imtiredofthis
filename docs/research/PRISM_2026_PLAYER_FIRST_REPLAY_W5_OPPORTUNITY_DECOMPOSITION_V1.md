# PRISM 2026 Player-First Replay — Week 5 Opportunity Decomposition V1

Date: 2026-10-17

Status: **DIAGNOSTIC DECOMPOSITION COMPLETE / NO MODEL CHANGE**

Parent protocol:
`docs/research/PRISM_2026_PLAYER_FIRST_REPLAY_W4_W5_V1_PROTOCOL.md`

Initial repeat-player evidence:
`docs/research/PRISM_2026_PLAYER_FIRST_REPLAY_W4_W5_V1_INITIAL_EVIDENCE.md`

Authorities:
- Week-5 player failure casebook run `38397361891`, artifact `11812403368`;
- Week-5 prospective opportunity-lock postmortem run `38396065511`, artifact `11812261490`.

No sportsbook refetch and no new OddsAPI call were used.

## 1. Purpose

The prior casebook labels a large miss as opportunity-dominant, efficiency-dominant,
or mixed.

PRISM needs a more football-specific question:

> When the named player's opportunity was wrong, how much of the miss came from
> the team/game parent opportunity quantity and how much came from allocation to
> this specific player?

This is the distinction between:
- the offense creating/running enough parent opportunities; and
- PRISM assigning the correct share of those opportunities to the named player.

## 2. Algebraic decomposition

For a market with:
- predicted parent opportunity quantity `N_pred`;
- predicted player probability/share `p_pred`;
- actual parent opportunity quantity `N_actual`;
- actual named-player share `p_actual`;

then:

`predicted_player_opportunity = N_pred * p_pred`

`actual_player_opportunity = N_actual * p_actual`

and the exact player-opportunity gap decomposes as:

`actual - predicted = (N_actual - N_pred) * p_pred + N_actual * (p_actual - p_pred)`.

The first term is the **parent opportunity quantity contribution**.
The second term is the **named-player allocation contribution**.

For RB rushing, parent quantity is team rushing attempts.

For pass-catcher targets, the production parent is predicted team passing
opportunities while the realized finite parent is observed team targets. The
decomposition remains an exact accounting identity for the stored predicted and
actual target counts, but the parent-source semantics are not claimed to be
literally identical. Treat that term as passing/target-opportunity quantity
context, not a newly authorized model object.

## 3. Repeat-player opportunity cases

### Aaron Jones — Week 5 rushing

Frozen production:
- predicted team rush attempts: **23.91**;
- final player rush probability: **45.61%**;
- predicted carries: **10.91**.

Realized:
- team rushes: **36**;
- Aaron carries: **25**;
- realized team-rush share: **69.44%**.

Opportunity gap:
- **+14.09 carries**.

Decomposition:
- parent team-rush quantity: **+5.51 carries**;
- named-player allocation: **+8.58 carries**.

Approximately **60.9%** of the carry gap is the allocation term.

Week-5 casebook label:
`OPPORTUNITY_DOMINANT`.

Strictly-prior RB-room context from the prospective lock:
- Aaron had **86.96%** of the recent RB-room carry share;
- the room was already highly concentrated before Week 5.

Important:
RB-room share and total-team rush share have different denominators. Do not
directly substitute one for the other. The room evidence is player-role context,
not an authorized formula.

### Jahmyr Gibbs — Week 5 rushing

Frozen production:
- predicted team rush attempts: **26.14**;
- final player rush probability: **49.50%**;
- predicted carries: **12.94**.

Realized:
- team rushes: **32**;
- Gibbs carries: **25**;
- realized team-rush share: **78.13%**.

Opportunity gap:
- **+12.06 carries**.

Decomposition:
- parent quantity: **+2.90**;
- named-player allocation: **+9.16**.

Allocation explains approximately **76.0%** of the carry gap.

Strictly-prior RB-room context:
- recent RB-room carry share **87.93%**;
- recent room was Gibbs-dominant before the target game.

Week-5 rush-yard label:
`OPPORTUNITY_DOMINANT`.

### Ollie Gordon II — Week 5 rushing

Frozen production:
- predicted team rush attempts: **22.20**;
- final player rush probability: **25.96%**;
- predicted carries: **5.76**.

Realized:
- team rushes: **34**;
- Gordon carries: **18**;
- realized team-rush share: **52.94%**.

Opportunity gap:
- **+12.24 carries**.

Decomposition:
- parent quantity: **+3.06**;
- named-player allocation: **+9.17**.

Allocation explains approximately **75.0%** of the carry gap.

Strictly-prior RB-room context:
- recent room carry share **49.15%**;
- De'Von Achane **38.98%**;
- Jaylen Wright **11.86%**.

Actual Week-5 RB-room carry share became:
- Gordon **75%**;
- Wright **25%**;
- Achane **0%**.

This case requires the next state-sheet step to determine what target-week
availability/role information was known pregame. Do not infer the reason from
the result itself.

### Braelon Allen — Week 5 rushing

Frozen production:
- predicted team rush attempts: **25.28**;
- final player rush probability: **27.37%**;
- predicted carries: **6.92**.

Realized:
- team rushes: **32**;
- Allen carries: **24**;
- realized team-rush share: **75.0%**.

Opportunity gap:
- **+17.08 carries**.

Decomposition:
- parent quantity: **+1.84**;
- named-player allocation: **+15.24**.

Allocation explains approximately **89.2%** of the gap.

This is the cleanest repeat-player allocation miss in the current queue.

The Week-5 prospective RB-room lock did not contain a NYJ room row, so the
strictly-prior teammate/room state must be reconstructed separately rather than
silently inferred.

### Malik Nabers — Week 5 targets

Frozen production:
- predicted parent passing opportunities: **34.22**;
- final target probability: **19.39%**;
- predicted targets: **6.63**.

Realized:
- team targets: **52**;
- Nabers targets: **20**;
- realized team target share: **38.46%**.

Opportunity gap:
- **+13.37 targets**.

Decomposition:
- parent passing/target quantity: **+3.45**;
- named-player allocation: **+9.92**.

Allocation explains approximately **74.2%** of the target gap.

Strictly-prior WR trajectory lock:
- recent-2 target share: **26.0%**;
- earlier share: **21.31%**;
- positive trajectory delta: **+4.69 percentage points**;
- frozen baseline entitlement: **17.88%**;
- the WR trajectory shadow made **no entitlement change** on this anchored WR row.

The historical Week-5 WR trajectory study was flat/slightly worse overall, so
this case does not authorize simply applying the recent trajectory formula.
It does show that the named player's current concentration was already moving
up before the 20-target game.

### Brock Bowers — Week 5 targets

Frozen production:
- predicted parent passing opportunities: **32.79**;
- `rules_tgt_share`: **27.77%**;
- final target probability after downstream transmission: **20.63%**;
- predicted targets: **6.76**.

Realized:
- team targets: **36**;
- Bowers targets: **10**;
- realized target share: **27.78%**.

Opportunity gap:
- **+3.24 targets**.

Decomposition:
- parent quantity: **+0.66**;
- named-player allocation: **+2.57**.

Allocation explains approximately **79.5%** of the target-count gap.

This is a particularly important architecture case:

**the upstream rules target share (27.77%) was essentially identical to the
realized share (27.78%), but a later production layer reduced the final target
probability to 20.63%.**

Relative attenuation:
- rules share -> final target probability: approximately **-25.7%**.

This is not evidence that PRISM failed to know Brock Bowers' role.
It is evidence that a strong player-role read was not preserved through the
full projection path.

Week-5 casebook label:
`MIXED_OPPORTUNITY_EFFICIENCY`.

Strictly-prior trajectory context:
- recent-2 target share **30.77%**;
- frozen trajectory shadow **19.36%**;
- actual **27.78%**.

## 4. Control cases show why a blanket role boost is wrong

### Jaylen Warren

Production:
- predicted team rushes **26.09**;
- final rush probability **44.18%**;
- predicted carries **11.53**.

Actual:
- team rushes **22**;
- Warren carries **15**.

Opportunity gap:
- **+3.47 carries**.

Decomposition:
- parent quantity contribution **-1.81**;
- allocation contribution **+5.28**.

The yard miss was nevertheless classified:
`EFFICIENCY_DOMINANT`.

Strictly-prior RB-room carry share was **77.59%**, but using that room share as a
naive total-team share is denominator-invalid and would overstate carries.

The lesson is not "recent role always fixes the projection."

### CeeDee Lamb

Production:
- predicted parent passing opportunities **35.05**;
- final target probability **23.57%**;
- predicted targets **8.26**.

Actual:
- team targets **42**;
- Lamb targets **5**;
- realized target share **11.90%**.

Opportunity gap:
- **-3.26 targets**.

Decomposition:
- parent quantity contribution **+1.64**;
- named-player allocation contribution **-4.90**.

Earlier season extreme misses had been large high-side misses, but Week 5 flipped
direction.

Strictly-prior target trajectory was still elevated:
- recent-2 target share **34.94%**;
- earlier share **27.87%**.

This is exactly why PRISM needs state/reliability/teammate context rather than
permanent player-specific mean boosting.

## 5. Downstream attenuation audit

For the repeat-player queue, the stored production path also shows how much
`rules_*_share` was reduced before the final opportunity probability:

- Aaron Jones rush: **48.62% -> 45.61%** (~6.2% relative attenuation);
- Jahmyr Gibbs rush: **58.41% -> 49.50%** (~15.3%);
- Ollie Gordon II rush: **26.58% -> 25.96%** (~2.3%);
- Jaylen Warren rush: **53.36% -> 44.18%** (~17.2%);
- Braelon Allen rush: **27.37% -> 27.37%** (no attenuation);
- Malik Nabers target: **23.43% -> 19.39%** (~17.2%);
- Brock Bowers target: **27.77% -> 20.63%** (~25.7%);
- CeeDee Lamb target: **25.28% -> 23.57%** (~6.8%).

These are descriptive transmission facts.

They do not authorize removal of the downstream layer. The correct next question
is player-by-player:

> Was the upstream player-state estimate already good, and if so why did the
> downstream path attenuate it? Or was the upstream role estimate itself wrong?

Bowers is currently the clearest example of the first class.

## 6. Scientific interpretation

For the high-priority repeat opportunity cases, the Week-5 opportunity miss is
not mainly one generic team-volume problem.

Named-player allocation is the larger algebraic component for:
- Aaron Jones;
- Jahmyr Gibbs;
- Ollie Gordon II;
- Braelon Allen;
- Malik Nabers;
- Brock Bowers.

This is directly consistent with the governing PRISM model object:

**the named player in his current room/team situation.**

It also explains why aggregate position MAE is insufficient. Two players at the
same position can fail for completely different reasons:
- parent opportunity quantity;
- named-player allocation;
- downstream attenuation;
- efficiency;
- reliability/state reversal.

## 7. Exact next action

Build strictly-prior state sheets for:
1. Aaron Jones;
2. Jahmyr Gibbs;
3. Ollie Gordon II;
4. Braelon Allen;
5. Malik Nabers;
6. Brock Bowers.

Each state sheet must reconcile:
- prior role and trajectory;
- teammate room competition;
- target-week availability/inactive state available pregame;
- canonical entitlement/share entering production;
- downstream share/probability transmission;
- parent opportunity quantity;
- realized opportunity only after pregame state is frozen;
- whether the eventual failure is `KNOWN_BUT_NOT_TRANSMITTED`,
  `STATE_CHANGE_NOT_CAPTURED`, `PARENT_QUANTITY_MISS`,
  `ALLOCATION_MISS`, `EFFICIENCY_TAIL`, or `NO_CLEAR_MECHANISM`.

No player-specific correction may be fitted from these cases.

## 8. Integrity

- paid OddsAPI calls: **0**;
- sportsbook refetch: **0**;
- Week-6 outcomes used: **0**;
- fitted coefficients: **0**;
- fitted thresholds: **0**;
- production changes: **0**;
- projection means changed: **0**.

Disposition:

`PRISM_W5_REPEAT_PLAYER_OPPORTUNITY_DECOMPOSITION_COMPLETE__STATE_SHEETS_NEXT`.
