# PRISM 2026 Week-5 Named-Player Situation Forensic Replay V1 — Result

Date: 2026-10-17

Status: **FORENSIC CURRENT-SEASON PLAYER-STATE REPLAY COMPLETE / NO PROMOTION**

Model:
**PRISM — Player Role & Individual Situation Model**

Disposition:
`PLAYER_STATE_FORENSIC_REPLAY_COMPLETE__NO_PROMOTION`

## 1. Why this replay exists

The governing object is the named player in his actual football situation:

`PLAYER IDENTITY -> CURRENT ROLE/STATE -> TEAMMATE COMPETITION -> OPPORTUNITY ALLOCATION -> OPPORTUNITY QUALITY -> OPPONENT/ENVIRONMENT -> EFFICIENCY -> RELIABILITY -> MARKET-SPECIFIC OUTCOME`.

This replay was created because the project already has completed 2026 evidence and
should not wait for another shadow week merely to inspect whether PRISM
understands the player.

This is deliberately more than an aggregate MAE exercise.

It asks:
- does the same player's historical projection difficulty carry forward?
- what failed on the named player this week: opportunity, efficiency, or both?
- did strictly-prior current player state point in a different direction from
  the production rule state?
- was opponent context available at the trace boundary?
- can the preserved evidence legitimately reconstruct teammate/injury context?

## 2. Exact source authorities

### W1-W4 player decomposition

- run `37777319154` — SUCCESS;
- exact head `cb1e511adcb58e40fc41d0949306083a304e83dd`;
- artifact `11550446909`;
- digest `sha256:7acfc453258a49f3aadf0d0ea386b13827eb715f1dfb055b96505f733320d204`.

### Week-5 settled full board

Canonical pregame Full Slate:
- run `37860592615`;
- artifact `11585973062`;
- digest `sha256:bf52aa4bc30bf6e8cc6029a0a86c0f33cdfaa09f587c38a138f23cef6b7436f4`.

Settled postmortem:
- run `38395384228` — SUCCESS;
- artifact `11812200405`;
- digest `sha256:5888cab39506ae0bf70034dfb60f297defd28dd3219ea8118870738706f55ac3`.

### Week-5 frozen player-failure casebook

- run `38397361891`;
- artifact `11812403368`;
- digest `sha256:4cdef29912f35368473d57e15b38efdc3dd0f75e897bbbdc3b97fe1ff8f67670`.

### Week-5 pregame player-landscape state

- run `37717016519` — SUCCESS;
- exact head `35820953a255ce0f8413c2c468501df7f267fa96`;
- artifact `11524296325`;
- digest `sha256:e043133d4b9a48ce472dde8810d4d77d636f4ce2fbcee14708b23fe0ea35f6a0`;
- disposition parent:
  `INDIVIDUAL_PLAYER_CORE_CONFIRMED__LANDSCAPE_TRANSMISSION_INCOMPLETE`.

Replay implementation:
`scripts/research/audit_prism_week5_player_state_forensic_replay_v1.py`.

## 3. Strict-prior reliability support

Using only Weeks 1-4 for the same named player and same market:

- Week-5 replay rows: **318**;
- unique players: **166**;
- frozen Week-5 top-failure cases: **40**;
- top-failure cases with all four prior same-market weeks: **30 / 40**.

This is the same strict-prior support behind:
`docs/research/PRISM_2026_WEEK5_STRICT_PRIOR_PLAYER_REPLAY_V1_RESULT.md`.

Reliability remains a trust layer. It is not an over/under or directional rule.

## 4. Main finding — the same player does not fail for the same reason every week

Take the deterministic top-40 Week-4 decomposition misses and the already-frozen
top-40 Week-5 failure casebook.

There are **9** repeated **same player + same market** cases.

**7 / 9 = 77.78% changed component state** between:
- opportunity dominant;
- efficiency dominant;
- mixed opportunity/efficiency.

| Player | Market | Week 4 | Week 5 |
|---|---|---|---|
| Joe Burrow | pass yards | opportunity | efficiency |
| Malik Willis | pass yards | mixed | efficiency |
| CeeDee Lamb | rec yards | opportunity | mixed |
| Jacoby Brissett | pass yards | efficiency | opportunity |
| Dak Prescott | pass yards | opportunity | opportunity |
| Ollie Gordon II | rush yards | mixed | opportunity |
| Malik Nabers | rec yards | efficiency | opportunity |
| Matthew Stafford | pass yards | opportunity | opportunity |
| Jordan Love | pass yards | mixed | efficiency |

Week-4 component attribution uses the exact multiplicative
opportunity × efficiency Shapley identity.

Maximum identity gap:
**5.684e-14**.

### Interpretation

A static rule such as:
> "this player is volatile"

is not enough.

PRISM must understand:
> what state of this player is active **this week**?

Player identity is the history/playbook.
Current role and situation determine which part of that identity is relevant now.

This directly supports the governing PRISM architecture.

## 5. Opportunity compression is still visible inside the Week-5 major misses

Restrict the frozen Week-5 top-40 failure casebook to direct skill-player markets:
- rush yards;
- receiving yards;
- receptions.

There are **23** such cases.

Realized opportunity exceeded predicted opportunity in:

**22 / 23**.

For the **8 opportunity-dominant** direct skill cases:

**8 / 8** had realized opportunity above the pregame projection.

This is not a position-average statement.
These are named-player case failures.

## 6. Current role state often pointed above the production rule state — but the blunt fix remains CLOSED

Among those 8 opportunity-dominant direct skill cases:

- **6 / 8** had the strict-prior raw current player share above the share used
  by the production rules;
- median `current_share - rules_share`:
  **+0.07263** — about **7.26 percentage points**.

Examples:

| Player | Market | Pred opp | Actual opp | Current share - rules share |
|---|---:|---:|---:|---:|
| Jahmyr Gibbs | rush yards | 12.94 carries | 25 | +0.2523 |
| Braelon Allen | rush yards | 6.92 | 24 | +0.1393 |
| Aaron Jones | rush yards | 10.91 | 25 | +0.1535 |
| Malik Nabers | rec yards | 6.63 targets | 20 | +0.0063 |
| Puka Nacua | rec yards | 7.13 | 24 | +0.0271 |
| Ollie Gordon II | rush yards | 5.76 carries | 18 | +0.1182 |

Two opportunity-dominant cases did **not** have current share above rules:
George Pickens and Germie Bernard.

Therefore the lesson is **not**:
> replace rules/Bayes with raw recent share.

That exact obvious family is protected CLOSED:
- `BAYESIAN_CURRENT_STATE_TRANSMISSION_SYSTEMIC_MISMATCH_CONFIRMED`;
- Opportunity Authority Priority V1 subsequently **FAILED CLOSED**;
- do not rescue via RB-only / TE-only replacement, alternate Bayes weights,
  position exceptions, or retuned group strengths.

The new forensic point is narrower:

> In many major named-player misses, a faster current-role signal existed
> pregame, but PRISM still needs a validated mechanism for deciding when that
> state represents a real role transition rather than noise.

The already-qualified Stage-1 / Stage-2 PRISM science remains protected:
- active-role participation QUALIFIED ALL;
- conditional allocation QUALIFIED for RB rushing, RB receiving and TE receiving.

No existing allocation result is reopened here.

## 7. Opponent context existed, but case-level transmission remained weak at this rules field

All **23 / 23** direct skill cases had the relevant opponent defensive EPA
available in the Week-5 parity player-state trace.

Yet the relevant rules-layer efficiency multiplier was exactly **1.00** in:

**21 / 23** cases.

This does **not** prove the model ignored the opponent:
- other production paths consume opponent context;
- the prior landscape audit already confirmed opponent information is consumed
  somewhere in all required markets.

It does prove that, for these specific failure rows, the preserved rules-layer
multiplier usually transmitted **no net efficiency adjustment** despite opponent
state being available.

Do not turn this into a generic defense-vs-position multiplier.
The opponent/environment boundary audit remains binding.

A future matchup mechanism must be player-conditioned on the named player's
actual role/opportunity subtype.

## 8. Injury / teammate context cannot be reconstructed completely from the currently preserved forensic artifact set

The Week-5 parity player-state artifact contains:
- `injury_status` non-null in **0 / 40** frozen failure cases;
- `injury_designation` non-null in **0 / 40**.

This is an **evidence-preservation limitation**, not proof that Week-5 production
ignored injuries.

The canonical Week-5 production handoff records that the priced Full Slate
refreshed a bye-week-aware official NFL.com injury scope before availability and
certified all 30 scheduled teams.

However, the original canonical Week-5 run artifact `11585973062` is no longer
retrievable through the current GitHub artifact API, so this replay cannot
legitimately reconstruct exact row-level injury facts from that artifact now.

Therefore:
- do not invent teammate/injury narratives for these historical cases;
- do not treat blank parity fields as healthy;
- future PRISM pregame evidence should persist row-level availability and room
  state in the same immutable player trace used for postmortem.

## 9. Why this matters for the Dobbins-style model standard

The full model cannot stop at:

> "What is the average error for RB rushing?"

It must be able to explain, before kickoff:

1. who this player historically is;
2. what role is active now;
3. what changed in his room;
4. how much finite opportunity belongs to him;
5. what kind of opportunity he is likely to receive;
6. how this opponent interacts with **his** path to production;
7. how efficiently he converts it;
8. how reliable this named player's projection is;
9. which exact market that football state answers.

The Week-4/Week-5 repeat analysis shows why:
the same named player can be an opportunity miss one week and an efficiency miss
the next.

## 10. Anti-retest / integrity boundary

This replay was designed after Week-5 outcomes existed.

Therefore it is:
- valid forensic QA;
- valid current-season evidence about failure structure;
- not a preregistered prospective mechanism test;
- not permission to fit a new mean adjustment from Week 5;
- not permission to use top-miss examples as a tuned selector.

Protected:
- direct recent-share replacement remains CLOSED;
- opportunity-state disagreement uncertainty remains CLOSED;
- exact QB Stage-2 remains CLOSED;
- WR Stage-2 source block remains protected;
- role-elasticity V1 source block remains protected;
- generic DvP / simple matchup transmission remains closed;
- Week-6 locks remain immutable/ungraded;
- Weeks 7-10 WR dual reliability prospective confirmation remains intact.

Integrity:
- new fitted model: **0**;
- thresholds fit from Week-5 results: **0**;
- sportsbook predictor: **false**;
- OddsAPI calls: **0**;
- production changed: **false**.

Output digests:
- strict-prior replay rows:
  `sha256:da1324f65afea94fce0fe34e564b53396bed0602d7e9016b1b6e415306289ef5`;
- Week-5 situation case rows:
  `sha256:04b2f4faa064807481db37be8f443972e1763e0d6c8a52b40c538433cdcee9c4`;
- W4/W5 repeat-state rows:
  `sha256:02287dd6f4a55c3840891587aada95d9e4045f997ca8d5fd1e05c5506db74268`.

## 11. Forward implication

The immediate PRISM task is **not another generic aggregate correction**.

The next work should use these forensic results to test genuinely new,
source-legal state mechanisms that answer:

> When should a named player's current role/state be treated as a real structural
> change rather than temporary noise, while respecting teammate competition and
> finite opportunity?

That work must anti-retest against:
- failed raw recent-share replacement;
- blocked teammate-availability role-elasticity V1;
- qualified Stage-1 / Stage-2 allocation;
- closed generic matchup families.

Separately, future pregame runs should preserve an all-player situation trace so
that role, room, availability, opponent state, efficiency and reliability can be
audited on the exact named player without reconstructing them after the fact.

Aggregate MAE remains downstream QA only.
