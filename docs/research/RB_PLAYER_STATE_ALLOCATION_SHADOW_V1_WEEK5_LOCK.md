# RB Player State Allocation Shadow V1 — Week-5 Pregame Lock

**STATUS: IMMUTABLY FROZEN BEFORE WEEK-5 OUTCOMES**

Frozen contract:
`docs/research/RB_PLAYER_STATE_ALLOCATION_SHADOW_V1_CONTRACT.md`

Authority:
- branch: `research-rb-player-state-allocation-shadow-v1`
- run: `37560824479` — **SUCCESS**
- source SHA: `48f718b075b9f417145042c5cbb48ea6ba8beb2e`
- artifact: `11456916566`
- artifact digest: `sha256:edf51bbd93920ef0af580a0396422af3288cf511785be59deeea95c117062af4`
- canonical row CSV digest: `sha256:66a428f0c19ee1dc356db117fa8e39092204896356461358667224826826e90a`
- generated: `2026-10-07T02:12:40.776560+00:00`

Status:

`WEEK5_PREGAME_LOCK_FROZEN`

## Locked universe

- teams: **30**
- locked RB/HB/FB player identities: **98**
- teams with a non-zero shadow change: **30 / 30**
- parameters fit: **0**
- sportsbook inputs: **0**
- Week-5 outcomes read: **0**
- production changed: **false**

All locked team control and shadow shares conserve to 1.0 within floating-point tolerance.

## Control

`CONTROL_RECENT_CARRY`

For each locked team:
- take each locked player's strictly-prior last-three RB-room carry share;
- renormalize among locked players to 1.0.

## Shadow

`RB_PLAYER_STATE_ALLOC_V1`

For each locked team:
- `carry_state` = normalized recent RB-room carry share;
- `snap_state` = normalized recent RB-room offensive-snap fraction;
- `shadow_share = 0.50 * carry_state + 0.50 * snap_state`.

No coefficient was fit. The equal weight was frozen before Week-5 outcomes.

## Materiality of the pregame difference

Across teams:
- every team changes at least one player's predicted allocation;
- median team maximum absolute player-share change:
  **0.09455** (9.46 percentage points).

Examples of maximum player-level share changes:
- CLE: 17.53 points
- DEN: 17.02
- BUF: 16.20
- SF: 15.79
- SEA: 14.79
- LAC: 14.68
- BAL: 14.19
- ARI: 12.70
- TB: 12.66
- NYG: 12.58
- MIA: 12.04
- WAS: 12.03

Small-change rooms also remain represented:
- IND: 0.74 points
- HOU: 0.85
- ATL: 1.82

Nothing was filtered based on the size or direction of the shadow adjustment.

## Grading boundary

Do not grade until target-game results are available.

A team-game will fail closed if an RB/HB/FB who was not in the pregame locked set records a carry.

Week 5 is only prospective lock #1.

No scientific PASS/FAIL is permitted until at least:
- 4 distinct future locked weeks;
- 80 scoreable team-games;
- 200 scoreable RB player-games.

No weight, feature, row rule, or grading rule may change during accumulation.

## Interpretation

This shadow isolates **individual workload allocation**.

It does not change:
- projected team rush volume;
- QB/WR rushing;
- RB YPC;
- opponent rushing efficiency;
- rushing-yard distributions;
- receiving;
- RB Rush+Receiving V2 conservation;
- betting/pricing.

That isolation is intentional.

If this prospective player-state shadow beats recent-carry allocation, the next question becomes whether the improved room allocation can be integrated into the existing production stack without harming its validated team-volume, efficiency, and conservation science.

No production change is authorized by this lock.
