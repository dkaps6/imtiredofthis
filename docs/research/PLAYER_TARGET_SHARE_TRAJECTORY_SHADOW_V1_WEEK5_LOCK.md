# Player Target Share Trajectory Shadow V1 — Week-5 Pregame Lock

**STATUS: IMMUTABLY FROZEN BEFORE WEEK-5 OUTCOMES**

Frozen contract:
`docs/research/PLAYER_TARGET_SHARE_TRAJECTORY_SHADOW_V1_CONTRACT.md`

## Certified authority

- branch: `research-player-target-share-trajectory-shadow-v1`
- corrected lock run: `37654382316` — **SUCCESS**
- job: `112905655806`
- source SHA: `e2c57e17c1df95130b70d3a3926b090ccd26e847`
- artifact: `11497153776`
- artifact digest: `sha256:c12878ed97a4543a907ac54f4cbadadda7178db068491373091d2c048f176e88`
- canonical row CSV digest: `sha256:afbfd7f360c50fcd4850c0967be40f9a333da1bd5f835cdc676c2e88d777c1f3`

The preceding run `37654110378` produced the same canonical row CSV digest. The corrected run changed only the reporting scope of the identity-coverage metadata from the entire football universe to the locked WR/TE cohort; the immutable scientific rows and entitlement values did not change.

## Lock status

`WEEK5_PREGAME_LOCK_FROZEN`

Target:
- season 2026
- Week 5
- target outcomes read: 0
- sportsbook inputs used: 0
- fitted parameters: 0
- production changed: false
- same/future feature violations: 0

## Locked player universe

- 277 WR/TE rows
- WR: 167
- TE: 110
- stable-ID coverage on locked WR/TE cohort: **83.0325%**
- trajectory feature available: 230 / 277 = **83.0325%**

Players without a valid trajectory feature are retained with zero trajectory adjustment under the frozen fail-safe rule.

## Frozen transformation

Parent:
- current promoted M38 -> TE-R5P -> WR-R15 entitlement.

Player state:
- RECENT2 target share minus EARLIER same-season same-team target share.

No coefficient is fit:

`weight_i = baseline_entitlement_i * exp(trajectory_delta_i)`

Then renormalize inside the protected room.

TE:
- preserve exact TE-room pool.

WR:
- preserve exact M38 WR1 anchor;
- redistribute only WR2+;
- preserve exact WR2+ pool and total WR room.

Other positions:
- unchanged.

## Pregame materiality

- changed players: **247**
- changed protected rooms: **60**
- median absolute player entitlement change: **0.0018360**
  - about **0.184 target-share percentage points**
- maximum absolute player entitlement change: **0.0257566**
  - about **2.576 target-share percentage points**

This is intentionally a conservative redistribution: the confirmed player-state signal changes individual allocation without creating new team target volume.

## Conservation / integrity

- max room-pool gap: `5.551115123125783e-17`
- max team modeled target-mass gap: `1.1102230246251565e-16`
- TE parent model: `TE_R5P_PRODUCTION_MODEL_V1`
- WR parent model: `WR_R15_PRODUCTION_MODEL_V1`

Therefore the shadow does **not**:
- change team pass volume;
- change total modeled team target mass;
- change the M38 WR1 anchor;
- change TE-room mass;
- change WR2+ pool mass;
- change receiving efficiency;
- change QB science;
- change production.

## Scientific boundary

Week 5 is prospective lock #1.

No final PASS/FAIL is permitted until at least:
- 4 future locked weeks;
- 400 scoreable WR/TE player-games;
- 120 distinct WR/TE identities;
- 80 scoreable team-position rooms.

The formula, scale, window, eligibility rules, room definition, and grading rules are frozen during accumulation.

No rescue after target outcomes with:
- alternate exponent scale;
- recent1/recent3/recent4;
- clipping;
- rising-only or falling-only routing;
- WR-only / TE-only rescue;
- WR1 unfreezing;
- sportsbook conditioning.

No production change is authorized by this lock.
