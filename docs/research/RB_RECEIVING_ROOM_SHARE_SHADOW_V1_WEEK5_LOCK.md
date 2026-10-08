# RB RECEIVING ROOM SHARE SHADOW V1 — WEEK-5 LOCK

Date: 2026-10-07  
Branch: `research-rb-receiving-room-share-week5-lock-v1`  
Successful run: `37705464974` — **SUCCESS**  
Successful head: `a9ecd2b41c1ae1843ebc2db6047235901897b49d`  
Artifact: `11518604971`  
Artifact digest: `sha256:2174af9aeb75100f29f9965f808be03d9d1d3d38d91366b539d7d25909cfe029`

## Status

`WEEK5_PREGAME_LOCK_FROZEN`

The exact no-fit RB receiving-room share rule that improved 2026 Weeks 1-4
individual RB receiving projections has now been frozen prospectively for Week 5.

## Frozen population

- Week: **5**
- scheduled RB rooms: **30**
- frozen RB identities: **98**
- strict-prior receiving history available: **98 / 98 (100%)**
- identity authority: frozen Week-5 RB player-state lock
- history bridge: exact GSIS identity
- parameters fit: **0**
- sportsbook inputs: **0**
- Week-5 outcomes read: **0**
- production changed: **false**

## Frozen receiving state and rule

Receiving state:

`prior_rb_room_share`

Rule:

> normalize each frozen RB's strict-prior `prior_rb_room_share` within the exact
> frozen Week-5 RB room.

No coefficient, threshold, subgroup, or history window was altered after the
Weeks 1-4 retrospective result.

`retrospective_result_used_to_change_rule = false`

## Provenance

Parent live player-state authority:
- run `37560311001`
- artifact `11456556226`
- digest `sha256:a39b958e492a781e310de0f14d34153e21ca589a76cd226479b3b39e10f9328e`

Parent frozen RB identity lock:
- run `37560824479`
- artifact `11456916566`
- digest `sha256:edf51bbd93920ef0af580a0396422af3288cf511785be59deeea95c117062af4`

Retrospective W1-W4 impact authority:
- run `37703522415`

## Identity repair during lock construction

The first lock attempts correctly failed closed because the generic historical
roster universe did not contain seven identities from the already-certified
live Week-5 RB universe.

A subsequent name-based receiving-history bridge exposed six suffix/alias
misses despite those players having real 2026 history.

The final lock did **not** drop, replace, or fabricate any player.

Instead:
- retained the exact 98-player frozen live RB universe;
- used each player's frozen GSIS ID as the primary receiving-history bridge;
- added a regression safeguard proving exact target-week history cannot be
  consumed.

This is an identity-source repair only. It does not alter the scientific
receiving-share mechanism.

## Conservation and evidence hashes

- rooms locked: **30**
- maximum candidate room-share conservation gap:
  `2.220446049250313e-16`
- median candidate room HHI: **0.39213989267197824**
- canonical row CSV digest:
  `sha256:4c5d325a552673d64c91e21b812dd05f5b15adb28c479640e382d4b4baf4ffd7`
- team summary digest:
  `sha256:13d0fc3cbf3780350dcc8f7f748666727908d69d7d7dafc20d3d43b62dbc06cc`

Strict repository audit: **PASS**

## Interpretation boundary

The evidence now separates cleanly:

### Weeks 1-4
Retrospective mechanism-impact evidence.

Confirmed individual-player improvements:
- target MAE: **7.60%**
- receptions MAE: **3.83%**
- receiving-yards MAE: **2.89%**
- rush+receiving-yards MAE: **1.66%**
- rush-yards projections: **exactly unchanged**

### Week 5
Prospective untouched lock.

Week-5 results may grade this exact frozen rule later, but no Week-5 outcome was
used to construct, alter, or select the lock.

No automatic production promotion is authorized by this document.
