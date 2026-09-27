# RB-PD2 Week-3 Prospective Lock V1 — Operational Result

**STATUS: VALID PROSPECTIVE WEEK-3 LOCK RECOVERED FROM EXACT PREGAME CAPTURE. RESEARCH ONLY. NO PRODUCTION CHANGE.**

## Source Full Slate / exact capture

Authorized paid Full Slate:
- run: `36330757181`
- production SHA: `37deff5b51b5ec48117556b91d12dba1d4ca1fad`
- live odds explicitly authorized
- `rb_pd2_shadow_capture=true`

The canonical pricing step later failed mechanically, so this run is **not** a replacement betting-board authority.

The research-only same-process shadow hook nevertheless completed before the pricing failure and wrote an immutable exact baseline capture:
- capture session: `2026-20260927T155118Z-1a6bfca1`
- capture rows: **54**
- exact empirical draws: **25,000 per captured player**
- outcome present at capture: **false**
- sportsbook inputs used in candidate: **false**
- production output mutated: **false**
- source capture artifact: `10934814338`
- source capture artifact digest: `sha256:f4f215eaa1ba4fd6931dd327300599e4052cf984ba0f887d0bd1755c2c0a16ba`

## Mechanical failures discovered

Two operational defects prevented the original Full Slate run from producing a valid lock directly.

### 1. History restore was skipped after pricing failure

The Full Slate history-restore step had a normal success-gated `if` expression while the downstream lock step used `always()`.

Therefore:
- pricing step failed;
- GitHub implicitly skipped the history restore;
- the lock assembler ran but correctly failed closed because history was absent.

This is workflow control-flow plumbing, not football/model science.

### 2. Frozen history manifest digest was not CSV-roundtrip reproducible

Pinned history authority:
- run: `35732782075`
- artifact ID: `10696312599`
- artifact digest: `sha256:1205ea152b4702e19d12379c79e29ff7fc20fd2a632a61f7c519980f64b648fb`
- state rows: **1,608**
- completed 2026 history: exactly **Weeks 1-2**
- zero prospective outcomes used: **true**

The producer recorded:
`history_state_sha256=f79f67dc8b36bc481638c648e33dd690cb89ef9d7e7a2c17c78fa2940a88dd0f`

That value was computed from the pre-CSV in-memory DataFrame. The later live consumer reads the persisted CSV and recomputes the same logical digest. Its reproducible parsed-CSV digest is:
`3cc7f7c87b9b267db19a1bef9a4a52c9aa4c2ffc731bc846b135dedb7de78542`

No history row changed. The original manifest was retained. Mechanical recovery created a derived manifest that changes only the reproducibility digest and records:
- original manifest digest;
- exact source artifact ID/digest;
- `history_rows_changed=false`;
- `science_fields_changed=false`.

## Authoritative mechanical recovery

Recovery run:
- `36331514633` = **SUCCESS**
- artifact: `10935529140`
- artifact digest:
  `sha256:94f168125dc889b6a747da5f3b3e3829d2fd4bf7a9b6b3d9f4d2d79ac0b59b6e`

The recovery did **not** regenerate production draws or make another OddsAPI request.

It used:
1. the exact same-process empirical baseline capture from Full Slate `36330757181`;
2. the exact pinned frozen history artifact;
3. the unchanged frozen candidate transform and lock assembler.

## Valid Week-3 prospective population

Recovered receipt:
- `valid=true`
- target season/week: **2026 Week 3**
- history completed through: **Week 2**
- capture rows: **54**
- locked rows: **46**
- population exclusions: **8**
- integrity failures: **0**
- production changed: **false**
- sportsbook inputs used in candidate: **false**
- outcome present at lock: **false**

All 8 exclusions are frozen scientific-population exclusions for insufficient prior games:
- Emmett Johnson
- Sione Vaki
- Jadarian Price
- Mike Washington Jr.
- Travis Etienne Jr.
- Jeremiyah Love
- Kaelon Black
- Kenny Gainwell

Candidate application:
- locked rows with width multiplier > 1.00: **34 / 46**
- maximum frozen width multiplier: **1.2942418426x**

## Pregame timing proof

Exact persisted-lock timestamp:
`2026-09-27T15:59:26.120129Z` = **11:59:26 AM ET**

Earliest locked-game kickoff:
`2026-09-27T17:00:00Z` = **1:00 PM ET**

Minimum actual persistence buffer:
**60.56 minutes**

Frozen minimum required buffer:
**15 minutes**

Therefore the recovered artifact is prospectively locked with substantial pre-kickoff headroom.

## Scientific consequence

Week 3 is the **first valid prospective observation week** for the already-frozen RB-PD2 yard-difficulty MC-width forward confirmation.

It does not authorize any production change.

The scientific PASS/FAIL gate still requires:
- at least **8 distinct prospectively locked weeks**; and
- at least **400 unique eligible locked RB/HB/FB rushing-yard player-games**.

Week-3 outcomes may be joined only after completion to these immutable pregame locks.

No coefficient, onset percentile, width cap, population definition, metric, or threshold may be changed after observing Week-3 results.

## Permanent mechanical repair in this PR

This PR also repairs future collection mechanics without changing science:

1. Full Slate history restore uses `always()` so a downstream pricing failure cannot suppress the independent research-history restore after an exact shadow capture.
2. The history producer hashes the **persisted CSV round-trip representation** that the consumer actually reads.
3. The manifest also records a raw history-state file SHA-256.
4. Tests require the persisted state to reproduce the manifest digest.

No football mean, carry allocation, YPC, simulation distribution, width formula, sportsbook logic, or production recommendation is changed.
