# QB TEAM PASS INTENT — COMMON CRAWL ARCHIVE TRANSPORT V1I RESULT

Date: 2026-10-11

Status: **COMMON CRAWL ARCHIVE TRANSPORT RESCUES THE FROZEN V1H GATE. SOURCE-ONLY RESEARCH. FINAL V1 SOURCE QUALIFICATION NOT YET CLAIMED. NO PREDICTIVE QB CANDIDATE AUTHORIZED.**

Frozen plan:
- `docs/research/QB_TEAM_PASS_INTENT_COMMON_CRAWL_ARCHIVE_TRANSPORT_V1I_PLAN.md`

Canonical authority:
- workflow run: `38173788868`
- exact head: `2c038cdca11a322805744f73c4aeff8d4d11124a`
- artifact: `11716151694`
- artifact digest: `sha256:4464dafff9928d0bd04774df44b56efef02fe876e91595538a801edbdff17470`
- workflow conclusion: `SUCCESS`

## Frozen disposition

`COMMON_CRAWL_ARCHIVE_TRANSPORT_RESCUES_V1H_GATE`

## Result

Inherited repaired V1H baseline:
- unresolved rows: **158**
- direct-local transport-ready: **123**
- archive-needed: **35**
- unchanged combined transport gate: **>=127 / 158 (80%)**

Common Crawl archive result:
- archive-rescued rows: **23 / 35**
- combined direct-or-archive transport-ready: **146 / 158**
- combined rate: **92.4051%**
- margin over frozen 80% gate: **+19 rows**
- Common Crawl index HTTP calls: **295**

Archive rescue by franchise:
- DAL: **18 / 18**
- GB: **4 / 14**
- SEA: **1 / 3**

Rows still without direct or archive transport: **12**
- GB: 10
- SEA: 2

These 12 remain unresolved for final negative-disposition handling under the original V1 protocol; V1I does not finalize them.

## Gate readout

All frozen V1I gates passed:
- exact unresolved universe = 158;
- exact direct-ready baseline = 123;
- exact archive-needed rows = 35;
- combined transport-ready >=127: observed 146;
- frozen publisher map unchanged;
- every archive-ready record belongs to a frozen mapped publisher domain;
- no current publisher access-control bypass attempted;
- football outcomes read = 0;
- model residuals read = 0;
- sportsbook/game-market inputs used = false;
- paid OddsAPI calls = 0;
- 2026 Week-5 outcomes read = 0;
- football predictive models fit = 0;
- production changed = false.

## Interpretation

The QB public-intent source family is no longer blocked on historical transport.

The transport stack now supports:
1. official first-party direct archives (V1G);
2. frozen local-publisher direct archives where available (V1H);
3. frozen local-publisher Common Crawl archive capture discovery where direct access is unavailable (V1I).

This clears the original local-transport completeness threshold without:
- lowering the 80% gate;
- replacing publishers after seeing results;
- adding team-specific search hacks;
- contacting public search-engine HTML;
- bypassing publisher controls;
- using football outcomes or sportsbook data.

Transport qualification alone still does **not** authorize predictive QB modeling.

## Next legal frontier

Freeze the final source-qualification bridge.

It must:
1. preserve all 536 original team-week rows;
2. validate that V1G AUTO_REVIEW_READY official rows can be promoted to final V1 eligible-source status with acceptable semantic precision;
3. fetch and qualify local direct/archive candidates under the original pre-kickoff publication and source rules;
4. complete negative dispositions only after the frozen official + local transport sequence is exhausted;
5. apply the original parent V1 source-qualification gates unchanged;
6. still prohibit football outcomes, residuals, sportsbook data and predictive fitting.

Only if the original parent V1 source audit formally qualifies may a separately frozen predictive QB pregame-intent candidate be designed.

## Protected science

Unchanged:
- M89/M90;
- QB C2;
- generic QB mean/YPA reopening CLOSED;
- same-data attempts repackaging CLOSED;
- schedule/rest D1 CLOSED;
- PBP D2 CLOSED;
- sportsbook downstream only;
- no Week-5 prospective grading;
- production unchanged.
