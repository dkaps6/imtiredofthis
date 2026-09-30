# NFL HANDOFF — 2026-09-30 — WR/CB ARCHIVE BREAKTHROUGH / SOURCE INTEGRITY CURRENT

GitHub is canonical. This handoff is designed so the next chat can resume without relying on chat memory.

## 0. READ / EXECUTION CONTRACT

Start with:
1. `AGENTS.md`
2. newest top checkpoint in root `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. **this file**
4. Issue #535 comments from **5913566949 onward**, especially:
   - 5913983548
   - 5914141956
   - 5915023985
   - 5915207422
   - 5918591205
   - 5918732129
   - 5918857679
   - anything newer
5. query GitHub live for `main`, PR #663, PR #665, PR #662, branch `repair-specialist-rng-isolation-v1`, branch `research-wr-cb-historical-snapshot-discovery-v1`, and Actions.

Do **not** recursively read older handoffs. Do not restart closed science. When comments conflict, branch code + frozen result docs + successful Actions are authoritative for that lane.

---

## 1. LIVE CANONICAL STATE AT HANDOFF

### main
At handoff preparation, live main:
`50f6624be68fcac3af88fe40a258b43eec71402c`

This is a docs/continuity descendant of production code with the retired WR coverage heuristic already removed. Verify live before editing.

### PR #665 — free WR/CB historical source audit
Draft/open:
- branch: `research-wr-cb-free-archive-v1`
- live head at handoff: `1a39b519d472d68a10096f2d9faa992d69e27d60`
- latest integrated Repo CI: **SUCCESS**
- latest integrated WR-CB source audit: **SUCCESS**

Do not merge or promote just because CI is green. The source/model gate is still **CLOSED**.

### Separate archive recovery branch
`research-wr-cb-historical-snapshot-discovery-v1`

Live head at handoff:
`ca29322430a2600eef195ca7fc2704c6e94fe13c`

Latest message:
`Research: fix exact 2022W5 archived source URL key`

This branch is the active no-paid historical point-in-time recovery lane. It is intentionally separate from #665 to avoid racing source-quality code.

### PR #663 — CLV / market snapshot history
Draft/open:
- head `2a45bc79887d83a2172520efbb3f4fef3bc0242b`
- Repo CI `36659377992` = **SUCCESS**
- replay workflow remains red because its pinned Week-2 source artifact expired; do not substitute Week 3 or weaken the frozen replay contract.
- same-book <=30-minutes-before-kickoff is the only valid CLV label.
- no paid OddsAPI pull solely for CLV without explicit approval.

### Specialist RNG repair
Branch:
`repair-specialist-rng-isolation-v1@30333ec2b853d5ba7bd420b74d63c53f097b5fff`

Run:
`36633421579` = **SUCCESS**

Disposition:
`RNG_ISOLATION_PRODUCTION_CANDIDATE_PASS`

It passed the exact frozen validation, but was **not production-promoted**. Do not casually merge/relabel it.

### PR #662 — GSIS research contract
Draft/open:
- head `960f1e4ad250ec910280e44e7c4a8362422d5c01`
- title: `Research: add safe GSIS point-in-time archive contract`

Raw/private GSIS data must never be uploaded to the public repo.

---

## 2. WEEK 3 / WEEKS 1-3 — CLOSED SCIENCE, DO NOT REOPEN

Canonical postmortem:
- branch `research-week3-postmortem-execution-v1`
- findings commit `acfd97b41e107c12a96cb250220a9a7cba739ae7`
- run `36622608145` = **SUCCESS**
- artifact `11059171429`
- artifact digest `sha256:5e05afca1f94afaa04a4da6f93fa396714df2234d283cc75302b87eac97d38a6`

Settled record:
- Week 1: **204-205, -20.92u**
- Week 2: **192-198, -22.06u**
- Week 3: **233-208, +2.26u**
- combined: **1,260 selected rows / 1,240 decided / 20 voids / 629-611 / 50.7% / -40.72u**

Already-known severe calibration problem:
- 70-100% stated-probability band: 502 bets
- mean stated probability 81.0%
- realized 51.8%
- gap -29.2 percentage points
- 54 tested sliced hypotheses; **zero** survived game-cluster-aware BH-FDR q=0.10.

No Week-3 outcome-driven threshold tuning, carveout, or retroactive model repair is authorized.

### Claude DNP reconciliation — CLOSED, no new void defect
Claude initially inferred 6 additional Week-3 voids because Elijah Arroyo, Erick All, and Blake Whiteheart were absent from nflverse weekly player-stat rows.

That inference was disproven by the canonical settlement contract:
- weekly stats can omit players with zero box-score usage;
- grader also uses snap-count participation;
- preserved ledger has these players `ACT`, `snap_participated=True`, `actual_source=snap_confirmed_verified_zero`;
- four true DNP players were properly `INA` / `snap_participated=False` / sportsbook void.

Keep canonical **20 voids** and Week-3 **233-208 +2.26u**. Do not regrade.

---

## 3. WR/CB SOURCE FRONTIER — WHAT CHANGED

Production remains fail-closed for unsupported player-level WR/CB assignment. The old static `coverage_penalty()` / 0.92/0.94/1.06/1.04 heuristic was retired in merged PR #664 and must **never be restored**.

FantasyAlarm public reports are **editor-projected pregame WR-CB alignments**, not observed route-by-route coverage responsibility. Their editorial matchup labels are not eligible football-model features.

### Initial public archive inventory
Current public source corpus on #665:
- 71 season-week pages
- 4,459 factual observed pairing rows
- 2021: 2/18 weeks
- 2022: 14/18
- 2023: 17/18
- 2024: 17/18
- 2025: 18/18
- 2026: W1-W3
- older slot coverage can be selective; missing rows are **missing**, never zero/no matchup.

### Article modification provenance
A key source-quality discovery: article publish time alone does not prove the current factual table is the same as the pregame table.

Metadata audit found:
- 4,459 rows total
- 660 rows from pages whose recorded modification time was after game kickoff
- earlier 4,063 provisional-ready diagnostic contained 479 such rows
- post-provider-reuse repair: 4,034 provisional-ready
- post-CB weekly-opponent guard: **3,999 provisional-ready**
- among those 3,999:
  - **3,526** only have modification timestamps compatible with pregame; still NOT snapshot proof
  - **473** are from pages with post-kickoff modification metadata
  - **0** rows in the normal live-page audit are independently proven original pregame snapshots.

The source/model gate therefore remains **CLOSED**.

---

## 4. WR/CB SOURCE IDENTITY BUGS FOUND AND FIXED IN #665

### A. Provider-ID reuse defect
Claude's provider-ID census surfaced provider IDs reused across different players.

Concrete reproduced defect:
- FantasyAlarm WR provider ID `300936`
- Marquise Goodwin anchored to GSIS `00-0030068`
- same provider ID later appears for Allen Robinson II
- old bridge propagated Goodwin's GSIS ID onto **9** Allen Robinson 2022 rows
- **6** had counted provisionally source-ready.

Fix:
- commit `abd83eea9823848ae1eb5d798bf9e44b912c4d62`
- reused / multi-person provider IDs veto bridge assignment;
- composite corner assignments cannot certify one stable identity;
- independent exact roster matches remain intact.

Validation:
- Repo CI `36734473138` = SUCCESS
- source audit `36734472417` = SUCCESS
- artifact `11106407106`
- digest `sha256:a6a1bfde25f3587a235e63f7c78a9d051c7a690429306ab731c7be6a66db2077`

Effect:
- 4,063 -> **4,034** provisional-ready
- 29 uncertain bridge-dependent rows removed.

Result doc:
`docs/research/WR_CB_PROVIDER_ID_REUSE_GUARD_V1_RESULT.md`

### B. Wrong-opponent CB bridge defect
Independent same-week defensive roster cross-audit confirmed bridge rows could assign a real CB to the wrong opponent.

Lower-bound artifact audit first found:
- 10 demonstrably contradicted currently-ready rows
- all 10 bridge rows.

Full strict weekly defensive-roster guard then measured:
- 4,034 -> **3,999** provisional-ready
- **35** newly quarantined
  - **33** defender GSIS IDs on another team's same-week defensive roster
  - **2** absent from weekly defensive roster -> UNKNOWN, not declared wrong.

Integrated #665 repair:
- commit `2f09c696592aa05718ed1f858bea6a4c0eceb88d`
- docs-only descendant/head at handoff `1a39b519d472d68a10096f2d9faa992d69e27d60`.

Validation:
- full guard run `36741822792` = SUCCESS
- integrated Repo CI `36742368817` = SUCCESS
- integrated source audit `36742368919` = SUCCESS.

Result doc:
`docs/research/WR_CB_WEEK_OPPONENT_ROSTER_GUARD_V1_RESULT.md`

Do not undo either guard to recover row count.

---

## 5. HISTORICAL POINT-IN-TIME ARCHIVE BREAKTHROUGH

This is the biggest scientific/source update.

Broad live-page timestamps were insufficient, but exact public Wayback captures can prove point-in-time source contents. A parser was developed for archived bodies, including JSON-LD `articleBody` when rendered replay HTML lacks tables.

### 2024 Week 1 — first verified historical pregame recovery
Wayback capture:
- `2024-09-07 00:57:14 UTC`

Hashes:
- replay body SHA-256:
  `0854732b5af783b1d67fd64bcf6598a86c03a0f3669e70e4399b903c441dc33f`
- JSON-LD articleBody SHA-256:
  `85998a169935efd81dece50d0d162d41d4f88c58bfdd18d1239b2afbe497680d`

Run:
- `36768720380` = SUCCESS
- artifact `11122412408`
- artifact digest `sha256:19986b8ff1890ca0d507a029d854d400d69abd8c3edc7a2affa632def6bf8a8f`

Rows:
- 66 explicit factual pairings recovered
- 57 still pregame after per-player kickoff isolation
- **46** strict exact-week WR-team + opponent-CB roster rows
- 11 quarantine
- 3 WR identity failures
- 9 CB-opponent identity failures (one overlap)
- 0 schedule mismatches.

Durable lock:
`data/research/wr_cb_verified_snapshot_lock_2024w01_v1.csv`

Lock SHA-256:
`99f37f47e9f9b3ccdbbb76175ee86c944fdcd17ed6e75fce48e7bb71dede6dd9`

Result doc:
`docs/research/WR_CB_2024W1_VERIFIED_ARCHIVE_RESULT.md`

Important drift cross-audit:
- 57/57 archived pregame pairing keys persisted on later/current page;
- 46/46 strict rows too.
Thus post-kickoff `dateModified` is a **risk flag**, not proof the pairing table changed.

### Protected 2025 Week 14 — independently verified confirmation-source lock
This is SOURCE/PROVENANCE QA only. **Do not access or use 2025 target outcomes yet.**

Capture:
- `20251206131409`

Exact CDX digest:
- `2VNA2WFCUG3X347ZKOY5MGW7Z4BEEFQN`

Run:
- `36770684037` = SUCCESS
- artifact `11123566159`

Hashes:
- replay body SHA-256:
  `e6f0bc1bd5a676f542ab086ed4aecdc5dba2dd30579a9ec9f59c5223ff134816`
- JSON-LD articleBody SHA-256:
  `03487ac0eb907191e1d68b2cda5cc125aa6e1e4ec4b99705f511c6be0f62402d`

Rows:
- 60 factual rows
- 56 still pregame
- **47** strict exact-week two-sided roster rows.

Verifier explicitly records:
- `confirmation_outcomes_accessed=false`
- `target_game_outcomes=false`
- `parameters_fit=0`
- `editorial_grade_used=false`.

Durable protected lock:
`data/research/wr_cb_verified_snapshot_lock_2025w14_v1.csv`

Lock SHA-256:
`c5b4875aa1d78fefeffb616b9a0684320d29c64babaaa4db004bcf3867509ed2`

Both 2024W1 and 2025W14 lock/result checkpoint committed on archive branch:
`500c2135dfe80a85e5ffa8337daec921ad8826cf`

Never edit lock files in place. Any correction = new version + new hash.

### Broad Wayback scanning warning
A broad 2024-2025 exact-URL CDX scan run `36769614598` suffered heavy ReadTimeout / ConnectTimeout / ConnectionError / HTTP 503.

It returned zero usable candidates, BUT known-positive 2024 W1 itself timed out in that scan. Therefore:
- **ACCESS-INCONCLUSIVE**
- NOT a zero-snapshot result
- do not hammer broad CDX endpoints
- use sparse exact discovery with known-positive control.

Frozen result:
`docs/research/WR_CB_WAYBACK_2024_2025_INDEX_EXPANSION_V1_RESULT.md`
commit `a4a7748db782b5555c9c655538fa1c8189b4ae4a`.

---

## 6. LATEST FRONTIER HIDDEN BY CHAT TIMEOUT — 2022 WEEK 5 VERIFIED

This happened after the last Issue #535 archive milestone comment and MUST be carried forward.

Latest archive branch:
`research-wr-cb-historical-snapshot-discovery-v1@ca29322430a2600eef195ca7fc2704c6e94fe13c`

Latest successful workflow:
- name: `Verify 2022W5 full-week WR-CB Wayback body V1`
- run `36773818193` = **SUCCESS**
- job `110086587401` = **SUCCESS**
- artifact `11124612896`
- artifact ZIP digest:
  `sha256:d84d594af003a118a5e06560bea678b2bd61ee926be30b180f638c5497cdda54`

Exact source:
`https://www.fantasyalarm.com/articles/nfl/wide-receivers/2022-fantasy-football-wr-cb-match-up-report-week-5-tyreek-hill-to-burn-the-jets-in-week-5/134887`

Archive timestamp:
- `2022-10-05 19:51:42 UTC`

Exact replay URL:
`https://web.archive.org/web/20221005195142id_/https://www.fantasyalarm.com/articles/nfl/wide-receivers/2022-fantasy-football-wr-cb-match-up-report-week-5-tyreek-hill-to-burn-the-jets-in-week-5/134887`

Source publication:
- `2022-10-05 16:29:08 UTC`

CDX:
- status `EXACT_CDX_DIGEST`
- digest `HKCXAO5CXTWGRFTPRIDA4OBCTJLANM7F`

Body SHA-256:
`f9db298ffe8ae369acc2753897a6391e579f29b96bab7008ac803186ad9d3d5e`

Result:
- status `ARCHIVE_BODY_HASH_MATCH_WITH_STRICT_DISCOVERY_SOURCE_ROWS`
- parse source `ARCHIVED_OUTER_HTML`
- 54 parsed factual pairing rows
- **41 verified pregame factual rows**
- WR exact-week rows: 38
- CB opponent exact-week rows: 39
- **36 strict exact-week identity rows**
- 5 quarantined rows
- archived digest verified = true
- target game outcomes = false
- confirmation outcomes accessed = false
- parameters fit = 0
- editorial grade used = false
- source/model gate = false.

This is **source-discovery evidence**, not a promoted feature and not yet a locked 2022 training dataset.

### Immediate next action on this lane
1. Freeze a durable 2022W5 result document + sanitized immutable source lock from the successful artifact/run if not already present when next chat starts.
2. Hash that lock and never mutate in place.
3. Continue sparse exact historical archive discovery, prioritizing DISCOVERY years first.
4. Keep protected 2025 outcome data untouched.
5. No model fitting until a preregistered source population/coverage/semantics contract is frozen.

---

## 7. PROSPECTIVE 2026 CAPTURE

A point-in-time prospective capture path exists on #665:
- `scripts/research/capture_fantasyalarm_wr_cb_pregame_v1.py`
- strict exact domain/article season/week validation
- network fetch start/end timestamps
- rowwise real kickoff/schedule isolation
- exact source SHA-256
- editorial grade excluded from factual lock
- no model eligibility on capture.

The active PR workflow supports one explicitly requested source URL via:
`data/research/wr_cb_pregame_capture_request_v1.csv`

Default is header-only -> **no request**.

At last validated empty-request run:
- Repo CI `36723815933` = SUCCESS
- source audit `36723816125` = SUCCESS
- log explicitly: `No explicitly requested prospective article. No fetch.`

As of the latest research checkpoint, public search had found a 2026 Week-4 DFS WR article but **not** a verified Week-4 WR/CB report. Do not guess a URL. If a genuine Week-4+ WR/CB report appears before kickoff, capture it prospectively and freeze the sanitized fact lock/hash before the applicable game.

---

## 8. CLAUDE COORDINATION STATE

Issue #535 is the communication bus.

Claude has:
- retracted the incorrect claim that his partial DK subset was the first Week-3 live calibration confirmation;
- accepted canonical Week-3 postmortem;
- delivered provider-ID collision census;
- surfaced the useful provider reuse mechanism;
- DNP inference closed after snap-count proof;
- archive parser/source-lock lane should NOT be duplicated by Claude;
- CB census/settlement lane is effectively closed unless a new discrepancy appears.

Latest relevant coordination comments listed in section 0.

If Claude posts something newer:
- inspect it,
- cross-audit mechanically,
- fix real source/system defects,
- do not duplicate already integrated repairs,
- do not let Claude reopen closed W3 science.

---

## 9. OTHER CLOSED / HOLD LANES TO PRESERVE

- RB-PD2: HOLD, 1/8 weeks, 46/400 rows; do not declare pass/fail early.
- RB vacancy / public intent / receiving semantics Week-3 work remains observational/frozen as previously recorded.
- projection-authority line crossing primary hypothesis failed; same-side strength direction remains forward-only Week-4+ observation.
- no QB carveout, RB-only rescue, top-N special rule, share threshold hack, depth-chart hack, evidence-state retuning, or Bayesian outcome retuning from Week 3.
- no paid WR/CB data.
- no paid OddsAPI without explicit user approval.
- do not restore `coverage_penalty()`.
- do not treat FantasyAlarm editorial Safe/Moderate/Risky or Upgrade/Neutral/Downgrade as football features.
- do not infer missing historical pairings as zero exposure.

---

## 10. EXACT NEXT EXECUTION ORDER

1. Query live GitHub state before editing:
   - main
   - Issue #535 newest comments
   - PR #665
   - archive branch `research-wr-cb-historical-snapshot-discovery-v1`
   - PR #663
   - PR #662
   - RNG branch
   - Actions.
2. On archive branch, preserve the successful **2022W5** run as a result doc and immutable sanitized source lock if still missing.
3. Keep expanding historical source availability **sparsely**, not by broad Wayback hammering. Known-positive controls matter because CDX has false-negative access failures.
4. Discovery/source years first. Protected 2025 remains source-only; do NOT inspect target outcomes until the source population and preregistered test are frozen.
5. Once enough discovery-source weeks are independently locked, freeze the scientific design BEFORE joining outcomes:
   - exact eligible population,
   - missingness semantics,
   - projected-alignment interpretation,
   - strict prior CB-effect construction,
   - baseline/opportunity control,
   - discovery vs untouched confirmation split,
   - cluster-aware inference / stopping rule.
6. Capture current 2026 WR/CB pages prospectively if/when a verified URL appears before kickoff.
7. Separately keep #663 and RNG/GSIS lanes honest; do not conflate their status with WR/CB source progress.
8. Do not merge #665 merely because source code is cleaner; source/model gate is still CLOSED.

---

## 11. CURRENT SCIENTIFIC INTERPRETATION

We have moved from:
`NO_FREE_HISTORICAL_WR_CB_SOURCE_KNOWN`

to:
`FREE_PUBLIC_PROJECTED_WR_CB_ARCHIVE_EXISTS_AND_POINT_IN_TIME_RECOVERY_IS_PROVEN`

but **not** to:
`TRUE_OBSERVED_COVERAGE_DATA_AVAILABLE`
or
`WR_CB_FEATURE_PROMOTED`.

What is now proven:
- free public pregame projected WR-CB pairing snapshots can be independently recovered and hashed;
- strict per-game prekickoff isolation is possible;
- strict weekly WR/opponent-CB roster identity gates catch real bad rows;
- at least 2022W5, 2024W1 and protected 2025W14 have successful point-in-time recovery evidence.

What is not proven:
- that projected pairing equals actual route responsibility;
- complete historical coverage;
- slot completeness;
- predictive incremental value;
- production model benefit;
- protected 2025 confirmation performance.

That is the exact frontier.

