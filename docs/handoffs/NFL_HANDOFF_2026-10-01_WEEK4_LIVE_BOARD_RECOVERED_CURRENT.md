# NFL HANDOFF — 2026-10-01 — WEEK 4 LIVE BOARD RECOVERED / GAME-DAY CURRENT

**GitHub is canonical. Chat memory is secondary.**

This is the authoritative current execution handoff for the NFL project as of the end of the 2026-10-01 game-day live-odds work.

Do **not** recursively load older handoffs. Use the read order at the bottom.

---

## 1. CURRENT PRODUCTION / MAIN

Canonical repo:

`dkaps6/imtiredofthis`

Main immediately before this continuity commit:

`0e6d8ab63c395b6bb9e88b0d55844083ac3221a5`

That main includes:

- validated RNG-isolation candidate code from merged PR #666 at `2483c5a5b787089a120c6b5a258233d33600a670`;
- ESPN availability semantics repair from merged PR #670 at `ab16d90d55c92150bde358b0a28f292c854cd222`;
- Week-4 unresolved backup-QB final-board quarantine from merged PR #671 at `0e6d8ab63c395b6bb9e88b0d55844083ac3221a5`.

Important RNG boundary:

- PR #666 landed the validated candidate implementation / verifier / tests.
- It did **not** silently switch canonical Full Slate routing to the new RNG-isolation candidate.
- Do not assume the RNG repair is activated in production merely because the candidate code is on main.

Latest clean no-live production after #671:

- Full Slate run `36936244158` = **SUCCESS**
- Repo CI `36936244165` = **SUCCESS**
- event = push
- head = `0e6d8ab63c395b6bb9e88b0d55844083ac3221a5`
- push mode means `FETCH_LIVE_ODDS=false`.

---

## 2. WEEK 4 LIVE ODDS — AUTHORIZED PAID RUN

The user explicitly authorized a live-odds Full Slate run on 2026-10-01.

One-shot dispatcher:

- commit `b8e233192033a8fa44742b7c069ad2a42968ceec`
- dispatcher run `36934550541` = **SUCCESS**
- this launched canonical `.github/workflows/full-slate.yml` with `fetch_live_odds=true`.

Paid canonical Full Slate:

- run `36934563481`
- event = `workflow_dispatch`
- head = `b8e233192033a8fa44742b7c069ad2a42968ceec`
- conclusion = **FAILURE**, but **only after the live sportsbook acquisition completed successfully**
- artifact = `11197726111`
- artifact name = `run_36934563481`
- artifact digest = `sha256:202aa5205e71ad0acedf1910f505c2b362f564d8593a0f82dd0928fb45175aae`
- artifact remains the exact paid Week-4 source authority.

### What succeeded before the failure

Live odds acquisition/gating succeeded.

The paid run produced:

- 16 game identities;
- 392 unique sportsbook players in compact model-facing offers;
- 879 model-facing compact rows;
- 149 quarantined sportsbook rows;
- 1,740 exact bookmaker-line pricing offers;
- 3,480 side rows after pricing;
- live market families:
  - player_pass_yds
  - player_reception_yds
  - player_receptions
  - player_rush_reception_yds
  - player_rush_yds
  - player_anytime_td

The football model remained upstream / sportsbook-independent.

The live compact boundary reported:

- player_anytime_td: 392 rows
- player_pass_yds: 32
- player_reception_yds: 166
- player_receptions: 162
- player_rush_reception_yds: 37
- player_rush_yds: 90

Pricing offer adapter reported:

- 1,740 book-line rows
- 3,480 side rows
- duplicate book-line rows = 0.

### Exact paid-run failure

The run failed in QB-C2 pricing-lineage stamping, not during odds fetch.

Exact exception:

`RuntimeError: priced pass-yard rows missing QB C2 audit identity`

The unresolved live pass-yard offer identities were:

- CHI — Tyson Bagent
- WAS — Marcus Mariota

The football-only C2 starter authority for those teams remained:

- CHI — Caleb Williams
- WAS — Jayden Daniels

This is the correct fail-closed behavior: sportsbook offers must **not** choose or overwrite the upstream football starter.

Do not “fix” this by letting the sportsbook define the QB starter.

---

## 3. WEEK 4 LIVE BOARD — RECOVERED WITHOUT A SECOND ODDS FETCH

The exact paid artifact was recovered offline.

Successful recovery:

- branch used: `repair-week4-qb-starter-uncertainty-v1`
- recovery run `36935917903` = **SUCCESS**
- source paid artifact = `11197726111`
- source digest = `sha256:202aa5205e71ad0acedf1910f505c2b362f564d8593a0f82dd0928fb45175aae`
- recovered artifact = `11197776900`
- artifact name = `week4-live-recovered-36935917903`
- recovered digest = `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`
- **no second OddsAPI fetch was used**.

Recovery actions:

1. restored exact paid Week-4 artifact;
2. preserved football/pricing state;
3. stamped QB lineage with the Week-4 final-board quarantine;
4. re-ran certified downstream audits;
5. removed unresolved backup-QB rows at the final board only;
6. rebuilt the live Week-4 workbook;
7. uploaded the recovered board.

Final-board quarantine:

- CHI : Tyson Bagent
- WAS : Marcus Mariota

Rows removed:

- 20 side rows total.

Recovered live board:

- 3,460 side rows remaining;
- 1,730 priced offers;
- 873 player-market rows;
- 392 priced players in the underlying certified stack before player-market aggregation;
- 16 games;
- workbook pricing status = `CURRENT`;
- sportsbook role = downstream comparison only;
- unresolved position rows = 0.

Master workbook audit:

`MASTER_BETTING_WORKBOOK_PUBLISHED`

Recovered artifact `11197776900` is the **current Week-4 live board authority** for analysis in the next chat.

### CRITICAL COST RULE

Do **not** fetch live odds again merely to inspect or analyze today's board.

Use recovered artifact `11197776900` first.

A second paid OddsAPI acquisition requires fresh explicit user approval.

---

## 4. MERGED WEEK-4 FINAL-BOARD REPAIR — PR #671

PR #671:

`Repair: quarantine unresolved Week 4 backup-QB markets`

Status:

- **MERGED**
- merge commit `0e6d8ab63c395b6bb9e88b0d55844083ac3221a5`
- source branch head `03741d41f4090ac587131681ea851a76f90ab963`

The repair adds the existing season/week-scoped manual final-board quarantine entries for:

- CHI Tyson Bagent
- WAS Marcus Mariota

It does not change football projections, distributions, coefficients, or sportsbook acquisition.

It exists so unresolved sportsbook backup-QB offers cannot survive into the published Week-4 board.

PR checks:

- Repo CI = green
- Weeks 1-2 full-board backtest = green
- old “Replay Paid Full Slate Artifact Once” red remains the known expired pinned-artifact dependency and is not evidence against the repair.

---

## 5. OPEN DYNAMIC QB STARTER-CONFLICT REPAIR — NOT FINISHED

There is a newer mechanical branch attempting to make the QB starter-conflict quarantine dynamic instead of relying on the Week-4 manual final-board entries.

Branch:

`repair-week4-qb-live-starter-conflict-v1`

Current head:

`d1003a5809bb17b02592b533eb1f42bc247fb88c`

Compared with current main:

- 5 commits ahead
- 0 behind at the last query
- modified:
  - `.github/workflows/verify-week4-qb-live-starter-conflict-v1.yml`
  - `scripts/stamp_qb_c2_pricing_lineage_v1.py`
  - `scripts/operations/quarantine_final_priced_props_v1.py`
  - `tests/test_quarantine_final_priced_props_v1.py`

Latest verifier:

- run `36937252724` = **FAILURE**

What passed in that run:

- focused quarantine regression tests;
- exact paid artifact identity;
- exact paid artifact download;
- paid data/output restore;
- dynamic C2 starter-conflict stamp.

Dynamic stamp correctly found:

- CHI Tyson Bagent
- WAS Marcus Mariota

and correctly reported:

- 6 pass-yard side rows dynamically quarantined at the lineage-stamp stage;
- pricing values modified = false;
- sportsbook inputs used to stamp = false;
- football QB authority remained independent.

### Exact remaining failure

The verifier then failed in market-model-lineage governance:

`RuntimeError: final-board-quarantined QB rows missing quarantine authority marker`

So the dynamic branch is **not ready to merge**.

Exact next mechanical action if this lane is resumed:

1. inspect `audit_market_model_lineage_v1.py::_certify_qb_c2`;
2. identify the exact quarantine-authority marker required for final-board-quarantined QB rows;
3. make the dynamic stamper populate that authority without changing any pricing/projection value;
4. rerun the same preserved paid artifact verifier;
5. merge only if all governance + workbook gates pass.

Do not refetch odds to validate this branch. Use paid artifact `11197726111`.

This dynamic repair is secondary to using today's already-recovered board.

---

## 6. GAME-DAY BET-SELECTION SCIENCE — IMPORTANT BOUNDARY

The user explicitly raised the core problem:

> Live Full Slate produces giant edges on too many props, but Weeks 1-3 show those raw edges are not trustworthy enough to say “bet everything.”

That problem has now been attacked with two genuinely different frozen historical selector studies.

### A. Market-Relative Bet Selector V1 — terminal null

Draft PR #667.

Canonical run:

- `36800239299` = SUCCESS
- artifact `11135730047`
- digest `sha256:1a48ca7c1b333a735b5348cc87a49cbd385e9d80e983bb87613858130f09976c`

Test:

`fair_line = market_consensus + beta * (model_projection - market_consensus)`

Two-way 2024 <-> 2025 holdouts.

Results:

- QB pass yards: one direction pass / reverse fail -> terminal fail
- TE receiving yards: fail
- TE receptions: fail

Terminal:

`NO_VERIFIED_INCREMENTAL_MODEL_LEVEL_SIGNAL_V1`

Do not rescue, retune beta, invert TE, or create an edge threshold.

### B. Market Offer Residual Probability V1 — terminal null

Draft PR #668.

Canonical run:

- `36801531063` = SUCCESS
- artifact `11136326900`
- digest `sha256:fca5a66648047e2c17aaf02a01de7d449d12b39effb2d478fd72e18843cc2695`

Fixed 5-feature offer-level logistic architecture versus book no-vig probability.

Terminal for:

- QB pass_yards
- TE rec_yards
- TE receptions

All:

`NO_VERIFIED_MARKET_OFFER_RESIDUAL_PROBABILITY_SIGNAL_V1`

Do not tune C, add interactions, split by side/role/week/edge bin, or move to sparse actionability from this failed model.

### Meaning for today's Week-4 board

Do **not** interpret huge raw model-vs-book edge as validated betting confidence.

The live board is valid as a projection/pricing artifact, but the project does not currently possess a historically validated downstream selector that proves the largest displayed edges are the best bets.

When analyzing today's board, be explicit about this limitation.

Do not silently invent a “safe / best / high-confidence” rule from raw edge magnitude.

---

## 7. WEEKS 1-3 POSTMORTEM — CLOSED

Canonical combined Weeks 1-3:

- W1: 204-205, -20.92u
- W2: 192-198, -22.06u
- W3: 233-208, +2.26u
- total selected: 1,260
- decided: 1,240
- voids: 20
- record: 629-611
- hit rate: 50.7%
- units: -40.72u

Severe overconfidence remains established:

- 70-100% stated-confidence band: 502 bets
- mean stated probability: 81.0%
- realized: 51.8%
- calibration gap: -29.2pp

54 tested slices produced no survivor after game-cluster-aware BH-FDR q=.10.

Do not reopen / retune Weeks 1-3 science.

No QB carveout.
No RB-only rescue.
No top-N rule.
No share-threshold hack.
No depth-chart hack.
No outcome-driven evidence-state tuning.

---

## 8. GSIS RB SUCCESSOR LINEUP V1 — SEALED PROSPECTIVE RESEARCH

Draft PR #669:

`research-gsis-rb-successor-lineup-v1`

Current known branch head:

`a62ae4bba9af5190c95409af0f774fe3689c421a`

The first genuine prospective Week-4 private lock was corrected before outcomes and supersedes the earlier V1 lock.

Authoritative private V2 allocation lock:

- finalized `2026-10-01T17:14:21.114582+00:00`
- allocation SHA256 `38a79c295a85f5f58c7a77473aae9ea62122da0c5f06d99920f868f1ef3e6b4e`
- event-audit SHA256 `2d01fa956319210a420ac0617e9d2a422abb7dbc101e8610974df8c5536c0461`
- target outcomes read = false at lock time
- sportsbook inputs read = false
- private/raw GSIS remains outside public GitHub.

Support currently locked:

- 1 / 6 future weeks
- 2 / 10 vacancy team-games
- 5 / 20 successor player-games

Pre-outcome review fixes already applied:

1. three-arm projection routes through current promoted Week>1 Full Slate simulation seam;
2. the lock's own finalization time must be pre-kickoff;
3. full unavailable set retained for multi-vacancy conditioning;
4. active RB/FB successor pool can include a player with zero Vacancy-V1 snap weight;
5. GSIS can assign that player successor mass if exact lineup exposure supports it.

Do not change formula, cohort, support thresholds, or source semantics after outcome exposure.

Do not grade an event until its game is final.

Private GSIS files live in the user's Library under:

`/NFL stuff/Private GSIS Locks/`

Never publish raw/private GSIS rows to public GitHub.

---

## 9. AVAILABILITY SEMANTICS REPAIR — MERGED

PR #670 is merged at:

`ab16d90d55c92150bde358b0a28f292c854cd222`

Correct semantics now:

ESPN core team injury-log `OUT`:

- is definitive **reported** unavailability;
- becomes `UNAVAILABLE_REPORTED`;
- overrides older QUESTIONABLE/DOUBTFUL when ESPN's latest status is OUT;
- is **not** official game-day inactive-section authority;
- cannot satisfy the T-75 official-inactives certification gate.

Do not restore the old mislabeled semantics.

---

## 10. WR/CB, CLV, RB-PD2, OTHER OPEN LANES

### WR/CB

Projected pregame FantasyAlarm alignment source is real but too sparse for the frozen strict-prior V1 source-readiness threshold.

Historical immutable strict locks remain:

- 2022W5 V2: 36 strict
- 2024W1: 46 strict
- protected 2025W14: 47 strict, outcomes protected

Do not resume blind archive scanning.

Resume only with:

- a new exact URL/timestamp lead;
- a genuinely different free archive exposing an exact version;
- a verified prospective current report captured pre-kickoff.

Legacy `coverage_penalty()` is retired. Never restore it.

### CLV PR #663

Still draft/passive.

Same-book <=30m before kickoff only may be labeled CLV.

Do not fetch OddsAPI solely for CLV.

### GSIS PR #662

Research-only safe archive contract.

Private/raw GSIS never goes to the public repo.

### RB-PD2

Still HOLD.

Do not lower its prospective support floor.

---

## 11. TODAY'S IMMEDIATE NEXT ACTION

The next chat should **not** spend more live-odds credits first.

Priority order:

### First — use the recovered Week-4 live board

Pull artifact:

- run `36935917903`
- artifact `11197776900`
- digest `sha256:3f058570037ca016a5cbf1fa79e6e6abc4d845384de4dbdfaf477c8f0a3160a8`

Inspect:

- `outputs/NFL_BETTING_MODEL_MASTER.xlsx`
- `outputs/props_priced_clean.csv`
- relevant live audit/status files

Confirm:

- pricing status CURRENT;
- 1,730 priced offers;
- 873 player-market rows;
- Bagent/Mariota absent from final published board;
- no other quarantine/governance failure.

Then answer the user's game-day board questions from that recovered artifact.

### Second — be careful with “best bets”

Raw edge magnitude is **not validated confidence**.

If presenting candidates:

- distinguish model projection / market line / model probability / price;
- do not claim the biggest raw edges are proven best bets;
- make uncertainty visible;
- do not pretend the two failed selector studies provide a betting confidence layer.

### Third — optional mechanical cleanup

If continuing the dynamic QB repair:

- branch `repair-week4-qb-live-starter-conflict-v1@d1003a...`
- fix missing quarantine-authority marker;
- rerun exact preserved paid artifact;
- no new sportsbook acquisition.

### Fourth — preserve research locks

GSIS Week-4 remains sealed until final outcomes.
WR/CB remains source-blocked.
CLV passive.
RB-PD2 HOLD.

---

## 12. COST / ACQUISITION RULES

One paid Week-4 OddsAPI acquisition was explicitly authorized and executed.

Do not issue another live odds pull without fresh explicit user approval.

The exact paid snapshot is preserved in artifact `11197726111`, and the recovered current board is artifact `11197776900`.

Use those.

---

## 13. MEMORY-EFFICIENT NEXT-CHAT READ ORDER

Read exactly:

1. `AGENTS.md`
2. newest top checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`
3. this file:
   `docs/handoffs/NFL_HANDOFF_2026-10-01_WEEK4_LIVE_BOARD_RECOVERED_CURRENT.md`
4. Issue #535 from comment `5936680436` onward
5. query live:
   - main
   - PR #669
   - PR #668
   - PR #667
   - PR #665
   - PR #663
   - PR #662
   - branch `repair-week4-qb-live-starter-conflict-v1`
   - Actions run `36934563481`
   - recovery run `36935917903`
   - latest main Full Slate / Repo CI

Then work.

Do not recursively read old handoffs.
Do not ask the user to re-explain any of this.
Do not rerun closed science.
Do not spend OddsAPI again unless explicitly authorized.
