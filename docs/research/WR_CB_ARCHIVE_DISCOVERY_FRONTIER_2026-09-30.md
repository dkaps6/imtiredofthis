# WR-CB Archive Discovery Frontier — 2026-09-30

**Disposition: HISTORICAL SOURCE BREAKTHROUGH PRESERVED; ADDITIONAL BLIND ARCHIVE SCANNING PAUSED; SOURCE/MODEL GATE CLOSED.**

This document freezes the current no-paid-data source frontier without accessing protected target outcomes.

## Authoritative strict source locks

### 2022 Week 5 — discovery
- exact Wayback capture: `20221005195142`
- CDX digest: `HKCXAO5CXTWGRFTPRIDA4OBCTJLANM7F`
- exact body SHA-256: `f9db298ffe8ae369acc2753897a6391e579f29b96bab7008ac803186ad9d3d5e`
- verifier run: `36773818193` SUCCESS
- artifact: `11124612896`
- 54 parsed factual rows
- 41 rowwise pregame factual rows
- 36 strict exact-week WR + opponent-CB identity rows
- authoritative lock: `data/research/wr_cb_verified_snapshot_lock_2022w05_v2.csv`
- authoritative lock SHA-256: `4eb261dac8b45f04b93ac427fce8666fb72625fbd1cf67c2f4351bfe0e501425`
- v1 is superseded because one manually transcribed CB GSIS ID did not match the successful artifact; v1 was not edited in place. See `WR_CB_2022W5_VERIFIED_ARCHIVE_LOCK_CORRECTION_V2.md`.

### 2024 Week 1 — discovery
- 46 strict pre-kickoff exact-week identity rows
- immutable lock: `data/research/wr_cb_verified_snapshot_lock_2024w01_v1.csv`
- preserved result: `docs/research/WR_CB_2024W1_VERIFIED_ARCHIVE_RESULT.md`

### 2025 Week 14 — protected confirmation source only
- 47 strict pre-kickoff exact-week identity rows
- immutable lock: `data/research/wr_cb_verified_snapshot_lock_2025w14_v1.csv`
- **protected 2025 target outcomes remain unopened for this lane**

## Current discovery-row readiness

Across the two discovery locks:
- total strict rows: **82**
- 2022W5: 36 rows = 15 LWR/RCB + 15 RWR/LCB + 6 slot
- 2024W1: 46 rows = 19 LWR/RCB + 23 RWR/LCB + 4 slot
- unique source CBs: 36 in 2022W5 and 46 in 2024W1
- only 6 CB identities appear in both locks
- the locks are separated by almost two seasons

Therefore these locks prove source validity but do **not** provide a usable same-season strict-prior CB history. Under the frozen preregistration, current eligible scored target rows = **0**.

## Sparse Wayback availability probes

All probes preserved the known 2024W1 positive control and fetched no archive bodies/outcomes.

- 2022W4 — run `36775930453`, artifact `11125603072`: no available candidate; access-inconclusive.
- 2022W7 — run `36776073453`, artifact `11125743279`: no available candidate; access-inconclusive.
- 2022W3 — run `36776412522`, artifact `11125678735`: no available candidate; access-inconclusive.
- 2022W11 — run `36777085212`, artifact `11125199343`: no available candidate; access-inconclusive.

These are not archive-absence claims.

## Sparse exact CDX probes

A generic exact-URL, bounded-time, sequential CDX probe was added with the already verified 2022W5 timestamp as the positive control.

- Initial run `36776678489` exposed a local control-timestamp dtype bug and network timeouts. The dtype bug was repaired at commit `89a049a4b2ff46db6f3b43465028a60d4bcc56f0`.
- Fixed run `36777079292`, artifact `11126161872`:
  - W5 control recovered exactly: timestamp `20221005195142`, digest `HKCXAO5CXTWGRFTPRIDA4OBCTJLANM7F`;
  - W3 returned `NETWORK_ConnectTimeout`, therefore access-inconclusive.
- W11 run `36777957891`, artifact `11126301549`:
  - W5 control recovered;
  - W11 returned `NO_EXACT_INDEX_MATCH_NOT_PROOF_OF_ABSENCE`.
- W4 run `36778143986`, artifact `11126038208`:
  - W5 control recovered;
  - W4 returned `NO_EXACT_INDEX_MATCH_NOT_PROOF_OF_ABSENCE`.

This demonstrates that the exact CDX method can recover a known snapshot while still failing or returning no record for neighboring weeks. Do not convert these results into a broad historical absence conclusion.

## Alternate archive probe: Arquivo.pt

A free exact-URL metadata-only probe was added after confirming Arquivo.pt publicly supports URL version-history search.

Run `36777455913` SUCCESS, artifact `11126570485`:
- 2022W3: 0 exact Arquivo.pt versions
- 2022W5: 0 exact Arquivo.pt versions
- 2022W11: 0 exact Arquivo.pt versions

Because Arquivo.pt also lacks W5, which is independently proven in Wayback, these zeroes only describe Arquivo.pt holdings. They do not contradict the W5 source lock and are not global absence evidence.

## 2026 prospective capture

Public search on 2026-09-30 found the verified FantasyAlarm Week-3 report but did not surface a Week-4 WR/CB report.

Rule remains:
- never guess a current-week URL;
- freeze only a verified public report actually captured before its applicable kickoff;
- prospective 2026 captures are source locks, not permission to inspect future outcomes early.

## Acquisition stopping rule for this session

Do not continue blind or broad archive scans. Resume historical acquisition only when one of these occurs:
1. a new exact URL/timestamp lead is independently identified;
2. a different free archive exposes an exact historical version;
3. a prospective 2026 report is publicly verified before kickoff.

This avoids spending Actions on repeated access-inconclusive scans.

## Boundary

No sportsbook data was purchased or fetched.
No protected 2025 outcomes were accessed.
No model parameter was fit.
No editorial FantasyAlarm matchup grade is model eligible.
The retired `coverage_penalty()` / static WR-CB heuristic remains retired.
The source/model gate remains **CLOSED**.
