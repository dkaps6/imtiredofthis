# 2026 Weeks 3-5 priced boards — forensic closure

**Date:** 2026-11-04
**Question:** Weeks 1-8 of 2026 were all genuinely run with full sportsbook
odds and the model priced. Why does `data/market_track_record/boards/` hold
only W1, W2, W6, W7, W8?

**Answer:** nothing was deleted. Two separate causes, both now fixed.

---

## 1. What the archive actually holds

```
2026_wk01.csv  committed 2026-09-22  4190c704
2026_wk02.csv  committed 2026-09-22  4190c704
2026_wk06.csv  committed 2026-10-19  a8f4e8bf  run 38792341891
2026_wk07.csv  committed 2026-10-22  c8f6f057  run 39849371821
2026_wk08.csv  committed 2026-10-29  be975b1e  run 40953584718
```

`git log --diff-filter=AD -- data/market_track_record/boards/` returns only
ADDs. **W3, W4 and W5 boards were never committed.** There is no deletion to
recover.

## 2. Week 3 — the board never existed

`docs/research/RB_PD2_WEEK3_PROSPECTIVE_LOCK_V1_RESULT.md` records run
`36330757181` with live odds authorized, and states the canonical pricing step
failed mechanically. The odds were paid for and acquired; the priced board was
never produced.

That run's artifacts were checked. Exactly one survives:
`rb-pd2-forward-lock-36330757181`, 10,149,354 bytes, `expired: false`,
`expires_at: 2026-12-26`. That is the opt-in RB PD2 research capture, which the
workflow uploaded at `retention-days: 90`. It is not a board.

## 3. Weeks 4 and 5 — the boards existed and expired

Both weeks produced priced boards and both are documented by artifact id and
sha256 digest:

| week | run | artifact | content | artifacts now |
|---|---|---|---|---|
| W4 paid source | `36934563481` | `11197726111` | 1,740 book-line / 3,480 side rows | **0** |
| W4 recovered board | `36935917903` | `11197776900` | recovered, no second fetch | **0** |
| W5 paid source | `37852811339` | `11582178322` | paid acquisition | **0** |
| W5 canonical replay | `37860592615` | `11585973062` | 1,472 priced offer / 754 player-market rows | **0** |

`list_workflow_run_artifacts` returns `{"total_count":0}` for all four. The
digests in the handoffs are now digests of nothing.

## 4. Root cause

`full-slate.yml` carried two uploads with different retention. At the W3-W5
commits the research capture used `retention-days: 90` and the main bundle
holding the priced board used `retention-days: 7`:

```
git show f39374a7:.github/workflows/full-slate.yml | grep -n retention-days
528:          retention-days: 90
587:          retention-days: 7
```

`retention-days: 7` survived in the workflow until **2026-10-19**, when
`a6068d6a` raised it to 90 and `584782f7` added archive-on-pricing-failure.
W6's board was preserved by hand the same day (`a8f4e8bf`).

That date is the whole explanation for the shape of the archive:

- W1/W2 survive because they were committed by hand inside the 7-day window;
- W3 has no board because pricing crashed;
- W4/W5 boards lived exactly 7 days and GitHub deleted them;
- W6 onward survive because retention is 90 **and** the board is committed.

## 5. What is unrecoverable

The W4 and W5 sportsbook lines came from paid OddsAPI acquisitions that cannot
be refetched at any price. Row-level grading of those two slates is permanently
impossible. W3 was paid for and produced nothing gradeable.

Five of the eight 2026 slates are gradeable at row level: W1, W2, W6, W7, W8.

## 6. Already fixed

- retention at the 90-day public-repo ceiling on every upload;
- paid boundaries archived to `data/market_track_record/` on every live run,
  including when pricing fails;
- AGENTS.md section 5a states the durability contract: an artifact is transport,
  not storage.

No further action is available on W3-W5. This document exists so the gap is
never re-investigated.
