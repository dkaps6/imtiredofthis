# NFL PRISM — Week 8 Sunday bet-slip construction — recovered chat checkpoint
**Recovered:** 2026-10-31 ~22:45 EDT; Sunday slate is 2026-11-01.
**Original interrupted chat:** "Resume NFL Project Audit".
**Repo:** dkaps6/imtiredofthis, canonical research handoff: root `CURRENT_NFL_RESEARCH_HANDOFF.md`, coordination: Issue #535.
**Status:** DISCRETIONARY BET CONSTRUCTION IN PROGRESS; **NO FINAL TICKETS APPROVED**. This is a chat-continuity snapshot, NOT a research promotion or change to the root canonical science ledger.
**Source snapshot:** archival 2026 Week-8 board, `data/market_track_record/boards/2026_wk08.csv`, blob SHA `a40139684b98230e5adf4b87461ffa74d9accebd`, Full Slate run **40953584718**, original quotes `2026-10-29T21:04:02Z`. **Historical bookmaker quotes — NOT current Oct 31 prices.** Some candidates' last chat-visible lines matched this source; verify all live prices, injury/role changes before betting. **No OddsAPI credits used in this recovery.**

## User's exact betting goal and methodological contract
The user has run Week 8 PRISM and wants **sharp actual Sunday prop parlays**, not another model-development tangent. Weeks of model development did not translate into profitable bet construction; Week 7 was especially disappointing, so **ticket selection and price discipline are the immediate assignment**. Every proposed leg must pass BOTH (1) a traceable PRISM named-player projection and (2) an **independent real-football player-specific assessment**: availability/anomaly state, current opportunities, teammate competition, quarterback/offensive scheme, opponent defensive style and matchup, likely script, relevant market-specific pathway, and injury/rotation volatility. This second gate must be capable of REJECTING a superficially high model edge. Finally price the specific offered line, vig, sportsbook and any alt line; do not equate high modeled probability with validated hit rate. If football evidence conflicts or cannot be verified, downgrade/remove leg. Betting is not a production science approval.

**No arbitrary leg count.** Don't force parlays into 2/3/4 picks, but avoid unwieldy 19-leg lottery tickets. Build coherent Safe / Moderate / Medium / High / Lottery tickets as warranted by genuinely approved legs, diversifying across names/games and accounting for same-game/shared-script correlation and exposed bankroll. Avoid heavy concentration in one player or repeated dependencies. Properly distinguish standard vs alternate lines: an easier threshold only belongs on a ticket when its new price and robust football floor justify it. Confirm actual sportsbook lines from user screenshots when available, never invent current lines or -110. Don't claim independence or multiply raw leg probabilities without correlation checks.

**Personal user preference/history:** Successful 2026-10-19 MNF ticket(s) used individual-player script/usage analysis; Week 7 Sunday only one ticket reportedly won, causing renewed scrutiny. Do not treat one win or loss as validation of a technique. W7 original model-signal retrospective report on #535: 200-161, 8 void, +21.10u / +5.85% modeled flat-unit ROI from original workbook at historical quotes; this is NOT proof that user's submitted parlays won or that this week's calibration is reliable.

## Where exactly the preceding conversation stopped
Screenshot of the last completed assistant message:
> "Where we stand: Daniel Jones, Keon Coleman, Goff and Diggs are the priority investigations; Gunnar Helm is a more speculative fifth candidate. More importantly, we've already prevented several bad decisions by catching stale prices and changed player availability."
> "each player earns the leg, each price earns the wager, and only then do we build the parlay."

The user attempted a GitHub handoff twice; both requests caused `APICore.ConversationDetailError.DetailError error 1` after the conversation limit was reached. New chat resumed access to repo, root V20 handoff, #535, and original W8 board. This document restores the missing BET-CONSTRUCTION session-specific handoff; it **does not supersede** root V20's ongoing scientific research priority/constraints.

## Verified model rows and archived sportsbook offers (as-of Oct 29, not live)
These are the original archived board values *not current odds*:

| Candidate | Market/side | PRISM final mean | Archived book, threshold, odds | Raw board fair probability at line | Status |
|---|---|---:|---|---:|---|
| Daniel Jones (IND @ JAX) | OVER pass yards | 230.2602 | DraftKings **187.5 -112**; FanDuel **188.5 -114** | 76.816% DK / 76.308% FD | **priority investigation; no final approval** |
| Keon Coleman (BUF vs BAL) | OVER rec yards | 42.0646 | DraftKings **32.5 -110**; FanDuel **29.5 -114** | 62.744% DK / 68.240% FD | **priority investigation; no final approval** |
| Jared Goff (DET vs MIN) | UNDER pass yards | 228.9553 | DraftKings **263.5 -112**; FanDuel **259.5 -114** | 74.388% DK / 72.348% FD | **priority investigation; no final approval** |
| Stefon Diggs (NE @ MIA) | OVER rec yards | 50.8227 | DraftKings **41.5 -114**; FanDuel **42.5 -114** | 62.080% DK / 60.456% FD | **priority investigation; no final approval** |
| Gunnar Helm (TEN @ CIN) | OVER rec yards | 20.4872 | DraftKings **18.5 -108**; FanDuel **15.5 -114** | 51.820% DK / 61.788% FD | **speculative fifth; no final approval** |

IMPORTANT: `fair_prob` is PRISM's **raw uncalibrated tail estimate**, not a verified real-world chance. Root science record says live 2026 model probabilities were overconfident (Brier 0.2800 vs market 0.2491). Treat apparent extreme probability edges carefully; get actual prices at bet placement. These archived offers are from **Thursday Oct 29** and can be unbettable/stale by Saturday night / Sunday morning.

**Previously excluded in interrupted conversation:** Hunter Henry over after the offered line rose substantially; Chris Godwin over after line rose from ~29.5 to ~40.5; RJ Harvey overs after reported out status. These are **past chat decisions**, not independent injury confirmation here. Reverify official inactives and current lines before considering any reversal. **No reason to reopen solely because the old board has a superficially favorable model mean.**

## Immediate next actions — do this without re-interviewing user
1. Reopen and respect `AGENTS.md`, root `CURRENT_NFL_RESEARCH_HANDOFF.md`, and this newly persisted betting-specific checkpoint. Science lane remains separate; no production model changes/promotions, no OddsAPI fetch.
2. Verify Nov 1 2026 schedule, official injury/practice reports and latest player participation/role/news, defensive personnel, Vegas game totals/spreads and weather when material. Pay special attention to Jones's starter/attempt expectation, Coleman target-route role, Goff MIN matchup/game script, Diggs target/coverage environment, and Helm route share / game role; identify anomalous recent game states before drawing trend inference.
3. Evaluate the **five in order**, one by one, showing model view versus football view, evidence quality, risk/mismatch and PASS/REJECT. Investigate whether there are objectively stronger alternatives on the full Week 8 slate rather than forcing these five.
4. Check actual up-to-date offered odds/alternate thresholds from available sources or user screenshots. Do not invent price. Distinguish model-side directional advantage from value at price and practical parlay leg; use mathematically correct implied breakeven and correlation reasoning.
5. Produce rational tiered, clearly placeable Sunday slips using **only** football-approved players and confirmed lines (safe/moderate/medium/high/lottery as fits evidence, not obligatory fill). Include player/team/game, exact line and direction, bookmaker/price if verified, the football thesis, failure mode, and reason for each alt. Limit overlap and avoid one player appearing repeatedly at multiple thresholds. A PASS is allowed.
6. After each substantive checkpoint, preserve this betting work in GitHub separately from root science TOP (and note cross-link to #535 as appropriate). Keep a source+timestamp ledger for bookmaker screenshots and the final tickets.

## Integrity notes / not yet done
- No refreshed Oct 31/Sunday Nov 1 sportsbook prices obtained.
- No individual football deep dives completed since resumption; candidates not cleared.
- No final tickets generated or actual wagers placed by the assistant.
- The canonical root handoff V20 is dated Oct 28 and reflects a different scientific audit lane; it must not be mistaken for the status of the separate Saturday-night bet-construction conversation.
- Week8 archival board includes ex-ante injury risks; don't assume all originally listed players are now active.
- No claims that the assistant can continue working offline or overnight; save durable outputs during each interaction.
