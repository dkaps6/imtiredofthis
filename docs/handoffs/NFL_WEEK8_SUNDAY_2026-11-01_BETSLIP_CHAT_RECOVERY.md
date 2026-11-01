# PRISM Week 8 Sunday bet-slip recovery — 2026-10-31 22:45 ET

## Purpose and boundary
This recovers the interrupted ChatGPT conversation **"Resume NFL Project Audit"** in the NFL stuff project on October 31, 2026, immediately before Sunday November 1 kickoff. User expressly wants **Sunday manual bet-slip construction**, not a restart of research, a new science path, or paid odds. The original chat hit its conversation limit and iOS showed `APICore.ConversationDetailError.DetailError error 1`. This file is a durable **bet-construction conversation handoff**, not a scientific promotion or authorized modification to Full Slate; canonical root `CURRENT_NFL_RESEARCH_HANDOFF.md` remains unchanged and authoritative for research. Do not confuse these lanes.

Repository: `dkaps6/imtiredofthis`. Coordination issue: https://github.com/dkaps6/imtiredofthis/issues/535. Read `AGENTS.md` and `CURRENT_NFL_RESEARCH_HANDOFF.md` before any code/model changes. The research handoff currently says "betting CLOSED": that remains the automated science/production authorization state; user-directed manual ticket exploration is **not** authorization to reopen paid OddsAPI, alter production, or claim promoted betting-grade probability calibration.

## The user's non-negotiable construction contract
1. Named player FIRST: strictly-prior identity, current usage/role and anomaly checks, opportunity quantity and quality, teammate competition, QB/coaching/scheme/defense/box/coverage, game script and pace; market-specific reliability, not position averages.
2. Every leg earns an individual football evaluation **AND** independent archived PRISM support. A raw projection/line gap does not earn a bet by itself.
3. Every PRICE earns a wager separately: live book, true line, odds, P(clear), estimated probability reliability, and conditional EV; detect stale markets and movements, and reject if no longer attractive. Never present bookmaker pricing frozen on Oct 29 as currently actionable.
4. **No predetermined number of parlay legs.** User rejects arbitrary 3/4/5-leg limits; also rejects absurd ~19-leg tickets. Construct natural groupings of surviving legs based on quality, exposure and correlation, not forced count.
5. Do not force the user's historic Safe/Moderate/Medium/High/Lottery tiers or alt lines to manufacture odds. Original main line first; alt line only if changed clearing chance and marginal payout justify it. No multiple highly correlated legs without discounting ticket probabilities. Do not let one player overweight every ticket. PASS is valid.
6. User has had poor *actual-ticket* outcomes despite weekly model improvement. Week 7 **original-workbook signal grade** (not actual tickets): 200–161 settled, 8 void, +21.103u flat illustrative, 55.4017% hit, 5.8458% illustrative ROI, #535 comment 6480266032; this is not proof that assembled parlays will profit or that model probabilities are calibrated. W1–4 actual user-reported results were negative. Preserve distinct model-signal vs submitted-ticket ledgers.
7. Do not claim approved slips or verified parlay payouts without current on-screen sportsbook selections/odds. Verify official inactives Sunday ~T−75 minutes and any last-minute changes before final bets.

## Exact interrupted conversation checkpoint (recovered from prior messages)
As of October 31 around 22:41 Eastern, five principal Sunday candidate *investigations* remained:
- **Daniel Jones, IND @ JAX — OVER 188.5 passing yards**. PRISM `230.2602296`. Strong raw model gap; JAX has allowed 273.8 opposing pass yards per game (rank 30 per contemporaneous October 30 coverage). Football caution: Jones averaged ~199.7 passing yards in seven games, completed only 59.6%, 4 TD / 7 INT, just 108 pass yards on Oct 25; Jacksonville can force turnovers and Colts offense struggles. Individual role/game-state threshold analysis still required.
- **Keon Coleman, BUF vs BAL — OVER 29.5 receiving yards**. PRISM `42.0646050`. Volatile route/target floor, Bills room competition (DJ Moore and Dalton Kincaid available per updated injury reports), Lamar back changes likely script. Must establish actual routes/targets, not aggregate weak WR defense.
- **Jared Goff, DET vs MIN — UNDER 259.5 passing yards**. PRISM `228.9552996`. Strong raw gap, but specifically flagged football counterarguments: competitive Vikings/Lions game may force passing; Goff's recent production and volume could break the under. Not approved.
- **Stefon Diggs, WAS vs PHI — OVER 41.5 receiving yards**. PRISM `50.8227425`. **CORRECT TEAM/OPPONENT: Washington Commanders hosting the Philadelphia Eagles on Sunday Night Football, not New England at Miami.** Verified by the Washington team roster, its official injury report and PRISM's archived board. Washington WR Terry McLaurin is on IR, while Diggs practiced fully Thursday/Friday despite a hand/hamstring/rest listing and has no final game designation. Diggs' five 2026 game yardages: 55, 47, 33, 35, 44; mean 42.8. This leaves 3/5 above 41.5, but do not confuse this descriptive streak with a calibrated probability. **A.J. Brown's New England injury is entirely irrelevant to Diggs.**
- **Gunnar Helm, TEN @ CIN — OVER 15.5 receiving yards**. PRISM `20.4872288`. Speculative fifth, thinner margin and contingent on actual TE role/route usage. Titans WR Carnell Tate OUT and Calvin Ridley questionable (Oct 31 NFL report). Do not elevate solely due to others missing.

Candidate numbers above are the original Oct 29 cached Full Slate projections; never imply recalculated on Oct 31. The exact stored authoritative source is `data/market_track_record/boards/2026_wk08.csv` (git blob SHA `a40139684b98230e5adf4b87461ffa74d9accebd`, Full Slate source run `40953584718`, fetched stamp `2026-10-29T21:04:02Z`; contents retrieved via GitHub `fetch_blob` even though larger `fetch_file` returns empty).

### Historical archived book terms, NOT currently valid offers
| Player / market / direction | Archived book | Archived line | Archived odds | Raw archived model P(side) |
|---|---|---:|---:|---:|
| Jones pass yards OVER | FanDuel | 188.5 | -114 | 0.76308 |
| Coleman rec yards OVER | FanDuel | 29.5 | -114 | 0.68240 |
| Goff pass yards UNDER | FanDuel | 259.5 | -114 | 0.72348 |
| Diggs rec yards OVER | DraftKings | 41.5 | -114 | 0.62080 |
| Helm rec yards OVER | FanDuel | 15.5 | -114 | 0.61788 |

These are raw model empirical probabilities, NOT reliable ticket win chances or calibrated 'safe' percentages. The existing research handoff found overconfident model probability calibration (2026 Brier 0.2800 vs market 0.2491). Do not multiply the raw Ps uncritically or claim a proven edge.

### Previous tentative slips from lost chat
- **Ticket A / provisional:** Daniel Jones OVER 188.5 pass yards + Keon Coleman OVER 29.5 rec yards + Stefon Diggs OVER 41.5 rec yards.
- **Ticket B / provisional:** Daniel Jones OVER 188.5 pass yards + Jared Goff UNDER 259.5 pass yards + Keon Coleman OVER 29.5 rec yards.
These were *preliminary scenarios*, NOT finally approved tickets or placed wagers. Do not simply submit them. They share Jones+Coleman and are overconcentrated if both used as main tickets.

### Candidates already removed (do not re-add from Oct 29 board)
- Hunter Henry OVER rec yards: original ~38.5 line advanced to ~44.5–45.5; earlier apparent gap eroded. With A.J. Brown out, reconsider only if current lines/real football evidence actually restore value.
- Chris Godwin OVER rec yards: line moved ~29.5 to ~40.5; earlier bargain disappeared.
- RJ Harvey OVERS: officially OUT (knee) for Chiefs at Broncos, NFL.com Oct 31; do not use. Any archived bookmaker row for him is invalid.

## Latest factual checks as of Oct 31 night
- NFL.com Week 8 official injury roundup updated Oct 31 5:25pm: https://fantasy-www.nfl.com/news/nfl-week-8-injury-report-2026-season
- Washington official Diggs player page, 2026 game logs: https://www.commanders.com/team/players-roster/stefon-diggs/
- Official Eagles pregame injury report verifying Diggs' Washington role: https://www.philadelphiaeagles.com/news/eagles-at-commanders-injury-report-2026-nfl-week-8-sunday-night-football-saquon-barkley-devonta-smith
- Washington injury report: https://www.commanders.com/news/commanders-vs-eagles-week-8-injury-report-2026
- Patriots A.J. Brown absence belongs ONLY to NE vs MIA; it is **not** Diggs context: https://www.patriots.com/news/roster-analysis-patriots-downgrade-wr-a-j-brown-rb-treveyon-henderson-to-out-for-week-8-vs-dolphins
- FanDuel Oct 30 Daniel Jones preview: https://www.fanduel.com/research/nfl/player-news/game-updates/daniel-jones-and-colts-play-jaguars-in-week-8-11-1-2026 (188.5 as Friday, and 1,398 pass yards/7 games)
- Week 8 Sunday matchups: Jones IND @ JAX 1pm ET; Coleman BUF vs BAL 1pm; Goff DET vs MIN 1pm; Helm TEN @ CIN 1pm; Diggs WAS vs PHI 8:20pm ET. 2026-11-01 DST shift: local Eastern time = EST.

## Immediate continuation — execute in this order
A. Re-open this specific 5-player board, not a new sweeping audit. Inspect player-specific last 4-7 games, per-game snaps/routes/targets/attempts, injury + returns (including anomalies), relative teammate roles, opposing coverages/pass rush and offense likely game script, then assign each leg `FOOTBALL PASS / CONDITIONAL / REJECT` with explicit counter-case.
B. Check the most recent legitimately available betting lines, prices and alt lines from sportsbook or new user screenshots. The Oct29 archival values are snapshots and Wednesday-to-Saturday repricing already killed Henry/Godwin. Mark unavailable prices as `UNVERIFIED`, never assume a price still exists.
C. Run final official game-day activity checks before 1pm/4:25pm; address player or teammate news, weather and role shifts.
D. Build transparent tickets only from survivors, across user tiers as appropriate, with a reason why the exact price and specific line clear the gate; balance players, games and exposure. No forced leg limit, no silly 19-leg padding. Explain ticket correlation and how alt vs main lines differ.
E. If requested, preserve actual finalized picks/screenshots with source timestamp and grade afterwards, distinct from original workbook's signal grade. Avoid paid OddsAPI credits; do not modify production or frozen model without express user authorization.

**Critical continuity statement**: We are in SUNDAY BET CONSTRUCTION, **not** new RB-science work. The last chat timed out before final individual football approval and final priced slips. Resume there.


## Oct 31 independent individual-football audit — first-pass flags (NOT final approval)
- **Jones conditional positive:** Jaguars concede 273.8 pass yds/game (rank 30), official Colts 2026 average for Jones 1,398/7=199.7. Contrarian check: 59.6% completions, 4 TD/7 INT and 108 passing yards last week; line 188.5 is below his season mean, but passing efficiency/turnovers and 2026 QB player-state uncertainty remain. https://www.fanduel.com/research/nfl/player-news/game-updates/daniel-jones-and-colts-play-jaguars-in-week-8-11-1-2026 ; https://www.colts.com/team/stats/2026/REG
- **Coleman CONCERN / role degradation:** October 18 six targets, 59 yards came with DJ Moore sidelined; he ran 94% of routes in that absence, but Moore and Kincaid have returned/cleared practice for November 1. Do not project the Moore-absent role as unconditional. Coleman 24 targets in six games (4/g), with yards very volatile: Week 4 116 vs Week 5 7. https://www.draftsharks.com/fantasy-football-news/83553/keon-coleman-keeps-scoring-while-dj-moore-sidelined ; https://fantasy-www.nfl.com/news/nfl-week-8-injury-report-2026-season
- **Goff HIGH-CONFLICT / REMOVE FROM MAIN TICKET pending override evidence:** PRISM under 259.5 despite Goff's 2026 1,788 pass yds in six games (298/g) and five consecutive games 269+; Minnesota yields very few passing yards and high pressure, but Goff's demonstrated high-efficiency blitz performance complicates this under. Do not label this “safe” just because PRISM returned model P 0.72348. https://www.vikings.com/news/detroit-lions-game-preview-nfl-week-8-2026 ; https://www.fantasypros.com/nfl/notes/480133/jared-goff-2026-week-8-outlook.php
- **Diggs corrected to Washington:** Washington WR McLaurin on IR, Diggs 214 yds / five games (42.8); line 41.5 is not a large margin relative to his *actual* profile. Await final injury confirmation, Eagles coverage matchups, and new price before inclusion. Unlike prior mistaken interpretation, A.J. Brown's New England absence offers zero support.
- **Helm LOW-CONFIDENCE / exclude main board:** Titans official 2026 game logs 28,13,14,5,0,0,24 yds (84 over seven = 12/g), only 2/7 above 15.5 (28 and 24). Week 7 24 may be a rebound, but two successive zeroes require investigation; Cincinnati conceding TE production alone is insufficient. https://www.tennesseetitans.com/team/players-roster/gunnar-helm/

**Source-integrity ruling:** The archived PRISM Oct 29 board already lists Stefon Diggs as **WAS vs PHI**, not NE. The wrong Patriots association arose from the attempted chat reconstruction, not a PRISM roster bug. Do not blame or change the model for that assistant error. Previous Ticket A and B remain historical notes only. Current viable research pool should be reconsidered on the merits, not blindly recycled as tickets.
