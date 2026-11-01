# Week 8 Sunday — overnight check 04 — ~03:00 ET Nov 1 2026

**EVIDENCE ONLY. No bet placed; no paid OddsAPI; no model science changes. Do not stop hourly review.** Canonical `AGENTS.md`, root handoff, `WEEK8_8AM_FINAL_LOCK_PREP_20261031.md`, RUN01–03, and archived original 2026_wk08 board (Oct29 run 40953584718, blob `a40139684b98230e5adf4b87461ffa74d9accebd`) consulted. The user wants exact threshold right-side, individual football scrutiny and five diversified tiers + separate $1M entertainment card.

## NEW: Quarterback pass-yard false-positive audit

The largest unexamined pass-yard PRISM signals were compared against 2026 **pregame** official game logs and recent posted lines. Counts are game-by-game 2026 actual outcomes, NOT independent probability estimates; abnormal games need anomaly gate before claiming a trend.

| Player / archived side | PRISM original | 2026 actual evidence / current market | Gate |
|---|---|---|---|
| **Jared Goff DET UNDER263.5 (DK), UNDER259.5 (FD)** | mean **229.0**, raw no-vig edge +24.4pp/+22.3pp | 2026 pass yards **206,327,269,412,271,303**. **Only 1/6 UNDER259.5 and 1/6 UNDER263.5**. Market now around **252.5** vs MIN (more demanding UNDER). Vikings allow fewest pass yds to QBs; that's the genuine countercase, but model direction grossly contradicts Goff's 2026 usage and performance. | **HARD REJECT as regular under**. Do not let matchup alone rescue 72–74% uncalibrated model P. Sources https://fantasydata.com/nfl/jared-goff-fantasy/17922 and https://statz.ai/nfl/players/jared-goff-32/passing-yards |
| **Kirk Cousins LV UNDER240.5 DK / UNDER239.5 FD** | mean **220.8**, raw edge +15.8pp/+15.3pp | 2026 yards **160,253,248,365,243,313,295** = **1/7 under240.5**; Jets allow fifth-fewest passing yards (187.7/team/g), so opponent strength real, but Cousins averages **36.7 pass attempts** and 268.1 yards. Current market around **233.5**, making under more fragile. | **HARD REJECT as regular under**. Do not bet against established passing volume just because original mean is low. Official https://www.nfl.com/players/kirk-cousins/stats/logs/2026/ and https://statz.ai/nfl/players/kirk-cousins-26/passing-yards |
| **Cam Ward TEN OVER192.5 FD** | mean **213.8**, raw +13.0pp, uncalibrated P **63.0%** | 2026 yards **140,183,181,222,170,102,181** = **1/7 over192.5**; CIN allows 7th-most QB pass yards and TEN ~10-point underdog, genuine volume upside, but 2026 efficiency/production deeply contradicts 63% estimate. Current public FD O192.5 around -114 (snapshot). | **HARD REJECT as regular over**, no upside chase solely from game script. https://www.pro-football-reference.com/players/W/WardCa00.htm and https://slipsurgeon.com/props/nfl/2026-11-01-titans-at-bengals/ |
| **Lamar Jackson BAL UNDER237.5** | mean209.8, raw +21.7pp | 2026 healthy starts yards **324,235,186,222** = **3/4 under237.5**; but BUF defense allows **298.7 QB passing yards/game, 2nd-most**, and return/ankle context; high variance. | **HOLD**, not a certified lottery anchor. https://statsnext.com/nfl/players/lamar-jackson-d62fd965/passing-yards |
| **Joe Burrow CIN UNDER272.5** | mean258.3, raw +11pp | 2026 **254,207,282,428,190,292** = **3/6 under272.5**, latest 292, TEN poor defense. | **HOLD/REJECT from safe tiers**. https://www.bengals.com/team/players-roster/joe-burrow/logs/ |
| **Aaron Rodgers PIT OVER218.5** | mean **232.3**, uncalibrated P **57.89%** | 2026 yards **221,187,292,299,235,253,212** = **5/7 over218.5**. Week4 **299 at Cleveland** on 40 attempts. Cleveland defense sixth/fewest team passing yards (~201.7) and opponent QB ~220.7, genuine countercase; Pittsburgh 37.6 pass attempts/game; role intact. Fresh public **FanDuel 218.5 OVER -113** (implied **53.05%**) on BetSharp, also FD Research published 218.5 Friday. | **PROMOTE to conditional HIGH-tier candidate**, ahead of weaker Strange 4+ or Dulcich 28.5. Avoid same-ticket pairing with Warren reception props; can allocate Rodgers HIGH and Warren SAFE/MED (cross-ticket shared game risk still matters). Sources https://www.nfl.com/players/aaron-rodgers/stats/logs/2026/ ; https://www.fanduel.com/research/nfl/player-news/game-updates/aaron-rodgers-and-steelers-play-browns-in-week-8-11-1-2026 ; https://betsharp.com/nfl/game/2026-11-01-cleveland-browns-at-pittsburgh-steelers ; https://www.reuters.com/sports/nfl/browns-blow-11-point-lead-rally-past-steelers--flm-2026-10-02/ |

### IMPORTANT CORRECTION: archived `edge_pct` is a no-vig market gap, NOT profit margin at bookmaker price

On the archived board the symmetric FD -114/-114 pair has `market_prob=0.5`, so an example `edge_pct=0.0789` represents `PRISM P57.89%-no-vig market50%=7.89pp`, **NOT** a 7.89pp advantage over the actual price break-even of 53.27%. The correct break-even comparison must use the exact offered odds, and PRISM raw fair_prob is **overconfident** (Week7 Brier0.280 vs market0.249). The following are **raw uncalibrated model P minus posted single-leg break-even** (not actual verified value/EV):

| Player side | PRISM raw P | public/archived price | price breakeven | raw P − breakeven |
|---|---:|---:|---:|---:|
| Rodgers O218.5 pass | 57.89% | public FD -113 | 53.05% | +4.84pp |
| Tre Tucker O3.5 rec | 61.8% | public FD -120 | 54.55% | +7.25pp |
| Jaylen Warren O2.5 rec | 64.3% | Oct29 archived FD -148 | 59.68% | +4.62pp |
| Omarion Hampton O5.5 rec yds | 72.44% | public FD -114 | 53.27% | +19.17pp |
| Mason Taylor O22.5 rec yds | 63.43% | Oct29 archived FD -114 | 53.27% | +10.16pp |
| Jake Ferguson 2+ rec (O1.5) | 79.43% | Oct29 archived FD -188 | 65.28% | +14.15pp |
| Travis Kelce U4.5 rec | 65.5% | Oct29 archived FD -132 | 56.90% | +8.60pp |

These gaps are **not trustworthy profit estimates** because probability calibration is unproven and some prices old. Treat as a ranking aid subject to football role, independent hit path, and current in-app price. **New explicit reject:** Brenton Strange 4+ receptions = archived FD O3.5 at -138, PRISM P only **49.02%**, sportsbook break-even **57.98%**. Last three 5/7/7 could represent role change but original model does **NOT** endorse that over at this price. Do not call Strange 4+ a model-approved HIGH leg; replace with Rodgers O218.5 conditional. Strange 2+ can remain SAFE only if alt quote is not punitive.

## New weather and status verification ~03:00 ET

- Stadium-hour Jacksonville (IND/JAX): **16mph wind, gust24, rain chance30%, precip only0.01in/hr**; much less precipitation than older broad daily reports, but sustained wind still significant for Daniel Jones pass overs. https://weather.football/nfl/game/2026-08-ind-at-jax
- Miami (NE/MIA): stadium-hour **13mph wind, gust24, rain chance71%, 0.16in/hr average during 3-hour game**; continue weather risk to Maye/Henry and avoid heavy QB passing ladders. https://weather.football/nfl/game/2026-08-ne-at-mia
- Official NFL injury report last updated **Oct31 4:25pm ET** still lists BUF DJ Moore and Dalton Kincaid Q, JAX Jakobi Meyers Q, DAL George Pickens Q, NYJ Breece Hall Q. No confirmed Sunday game-day final inactive list. https://amp.nfl.com/news/nfl-week-8-injury-report-2026-season
- Current public FanDuel Rodgers 218.5 O -113, Tucker O3.5 rec -120, Hampton O5.5 rec -114, Taylor O49.5 rush -113, Warren O2.5 prices vary -141 to -156; all need actual FanDuel user receipt and T-90 inactive check. No combined slip prices confirmed.

## Revised six-ticket DRAFT architecture, replacing an objectively weak HIGH leg

**SAFE** Warren 2+ rec (PIT/CLE), Ferguson 2+ rec (DAL/ARI), Strange 2+ rec (JAX/IND). Prices of low alts unverified; not guaranteed safe.

**MODERATE** Hampton O5.5 receiving yds (LAC/LAR), Shakir 30+ receiving yds (BUF/BAL), Tucker O3.5 receptions (LV/NYJ), Jonathan Taylor 40+ rushing yds (IND/JAX). Tucker FD public -120, Shakir 30+ expensive, Taylor 40+ alt unverified.

**MEDIUM** Coleman O29.5 receiving yds (BUF/BAL), Mason Taylor O22.5 receiving yds (NYJ/LV), Warren 3+ rec (PIT/CLE), Kelce U4.5 rec (KC/DEN). Coleman teammate Q, Mason receiving role emerging, Warren 3+ 5/7.

**HIGH (revised)** Jonathan Taylor O49.5 rushing (IND/JAX), **Aaron Rodgers O218.5 passing (PIT/CLE)**, Hunter Henry 3+ receptions (NE/MIA), Chris Godwin 30+ receiving yds (TB/ATL), Ja'Marr Chase U79.5 receiving yds (CIN/TEN). All five distinct games; Rodgers replaces Strange 4+ which has negative PRISM price threshold evidence. Weather risk Henry; Godwin 30+ alt quote missing; Chase injury-truncated game and soft TEN pass defense make under volatile. Not approved until all quotes validated; may trim if needed.

**LOTTERY** Coleman O29.5 receiving yds (BUF/BAL), Hampton O43.5 combined rush+rec (LAC/LAR), Mason Taylor 30+ receiving (NYJ/LV), Ferguson 3+ rec (DAL/ARI), Daniel Jones 200+ passing (IND/JAX), Kelce U4.5 rec (KC/DEN), Chase U6.5 rec (CIN/TEN). Seven distinct games; Jones weather, Hampton RB opportunity mass, Mason 30+ above PRISM mean, Ferguson 3+ 4/7. Still provisional.

**FUN $1M** Preserve separate 14-leg exhibition `WEEK8_MILLION_DOLLAR_MOONSHOT_EXHIBITION_V1_20261031.md`, not priced. On $100 stake $1m gross requires **>=10,000x decimal**, and 14 identical -110 independent legs would only multiply to **~8,542x**, before any SGP+ correlations or restrictions. No real $1m quote and cannot promise this payout. See exhibition file for exact 14 markets.

**Player exposure:** Warren SAFE/MED, Ferguson SAFE/LOT, Strange SAFE only, Hampton MOD/LOT, Shakir MOD, Tucker MOD, Taylor MOD/HIGH, Coleman MED/LOT, Mason MED/LOT, Kelce MED/LOT, Rodgers HIGH, Henry HIGH, Godwin HIGH, Chase HIGH/LOT, Jones LOT. Max two regular tickets/player. No shared game in HIGH; all six tickets unplaced and quote-conditional.

## Next

Morning report by 8AM ET target, but official 1PM inactives at ~11:30AM. Re-check updated injury statuses, market changes, quote actual FD alternates and combined odds, avoid false PRISM raw edge. Do not stop hourly task until full deliverable with gates is sent. No need to send user a progress notification for this internal rejection/repair checkpoint unless a meaningful new verified leg requires action.
