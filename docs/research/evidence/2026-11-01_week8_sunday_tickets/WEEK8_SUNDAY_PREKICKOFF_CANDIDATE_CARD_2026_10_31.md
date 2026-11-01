# PRISM W8 Sunday — prospectively drafted five risk-tier bet slips (Oct 31, 2026)

**Scope:** Sunday Nov 1, 2026 only. **PREKICKOFF DRAFT, NOT PLACED, NOT FINAL PRICED**. User has authorized bet-slip selection, not new model code, and not a paid OddsAPI call. Sources: 2026 W8 archived `data/market_track_record/boards/2026_wk08.csv` run 40953584718 quoted Oct 29, plus public injury, game-log and bookmaker market checks Oct 31. Main sportsbook quotes can change before kickoff. Exact FanDuel milestone pricing **UNCONFIRMED** wherever stated. Never invent odds or payout. Read `docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`; Week7 forensic: do not pick bought-down OVERs when model canonical main side UNDER; no raw ATDs or blindly fading every RB rushing market.

## Individually screened directions (player/market, PRISM mean; independent football reason and risk)
1. **Jaylen Warren OVER receptions**, PRISM 3.3, 2.5 FD main around -148, p(over2.5)=64.3% raw; official 2026 REC 5,2,3,3,6,2,3; 7/7 reached 2+, 5/7 reached 3+. Steelers offensive involvement; not yardage over (PRISM 25.5 rec yards was under main ~23.5). 2+ is high-juice only at acceptable actual price.
2. **Drake Maye OVER pass yards**, PRISM 260.6, FD main ~237.5 -114 raw p~62.8%; 2026 yardages 178,208,199,269,374,235,321, all 7 reach 175+ and five >=200; MIA weak QB EPA. **A.J. Brown QUESTIONABLE, not OUT (official Oct30); Romeo Doubs / Mack Hollins IR**. Can make high mainstream over brittle, so 175+ alt and high version main with roster check.
3. **Dylan Sampson OVER rush yards**, PRISM 21.2, Thursday model board line12.5, latest main ~11.5, model p(over12.5)=72.2% raw. Came back injury Week7 with 6 carries 45yd and similar offensive snap rate to Judkins, Friday full practice. 10+ core, 20+ speculative, not automatic workhorse (tiny sample).
4. **Keon Coleman OVER receiving**, PRISM 42.06, FD main 29.5 -114 raw p~68.2%; 2026 4/6 at least30, Ravens allowed high WR yardage. BUT BUF WR DJ Moore and TE Dalton Kincaid both QUESTIONABLE Oct30, and Coleman only ~4 targets/g; **role/availability reevaluation is essential when those players' statuses resolve**. 20+ risk-down, main29.5 aggressive.
5. **Daniel Jones OVER passing**, PRISM 230.26, latest FD main ~192.5 -113 (Oct29 source 188.5), 2026 actual166,210,235,143,267,269,108; 4/7 reached175, 4/7 200, 5/7 150. JAX near bottom pass yds allowed, Colts underdog, but weak completion rate, last108 and JAX gusts22mph. **Use 175+ (medium), 200+ (lottery) rather than safe**.
6. **Ja'Marr Chase UNDER receiving yds / receptions**, PRISM rec yds 55.4 and catches4.5; original line 80.5 under raw p85.4%, under6.5 catches raw88.7%; in 2026 5 of6 games under80.5, 4 of6 under6.5 catches. Titans allow high WR production (real anti-thesis); Cincinnati heavily favored (less late volume), Chase full practice despite ankle note. FD observed UNDER 79.5 ~-114 and UNDER6.5 rec ~-102; independently verify.
7. **Justin Jefferson UNDER rec yards / catches** PRISM rec yds49.9 and catches4.1; original under77.5 raw p88.2%; 2026 under77.5 in3/5, avg60.4 (44,79,32,55,92). Lions have favorable WR environment, new QB Kyler Murray connection and explosive upside—**speculative**, not safe; FD under76.5 ~-114, under5.5 catches reportedly +112 at last reported snapshot (recheck market).

**Do not promote high model fair_prob as validated chance**; 2026 calibration overconfident and risk independent of Brier aggregates. Historic prices not contemporaneous confirmed lines.

## DRAFTS — explicit ordered candidate legs; each requires in-app book-line/price confirmation and final Sunday availability
**SAFE — 2 legs (both ALT):**
- Jaylen Warren **2+ receptions** PIT vs CLE (FD milestone price unknown).
- Drake Maye **175+ passing yards** NE @ MIA (FD milestone price unknown).
**MODERATE — 2 legs:**
- Dylan Sampson **10+ rushing yards** CLE @ PIT (FD milestone quote unknown; bookmaker has milestone by independent Oct31 report).
- Keon Coleman **20+ receiving yards** BUF vs BAL (FD milestone quote unknown; conditional on DJ Moore/Kincaid).
**MEDIUM — 3 legs:**
- Daniel Jones **175+ passing yards** IND @ JAX (FD milestone quote unknown).
- Ja'Marr Chase **UNDER 6.5 receptions** CIN vs TEN (FD about -102 last public check).
- Justin Jefferson **UNDER 76.5 receiving yards** MIN @ DET (FD about -114 public last check).
**HIGH — 3 legs:**
- Jaylen Warren **3+ receptions** PIT vs CLE (=OVER2.5 FD about -148).
- Drake Maye **OVER 237.5 passing yards** NE @ MIA (FD ~-114, condition Brown active and roster state; if Brown OUT reconsider/PASS).
- Dylan Sampson **20+ rushing yards** CLE @ PIT (FD price unverified).
**LOTTERY — 4 legs:**
- Daniel Jones **200+ passing yards** IND @ JAX (FD alt price unverified, weather exposure).
- Keon Coleman **OVER 29.5 receiving yards** BUF vs BAL (FD ~-114; Moore/Kincaid condition).
- Ja'Marr Chase **UNDER 79.5 receiving yards** CIN vs TEN (FD ~-114; Titans WR pass D vulnerability).
- Justin Jefferson **UNDER 5.5 receptions** MIN @ DET (some published FD alt +112; must confirm currently offered).
Exactly 7 distinct players, each used <=2 tickets; no same-game pairs in each ticket. No raw ATDs, no invented exact combined probabilities or payouts. User deliberately disallowed forced leg count; each tier has its natural 2–4 legs here. Do not put a ticket on a bet slip if any leg is unavailable or too expensive; PASS rather than force replacement.

## Cleared rejects / valuable negative evidence
- Stefon Diggs NE rec prop **INVALID TEAM** (actual WAS; see separate archived erratum and Issue #535 comment 6552515394).
- RJ Harvey on DEN IR; no plays.
- J.K. Dobbins 21.5 over has Harvey IR but Broncos rookie Jonah Coleman recently out-snapped/produced more, share not proven.
- Ray Davis BUF 4.5 rush over (model huge) has almost no real rushing usage; no.
- Dawson Knox BUF receiving over7.5 (model huge), actually <=7 yds five of six; no.
- Jalen Nailor LV 17.5+ receiving (model huge), current season 1/6; no.
- Ladd McConkey LAC UNDER3.5 catches mean2.0 looks profitable but catch stats were **CONCUSSION / PARTIAL GAME contaminated** (Week6 ~12% snaps); fully practiced now, no projection reliability gate. REJECT.
- Jared Goff passing UNDER259.5 model huge but actual 5/6 over259.5; avoid.
- Bowers/Puka/JSN overs bought down when canonical model favors under; avoid.
- Gunnar Helm rec OVER originally 15.5 but latest line 21.5, ~14 yds/game, little target evidence; no.

## Final-lock preflight
- 1pm official NFL inactives at T-90; 4:25pm later. Check DJ Moore, Dalton Kincaid, A.J. Brown and any changed QB/starters; update role projections qualitatively, do not transplant other player's invalid model.
- Verify actual FanDuel *all leg names + alt milestones + quote prices*; alt existence is not yet confirmed. Show price of each + combined offered payout before deciding if expected value adequate. Duplicating a high-juice -500 alt may make a nominal safe card economically unattractive.
- Gust/weather IND@JAX, price line may move; remove if new role game context materially conflicts.
- Grade prospectively vs actual odds and final active player state, distinguish model signal vs user-placed wagers. No OddsAPI and no code/model production changes.

Status: user requested concrete tickets; these are conservative **drafts**, not authorized claimed positive-EV wagers until fresh pricing.
