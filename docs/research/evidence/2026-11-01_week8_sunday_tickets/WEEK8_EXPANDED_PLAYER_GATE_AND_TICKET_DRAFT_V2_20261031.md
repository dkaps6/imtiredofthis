# Week 8 Sunday (Nov 1, 2026) — expanded PRISM football gate and 5-tier V2 betting draft

**Status:** 2026-10-31 late evening. This V2 *supersedes the seven-player abbreviated draft*, not the root science handoff; DO NOT call these placed/fully approved odds-confirmed bets. User wants FIVE distinct parlay tiers and generally more than two legs/tier but no arbitrary cap; selection must lead construction. Last Week7 original pregame **369 actionable decisions** yielded 361 settled, 200W/161L/8 void, ~55.4% *individual original signal selection*, +21.10 illustrative flat units, not 55% of all board rows, not a parlay hit rate or proof future odds are good. 2820 rows in W8 archival board include two sides & multi-book/market duplicates, and stale line/team states; do not count as 2820 unique qualified bets.

**Source:** committed PRISM W8 board blob a40139684b98230e5adf4b87461ffa74d9accebd (`data/market_track_record/boards/2026_wk08.csv`, source date Oct29). Public sportsbook aggregator `https://karmaedge.ai/nfl/player-props-today` as-of Oct31 10:27 PM ET for reference; actual FanDuel alt milestone pricing, availability, parlay payouts NOT VERIFIED. NFL 2026 injury report updated Oct31 5:25 ET. GitHub source `docs/research/MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1.md`; separate steering: football projection -> reliability -> current main market direction -> exact price -> alt threshold/correlation/ticket construction. **Never flip an UNDER to an OVER merely because buying down permits an easier milestone.**

## 13 current unique player *investigations* screened, not all independently certified to wager; model mean = archival raw mean.
| Player/side | PRISM mean | Last archived main & market/line | Independent football evidence to use | Verdict |
|---|---:|---|---|---|
| Jaylen Warren PIT OVER receptions | 3.3 catches | over2.5 approx -149 DK Oct29; Oct31 market2.5 | 2026 receptions 5,2,3,3,6,2,3. 2+ 7/7 and 3+ 5/7; Browns official says 24 rec / 294 yd, Steelers no2 receiver, pass-check role stable | Strong ALT 2+; main 3+ higher variance |
| Drake Maye NE OVER passing | 260.6 yards | main ~239.5 Oct31, source 233.5 Oct29 | 175+ in 7/7 2026 incl games A.J. Brown missed. Oct31 **A.J. Brown OUT**, WR Doubs/Hollins on IR; model mean stale. DON'T trust main >239.5; 175+ conditional floor, 200+ higher risk | Strong/conditional alt, not main |
| Brenton Strange JAX OVER receptions | 3.8 rec | main3.5 (book prices Oct31); Oct29 3.5 | 2026 catches 2,2,3,7,7,5 (6/6 >=2; 4/6 >=3); recent 5/7/7 catches and 5-8 targets; Jakobi Meyers Q may concentrate targets; if Meyers fully active re-check | Strong 2+; 3+ more variance |
| Jonathan Taylor IND OVER rushing | 68.6 yards | main now ~50.5 (Oct29 model source 59.5–60.5) | 2026 mean 83.9, 6/7 >55.5, 7/7 >=40; BUT JAX toughest vs RB rush yards and Colts projected big underdogs, so 40+ alt preferable to main50.5 unless Sunday circumstances reinforce carries | Strong alt 40+, medium main |
| Jake Ferguson DAL OVER receptions | 2.6 | main 1.5 at FanDuel Oct29; BetMGM Oct30 1.5 | 2026 2,4,3,3,3,2,**0**. 6/7 >=2; W7 0 is an anomaly to INVESTIGATE: Dallas only 27 pass attempts while winning vs PHI; no certainty about routes. Arizona matchup, DAL WR Pickens Q | Conditional 2+, 3+ speculative |
| Khalil Shakir BUF OVER receiving | 48.5 yards | archived38.5/39.5, reported ~38.5–41.5 Oct31 | weekly 50,38,6,82,80,34; 30+ 5/6, full 38.5 3/6. Avg48.3/5.3 targets; BAL fifth-most WR receiving yds, two teammate Qs Moore/Kincaid; need check allocation, avoid automatic Coleman+Shakir same slip | Favor 30+ ALT, full over only aggressive |
| Greg Dulcich MIA OVER receiving | 39.0 yards | archived28.5–30.5, Oct31 ~28.5–29.5 | 2026 yards18,17,55,0,37,86, 15+ 5/6, main over28.5 only3/6; W4 zero TARGETS despite >50% snap => role volatility; ankle limited early this week, official final status check required; MIA passing low | Conditional 15+ alt; main risky |
| Ja'Marr Chase CIN UNDER receiving / catches | 55.4 rec yds, 4.5 catches | FD under79.5 rec yds ~-114 Oct31; under6.5 catches referenced Oct29 | 2026 rec yds12,75,98,27,77,62 => 5/6 <79.5; catches2,7,9,3,6,5 => 4/6 under6.5. But TEN permits sixth-most WR receiving, possible quick 80-yard day; team 12.5 pt favorite supports reduced second-half attempts. Treat low 27 game as possibly contaminated, not proof of lasting decline | Main 79.5 under conditional; catches under6.5 higher risk |
| Keon Coleman BUF OVER receiving | 42.1 yards | FD over29.5 ~-113 Oct31 | 2026 yards1,63,37,116,7,59; 20+ and 29.5+ BOTH 4/6! Bought-down 20+ has not improved empirical survival, just degraded payout at unknown price; BAL #5 soft vs WR receiving, mean47.2, but downfield targets ~14.3 yards and 4 targets/game volatile; Moore/Kincaid Q | Main 29.5 value check, NOT 20+ buy-down |
| Daniel Jones IND OVER passing | 230.3 yards | archived188.5, current main192.5 | 2026 pass166,210,235,143,267,269,108, 150+5/7, 175+4/7, 200+4/7. JAX generous pass defense, IND underdog, but erratic completion/last108 and possible gusts. 150+ is ALT with evidence; main over192.5 volatile and conditional | 150+ alt; main aggressive |
| Dylan Sampson CLE OVER rushing | 21.2 yards | Oct31 11.5 (archived12.5) | Week7 return from injury, 6 carries 45 yards, now full practice. 10+ plausible if #2 work persists; one observed game is **not** a robust rush role, especially with Judkins primary and PIT defense | Conditional 10+, main11.5 speculative |
| Lamar Jackson BAL UNDER passing | 209.8 yards | archived237.5 | 2026 four healthy starts pass324,235,186,222; missed Weeks5–7 due HIGH ankle sprain, officially returning Nov1 with residual bearable pain, first post-injury QB state. Under237.5 happened 3/4 previous, but high-scoring BUF game and fewer QB scrambles could increase passing. **Do not put into Safe/Moderate** | Higher-risk conditional under |
| Darnell Washington PIT OVER receiving | 23.7 yards | archived main20.5 | 2026 rec yds18,7,67,27,12,25,4: 10+ 5/7 vs main20.5 only 3/7. Cleveland allowed 27 to him prior meeting. W7 had only 1 target, 4 yards, so volatile; 10+ if price fair, not core | Risky ALT 10+ |

## Other large apparent edges rejected or downgraded
- Kirk Cousins under240.5 passing despite PRISM under 220.8: 2026 over240.5 in **six of seven**, reject.
- Trey McBride under7.5 catches: PRISM 6.2, but McBride 2026 8.43 catches/game, 5 of 7 over7.5, 11+ targets; reject.
- DJ Moore under4.5 catches: shoulder questionable and rosters uncertain, defer not automatic.
- J.K. Dobbins over 21.5/29.5 rushing with Harvey OUT: Jonah Coleman first return was 28 offensive snaps, 9 carries54 yards vs Dobbins 6/23; do not project easy Dobbins workload without resolving that usage.
- Ja'Marr Chase 80+ receiving as lottery alternative is opposite PRISM side. Reject.
- Tee Higgins under70.5 receiving: model thinks under (57), but season actual >70.5 in **4/6**; football contradicts, reject.
- Keon Coleman 20+ bought-down over: same 4/6 outcomes as 29.5, no documented outcome improvement in current season; reject unless price/unique distribution justifies.
- Courtney Sutton Over40.5: 2026 only 2/7 40+ while PRISM no reliable board signal, no.
- Puka Nacua under92.5: high recent target volatility and 125/138 explosive starts, less clean than raw model.
- Devontez Walker 0.5 over: outside football experts now label him WR4, reliable role questionable despite raw PRISM 17.4.
- RJ Harvey injured reserve; Stefon Diggs archived NE lineup **invalid**; no alt.
- Monday players excluded (Rome Odunze, Sam Darnold).
- Previously rejected Goff UNDER (5/6 over threshold), Helm over, Henry moved, Godwin moved, rush-UNDER clusters driven by structural model defect.

## DRAFT five-tier parlays with >2 legs / natural variance, no forced fixed N
The following are **proposed selection architecture**, not bets or verified + FanDuel milestones: check that each alt exists and its true exact price. 13 distinct players appear <=2 slips each.

**SAFE (3 legs):** Warren 2+ receptions; Maye 175+ passing; Strange 2+ receptions. Each proposition has clear recurring opportunity. Alt prices unknown, heavy juice may make the ticket unattractive.

**MODERATE (4 legs):** Jonathan Taylor 40+ rushing; Jake Ferguson 2+ receptions; Khalil Shakir 30+ receiving; Greg Dulcich 15+ receiving. Taylor opponent run D and Dulcich zero-target game make this less clean; take only with acceptable prices.

**MEDIUM (5 legs):** Chase UNDER79.5 rec yards; Coleman OVER29.5 rec yards; Warren 3+ receptions; Strange 3+ receptions; Jones 150+ passing. Two full-line sides carry variance, Coleman teammate news pending. Warren vs Strange different games.

**HIGH (6 legs):** Shakir OVER38.5 rec yds; Sampson OVER11.5 rushing; Taylor OVER50.5 rushing; Ferguson 3+ catches; Dulcich OVER28.5 rec yds; Darnell Washington 10+ receiving. Several fragile usage/target situations, classify high even if books priced at -110; not a safe ticket.

**LOTTERY (7 legs):** Jones OVER192.5 passing; Coleman OVER29.5 receiving; Chase UNDER6.5 receptions; Sampson 20+ rushing; Lamar Jackson UNDER237.5 passing; Darnell Washington 10+ receiving; Maye 200+ passing. Multi-game; some legs are high uncertainty and **if any price or player state fails, PASS** rather than force a replacement. Do not call this high probability just because each leg has a model side.

All 5 cards here are **proposals**, NOT approved actionable wagers. Until exact CURRENT FanDuel screenshot prices, initial run provenance reconciled with final injuries, and a full ticket implied breakeven check, we cannot call any ticket +EV, safe in absolute terms, or promise profits.

**Next action:** verify actual offered FD milestone pricing/screenshot, Nov1 gameday actives and key questionable (DJ Moore/Kincaid, Pickens, Meyers, Dulcich), then score price vs player-specific clearance at selected threshold; reject expensive or unavailable leg without bias. Preserve final placed ticket IDs separately from candidate research. Do not spend paid OddsAPI credits. User has expressly asked for the betting-layer work; this is not model promotion or science merge.
