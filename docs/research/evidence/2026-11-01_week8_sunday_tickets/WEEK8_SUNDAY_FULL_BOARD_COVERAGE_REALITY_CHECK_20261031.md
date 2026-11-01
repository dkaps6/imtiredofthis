# Sunday Week 8 PRISM coverage reality check — October 31, 2026

**Status: EVIDENCE ONLY. Existing five-tier tickets are drafts, not a completed selection audit or verified placed wagers.**

Source: canonical archived Week 8 CSV blob a40139684b98230e5adf4b87461ffa74d9accebd (October 29, run 40953584718). NFL Week 8 schedule confirms 12 Sunday games (November 1), excluding Thursday Carolina-Green Bay and Monday Chicago-Seattle. No new odds API calls or model changes.

## Quantified opportunity pool

| Scope | Count |
|---|---:|
| Full Week 8 quote-side rows | 2,820 |
| All Week 8 distinct non-touchdown player-markets | 386 |
| Sunday November 1 non-touchdown rows | 1,246 |
| Sunday unique non-touchdown players | 143 |
| Sunday unique non-touchdown player-markets | **327** |
| Sunday player-markets with at least one October 29 archived model edge >=5 percentage points | **261** |
| Unique players represented in the 261 model-edge candidates | **133** |

**261 are raw model/book screening signals, NOT 261 valid or profitable bets.** Book sides, prices, roles and injuries can be stale; model fair_prob is not validated calibration. Among the 261 best-edge player-markets: 92 OVER and 169 UNDER. Market breakdown: 86 receptions (16 over, 70 under), 83 rec_yards (41 over, 42 under), 17 pass_yards (10 over, 7 under), 49 rush_yards (17 over, 32 under), 26 rush_rec_yards (8 over, 18 under).

## Game-by-game raw opportunity inventory

| Sunday game | Unique non-TD player-markets | 5pp+ model-screen candidates |
|---|---:|---:|
| Arizona–Dallas | 25 | 21 |
| Atlanta–Tampa Bay | 21 | 10 |
| Baltimore–Buffalo | 43 | 37 |
| Cincinnati–Tennessee | 31 | 26 |
| Cleveland–Pittsburgh | 36 | 29 |
| Denver–Kansas City | 19 | 15 |
| Detroit–Minnesota | 25 | 18 |
| Indianapolis–Jacksonville | 21 | 15 |
| Chargers–Rams | 26 | 23 |
| Las Vegas–Jets | 26 | 22 |
| Miami–New England | 35 | 29 |
| Philadelphia–Washington | 19 | 16 |
| **Total** | **327** | **261** |

## Hard methodological correction

Previous conversations prematurely promoted about 15 players and proposed five tiers without full player-market-by-player-market review of this 327-opportunity Sunday population. The list was a sampled shortlist, **not proof that only 15 names qualify**. Existing five regular tickets and 14-leg exhibition are not final priced wagers. Important raw outliers include invalid team identity (archived Stefon Diggs NE), RJ Harvey on IR, other late injury/role changes, huge systematic rush/rush+rec UNDER clusters, and 2026 partial-game contamination.

The 200W/161L/8void (~55.4% on 361 settled) prior Week 7 priced-model signal grade was illustrative model-signal evidence, not proof that manual parlay selection works. Do not conflate model accuracy, risk-adjusted market entry, or correlated combined-ticket survival.

**Required corrected gate:** For all 327 Sunday player-markets, record football-side opinion, identity and injury validity, last usage/role, opponent/game-script fit, model family trust, exact main-line direction, distribution for chosen alt threshold, current sportsbook quote, then a **reason-coded disposition** (reject, hold, approved). Compare rejected vs accepted before kickoff without outcome leakage. Tickets only from approved priced legs; no forced leg count, universal anchors, posthoc winner rescue, uncertified ATD probability, invented payout, or new paid OddsAPI.

**Current completion:** Entire-board quantitative inventory COMPLETE; individual football and quote audit of all 327 NOT COMPLETE. The user rightly identified that we had prematurely selected slips before exhaustive candidate discovery.