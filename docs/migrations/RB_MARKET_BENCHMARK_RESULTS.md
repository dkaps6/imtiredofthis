# RB Market Benchmark — 2025 M94C vs Archived DK/FD Rushing-Yard Lines

## Purpose

Benchmark the frozen football-only M94C 2025 RB rushing-yard projection against real archived sportsbook rushing-yard lines and actual outcomes. This is a **downstream market audit**, not sportsbook-assisted model training and not a reopening of exposed-2025 M96 router retuning.

## Authoritative run

- workflow: `RB Market Benchmark`
- run: **`33499129109`**
- job: **`99828098063`**
- tested SHA: **`a26ad1a9991c2f9303d30e4f5b4cff25c3e9d30c`**
- artifact: **`9796956965`**
- artifact SHA256: **`6759e7d8157ade3d4f9237e21a30feacb2507f77f03904ca85740683b7f96475`**
- execution: success
- sportsbook inputs into football model: `0`
- football model changes: `0`
- feature/weight/threshold search: `0`

## Source definition and caveat

The market source is the same public Action Network-derived archive audited previously in M60B:

`gcampb41/nfl_data-/data/processed/football/nfl/player_props/2025.parquet`

Only exact full-game `rushing_yards` straight props from DraftKings (`book_id=68`) and FanDuel (`book_id=69`) were eligible.

The archive does **not** preserve a trustworthy fixed pre-kick timestamp. These lines are therefore labeled **`archived_latest_per_book` / closing-like** and must never be described as a 30-minutes-before-kickoff snapshot.

Source audit:

- raw archive rows: `1,405,406`
- exact `rushing_yards` rows: `29,192`
- full-game rushing-yard rows: `22,661`
- DK/FD full-game rows: `7,899`
- eligible consolidated book-player rows: `2,891`
- conflicting player/week/team/book groups dropped: `18`
- one-sided groups dropped: `2`

The first benchmark workflow run (`33498879907`) was mechanically green but scientifically unusable: its broad rush+yard source filter admitted milestone/combo markets and its full-name identity join produced zero M94C matches because the archive uses abbreviated player names. No usable benchmark metrics were exposed. Run #2 repaired only source/identity mechanics by requiring exact `rushing_yards` and joining by season/week/team plus first-initial/surname short key. Football/model logic and benchmark definitions were unchanged.

## Join / comparison universe

- frozen M94C 2025 RB/FB rows: `1,393`
- eligible market book rows: `2,891`
- matched market book rows: `1,724`
- unmatched book rows: `1,167` (principally non-RB rushing props such as QBs/WRs and unresolved `FA` source-team rows)
- exact market-covered RB player-games: **`899`**
- two-book DK+FD games: `825`
- DK games: `894`
- FD games: `830`

The primary M94C-vs-market comparison uses the exact same `899` player-games and is apples-to-apples.

## Headline result — Vegas is better overall on listed RB games

### M94C football-only projection

- n: `899`
- MAE: **`25.515051`**
- RMSE: `34.364907`
- bias: `-0.579911`
- correlation: `.453546`
- actual mean: `50.014461`
- projection mean: `49.434549`

### Vegas consensus

Consensus is the median of available DK/FD straight rushing-yard lines for the player-game.

- n: `899`
- MAE: **`23.701891`**
- RMSE: `32.493543`
- bias: `-4.327030`
- correlation: `.529751`
- actual mean: `50.014461`
- line mean: `45.687430`

### Direct interpretation

Vegas consensus beat M94C by **`1.813160` rushing yards of MAE** and about `1.8714` RMSE yards on this market-covered sample. Vegas also ranked/located player outcomes better (`corr .5298` vs `.4535`).

M94C had materially better aggregate mean calibration: model bias only `-0.58` yards versus the market's `-4.33`, meaning the archive lines systematically sat below realized mean yards despite being closer player-by-player.

Head-to-head on the same 899 games:

- M94C closer: `403`
- market closer: `496`
- ties: `0`
- M94C closer rate: `44.8276%`
- market closer rate: `55.1724%`
- mean market absolute error minus model absolute error: `-1.813160`
- median difference: `-1.285798`

## Individual books

These rows have slightly different sample composition, so the 899-game consensus remains the primary benchmark.

- DraftKings: n `894`, MAE `23.903803`, RMSE `32.705990`, bias `-4.081655`, corr `.521570`
- FanDuel: n `830`, MAE `24.142169`, RMSE `32.819027`, bias `-4.816867`, corr `.501870`
- two-book consensus: n `825`, MAE `24.229091`, RMSE `32.910917`

## The most important finding — the market edge is concentrated in disagreements

Absolute M94C-vs-consensus disagreement:

| disagreement | n | M94C MAE | market MAE | M94C gain vs market |
|---|---:|---:|---:|---:|
| `<5` | 277 | `24.5481` | `24.6390` | **`+0.0909`** |
| `5-<10` | 247 | `22.8366` | `21.9150` | `-0.9216` |
| `10-<15` | 164 | `22.8550` | `21.2470` | `-1.6080` |
| `15+` | 211 | `31.9875` | `26.4716` | **`-5.5159`** |

When model and market are within five yards, M94C is effectively as accurate as the market and has a tiny mean-MAE advantage. The market advantage grows rapidly as disagreement widens.

This is evidence that the remaining model problem is **not universal rushing-yard noise**. It is concentrated in player-games where M94C's pregame football state materially disagrees with the market's role/workload expectation.

## Directional disagreement asymmetry

The most concerning regime is **M94C materially ABOVE the market**.

When M94C was `15+` yards above consensus:

- n `144`
- actual finished above the market line only `52.78%` of the time
- M94C was closer only `36.11%`
- M94C MAE `33.0303`
- market MAE `25.3924`
- market advantage **`7.63798` yards**

So a large M94C-over-market disagreement did not translate into strong directional information and was usually a model error in magnitude.

When M94C was `15+` yards below consensus:

- n `67`
- actual finished below the market `59.70%` of the time
- M94C was closer `52.24%`
- M94C MAE `29.7462`
- market MAE `28.7910`

The model-low side is therefore much less damaging than the model-high side and even carries some independent directional information.

## Postgame workload diagnostic — Vegas did NOT solve the extreme tail

This slice uses actual carries only to explain the benchmark after the fact; actual carries may never be used as a pregame selector.

| actual carries | n | M94C MAE | market MAE | better |
|---|---:|---:|---:|---|
| `0-5` | 188 | `19.8329` | `15.6596` | market by **`4.1733`** |
| `6-10` | 232 | `22.1260` | `19.9655` | market by **`2.1605`** |
| `11-14` | 208 | `24.9692` | `24.0313` | market by `0.9380` |
| `15-19` | 177 | `29.4814` | `28.2034` | market by `1.2780` |
| `20+` | 94 | **`38.9831`** | `39.8032` | **M94C by `0.8201`** |
| `25+` | 23 | **`51.0291`** | `54.7826` | **M94C by `3.7535`** |

This is a critical correction to the naive conclusion “Vegas solved RBs.” It did not.

The market's overall advantage is heavily driven by being better at identifying listed RBs whose realized workload collapses into the 0-10 carry range. Once a player actually reaches 20+/25+ carries, M94C is slightly to materially better than the market, although **both** remain severely low on those extreme outcomes.

For actual 25+ games:

- actual mean `123.09` yards
- M94C mean `73.28`
- market mean `69.52`

The extreme workload/yardage tail remains unsolved by both systems.

## Week / role-initialization clue

The market advantage was particularly large early in the season:

- Week 1: market better by `7.4636` MAE yards
- Week 3: `4.0859`
- Week 6: `2.6349`
- Week 8: `3.1332`
- Week 10: `4.5940`

M94C beat market in Weeks 2, 12, and 13, with Week 16 essentially tied.

Week 1 alone contributed roughly 24% of the total 899-game market absolute-error advantage despite representing only about 6% of the sample. But removing Week 1 does **not** erase the market edge:

- Weeks 2-18: M94C MAE about `25.1926`, market `23.7335`, market advantage about `1.4592`
- Weeks 13-18: M94C about `24.4824`, market `23.4400`, market advantage about `1.0424`

So Week-1 initialization is a major issue but not the only one.

## Casebook clues

Large M94C-over-market failures include player-games such as Jerome Ford W1, Jonathan Taylor W1/W4, Jaylen Warren W3/W9, Rico Dowdle W8/W11, Derrick Henry W3, and Bijan Robinson W1. These are consistent with stale/overconfident role or workload expectations rather than a universal efficiency problem.

Large M94C-under-market cases in Week 1 included rookies/new roles such as Ashton Jeanty, Omarion Hampton, TreVeyon Henderson, Jacory Croskey-Merritt, RJ Harvey, plus Christian McCaffrey. This strongly implicates **rookie/new-team/depth-chart role initialization** as a distinct football-only data problem.

M94C also strongly beat the market in selected games, including James Cook W9/W12, Tony Pollard W2, Derrick Henry W1/W18, Saquon Barkley W7, and Kenneth Walker W5. Therefore the model carries independent signal and should not be replaced with sportsbook lines.

## Scientific conclusion

The retrospective M96 router stopping rule remains valid: do not continue changing exposed-2025 thresholds/features until they pass. But this downstream benchmark exposes a **separately justified new-data research path** that the stop rule explicitly allowed.

The principal football-only gaps suggested by the market benchmark are:

1. **False-high workload suppression / workload-collapse detection.** The market is dramatically better in realized 0-10 carry games.
2. **Week-1 / rookie / new-team role initialization.** M94C sometimes starts players near zero or at legacy workloads when the market already reflects current depth-chart/role information.
3. **Current availability and depth-chart transitions.** Large model-high disagreements suggest stale role/availability state in some games.
4. **Extreme workload tail remains open.** Vegas is not better in actual 20+/25+ games, so do not use the market result as evidence that the tail is solved.

The market itself must remain downstream. The correct next investigation is to identify **football-only pregame information that the market is apparently incorporating and M94C may not be ingesting quickly enough**: official/current depth charts, transactions/new-team status, injury/practice participation and game-status timing, inactive/active information available before kickoff, rookie/draft/college workload priors where appropriate, coaching/backfield usage priors, and offensive-line availability.

Those data sources should be audited first and only then tested under a fresh, precommitted football-only protocol. Do not use the sportsbook line as a feature, training target, or gating variable.
