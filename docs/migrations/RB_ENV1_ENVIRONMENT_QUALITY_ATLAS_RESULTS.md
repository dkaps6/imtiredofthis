# RB-ENV1 — Environment × Runner Quality Atlas Results

## Purpose

No-fit, no-search diagnostic answering whether leakage-safe pregame rushing environment and pregame runner quality correlate with actual RB rushing outcomes, and identifying exception classes where backs overcome bad spots or fail in good spots.

No sportsbook inputs. No model promotion. Good/bad environment is defined from the already-frozen M95C out-of-sample environment contribution (`role_plus_environment - role_baseline`); runner quality is defined from the frozen pregame M95B efficiency/explosive scores. Outcomes are used only after those grades are frozen.

## Authoritative run

- Workflow: `RB-ENV1 Environment Quality Atlas`
- Run: `33508752571`
- Job: `99859029342`
- Tested SHA: `375e6a776d91132563937fec1d02d85ea56ae69d`
- Artifact: `9800707982`
- Artifact SHA256: `2373625d06e58019d761acdd925dff7f1458835718d66cb1c3f206a3da143df6`
- Execution: success
- Fitted models: 0
- Feature/threshold search: 0
- Sportsbook inputs: 0
- Production change: 0

The first run failed mechanically because SciPy was not installed for Spearman correlation. Only the missing dependency was added; scientific logic was unchanged.

## Headline environment relationship

Across 2,580 out-of-sample RB games from 2024-2025:

- pooled BAD spot average rushing yards: `30.8866`
- pooled GOOD spot average rushing yards: `44.8932`
- GOOD minus BAD: **`+14.0065 yards`**
- BAD 75+ rate: `10.2484%`
- GOOD 75+ rate: **`21.0526%`**
- BAD 100+ rate: `4.6584%`
- GOOD 100+ rate: **`9.9071%`**

The relationship replicated by season:

- 2024: BAD `29.2795`, GOOD `44.2817`, difference `+15.0022`
- 2025: BAD `32.4938`, GOOD `45.5046`, difference `+13.0109`

However, environment is not deterministic:

- pooled environment vs actual rush-yards Pearson: `0.1250`
- pooled environment vs baseline residual Pearson: `0.0774`
- pooled environment vs YPC among 8+ carry games: `0.0524`

Interpretation: a favorable environment materially moves the rushing-yard distribution and tail rates, but much of the outcome remains driven by opportunity/allocation, runner quality, and explosive variance. Environment is an add-on to the football architecture, not a replacement model.

## Environment × runner quality

Pooled 2024-2025 examples:

- BAD + WEAK: `23.8582` yards, 75+ `5.3640%`
- BAD + MID: `34.7390` yards, 75+ `13.4897%`
- BAD + STRONG: `45.7436` yards, 75+ `15.3846%`
- GOOD + WEAK: `31.3333` yards, 75+ `10.4167%`
- GOOD + MID: `39.9560` yards, 75+ `18.5535%`
- GOOD + STRONG: **`52.9568` yards**, 75+ **`25.8993%`**

This supports interaction between player quality and environment, but not a deterministic global multiplier.

## Exception classes

Large bad-spot successes included 2025 Rico Dowdle W5 (23 carries/206 yards), David Montgomery W3 (12/151), Saquon Barkley W8 (14/150), Ashton Jeanty W4 (21/138), James Cook W2 (21/132), Christian McCaffrey W7 (24/129), Kimani Vidal W13 (25/126), and others. Many are explained by large/unexpected workload and/or explosive-run variance overcoming the environmental headwind.

Large good-spot failures included Christian McCaffrey W18 (8/23), James Cook W18 (2/15), Saquon Barkley W12 (10/22), Jahmyr Gibbs W1 (9/19), Derrick Henry W2 (11/23), TreVeyon Henderson W16 (5/3), De'Von Achane W5 (10/16), and Jerome Ford W3 (0/0). These reinforce that a good environment cannot rescue a bad player-allocation/opportunity forecast.

## Durable conclusion

Use environment after establishing the likely backfield state and individual workload. Architecture should remain decomposed:

1. team rushing opportunity,
2. current backfield/player allocation,
3. individual carry distribution,
4. runner ability + blocking/opponent environment,
5. explosive/variance distribution.

Do not treat ENV1 as a replacement for M94C or as permission for a universal matchup multiplier.