# M95M — RB Cross-Season Tail Postmortem Results

Research-only diagnostic. No model fit, no feature search, no coefficient search, no sportsbook inputs, and no production change.

## Authoritative run

- Workflow: `M95M RB Cross-Season Tail Postmortem`
- Run: `33433593731`
- Job: `99624596080`
- Tested SHA: `52b537dcd52561a1545a8c87b381c1ea5fca63da`
- Branch: `research-rb-m95m-cross-season-tail-postmortem`
- Artifact: `migration-95m-rb-cross-season-tail-postmortem`
- Artifact ID: `9773558859`
- Artifact SHA256: `4b7277e815db7a5bf90a6abb9ef61efa23b7919fcfec5ef43c5c0ca6c206aead`
- Execution: success
- M95L sealed status inherited: `failed_in_m95l`
- M95M role: `postmortem_only_no_model_change`
- Primary pattern: `cross_season_nonstationarity_same_window`

## Why M95M existed

M95K was the strongest stable-workhorse result in the 2025 research-validation population, but the frozen architecture failed the sealed 2023 W13-18 confirmation in M95L. M95M was deliberately descriptive: compare the already-produced traces and identify *what changed* without retuning M95K against the now-open 2023 confirmation labels.

## Population comparison

### 2025 full research validation

Stable workhorses: `237` games.

20+ carries:

- M95F AUC `.581185`
- M95K AUC `.641164`
- AUC gain `+0.059979`
- Brier `.186593 -> .171528` (`+0.015065` improvement)

25+ carries:

- M95F AUC `.591714`
- M95K AUC `.612631`
- AUC gain `+0.020917`
- Brier `.053017 -> .051386` (`+0.001630` improvement)

### 2025 Weeks 13-18 only — same calendar window as M95L

Stable workhorses: `85` games; 24 actual 20+ events; 6 actual 25+ events.

20+ carries:

- M95F AUC `.646858`
- M95K AUC `.732923`
- AUC gain **`+0.086066`**
- Brier `.194926 -> .186846` (`+0.008080` improvement)

25+ carries:

- M95F AUC `.521097`
- M95K AUC `.436709`
- AUC gain **`-0.084388`**
- Brier `.071105 -> .075821` (`-0.004716` regression)

This is important: the 20+ M95K effect remains strong even when 2025 is restricted to the same late-season W13-18 window. Therefore the 2023 20+ failure is not explained merely by using a late-season confirmation window.

The 25+ method is less robust: it improved on full-season 2025 but failed even in late-2025 W13-18, before also failing in 2023 W13-18. The 25+ conditional-ratio/mass-anchor branch should be treated as particularly unstable.

### 2023 Weeks 13-18 sealed confirmation

Stable workhorses: `73` games; 24 actual 20+ events; 10 actual 25+ events.

20+ carries:

- M95F AUC `.727041`
- frozen M95K/M95L AUC `.545068`
- AUC gain **`-0.181973`**
- Brier `.233221 -> .244446` (`-0.011225` regression)

25+ carries:

- M95F AUC `.533333`
- frozen M95K/M95L AUC `.442857`
- AUC gain **`-0.090476`**
- Brier `.123614 -> .126356` (`-0.002742` regression)

## Signal-stability finding

The strongest 2025 player-current-season carry-ceiling features are the clearest source of nonstationarity.

25+ univariate AUC:

- player current-season p95 ceiling: `2025 full .717015`, `2025 W13-18 .622363`, `2023 W13-18 .481746`
- player current-season p90 ceiling: `2025 full .684031`, `2025 W13-18 .544304`, `2023 W13-18 .486508`
- team current-season lead-RB p95 ceiling: `2025 full .654867`, `2025 W13-18 .526371`, `2023 W13-18 .547619`
- composite carry-ceiling95: `2025 full .623492`, `2025 W13-18 .531646`, `2023 W13-18 .528571`

For 20+ the player current-season ceilings also weakened sharply:

- player current-season p95: `2025 full .631029`, `2025 W13-18 .732923`, `2023 W13-18 .534439`
- player current-season p90: `2025 full .624272`, `2025 W13-18 .708675`, `2023 W13-18 .538265`

However, **not every feed signal disappeared in 2023**. Examples:

- `feed25_rate` remained useful for 2023 20+ (`AUC .640731`) and 25+ (`.586508`)
- team lead current-season p90 remained useful for 2023 25+ (`.580952`)
- composite carry-ceiling90 remained mildly positive for 2023 25+ (`.553968`)

Therefore the postmortem does **not** support the conclusion that workload/feed history is universally useless. It supports a narrower conclusion: the exact M95K global combination, especially the player-current-season ceiling relationship that was unusually strong in 2025, is not stationary across seasons.

M95M counted six feed/ceiling signals above `.60` AUC for 2025 25+, and seven feed signals that were strong in 2025 but inverse (< `.50`) in the sealed 2023 population.

## Sample depth is not the explanation

The 2023 M95K rerank hurt stable 20+ AUC in every sample-depth tercile:

- low depth: `-0.152778`
- mid depth: `-0.140625`
- high depth: `-0.243697`

For 25+:

- low: `-0.090909`
- mid: `-0.015873`
- high: `-0.225000`

So the sealed failure cannot be dismissed as simply "not enough player history." In fact, the worst 2023 20+ degradation occurred in the highest sample-depth slice.

## Casebook — what the failed reranker actually did

Representative harmful 2023 20+ reallocations:

- Kyren Williams W14: actual `25`; M95F `.24097 -> M95K .14356`
- Chuba Hubbard W14: actual `23`; `.20478 -> .11990`
- Rachaad White W14: actual `25`; `.22113 -> .14362`
- Jonathan Taylor W17: actual `21`; `.22770 -> .15378`
- Josh Jacobs W14: actual `13`; `.16818 -> .24094`
- Derrick Henry W18: actual `19`; `.09374 -> .18888`
- Derrick Henry W15: actual `16`; `.10814 -> .17157`
- Derrick Henry W17: actual `12`; `.09864 -> .15668`

Representative harmful 2023 25+ reallocations:

- Josh Jacobs W14: actual `13`; `.16818 -> .23484`
- Rachaad White W14: actual `25`; `.10090 -> .06350`
- Kyren Williams W14: actual `25`; `.06961 -> .04015`
- Derrick Henry W18: actual `19`; `.02693 -> .05257`

This pattern is consistent with a stale/persistent-ceiling problem: historical workload reputation sometimes caused M95K to boost an established back on a week that did not convert, while simultaneously downweighting a current-week high-volume game from another incumbent.

The casebook is evidence for a *conditional-response* problem, not permission to hand-edit these individual players after the fact.

## Scientific interpretation

M95K remains a legitimate strong 2025 research finding, but its exact stable-workhorse reranker is not general enough for promotion.

The most defensible interpretation after M95M is:

1. **20+ feed/ceiling information can be useful, but its mapping to current-week workload is conditional and season/context dependent.** The same late-season window works in 2025 and fails sharply in 2023.
2. **25+ is even less stationary.** The frozen 25+ transformation failed in both late-2025 and sealed late-2023 despite a positive full-2025 result.
3. **Player-current-season p90/p95 is not a universal carry-ceiling prior.** It was very strong in 2025 but near random/inverse in 2023.
4. **Some team/feed composites remain useful across the failed season**, so the correct conclusion is not to discard all workload-history information.
5. **Sample depth does not rescue the architecture.** The issue is more likely role/context/response heterogeneity than raw history quantity.
6. M94C remains the central carry reference. M95F remains the safer stable-workhorse tail baseline. M95I vacancy remains a separate unresolved/promising regime.

## Modeling consequence

Do not retune M95K on the opened 2023 labels.

The next research question should be whether the response to the same football signals is **conditional on a player's current micro-environment**, rather than assuming one global stable-workhorse mapping. Candidate approaches to audit include pregame-only regime/archetype interactions, hierarchical/random-slope effects, or a mixture-of-experts/gating architecture. The next step should test whether these conditional response regimes are real before fitting a new production candidate.

No M95M result is promoted to production.
