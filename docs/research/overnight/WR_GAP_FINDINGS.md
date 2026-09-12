# WR Gap Findings — Overnight Research Survey

**STATUS: RESEARCH ONLY — NOT PROMOTED — AWAITING USER APPROVAL.**

## What's already in production

WR-R15 (`WR_R15_PRODUCTION_MODEL_V1`) + M38 govern **target entitlement** (share of team targets) only. They do not touch efficiency (YPT/catch-rate/YAC) or distribution/tail shape — those are explicitly named as a remaining open lane in `scripts/validate_certified_full_slate_stack_v3.py`'s `remaining_science_lanes`.

## The landscape (18 unmerged branches, Sept 5–7, 2026)

Two chains, both frozen-protocol, walk-forward, zero-sportsbook:

### Chain A — where does the post-M38 error actually live?

1. **ND1** (`research-wr-nd1-post-m38-decomposition`) → `YARDS_PER_TARGET_DOMINANT`. YPT is the dominant error source overall, **except** in 10+ target games, where within-WR allocation dominates instead. Explicit next step: decompose YPT into catch-rate vs YPR (then route-depth/YAC if supported).
2. **`research-wr-post-m38-error-decomposition`** (ND2) → explosive yardage does NOT get the `EXPLOSIVE_YARDAGE_DOMINANT` disposition (29.68%/29.73% share, not the top component, 60% threshold not met). Stop-rules: *do not retune M38 hierarchy multipliers; do not reopen generic target-pool pruning; do not infer fake WR-CB assignments from participation; do not use sportsbook inputs upstream.*
3. **R5** (`research-wr-r5-target-catch-ypr-individual-decomposition`) → identifies a real, distinct **YPR-dominant subgroup** (`WR_TARGET_CATCH_YPR_INDIVIDUAL_MECHANISMS_MAPPED`).
4. **R7** (`research-wr-r7-ypr-mechanism-conditioned-efficiency`) → tested whether persistent player traits (explosive rate, YAC, air-yards history) explain that subgroup's game-to-game YPR misses. **They don't** — results point weakly the *wrong* direction. Explicit redirect: "the next materially new YPR lane should use different pregame information — route/coverage/tracking/QB-delivery or similarly richer matchup mechanics."
5. **R8** (`research-wr-r8-target-dominant-role-signals`) → confirms a real TARGETS-dominant player class, but static participation/depth variables (incl. snap level) aren't precise enough to predict per-game allocation. Explicit next step: "genuinely richer football information — route participation/route type/alignment/air-yard role/coverage matchup/separation or other tracking-quality context." Explicit stop: **do not run another ND5 variant.**
6. **R9 → R10 → R11** (`research-wr-r9-ngs-source-audit` → `-r10-strict-prior-ngs-availability` → `-r11-strict-prior-ngs-target-model`) → this IS the "richer tracking" attempt, using vendor Next Gen Stats (NGS) data. Result: **`WR_NGS_TARGET_MODEL_FAIL`** — targets MAE improved marginally (missed the ≥0.05 gate), but receiving yards MAE got **materially worse** (+1.16 yards). Scientific failure, not source failure; leakage/parity gates all passed. **Rejected, not authorized for production or tuning rescue.**

### Chain B — is there exploitable individual WR persistence? (Different question than R7/R8's mechanism search.)

- **R2** (`research-wr-r2-player-tracking-residuals`) → `NO_ACTIONABLE_PLAYER_TRACKING_SIGNAL`. Rejected.
- **R3** (`research-wr-r3-player-error-persistence`) → **`WR_PLAYER_ERROR_PERSISTENCE_DETECTED`. This is a genuine positive, unexploited finding.** All 3 frozen diagnostics passed on strictly-prior walk-forward history (last 8 games, min 4), 6/6 positive seasons, consistent 2024→2025:
  - Directional bias persistence: Spearman .089, +9.55 signed-yard quartile gap, 55.6% sign agreement — **PASS**
  - Individual difficulty persistence: Spearman .258, +13.9 abs-yard quartile gap — **PASS**
  - Extreme-miss persistence: 1.37x next-game 30+ yard miss enrichment — **PASS**
  - It explicitly names two legitimate integration lanes it did NOT itself execute: (a) prior signed bias → a conservative, shrunk mean-calibration layer; (b) prior difficulty/extreme-miss history → player-specific MC uncertainty/tail calibration. It states a later "predeclared combined full-stack candidate is legitimate" but requires testing through the exact M38 production architecture with its own frozen aggregate + individual-error gates — **that integration test does not appear to have ever been run** (no later branch references it).
- **R6** (`research-wr-r6-player-target-residual-persistence`) → `NO_ACTIONABLE_WR_PLAYER_TARGET_PERSISTENCE_2025` — rejected, but note this is about **target-share** persistence (entitlement), a different question from R3's **error/bias** persistence. Not a contradiction.
- **R4** (`research-wr-r4-individual-mechanism-decomposition`) → feeds the above; not independently load-bearing.
- **ND3** (`research-wr-nd3-dynamic-target-entitlement`) → `NO_ACTIONABLE_DYNAMIC_ENTITLEMENT_SIGNAL`. Two explicit rejects: don't repackage simple higher-usage-WR-absence counts as an entitlement feature; simple vacated-target mass doesn't explain the within-WR residual on top of the existing alpha-vacancy rule.
- **ND4/ND5** (`research-wr-nd4-role-participation-source-audit`, `-nd5-snap-depth-entitlement`) → feed into R8's conclusion above (snap-depth alone insufficient).
- **ND6** (`research-wr-nd6-player-level-explosive-ceiling`) → produced `WR_R1_MULTISEASON_REPLICATION_RESULT.md` (didn't dig deeper — flagged for your awareness, not load-bearing to the current frontier).

## Explicit stop-rules (verbatim, consolidated)

- Do not retune M38 hierarchy multipliers.
- Do not reopen generic target-pool pruning (M31–M32).
- Do not infer fake WR-CB assignments from participation.
- Do not run another ND5 (snap-depth) variant.
- Do not use sportsbook inputs upstream, ever.
- NGS as a tracking-data source for target/yardage modeling is **closed** (R11 fail) — don't re-attempt the same vendor-NGS approach.
- Simple player-absence-count or vacated-target-mass entitlement features are closed (ND3).

## Current open frontier (their own words)

Two live, unclosed threads:
1. **Efficiency/allocation mechanism**: R7 and R8 both independently conclude the next step needs *route/coverage/matchup* information richer than static participation or player-history traits — and NGS (R9-R11) was tried and failed as that richer source. **Nobody has tried the repo's own PBP-derived coverage-matchup pipeline for this** (`scripts/run_coverage_v2.py` → `data/cb_coverage_team.csv`, `data/cb_coverage_player.csv`, `data/wr_cb_exposure.csv`). This is a different data source than vendor NGS (charting/PBP-derived man/zone rates and WR-CB assignment exposure vs. player-tracking speed/separation), already computed every Full Slate run for WR-R15/QB-C2, and — as far as I can tell from the branch history — never tested as the R7/R8 "richer matchup mechanics" signal. I'd flag this with a caveat: I did not find a branch that rules this out, but I also can't be 100% certain some earlier, unlisted work didn't already try it under a different name — worth a quick confirmation before investing real effort.
2. **R3's WR error-persistence finding is validated but unintegrated.** It passed its own frozen gates and explicitly authorized a follow-up full-stack integration test that never appears to have happened.

## Proposed new research directions

1. **Run R3's own authorized next step**: build the predeclared combined candidate (shrunk mean-calibration from signed bias + MC tail-calibration from difficulty/extreme-miss history) and test it through the exact M38 production architecture against R3's own frozen aggregate + individual-error gates. This isn't a new theory — it's finishing a result that already passed its diagnostic stage and was explicitly cleared for the next step, then stalled. Lowest risk, highest alignment with existing team findings.
2. **Test Coverage v2 (`cb_coverage_player.csv`/`wr_cb_exposure.csv`) as the R7/R8 "richer matchup" signal**, framed identically to R7's frozen protocol (strictly-prior, walk-forward, zero-sportsbook) but swapping the failed NGS source for the repo's own coverage-matchup data. Cheap to sanity-check: these files are already produced in every Full Slate run, so no new data acquisition is needed — just a frozen diagnostic test structured the same way R7 was. If it also fails, that's still valuable (rules out the whole "richer matchup context" hypothesis, not just the NGS implementation of it).
3. **Apply the RB ceiling-compression lesson here as a sanity check.** RB's own research found tail/ceiling miscalibration was the final unresolved RB issue after allocation was otherwise solved. WR's R3 finding (extreme-miss persistence, 1.37x enrichment) is structurally the same shape of problem — before building a new mean-signal, check whether some of the "unexplained YPT/YPR miss" ND1/R7 keep finding is actually a tail-calibration problem rather than a point-estimate problem, the same way it turned out to be for RB.

## Open questions for you

- Do you want the R3 combined-candidate integration test built and run first? It's the most "shovel-ready" item on this whole list — the diagnostic already passed, the two integration lanes are already specified, nothing needs new data.
- Should I check whether Coverage v2 data has already been tried under a different name before investing effort in proposal #2? I could not fully rule this out from branch names/docs alone in the time available.
