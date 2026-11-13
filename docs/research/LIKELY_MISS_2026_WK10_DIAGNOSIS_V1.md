# Why the model said UNDER on Isaiah Likely (2026 week 10 TNF)

The operator's point: the model exists to identify a role like Likely's, and it
did not. That is correct, and the cause is in the model, not only in the
selection layer. Board: `data/market_track_record/boards/2026_wk10.csv`.

## Result status

No final box score was reachable from this session (box-score hosts blocked,
nflverse week-10 rows not yet posted). Two live blogs disagree on Likely's
first half (a 66-yard touchdown versus 5 catches for 39 yards at halftime), so
**his final line is unknown to this record.** A previous message stated he
"finished well over 100"; that was not supported by any source I had and is
withdrawn. Grade every leg from nflverse once week-10 data posts.

## What the model did

Model output for rec_yards, book line 46.5: `model_proj` 43.7, UNDER 46.5 at 59%.

| layer | value | note |
|---|---|---|
| ML component | **59.5** | his 2026 mean is 56.0, median 59.5 |
| MC component | 37.2 | simulation |
| State component | 36.4 | identical to Okonkwo and Theo Johnson |
| frozen rec_yards weights | MC 0.660, ML 0.292, State 0.048 | 0.660x37.2 + 0.292x59.5 + 0.048x36.4 = 43.6 |

The one component that matches his actual usage was the one the frozen blend
weights second.

## Three defects, in order of how clearly they are shown

1. **A prior from a different role and team was given one third of the weight.**
   PlayerForm target share: prior 13.3%, current 26.5%, blended 22.1%. The
   prior is 2025 Baltimore, where he was TE2 behind Mark Andrews (36 targets to
   Andrews' 70; 3.0 targets and 25.6 yards per game). In 2026 he is NYG TE1 at
   8.6 targets and 56.0 yards per game. A team change plus a TE2-to-TE1 jump is
   exactly the role/state trigger in AGENTS.md; the prior should have been
   discounted, and nothing in the output says it was not.
2. **State returns bucket values, not player projections.** Across 13 priced
   receivers there are 6 distinct `state_proj` values and 11 of 13 share one:
   Likely/Okonkwo/Johnson 36.4; Nabers/Mooney/Diggs/Jefferson 49.0;
   Fields/Williams 37.5; Ekeler/White 19.4. Its rec_yards weight is only 4.8%,
   so it did not decide Likely, but it is not individual-player projection.
3. **Model UNDER signals on heavy-usage receivers carry no information in the
   2024-2025 backtest.** `data/backtests/full_stack_vegas_benchmark_v1`,
   rec_yards, players ranked by strictly-prior season average:

   | usage tier | n | actual | model | book line | model says over | actually over line |
   |---|---|---|---|---|---|---|
   | bottom 50% | 2101 | 24.0 | 15.9 | 18.5 | 38% | 51% |
   | 50-80% | 1241 | 42.4 | 30.9 | 38.1 | 22% | 48% |
   | 80-92% | 496 | 59.2 | 44.8 | 54.6 | 22% | 49% |
   | top 8% | 347 | 70.7 | 55.3 | 68.7 | 14% | 49% |

   The book line is close to unbiased; the model projection is low in every
   tier and has higher MAE than the line in every tier. In the top tier the
   model says under 86% of the time and those players go under the line 51% of
   the time.

   **Caveat that matters:** the benchmark README states the WR and TE production
   models (`WR_R15_PRODUCTION_MODEL_V1`, `TE_R5P_PRODUCTION_MODEL_V1`) are not
   included in these numbers, so this describes the generic engine rather than
   the deployed stack. Both are wired into the production workflow and run
   inside the simulation's target allocation (not as board columns), so tonight's
   MC value already includes them. The backtest cannot say how much they close.

## What this is not

Not evidence for any new weights. `model_ensemble_weights.csv` is promoted and
frozen; changing weights or the prior blend is a validated migration. These
findings justify the migration being the next research item, not an edit.

## Selection-layer guards that do not touch promoted science

* **Role-change flag:** team or role differs from the prior season -> print the
  prior-vs-current usage for that player next to every leg on him.
* **Component-disagreement flag:** final projection more than ~15% from the ML
  component or from the player's current-season per-game mean -> flag it.
* **No model UNDER on a top-usage pass catcher** without a stated football
  reason; the backtest shows no information in that signal.
