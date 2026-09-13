# CURRENT NFL RESEARCH HANDOFF — READ FIRST

## ACTIVE PRODUCTION CHECKPOINT — 2026-09-11

Before doing anything else, read:

1. `docs/handoffs/NFL_HANDOFF_2026-09-11_FULL_SLATE_LIVE_REPAIR_MERGED_CURRENT.md`
2. `NFL_MASTER_CONTINUITY_RECORD.md`

GitHub is canonical; chat memory is secondary.

The Week-1 live Full Slate mechanical incident is now **repaired and merged**. PR #523 was merged to `main` at `f84c6242da02b1804b4b9675c3a3a3e679838e10`. Post-merge Repo CI run `34656484043` and no-live Full Slate run `34656483980` both passed. The exact preserved paid artifact from run `34650067599` previously replayed through repaired Steps 29-31 with zero certification blockers and zero additional OddsAPI acquisition.

Do **not** restart the roster/event-scope repair, preserved-artifact replay, player-identity audit, rematch investigation, Knight alias investigation, downstream current-availability certification repair, or broad historical M107/M108 search. Do not spend another paid live-odds call merely to rediscover mechanical bugs.

Production model science remains frozen and unchanged by the repair. The active merged handoff contains the exact integrity verdict, paid-run lineage, replay evidence, merge SHA, and next authorized step.

Important continuity correction: prior chat notes carried an `M108 = 26/26 PASS` label, but repository search did not recover an authoritative M108 workflow/script/run/PR proving this was a canonical repository gate. Do not invent or require an M108 test by name unless concrete GitHub lineage is later recovered.

---

## PARKED SCIENCE CHECKPOINT

The QB/WR shared-opportunity / first-down pass-propensity / public pregame-intent V1B lane remains preserved at:

- `docs/handoffs/NFL_HANDOFF_2026-09-11_QB_WR_SHARED_OPPORTUNITY_CURRENT.md`

Resume that lane only after confirming no newer production incident supersedes the merged checkpoint.

No production-science change is authorized by the live-repair work itself.

---

## RESEARCH LEDGER — Vegas-line / game-script lane (2026-09-13)

User-initiated question (Issue #535): does modeling maximum pregame accuracy beat Vegas, and specifically does Vegas's own spread/total/moneyline carry usable signal for player props. All work below is research-only; none authorizes a production/model/threshold change.

- **PR #557** `STRONG_GATE_PROBABILITY_CALIBRATION_V1` — merged `24173a261ce620eab6f4347ca3d4990f468da5c8`. Result: `STRONG_GATE_CALIBRATION_NO_IMPROVEMENT`. Isotonic calibration reduced STRONG coverage as expected but worsened held-out performance in both directions. Do not deploy.

- **PR #558** `MARKET_IMPLIED_GAME_SCRIPT_V1` — merged `20f535047c263ae632409ce9a3fb796b9b2fae78`. Corrected result: no meaningful incremental value from the tested game-market representation over a fitted prior-8 rolling baseline for team plays or dropback rate. Treat as a team-level negative, not a complete falsification of every position-specific game-script hypothesis.

- **PR #559** `VEGAS_LINE_GAMESCRIPT_CALIBRATION_V1` — merged `091651ce76598ba0d2dad8ba0fda89cd7470d4ec`. 2023-2025, 816 games: market spread/total have moderate direct correlation with realized margin/total, roughly 10-point MAE, monotonic bucket ordering, and no useful out-of-sample linear/median recalibration bias. Codex required the median/L1 arm because the study is judged on MAE; the negative recalibration conclusion survived that correction.

- **PR #560** `GAME_SCRIPT_CONFIRMED_PLAYER_USAGE_V1` — merged `2d55e512abb562714a86d761123a973928ab2178`. After a Codex-found same-side bug was fixed, the realized game-script -> player-usage effect is large and stable across 2023-2025: margin -> RB rush volume about d~1.0; total -> WR/TE volume about d~0.6-0.8. Raw unconditional pregame Vegas labels capture only roughly a third of that effect. Restricting in hindsight to games where the Vegas line is confirmed within the frozen tolerance moves both RB and WR/TE effects materially toward the ground-truth ceiling, especially at T=7. This is a ceiling diagnostic only because confirmation uses the actual outcome.

- **PR #561** `VEGAS_CONFIRMATION_LIKELIHOOD_V1` — **active research PR, not yet merged and no canonical candidate result accepted yet**. Branch `research-vegas-confirmation-likelihood-v1`, based on exact `main@2d55e512abb562714a86d761123a973928ab2178`. Frozen plan commit `27e3baa924ca730d6d3a84b0aa1018356e810049`; evaluator initially committed at `0b820064cda0b539b2f32a8897f804fa6de3825c` and mechanically corrected before the full six-feature run after Codex review. Objective: train on 2023+2024 and blind-test 2025 to determine whether small, pregame-only market geometry can predict the exact corrected #560 confirmation labels and, more importantly, whether selecting high-confirmation-likelihood games sharpens the 2025 RB-rush-attempt and WR/TE-target effects toward the ground-truth ceiling.

  Frozen features: signed spread, absolute spread, posted total, distance to key margins `{3,7,10,14}`, plus favorite no-vig moneyline probability and a train-only spread-vs-moneyline consistency residual when two-sided moneyline coverage is >=80% in both train and test. Frozen model is `StandardScaler -> L2 LogisticRegression`; selector is the 2023-24 predicted-probability Q75, never a 2025-tuned cutoff. T=7 is primary; T=3 is directional sensitivity. Primary classification gates are AUC > .55, Brier better than train-prevalence constant, selected confirmation lift >=10pp, and adequate class support. Downstream primary gates are RB rush attempts for the margin hypothesis and WR/TE targets for the total hypothesis, requiring adequate selected support and an effect that both exceeds the unconditional Vegas arm and moves closer to the ground-truth arm.

  Codex pre-result review found two substantive implementation-fidelity defects before the full six-feature output was accepted: the moneyline residual must use the preregistered `favorite_ml_prob`, and the downstream gate must explicitly enforce movement toward ground truth. Both are mechanical corrections to match the already-frozen plan, not scientific amendments. The canonical handoff was also updated here to prevent cross-chat duplication. No production change is authorized by #561.

Current next step: execute the corrected #561 evaluator unchanged, preserve the emitted artifact, then independently cross-audit the blind-2025 classification and downstream gates. If either primary confirmation model fails its frozen classification gates, disposition is `NO_ACTIONABLE_PREGAME_CONFIRMATION_STATE`; do not rescue it downstream or tune a new threshold. If classification passes but player-volume sharpening fails, stop that hypothesis as not useful for this objective. Only a hypothesis passing both becomes a research-only qualified candidate.
