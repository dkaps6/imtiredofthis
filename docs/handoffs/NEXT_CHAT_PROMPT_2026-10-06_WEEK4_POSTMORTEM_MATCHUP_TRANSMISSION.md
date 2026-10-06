Pick up my NFL project seamlessly from the canonical GitHub handoff.

Repo: dkaps6/imtiredofthis.

GitHub is canonical. Do not ask me to re-explain anything. Do not restart research. Do not recursively read old handoffs. Do not reopen closed science. I need this to feel like the exact same chat continuing.

Start by reading, in this exact order:

1. AGENTS.md
2. CURRENT_NFL_RESEARCH_HANDOFF.md — read only the newest top checkpoint
3. docs/handoffs/NFL_HANDOFF_2026-10-06_WEEK4_POSTMORTEM_MATCHUP_TRANSMISSION_CURRENT.md
4. docs/research/FOOTBALL_MATCHUP_TRANSMISSION_V1_PLAN.md
5. Issue #535 from comment 6021335699 onward, especially 6021798748 and 6022435377
6. Query GitHub live for current main, these research branches, and Actions before editing:
   - research-week4-postmortem-execution-v1
   - research-distribution-right-tail-asymmetry-v1
   - research-opportunity-state-conflict-uncertainty-v1
   - research-football-matchup-transmission-v1

Important current state:

- Week 4 is fully graded: 220-207, 51.52%, -3.38u.
- Weeks 1-4 cumulative: 849-818, 50.93%, -44.10u.
- Raw edge/fair probability remains overconfident and is NOT validated confidence.
- Failure-mode atlas found a severe asymmetric upper-tail problem.
- Historical Right-Tail Asymmetry V1 is now CONFIRMED for rush_yards, rec_yards, receptions, and rush_rec_yards in BOTH 2024 and 2025; pass_yards did not replicate.
- Opportunity State Conflict Uncertainty V1 is NULL/CLOSED.
- The user's current priority is the FOOTBALL layer, not another generic calibration exercise.

Primary active lane:
Football Matchup Transmission V1.

Phase A is already COMPLETE and green:
- run 37504432928
- artifact 11431601141
- digest sha256:bc20084e4dd9a5272ff9bc6acb922ce5395b40243a8f5380b8a88d42b151bdd8

Do NOT rerun Phase A.

Phase A proved that production collects a lot of matchup/game-state data but generic RB/WR/TE projections use only a narrow subset. Key confirmed gaps:
- def_rush_epa reaches TeamContext but is not consumed by generic matchup rules;
- explosive_play_rate_allowed is available but not consumed;
- WR/TE/RB YPT allowed and outside/slot YPT are upstream but dropped before TeamContext;
- YBC/stuff run-defense fields are upstream but dropped;
- generic project_game_script hardcodes pass_share = 0.57;
- rules_pass_rate therefore normally bypasses PROE/pass-tendency fallback;
- lead/trail probabilities do not alter the pass/rush split in simulation_v2;
- generic RB rushing matchup changes YPC mostly through coarse light/heavy box thresholds;
- QB M89/M90 is separate and richer; do not conflate it with generic skill-position logic.

Bijan Week-4 pregame architecture illustration:
- ATL neutral pass rate 46.5%
- true PROE -0.105
- Bijan PlayerForm rush share .6158
- Bayes/rules rush share .5421
- production rules pass rate .57
- projected ATL plays 59.54
- team rush attempts 25.60
- Bayes YPC 4.874
- NO light box .391, heavy box .250
- rules rush-efficiency multiplier exactly 1.0

The mechanical ~95.6-yard illustration using existing pregame neutral pass rate + PlayerForm share is NOT a candidate and NOT the 'correct' projection. It only proves information loss/transmission.

Anti-retest:
- Do NOT invent 'bad run defense => boost RB X%'.
- M95A/M95B already tested generic role×defense matchup ideas; descriptive football truth existed, but generic prospective models were mixed/inconsistent.
- M56/M83 generic QB matchup families are closed.
- Do not reopen Rush Pool, Opportunity Authority, Bayes retuning, retired WR coverage penalty, or sportsbook-conditioned football inputs.

Exact next action:
Continue research-football-matchup-transmission-v1 into the already-frozen Phase B/C historical audit.

Phase B:
- leakage-safe 2024 W2-18 and 2025 W2-18
- zero 2026 outcomes
- zero sportsbook lines/odds
- no coefficient fitting
- test whether predeclared defensive matchup variables explain residual error conditional on existing player/role projection

Markets:
- RB rush yards
- RB rush+receiving
- WR receiving yards
- TE receiving yards
- RB receiving yards
- QB pass yards control

Families:
- run defense: def_rush_epa, stuff rate, YBC allowed, box rates
- receiving defense: position YPT allowed, outside/slot YPT, pass EPA/YPA/success, coverage
- game environment: pace/plays, PROE/pass tendency, pass/rush tendency faced, pressure

Phase C:
test continuous usage×matchup interactions:
- RB rush opportunity × run-defense weakness
- WR target/route opportunity × WR/outside/slot weakness
- TE target/route opportunity × TE/zone/middle weakness
- RB target/route opportunity × RB receiving weakness

No threshold search. No top-N. No bellcow-only post-hoc carveout. No target-game usage.

Replication requirement:
expected-sign signal in BOTH 2024 and 2025 + clustered support + incremental to existing role/opportunity + same live semantics + not a closed family.

If something passes, freeze a separate integration candidate BEFORE scoring it. Diagnostic PASS does not equal production promotion.

Also:
- if no RESULT.md yet exists for Right-Tail Asymmetry V1, write/freeze one from canonical run 37493425352.
- if no Phase-A RESULT.md yet exists for matchup transmission, write/freeze one from run 37504432928.
- no paid OddsAPI pull without explicit user authorization.

Take over as research lead and keep moving. Do not stop at a status summary if there is actionable work.