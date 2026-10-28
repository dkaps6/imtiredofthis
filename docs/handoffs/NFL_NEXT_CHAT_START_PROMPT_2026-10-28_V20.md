# NFL PRISM — NEXT CHAT START PROMPT — V20

Pick up my NFL PRISM project seamlessly from the newest canonical GitHub checkpoint.

Repo: `dkaps6/imtiredofthis`.

Do **NOT** ask me to re-explain anything. GitHub is canonical. This needs to feel like the exact same assistant continuing the exact same project with no reset.

## READ / VERIFY IN THIS ORDER

1. Read `AGENTS.md` completely.
2. Read `CURRENT_NFL_RESEARCH_HANDOFF.md` completely. It is the **single active root TOP checkpoint**. Do not recursively read older handoffs unless the root explicitly points to one.
3. Follow the root handoff's current read order for the RB rushing research docs.
4. Open GitHub Issue **#535**, which is the GPT ↔ Claude Code work channel.
5. Read standalone catch-up comment **6477740922**, then the current sequence:
   **6477974816 → 6478435704 → 6478678129 → 6478823926 → 6478960995 → 6479098849 → 6479259425**.
6. Before doing any work, fetch **all comments newer than 6479259425** and check current `main` SHA. If Claude has posted a new commit/result, independently audit the exact code/artifact before accepting it.
7. GitHub is canonical over chat memory. If anything in prior chat conflicts with the newest root handoff or verified production code, stop and reconcile the contradiction rather than guessing.

## WHERE WE ACTUALLY ARE

The active sequence is:

**(1) roster / availability EXISTENCE integrity → (2) coverage-aware team rush-budget reconciliation → (3) Bayesian evidence-dependent shrinkage.**

Do **not** skip that order.

The most important current structural finding is that **Ourlads is the sole authority that a player exists on the production football path**. W7 row lineage is:

`roles_ourlads 458 → roles_ourlads_status_v1 458 → current_player_availability 458 → roles_current_production_eligible_v1 396 → football_simulation_universe 396`.

The availability layer never adds a player; it only narrows. Therefore a known/returning/elevated player omitted by Ourlads can be structurally unreachable.

Separately, production player rush shares are estimated independently and are not reconciled into a valid team opportunity budget before Monte Carlo. On the exact W7 production universe, post-top-5 `rules_rush_share` sums to **1.2729** on average; all 28 teams exceed the 0.95 cap; `_allocate_counts` applies a mean **0.7574** haircut; simulated RB carries land at **19.77/team vs 23.15 actual**, bias **−3.38**. Do **not** simply remove the cap and do **not** normalize an incomplete roster to 1.0.

Working design contract, **not promoted**:

```
team_modeled_mass_target = 1 - unresolved_roster_rush_mass
```

The residual must be state-derived and time-dependent, not a universal 0.05.

## IMMEDIATE OPEN AUDIT — TARGET 6

Continue here first:

**A. Authoritative player-existence feed / timing**
Determine whether a candidate roster/existence feed is genuinely **pregame-safe** for a Thursday and/or game-day run. `nflreadpy.load_rosters_weekly(2026)` resolves the W7 settled cases, but do not assume its weekly status is available with safe publication timing before kickoff. A post-hoc settled roster snapshot cannot become a pregame production feature.

We need a leakage-safe union existence universe using:
- depth-chart role authority;
- active roster / transaction authority;
- return-from-reserve / practice-window states;
- relevant practice-squad / elevation candidates;
- final game-day activations / inactives.

Ourlads may remain a role/depth source but must not remain the sole authority that a player exists.

**B. Check whether the existence ceiling also affects WR/TE**
The 458→396 narrowing is not RB-specific. Diagnose whether receivers/tight ends are being dropped at the same existence stage. Do not assume it; measure it.

## NON-NEGOTIABLE SCIENCE / USER DIRECTIVES

- **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**
- The scientific unit is the named player in his actual current football situation.
- Player identity → current role/state/injury → opportunity/teammate competition → opponent/environment → market-specific outcome → reliability/distribution.
- Positional MAE/RMSE is QA only, never the objective.
- **ANOMALIES ARE QUESTIONS, NOT TRENDS.**
- A partial/injury/context game cannot automatically become a new baseline.
- Preserve validated/promoted Bayesian/ML/State/MC/QB/RB/coverage/box/pressure/scheme layers. Additive precision work only unless evidence explicitly supersedes something.
- Before claiming what production does, trace the actual workflow/import graph. Multiple false claims in this project came from measuring off-path code.
- Production pricing uses **empirical Monte Carlo draws**, not fixed sigma / a normal CDF.
- MC dispersion candidate remains **INERT / NOT PROMOTED**.
- RB carry concentration candidate remains **INERT**.
- The tested RB level substitution changed **26/79 decisions but produced zero net side-accuracy gain (13/13 on flipped decisions)**. Do not repeat the false wording that it “changed no decision.”
- Betting lane remains **CLOSED** unless I explicitly reopen it.
- No new paid OddsAPI credits.
- No sportsbook variable upstream of football projections.
- Do not retune on W7 outcomes.
- Do not weaken truth/integrity gates to make a test pass.
- Preregister candidate form, eligibility, holdout, kill conditions, and naive controls **before scoring**.
- Do not call a research discovery a production improvement.

## NGS / CAPABILITY NOTE — DO NOT CHANGE LANES

The repo currently uses **zero NextGen Stats on the production path**, even though 2026 weeks 1–7 NGS is available for passing/rushing/receiving. This is a standing capability gap, **not the current priority**.

The old statement that 2026 “NGS is unavailable” was wrong; it conflated NGS with the 2026 **participation** endpoint, which genuinely is unavailable. Target-week NGS is outcome data. Any future use must be strictly prior-week walk-forward and publication timing must be proven safe.

Do not open NGS research in competition with the active order:
**roster existence → rush budget → Bayesian shrinkage**.

## HOW TO WORK WITH CLAUDE

Claude coordination happens through GitHub Issue **#535**.

If I say “Claude responded,” “Claude said something,” “Claude gave you something,” or similar:
1. immediately fetch comments newer than the latest GPT comment;
2. inspect the exact commit SHA / files / artifact;
3. independently audit the live production path;
4. state clearly what is:
   - **research/discovery**,
   - **production code change**,
   - **validated improvement**;
5. post your audit/ruling back to #535.

Do not blindly agree with Claude. Translate technical updates into normal English for me and tell me whether they actually move **named-player projections** forward.

## COMMUNICATION STYLE WITH ME

I do not want to keep re-teaching the project. Be decisive, skeptical, and concrete. If a result is only diagnostic, say so. If nothing has improved live production, say so. If something is a real improvement, explain exactly what changed and why we trust it.

Start by confirming you have read the canonical checkpoint, telling me the current active problem in plain English, then immediately continue the newest unfinished audit/work rather than asking me what I want to do next.
