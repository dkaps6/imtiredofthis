from pathlib import Path

p = Path('docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md')
s = p.read_text()

s = s.replace(
    '- Current research branch: `research-rb-m96b-modular-joint-synthesis`',
    '- Current research branch: `research-rb-m96c-m94c-efficiency-residual`'
)
s = s.replace(
    '- No M91-M96B RB research has been promoted to production.',
    '- No M91-M96C RB research has been promoted to production.'
)

m96b_bullet = '- **M96B formalized the modular/puzzle approach. Simple additive stacking did not produce a broad winner: M94C remained the point anchor; the transplanted M95C environment residual slightly worsened point MAE; M95F workload-tail fusion improved 75+/100+ metrics only directionally and below the retention gate; the isolated M95D upside residual was destructive when added to M94C; and M95I vacancy information remained a promising diagnostic only. The key lesson is that positive modules may be redundant, conditional, or interactive rather than directly additive.**'
m96c_bullet = m96b_bullet + '\n- **M96C trained efficiency residuals directly against M94C using strict 2025 expanding-week OOF evaluation. No global E/P/D block cleared the frozen gate. Opponent-defense efficiency D was best globally (MAE `21.5719 -> 21.3474`) and improved 0-14 carry games, but materially worsened true 15+/20+ workload games. E/P showed the same sign flip. This supports a pregame conditional efficiency router rather than a universal correction. Isolated explosive X again failed as a separable tail increment. M96D is next.**'
if m96b_bullet in s and 'M96C trained efficiency residuals directly against M94C' not in s:
    s = s.replace(m96b_bullet, m96c_bullet)

if '# Latest completed migration: M96C' in s and '# NEXT MIGRATION — M96D' in s:
    p.write_text(s)
    raise SystemExit(0)

start_marker = '# NEXT MIGRATION — M96C'
if start_marker not in s:
    raise RuntimeError('M96C next-migration marker missing and completed M96C marker not present')
start = s.index(start_marker)
end = s.index('\n## Fresh-chat startup procedure', start)

new = '''# Latest completed migration: M96C — M94C-anchored RB efficiency residual synthesis

Full results: `docs/migrations/M96C_RB_M94C_EFFICIENCY_RESIDUAL_RESULTS.md`.

Authoritative:

- workflow `M96C RB M94C Efficiency Residual`
- run **`33462888850`**
- job **`99716610968`**
- tested SHA **`708f9ff23b96cde8e023b6317fcaec30b76e76b0`**
- artifact **`9783799265`**
- artifact SHA256 **`6109a8b3afc6d2fdb963db9149bf3fb238cc476e291bf743cc4b496ad39abf72`**
- execution success
- disposition **`M96C_NO_GLOBAL_WINNER_CONDITIONAL_EFFICIENCY_SIGNAL_SUPPORTED`**
- model fit `1`; feature search `0`; weight search `0`; hyperparameter search `0`; sportsbook `0`; production change `0`

Source/protocol:

- frozen M94C player-level rush-yard point exists in the authoritative artifact for 2025 only, so M96C did **not** invent a synthetic 2024 M94C player point;
- strict expanding-week 2025 OOF: test Weeks 6-18, each week trained only on earlier 2025 weeks;
- M94C rush attempts and central rush-yard point frozen;
- residual model predicted YPC/efficiency error only; correction multiplied by frozen M94C carries;
- train residual winsorization/clipping used training-only 5th/95th percentiles;
- M95D->M94C 2025 join `1340/1357 = 98.7472%`; exact yard and carry truth parity max diff `0.0`.

Frozen blocks:

- E blocking/environment `14` features;
- P player-created efficiency `8` available features;
- D opponent run efficiency/resistance `16` features;
- X explosive/upside `16` features, tail-only primary role.

Weeks 6-18 OOF all-RB (`n=961`):

- C/M94C MAE **`21.5719`**, RMSE `30.4500`, bias `+0.3820`, corr `.6045`.
- E MAE `21.5063` (gain `+0.0656`), but RMSE worsened `0.2509`.
- P MAE `21.4261` (gain `+0.1458`), RMSE worsened `0.1654`.
- D was best: MAE **`21.3474`** (gain **`+0.2245`**), RMSE `30.4341` (gain `+0.0159`).
- E+P MAE `21.5880`; E+D `21.5676`; P+D `21.4684`; E+P+D `21.6526`. Additive block stacking did not create a winner.
- No arm reached the frozen `>=0.25` all-RB MAE gain and all arms failed the workload non-degradation gate.

The key sign flip was D by actual workload (postgame diagnostic only):

- 0-5 carries: MAE `13.4145 -> 12.7837`, gain `+0.6308`.
- 6-10: `21.8220 -> 21.3148`, gain `+0.5071`.
- 11-14: `25.7482 -> 25.0106`, gain `+0.7376`.
- 15-19: `29.5957 -> 30.3544`, **regression `0.7587`**.
- 20+: `39.7267 -> 41.8936`, **regression `2.1669`**.
- 25+ diagnostic: regression `1.3432`.

E and P showed the same broad pattern: low/mid-workload value, higher-workload damage. This means the efficiency information is not useless; it is **conditional**. M96A already showed 20+/25+ yard misses are opportunity-dominant, while middle workload games are more efficiency-sensitive. M96C independently fits that architecture.

Do **not** use actual carries as the future router. Actual workload is postgame truth and is used only to diagnose the sign flip. Any router must use pregame M94C/M95F/role-state signals and be frozen before outcome scoring.

X tail-only audit failed:

- 75+ AUC `.806478 -> .800355`, Brier `.114757 -> .115719`.
- 100+ AUC `.790822 -> .785170`, Brier `.067047 -> .067255`.
- X is rejected as an isolated separable increment in this form; prior native-model interaction evidence is not erased.

Capability ledger after M96C:

- **C/M94C:** RETAIN global center.
- **E:** CONDITIONAL_CLUE only.
- **P:** CONDITIONAL_CLUE only.
- **D:** CONDITIONAL_CLUE only; strongest simple efficiency block.
- **X:** REJECT isolated tail increment.
- **M95F:** still workload-distribution evidence, not a universal yard mean boost.
- **M95I/V:** remains vacancy/transition diagnostic evidence and must be compared directly against the best baseline in a separately frozen conditional test.

Scientific interpretation: M96C did not find a safe universal efficiency correction, but it found exactly the kind of module specialization the puzzle framework is designed to exploit. The next step is not another global coefficient blend; it is a pregame router that decides when an efficiency expert should be active without sacrificing high-workload games.

# NEXT MIGRATION — M96D

Name: **M96D — Pregame Conditional Efficiency Routing Audit**

Primary question:

> Can pregame workload/role-state information identify the player-games where an M94C-anchored efficiency expert (especially D, with E/P as controlled alternatives) should be active, preserving low/mid-workload gains without damaging high-workload/tail games?

Required design:

- M94C carries and central rush-yard point remain frozen.
- No actual carries, actual yards, or any postgame variable may enter the router.
- No sportsbook inputs.
- Do not reopen generic M95 carry-tail coefficient search.
- Router inputs must come from already-validated pregame workload/role state: M94C projected carries, M95F 20+/25+ probability/distribution summaries, stable-workhorse/role state, vacancy flag/entitlement state where available, and only predeclared workload indicators.
- The router must be precommitted and small. Use a diagnostic gate grid or simple leakage-safe conditional model; no per-player/week hand selection and no after-result threshold tuning.
- D is the primary efficiency expert because it was the strongest M96C simple block. E and P are controlled alternatives; do not assume combinations are additive.
- Compare C vs routed-D and any other predeclared routed arms on global MAE/RMSE/bias plus workload-tail/75+/100+ guards.
- Preserve high-workload behavior: a routed candidate cannot earn retention by improving low workload while recreating M96C's 20+ damage.
- 2025 remains development evidence. Any retained routed architecture must still face genuinely prospective 2026 confirmation before production.

Decision path:

- If a pregame router preserves the M96C low/mid efficiency gains while removing the high-workload regressions, freeze the smallest successful routed architecture for prospective confirmation.
- If routing cannot separate the regimes safely, retain C/M94C and stop retrospective RB efficiency refinement rather than opening unlimited variants.
'''

s = s[:start] + new + s[end:]
p.write_text(s)
