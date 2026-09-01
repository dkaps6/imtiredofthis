from pathlib import Path
p=Path('docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md')
s=p.read_text()
marker='# NEXT MIGRATION — M96E'
if marker not in s: raise SystemExit('M96E next marker missing')
prefix=s.split(marker,1)[0].rstrip()+'\n\n'
block=r'''# Latest completed migration: M96E — Role Router with Frozen Workload-Risk Guard

Full results: `docs/migrations/M96E_RB_ROLE_WORKLOAD_RISK_GUARD_RESULTS.md`.

Authoritative:

- workflow `M96E RB Role Workload Risk Guard`
- run **`33467630395`**
- job **`99730679349`**
- tested SHA **`db1a139a270b7c246d1b5b07dc1a3490cb8fa3a0`**
- artifact **`9785416331`**
- artifact SHA256 **`c73a728570516b77c04c4a68ec1541e4a94fb830e144f40f16df63dbcc36dfbe`**
- execution success
- disposition **`M96E_FINAL_RETROSPECTIVE_ROUTER_FAILED_STOP`**
- model fit `0`; threshold search `0`; feature search `0`; sportsbook `0`; production change `0`

M96E was the final precommitted retrospective efficiency-router test. It started from M96D's stronger non-entrenched role-based D router and suppressed D whenever frozen M95F/M95I pregame workload-risk evidence indicated meaningful 20+ workload or vacancy-transition risk. M94C carries and center were unchanged; M95F/M95I were not refit.

Weeks 6-18 all-RB (`n=961`): C MAE `21.571881`; M96E `21.430091`, gain `+0.141791`. RMSE improved `30.449965 -> 30.431137`; correlation improved `.604528 -> .605692`; late W13-18 MAE improved `+0.097105`. 75+ AUC changed only `-.000407`; 100+ improved `+.001508`.

Crucially, the safety guard worked: actual 20+ MAE regression fell to only `+0.059047` and 25+ to `+0.159106`, both inside the frozen <=`.50` gate. The guard protected 69/75 actual 20+ and 20/21 actual 25+ games in evaluation-only accounting.

However, the frozen all-RB materiality gate required MAE gain >=`.150000`; observed was `.141791`, short by `.008209`. Eight of nine checks passed, but the materiality line is **not waived**. M96E is not retained/promoted.

Final retrospective RB architecture:
- **C/M94C** remains the conservative global rushing-yard point and central opportunity anchor.
- **M95F** remains workload-distribution/stable-workhorse tail evidence, not a universal point-mean boost.
- **M95I** remains vacancy/transition diagnostic evidence.
- **D/M96C** is validated as conditional scientific signal but did not earn a retained point-module role after the final safety/materiality test.
- **E/P** remain conditional clues only.
- **X** remains rejected as an isolated separable tail increment.
- no M91-M96E RB research is production-promoted by this closure.

# AUTONOMOUS_RB_RESEARCH_STOP

Retrospective RB efficiency refinement is now frozen. The final precommitted router solved most of the high-workload safety issue but missed the predeclared global materiality gate. Further threshold/feature variants on exposed 2025 outcomes would be overfitting rather than independent evidence.

The next legitimate RB evidence must come from genuinely prospective/untouched 2026 games, or from a separately justified new data source tested without retuning against the exposed historical outcomes. Until then, do **not** open M96F or additional retrospective RB router variants.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first and verify the latest authoritative GitHub Actions run/artifact. Respect `AUTONOMOUS_RB_RESEARCH_STOP`: RB retrospective refinement is frozen pending genuinely prospective 2026 evidence. Preserve all modeling/validation rules and do not restart old research.
'''
s=prefix+block
s=s.replace('Current research branch: `research-rb-m96d-pregame-efficiency-router`','Current research branch: `research-rb-m96e-workload-risk-guard`')
p.write_text(s)
