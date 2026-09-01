from pathlib import Path

p = Path('docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md')
s = p.read_text()
marker = '# NEXT MIGRATION — M96D'
if marker not in s:
    raise SystemExit('M96D next marker missing')
prefix = s.split(marker, 1)[0].rstrip() + '\n\n'
block = r'''# Latest completed migration: M96D — Pregame Conditional Efficiency Routing Audit

Full results: `docs/migrations/M96D_RB_PREGAME_CONDITIONAL_EFFICIENCY_ROUTING_RESULTS.md`.

Authoritative:

- workflow `M96D RB Pregame Efficiency Router`
- run **`33467325153`**
- job **`99729782983`**
- tested SHA **`dc57217aaa8312edc6c97c43486330ba9894bbc4`**
- artifact **`9785311314`**
- artifact SHA256 **`e65ef0cc861e8f27863e5d5fda8ba91b34d921d37724cf431212a9cb4026bf30`**
- execution success
- disposition **`M96D_PRIMARY_ROUTER_FAILED`**
- model fit `0`; threshold search `0`; feature search `0`; sportsbook `0`; production change `0`

M96D tested one frozen deterministic pregame router: turn M96C opponent-defense efficiency D on only below 15 M94C projected carries and when the back is not an entrenched workhorse. It improved all-RB Weeks 6-18 MAE `21.5719 -> 21.4165` (`+0.1554`), preserved RMSE/bias/tail AUC gates and improved late-season MAE, but failed the high-workload safety gate: actual 20+ MAE regressed `0.7489` and 25+ `0.5373` yards. No threshold was retuned.

The controlled role-only diagnostic was stronger globally (MAE `21.3641`, gain `+0.2078`; RMSE/correlation also improved) but still leaked damage into rare unexpected high-workload games. Pregame strata showed why: non-entrenched backs had only `3.16%` actual 20+ incidence overall, yet those rare spikes matter disproportionately. This supports exactly one final router type using already-frozen M95F workload-tail distribution and M95I transition/vacancy evidence as a safety guard around the role-based D expert. It does **not** reopen carry-tail tuning.

# NEXT MIGRATION — M96E

Name: **M96E — Role Router with Frozen Workload-Risk Guard**

Primary question:

> Can the stronger non-entrenched role-based D router retain its global rushing-yard improvement while using frozen M95F/M95I pregame workload-risk signals to suppress D specifically when an unexpected 20+/25+ workload spike or transition is plausible?

Required design:

- M94C carries/yard center frozen; M96C D frozen.
- Start from the M96D role-only insight; do not reuse actual carry buckets as router inputs.
- M95F calibrated 20+/25+ probabilities/distribution summaries and M95I vacancy/transition state are safety guards only. No carry-tail coefficient/model retuning.
- Freeze one primary guard before scoring; no threshold grid selection after results.
- No sportsbook inputs or postgame router features.
- Preserve all-RB MAE/RMSE improvement and require actual 20+/25+ diagnostic non-degradation.
- 2025 remains development evidence; any survivor is research-only pending prospective 2026 confirmation.

Stopping rule:

- If M96E passes, freeze the routed architecture for prospective 2026 confirmation.
- If it fails, stop retrospective RB efficiency refinement, retain C/M94C as point architecture plus existing workload/vacancy diagnostics, write `AUTONOMOUS_RB_RESEARCH_STOP`, and move no further without prospective evidence.

## Fresh-chat startup procedure

Tell a new chat:

> Continue my NFL stuff project from the `research-current-state` branch in GitHub repo `dkaps6/imtiredofthis`. Read `docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md` first, verify the latest authoritative GitHub Actions run/artifact, and continue directly from the `NEXT MIGRATION` section. Preserve all modeling/validation rules and do not restart old research.
'''
s = prefix + block
s = s.replace('Current research branch: `research-rb-m96c-m94c-efficiency-residual`', 'Current research branch: `research-rb-m96d-pregame-efficiency-router`')
p.write_text(s)
