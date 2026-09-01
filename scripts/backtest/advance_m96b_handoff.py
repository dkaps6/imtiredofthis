from pathlib import Path

p = Path('docs/migrations/CURRENT_NFL_RESEARCH_HANDOFF.md')
s = p.read_text()

s = s.replace(
    '- Current research branch: `research-rb-m96a-opportunity-efficiency-attribution`',
    '- Current research branch: `research-rb-m96b-modular-joint-synthesis`'
)
s = s.replace(
    '- No M91-M96A RB research has been promoted to production.',
    '- No M91-M96B RB research has been promoted to production.'
)

m96a_bullet = '- **M96A attributed 2025 rushing-yard error jointly to opportunity and efficiency: perfect carries recovered 7.68 MAE yards, perfect efficiency 6.73, and opportunity was the larger absolute component in 59.7% of games. The precommitted 1-yard dominance margin was missed by 0.048 yards, so M96B must model workload and efficiency as separate distributions rather than declaring either side solved.**'
m96b_bullet = m96a_bullet + '\n- **M96B formalized the modular/puzzle approach. Simple additive stacking did not produce a broad winner: M94C remained the point anchor; the transplanted M95C environment residual slightly worsened point MAE; M95F workload-tail fusion improved 75+/100+ metrics only directionally and below the retention gate; the isolated M95D upside residual was destructive when added to M94C; and M95I vacancy information remained a promising diagnostic only. The key lesson is that positive modules may be redundant, conditional, or interactive rather than directly additive.**'
if m96a_bullet in s and 'M96B formalized the modular/puzzle approach' not in s:
    s = s.replace(m96a_bullet, m96b_bullet)

rule17 = '17. Broad QB mean research remains frozen after M90 while RB work is active.'
rule18 = '''17. Broad QB mean research remains frozen after M90 while RB work is active.
18. **Modular capability rule:** do not judge every experiment only as a whole-model replacement. Record the exact capability it improved, the regime where it improved it, what it damaged, and whether that capability can coexist with other validated modules. Before inventing a new model, test whether retained capabilities can be combined through precommitted ablations and non-degradation gates. A module may own a narrow job (center, tail, vacancy, efficiency, explosive upside) without being allowed to alter other jobs. Positive signals are not automatically additive; they may be redundant, interacting, or conditional experts. Never force a combination merely because each component was individually promising.'''
if rule17 in s and '18. **Modular capability rule:**' not in s:
    s = s.replace(rule17, rule18)

# M96A is no longer the latest migration, but preserve its full section/history.
s = s.replace(
    '# Latest completed migration: M96A — opportunity vs efficiency attribution',
    '# M96A — opportunity vs efficiency attribution'
)

start = s.index('# NEXT MIGRATION — M96B')
end = s.index('\n## Fresh-chat startup procedure', start)

new = '''# Latest completed migration: M96B — modular joint workload × efficiency synthesis

Full results: `docs/migrations/M96B_RB_MODULAR_JOINT_SYNTHESIS_RESULTS.md`.

Authoritative:

- workflow `M96B RB Modular Joint Synthesis`
- run **`33461369073`**
- job **`99711988023`**
- tested SHA **`a3018bf828bf0c78b09a2e0b8a6cd1af60b25f40`**
- artifact **`9783267179`**
- artifact SHA256 **`81f25e134a34a6b2d8b28195bb7f804f6ce54b5823bead5a2e4b717ee544718b`**
- execution success
- disposition **`M96B_MODULAR_SYNTHESIS_COMPLETE`**
- feature search `0`; weight search `0`; sportsbook `0`; production change `0`
- only model fitting in M96B was the precommitted one-dimensional Platt calibration for tail probabilities

M96B froze a capability ledger before the result:

- **C = M94C central opportunity/yard point.** Owns the point center; cannot be globally inflated for tails.
- **W = M95F workload-tail distribution.** Allowed to inform upper workload/tail probability, not replace C with a universally higher mean.
- **V = M95I vacancy/transition evidence.** Vacancy-only; not allowed on stable incumbents and not production-promoted here.
- **E = M95C mean efficiency/environment information.** Allowed to modify efficiency/yards only, never carries.
- **X = M95D explosive/upside context.** Tail/ranking role only; not allowed to universally boost point YPC/yards.

### Source integrity

- 2025 M95D OOS rows `1290`; exact M94C+M95D+M95F common rows `1274`; **98.7597%** coverage vs the frozen `>=97%` gate — PASS.
- 2024 W13-18 common M95D/M95F temporal-calibration rows `449` of `479` M95F holdout rows (`93.7370%`).
- shared rushing-yard truth parity passed.

### C — point anchor RETAIN

On the exact 2025 M96B intersection (`n=1274`):

- M94C/C MAE **`21.8440`**
- RMSE `30.5811`
- bias `+1.0954`
- correlation `.5853`

C remains the point anchor.

### E — additive M95C environment residual REJECT

Frozen test:

`CE = M94C rush-yard point + (M95C-environment prediction - role-baseline prediction)`

2025:

- C MAE `21.8440`
- C+E MAE `21.9095`
- gain `-0.0654`
- bias `+1.0954 -> +1.3486`
- worst ordinary-slice MAE regression only `0.4274`, but all-RB MAE failed to improve.

Important nuance: E/environment still improved its **native M95D role baseline** slightly in both 2024 and 2025. M96B therefore did not prove environment information useless; it proved that a residual learned around a weaker/different baseline is not plug-compatible as a direct additive correction to M94C. The next efficiency model must be trained directly against the M94C residual.

### W — M95F workload-tail fusion directional positive, but formal RETENTION GATE FAILED

Full-2025 75+ rushing yards:

- B AUC `.799739`
- B+W `.802295` — `+.002556`
- Brier `.112127 -> .111325` — gain `+.000803`
- logloss `.356596 -> .354201`

Full-2025 100+:

- B AUC `.799035`
- B+W `.799324` — `+.000288`
- Brier `.063428 -> .063343` — gain `+.000085`

All four full-season metrics moved in the right direction and late-2025 did not materially reverse, but the improvements did not meet the frozen materiality requirement (`+.005` AUC or `+.001` Brier). W is therefore **not retained as a rushing-yard tail fusion module from M96B**. This does not erase M95F's role as a carry/workload distribution baseline.

### X — isolated M95D upside residual REJECT in additive form

Frozen residual:

`X_delta = full_environment_matchup - M95C_environment`

When rank-fused onto B it was strongly destructive:

- 75+ AUC `.799739 -> .726578`, Brier `.112127 -> .120215`
- 100+ AUC `.799035 -> .720655`, Brier `.063428 -> .065507`
- B+W+X also regressed and was not preferred.

Important interpretation: the M95D full matchup model had shown better 100+ discrimination than its own environment-only control in its native architecture in both 2024 and 2025. M96B shows that this value is **interactive/native-expert signal, not a separable additive residual over M94C**. Modular does not mean every positive signal can be added as a coefficient.

### V — M95I vacancy signal RETAIN DIAGNOSTIC ONLY

2025 vacancy rows `n=105`:

- 75+ events `17`: frozen comparison AUC `.63035 -> .67213`, gain `+.04178`
- 100+ events `9`: `.73264 -> .75752`, gain `+.02488`

Critical caveat: the frozen parent comparison was the predeclared `B+W+X` arm, which itself failed globally. Therefore M96B **does not establish that V beats the best M94C-only yard-tail baseline**. V remains a promising vacancy-specific diagnostic that needs a direct-baseline, precommitted/prospective test before promotion.

### M96B scientific synthesis

The user's modular/puzzle framing is now a permanent research principle. M96B demonstrates four possible module relationships:

1. **Compatible but redundant / non-portable:** E contains real information in its native family but does not improve the stronger M94C point when transplanted.
2. **Helpful but too small:** W directionally improves the 2025 rush-yard tail but does not clear the materiality gate.
3. **Interactive, not additive:** X has native full-model tail signal but its isolated residual destroys M94C ranking.
4. **Conditional regime expert:** V remains promising specifically for vacancy/transition rows but is not yet proven versus the best baseline.

Do not conclude that M95 work was wasted, and do not force a combined stack merely because components once looked positive. The correct workflow is: capability ledger -> precommitted ablations -> incremental/non-degradation gates -> retain only compatible responsibility-specific modules.

**Current surviving global point architecture after M96B remains M94C/C.** No new global rushing-yard tail fusion earned retention in M96B. Generic carry-tail tuning remains closed.

# NEXT MIGRATION — M96C

Name: **M96C — M94C-Anchored RB Efficiency Residual Synthesis**

Primary question:

> Can leakage-safe player/offense/defense rushing-efficiency information predict the rushing-yard residual left after M94C's frozen opportunity/yard center, without changing carries and without damaging ordinary workload regimes?

Required design:

- Freeze M94C carries and M94C central rush-yard prediction.
- Fit directly on `actual_rush_yards - M94C_rush_yards` (or an algebraically equivalent efficiency residual). **Do not transplant the old M95C delta**; M96B just falsified that plug-in approach.
- Use only predeclared feature blocks from already-audited M95B/C/D lineage:
  - offensive/blocking/environment mean-efficiency block;
  - player-created efficiency block;
  - opponent run-efficiency/resistance block;
  - explosive/upside block only for tail unless it separately earns point value.
- Treat feature blocks as modules. Run frozen block ablations and compatible combinations; no broad feature soup and no after-result weight search.
- A block must show incremental value relative to its parent combination **and** satisfy non-degradation gates on ordinary workload slices before retention.
- No carry adjustment in M96C.
- Report MAE/RMSE/bias/correlation and 75+/100+ tail metrics by season and key workload regimes.
- Use strict temporal evaluation. 2024/2025 are already inspected/development evidence, not pristine confirmation.
- Keep V parked as a vacancy-specific diagnostic. A future conditional-expert test must compare V directly against the best M94C-based yard-tail baseline; do not promote it from the M96B parent comparison.
- Do not reopen generic M95 carry-tail coefficient search.
- Any M96C survivor remains research-only until genuinely prospective/untouched 2026 confirmation.

Decision/stopping path:

- If one or more predeclared efficiency blocks improve the M94C residual without material ordinary-game/season regression, keep the smallest compatible modular combination and freeze it for the next confirmation/conditional-tail stage.
- If different blocks help different regimes but conflict globally, test a precommitted conditional-expert architecture rather than averaging them blindly.
- If the predeclared efficiency blocks cannot improve M94C safely, retain the conservative point layer and move toward prospective confirmation / conditional routing instead of opening an unlimited M96 alphabet.
'''

s = s[:start] + new + s[end:]
p.write_text(s)

# Deliberate no-op marker: workflow exists before this push, so this commit triggers it.
