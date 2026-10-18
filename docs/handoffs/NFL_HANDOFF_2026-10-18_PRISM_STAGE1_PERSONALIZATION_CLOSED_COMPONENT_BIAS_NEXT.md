# NFL PRISM Handoff — 2026-10-18 — Stage-1 Personalization Closed / Signed Component-Bias Frontier Next

## Core architecture

PRISM remains an **individual-player projection model**.

> **THE PLAYER SHOULD HAVE THE RULES. THE POSITION SHOULD NOT HAVE THE PLAYER.**

Hierarchy:
`league prior -> role/archetype prior -> individual player evidence -> current player state`.

No unconstrained tiny-sample player models.
All personalization remains strictly prior, shrunk, market-specific,
support-gated and protected at named-player grain.

## Live research branch

Branch:
`research-individual-opportunity-roadmap-2026-10-08`

Branch head at this checkpoint:
`1d22ebc92f0de5999f1af0a90f503adce6989565`

PR #672 remains the research PR.

No paid OddsAPI.
No production mutation.
No Week-6 grading.

## Preserved first player-specific breakthrough

RB receiving Stage-2 remains the first qualified actual player-specific
projection rule.

Canonical bridge:
- run `39099474655`;
- artifact `12125534698`;
- digest `sha256:3d1f8043dc7a2ded0ed7fd9705b359d31a094a272ecce6958823ce660c2898e4`;
- 2025 CRPS gain +0.008520;
- player CI low +0.005441;
- game CI low +0.006117;
- equal-player CI low +0.005582;
- 78.51% named players improved;
- finite opportunity conserved.

RB rushing Stage-2 hard authority bridge remains CLOSED.
TE Stage-2 player utility signal remains CLOSED.

## Stage-1 player mechanism-benefit history — SOURCE READY ALL

Canonical source run:
- `39099745136` — SUCCESS;
- exact head `a1ad78254926da3a6ccc135a6726cc05f5dcd8d4`;
- artifact `12125863397`;
- digest `sha256:1b81c160ac1f8421d5eed83bb6a6389b5d6872f5a39f468df01b3a98a53d0d38`.

Result:
`docs/research/PRISM_STAGE1_PLAYER_MECHANISM_BENEFIT_HISTORY_V1_SOURCE_RESULT.md`.

Strictly-prior same-player Stage-1 mechanism history >=3:

| Family | 2024 rows / players | 2025 rows / players |
|---|---:|---:|
| QB | 620 / 71 | 781 / 76 |
| RB | 998 / 113 | 1,244 / 129 |
| WR | 1,457 / 189 | 1,889 / 209 |
| TE | 944 / 114 | 1,207 / 119 |

All source gates passed.

## Stage-1 hierarchical mechanism-utility signal — QUALIFIED PARTIAL

Canonical signal:
- run `39099930673` — SUCCESS;
- exact head `2e61f48ceace967c2533765eeb631c0f421617e9`;
- artifact `12126251820`;
- digest `sha256:9ef8c1c7e07fd4be381a745b5eb96d5ed976809d88aff478e623a5d74794f519`.

Result:
`docs/research/PRISM_STAGE1_PLAYER_HIERARCHICAL_MECHANISM_ELIGIBILITY_SIGNAL_V1_RESULT.md`.

Dispositions:
- QB: **CLOSED**
  - 2025 equal-player CI crossed zero;
  - 2025 RMSE gate failed.
- RB: **QUALIFIED**
  - 2025 meta-AE +0.013680;
  - player CI low +0.009691;
  - game CI low +0.010407;
  - equal-player CI low +0.009438;
  - 87.74% named players improved.
- WR: **CLOSED**
  - strong player diagnostics, but frozen 2024 development Spearman gate failed;
  - do not rescue.
- TE: **QUALIFIED**
  - 2025 meta-AE +0.001956;
  - player CI low +0.000819;
  - game CI low +0.001279;
  - equal-player CI low +0.000978;
  - 87.50% named players improved.

## RB/TE Stage-1 direct authority bridge — CLOSED ALL

Frozen action:
- low meta-support -> keep `p_candidate`;
- supported expected utility >0 -> keep `p_candidate`;
- supported expected utility <=0 -> revert to family `p_baseline`.

Canonical bridge:
- run `39100106463` — SUCCESS;
- exact head `e575aacae34d402af41755735dd78eac679a731a`;
- artifact `12126431332`;
- digest `sha256:d4046c51b47b6a0a17904f5c6331ffd35d51e8e66506310fbbc7545cbea0576f`.

Result:
`docs/research/PRISM_RB_TE_STAGE1_PLAYER_SPECIFIC_MECHANISM_AUTHORITY_BRIDGE_V1_RESULT.md`.

RB:
- 2024 Brier +0.000302;
- 2025 Brier **-0.002897**;
- 2025 player CI low -0.005788;
- game CI low -0.005570;
- equal-player CI low -0.004812;
- pooled Brier -0.001298;
- **CLOSED**.

TE:
- 2024 Brier **-0.000528**;
- 2025 Brier **-0.000590**;
- player CI low -0.002069;
- game CI low -0.002609;
- equal-player CI low -0.001273;
- pooled Brier -0.000559;
- **CLOSED**.

Do not rescue either Stage-1 bridge with soft blends, thresholds, support changes,
subsets, recency or alternate history windows.

## Important architecture lesson

We now have repeated proof that:

> A named player's prior history can predict whether an existing mechanism is
> likely to help, while a naive on/off or fallback intervention can still be the
> wrong projection action.

Therefore stop treating mechanism utility as automatically equivalent to
projection authority.

Qualified utility signal and qualified intervention are separate scientific
objects.

## Next distinct player-specific frontier

Move away from generic on/off/fallback logic.

The next candidate is a **signed player component-bias / sensitivity** lane:

> Does PRISM systematically over- or under-estimate the opportunity component or
> efficiency component for a named WR/TE, and is that signed bias stable enough
> strictly prior to predict the next component error?

This is materially distinct from CLOSED
`PRISM_PLAYER_MECHANISM_INSTABILITY_TRUST_V1`.

The closed instability test:
- used mechanism switch rate / dominant-state instability;
- target was total-error fragility/trust;
- changed no mean.

The new candidate, if source-valid:
- uses signed strictly-prior same-player opportunity / efficiency error;
- predicts the same signed component error on the next row;
- tests player-specific component **mean bias/sensitivity**, not trust or
  volatility;
- does not use switch rate as the predictive object;
- does not target absolute total error.

Canonical source authority already exists from mechanism-instability source:
- run `39061606501`;
- artifact `12111278777`;
- digest `sha256:f64c6e7efde9deaf91b7bc19076693a5106cb632338c461acc4d2cfe2ae37ea9`;
- WR historical rows contain prior8 signed opportunity and efficiency error
  means plus current signed component errors;
- TE same, plus catch-rate error;
- source-ready historical windows:
  - WR development 2023 / confirmation 2024;
  - TE development 2024 / confirmation 2025.

Before scoring:
1. freeze an anti-retest/source plan;
2. prove stable player support for signed component bias;
3. define role/family fallback excluding target player;
4. freeze one exact hierarchical signed-bias signal;
5. evaluate opportunity and efficiency components separately at player grain;
6. no automatic mean correction unless the signal itself qualifies and a later
   bridge is frozen separately.

Do not revive:
- mechanism-instability trust;
- generic efficiency volatility;
- raw recent-share replacement;
- WR target-quality closed chain;
- TE target-quality efficiency;
- same-QB allocation;
- generic DvP.

## Protected state

Still preserve:
- extended Week-4/5 replay complete;
- component cancellation proven;
- finite opportunity conservation;
- Stage-2 qualified only RB rushing / RB receiving / TE receiving;
- QB Stage-2 closed;
- WR Stage-2 source-blocked;
- Week-7 Stage-2 readiness fail-closed until legal Week-6 prior-state source
  matures;
- Weeks 7-10 WR dual prospective work protected;
- Week-6 locks immutable/ungraded;
- no paid OddsAPI without explicit authorization;
- sportsbook downstream only;
- no production mutation from research without authorization.

## Next-chat continuation rule

Read:
1. `AGENTS.md`;
2. only newest TOP checkpoint in `CURRENT_NFL_RESEARCH_HANDOFF.md`;
3. this handoff;
4. newest Issue #535 material comments;
5. reconcile live main / research branch / PR #672 / exact-head Actions.

Then continue the signed component-bias source/anti-retest audit if nothing newer
supersedes it.
