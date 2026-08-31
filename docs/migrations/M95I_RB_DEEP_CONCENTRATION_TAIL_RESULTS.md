# M95I — Calibrated Deep-Concentration + Workload-Tail Integration Results

## Authoritative run

- Workflow: `M95I RB Deep-Concentration Tail Integration`
- Run: `33402566592`
- Job: `99522191259`
- Tested SHA: `cd33d3dcd1a1148e216ea07131cf0b247c58e5f4`
- Research branch: `research-rb-m95i-deep-concentration-tail`
- Artifact: `migration-95i-rb-deep-concentration-tail`
- Artifact ID: `9761827238`
- Artifact SHA256: `2ec133f4a97b2207d678544e7bde11c98e2d93701e1977536122c5104180fb46`
- Artifact size: `153,295` bytes
- Execution: success
- Scientific disposition: `RETAIN_M95I_AS_DIAGNOSTIC_DO_NOT_PROMOTE`
- Production change: `0`
- Sportsbook inputs: `0`
- M94C central reference preserved: `1`

## Scientific question

Can the validated M95H probability that a specific RB commands at least 70% of team RB carries be calibrated by role-transition regime and combined with M95F's calibrated 20+/25+ workload-state signal plus M94C team opportunity to selectively expand the carry tail without materially damaging ordinary games?

## Frozen inputs and protocol

M95I froze:

- M94C as the central carry/opportunity reference;
- M95F calibrated 20+/25+ workload-state probabilities;
- only the validated M95H `P(RB share >=70%)` component, not M95H's failed exact-lead or >=60% components.

All calibration/integration/transform selection used 2024 only. The 2025 M95H share probability came from the frozen authoritative M95H artifact rather than being rebuilt from current source snapshots.

The selected architecture was:

- M95H share70 spec: `entitlement_competition`, `C=.03`
- incumbent/vacancy share70 calibration shrink: `10`
- 20+ meta model: `tail_share70_opportunity`, `C=.3`
- 25+ meta model: `tail_share70`, `C=.3`
- selective carry transform: `share65_env`
- historical 20-24 carry state mean: `21.4375`
- historical 25+ state mean: `26.269231`
- frozen operating thresholds: `0.15` for 20+, `0.15` for 25+

The carry transform did not use an arbitrary gamma. It applied a capped expected-state uplift only when deep-concentration, workload-tail, and projected team-rush-environment conditions aligned.

## Share70 calibration result

The regime calibration solved an important M95H problem in vacancy situations.

### 2025 vacancy >=70% share

Raw M95H:

- actual rate: `7.76%`
- mean probability: `18.33%`
- AUC: `.865005`
- Brier: `.080244`
- log loss: `.258858`

M95I calibrated:

- mean probability: **`10.18%`**
- AUC: `.865005`
- Brier: **`.057475`**
- log loss: **`.201991`**

Thus the earlier vacancy overconfidence was materially reduced without sacrificing ranking.

Overall share70 also improved slightly in 2025:

- Brier `.090868 -> .090004`
- log loss `.280397 -> .277906`
- AUC `.919599 -> .922962`

The incumbent subgroup became slightly underconfident after the global/regime mapping, so this is not a universal probability replacement by itself.

## 2024 integration selection

On the 2024 selection slice, adding calibrated share70 to the workload-tail signal was strongly positive.

20+ selected meta model:

- AUC `.917726`
- AUC gain `+.027366`
- Brier `.058157`
- Brier gain `+.007430`

25+ selected meta model:

- AUC `.963790`
- AUC gain `+.022321`
- Brier `.024802`
- Brier gain `+.001742`

The selected `share65_env` carry transform passed the pre-specified development protection gate.

## Untouched 2025 carry results

| Slice | M94C MAE | M95I MAE | Gain |
|---|---:|---:|---:|
| All RB | 3.411003 | 3.446095 | -0.035092 |
| 0-5 | 2.559242 | 2.564019 | -0.004778 |
| 6-10 | 3.248217 | 3.295814 | -0.047597 |
| 11-14 | 3.470493 | 3.587871 | -0.117377 |
| 15+ | 5.336313 | 5.354968 | -0.018655 |
| 20+ | 7.876590 | **7.643905** | **+0.232685** |
| 25+ | 11.954550 | **11.856825** | **+0.097725** |
| Bellcow60 | 5.309789 | **5.307823** | +0.001966 |

The ordinary-game damage is small, especially 0-5 and 6-10, but the 11-14 slice regresses by about 0.12 carry. The 20+ tail improves materially enough to confirm directional value, while the 25+ deterministic mean improvement is still too small.

On actual 25+ games the mean projection moves only from `15.045` to `15.143` against an actual mean of `27.0`. M95I therefore does not solve absolute extreme-workload compression.

## Projection distribution

M94C:

- max `24.989`
- p95 `17.973`
- p99 `20.083`
- projected >=18: `68`
- >=20: `16`
- >=22: `2`
- >=25: `0`

M95I:

- max `24.989`
- p95 **`18.657`**
- p99 **`21.339`**
- projected >=18: **`93`**
- >=20: **`35`**
- >=22: **`9`**
- >=25: `0`

The selective integration expands the upper central tail without creating a deterministic 25+ projection.

## 20+ probability integration — strong validated signal

All 2025 RB games:

M95F:

- mean probability `.100862`
- AUC `.846474`
- Brier `.062636`
- log loss `.208788`

M95I joint:

- mean probability **`.094612`**
- AUC **`.860145`**
- Brier **`.060469`**
- log loss **`.199115`**

This is a meaningful improvement in ranking and probability scoring simultaneously.

Incumbent 20+ also improves:

- AUC `.843120 -> .856310`
- Brier `.065613 -> .063613`
- log loss `.216530 -> .207589`

Vacancy 20+ calibration improves strongly:

- actual rate `2.59%`
- M95F mean `7.26%`
- M95I mean **`5.15%`**
- Brier `.029865 -> .025854`
- log loss `.123561 -> .105825`

Vacancy 20+ AUC slips `.884956 -> .870206`, so the benefit is calibration rather than ranking in that subgroup.

## 25+ probability integration — population split appears

Overall 25+:

- AUC `.844321 -> .860421` — better ranking
- Brier `.017985 -> .019310` — worse calibration
- log loss `.078526 -> .080255` — worse

The important result is that two very different populations emerge.

### Vacancy / role-transition 25+

- actual rate `0.86%`
- AUC **`.721739 -> .939130`**
- Brier **`.008840 -> .008445`**
- log loss **`.048953 -> .040330`**

This is an unusually strong isolated signal. Recipient-specific deep-concentration plus tail state information appears highly useful for ranking extreme workloads when a role is transitioning.

### Stable workhorse 25+

- actual rate `4.33%`
- M95F predicted `11.12%`
- M95I predicted **`13.95%`**
- AUC `.588851 -> .600823`
- Brier `.050569 -> .057826`

The model becomes more overconfident even though ranking improves slightly. This is the main failure preventing advancement.

Stable workhorse 20+ also worsens:

- actual `20.87%`
- M95F `29.53%`
- M95I `31.09%`
- AUC `.584812 -> .548766`
- Brier `.181596 -> .189046`

Thus the remaining problem for established workhorses is not identifying the player. It is identifying the particular **week** in which the normal lead back actually converts his role into an extreme workload state.

## Operating point audit

20+:

- M95F threshold `.20`: 60 TP / 199 FP / 38 FN, recall `61.2%`, precision `23.2%`
- M95I threshold `.15`: 77 TP / 261 FP / 21 FN, recall **`78.6%`**, precision `22.8%`

M95I finds many more true 20+ games but also broadens the flag population.

25+:

- M95F threshold `.10`: 12 TP / 139 FP / 12 FN
- M95I threshold `.15`: 8 TP / 102 FP / 16 FN

The selected 25+ operating point does not generalize and is another reason the integrated model cannot advance wholesale.

## False-positive / false-negative pattern

Many largest false positives are established lead backs in plausible high-volume football environments: James Cook, Quinshon Judkins, Christian McCaffrey, Jonathan Taylor, Derrick Henry, Saquon Barkley, Jaylen Warren, Ashton Jeanty, De'Von Achane, Kyren Williams, etc.

This reinforces that **role entitlement is already known** for many of these players. The missing variable is whether that particular game converts to the high-rushing state.

Major 25+ misses still include Derrick Henry 36, Kareem Hunt 30, Jonathan Taylor 32, Rico Dowdle 30, James Cook 32, Kyle Monangai 26, Cam Skattebo 25, Christian McCaffrey 28, Emanuel Wilson 28, Kimani Vidal 25, etc.

Several misses fail the selected transform because projected team rush environment is too low or the deep-share probability is below the chosen gate. This again suggests that the stable-incumbent and role-transition populations need different mechanisms.

## Downstream yard sensitivity

Changing carries alone while holding M94C expected YPC fixed is not a production model; this was a sensitivity audit.

All RB:

- rush-yard MAE worsens `21.6542 -> 21.8395`
- rush+rec worsens `25.7792 -> 25.9616`

20+:

- rush yards improve **`40.1352 -> 39.5217`** (+0.6135)
- rush+rec improves +0.1606

25+:

- rush yards improve **`49.3105 -> 48.8887`** (+0.4218)
- rush+rec improves +0.4218

The carry-tail adjustment propagates in the correct direction for the true high-workload games but damages the middle/overall population when applied as a central adjustment.

## Legacy guard

M95I does not modify the production rush-yard model. It inherits M94C's tiny all-player rushing guard failure:

- baseline MAE `7.758864`
- M94C candidate `7.762069`
- gain `-0.003205`

This gate is not waived.

## Scientific conclusion

M95I does not pass as one combined tail architecture. However, it isolates a highly important structural split:

1. **Role-transition/vacancy backs:** recipient-specific deep-concentration plus workload-tail information is unusually strong, especially for 25+ ranking.
2. **Stable incumbents/workhorses:** entitlement is already known, but the model still cannot reliably identify which specific week becomes a monster-workload game. Tail probability is overconfident in this population.

Therefore the next experiment should not add more generic entitlement features. It should split the workload-tail problem by regime and attack **week-specific workload conversion among stable incumbents** while preserving the strong role-transition signal.

## Recommended next migration — M95J

**M95J — Regime-Specific Workload Conversion**

Primary question:

> Can separate stable-incumbent and role-transition tail models preserve M95I's strong vacancy signal while using football-derived weekly game-environment variables to distinguish which established workhorse weeks actually convert into 20+/25+ workloads?

Stable-incumbent candidates should emphasize week-specific conversion variables rather than more role identity:

- M94C projected team rush volume;
- projected offensive plays;
- football-model lead / neutral / trail probabilities;
- neutral rush rate / early-down rush tendency;
- opponent run-defense matchup signal from M95A/B;
- offensive drive sustainability / efficiency;
- opponent offense strength as a script counterweight;
- QB rush share / mobile-QB siphoning;
- active RB2 competition even when RB1 remains incumbent;
- recent carry/share trend;
- practice/injury limitation;
- week 17/18 rest context;
- coaching tendency to continue feeding the lead RB in positive game states where leakage-safe history supports it;
- home/away and other football-only context.

The role-transition branch should preserve and separately calibrate M95I's deep-share + tail integration rather than forcing it through the stable-workhorse model.

No sportsbook inputs. No manual tail boost after seeing validation.
