# PRISM Week-7 SEA Receivers vs KC Environment-Transmission Decomposition V1

**Date:** 2026-10-26  
**Lane:** read-only architecture / forensic decomposition  
**Players:** Jaxon Smith-Njigba, Cooper Kupp, AJ Barner  
**Scientific status:** **OPEN — NOT A PREDICTIVE LOCK**  
**Outcome use:** **0**  
**Sportsbook reacquisition:** **0**  
**Production changes:** **0**

## 1. Purpose

This is the representative WR/TE continuation of the Walker-first generic-skill transmission audit.

It uses the exact already-paid Week-7 artifact:

- run `39849371821`
- artifact `12456298985`
- digest `sha256:d1a1fb48e5bb920c5291f35b7457b5870801b689751deb2d742f98c555603a85`
- artifact head `4dc178b59da8708dbea978b8e54e753b1e2f75c5`

No Week-7 outcomes are used.

The audit separates:

1. direct generic opponent rules;
2. team-room conservation side effects;
3. already-promoted TE-R5P / WR-R15 specialist allocation;
4. opponent-blind ML / State / Bayesian arms;
5. known receiving-semantics research that is already closed against post-hoc rescue.

## 2. KC defensive state available pregame

Canonical Week-7 KC TeamContext contained:

| Feature | Value | Source state |
|---|---:|---|
| success_rate_def | 0.384127 | current 2026 guarded PBP |
| explosive_play_rate_allowed | 0.107937 | current 2026 guarded PBP |
| coverage_man_rate | 0.179 | current-slate Coverage V2; provider semantics pending audit |
| coverage_zone_rate | 0.789 | current-slate Coverage V2; provider semantics pending audit |
| middle_open_rate | 50.7 percentage points | legacy merged; provider audit needed |
| pressure_rate_generated | 0.149738 | M89/M90 promoted rolling-8 |
| def_pass_epa_allowed | -0.070883 | M89/M90 promoted rolling-8 |
| def_pass_success_allowed | 0.428047 | M89/M90 promoted rolling-8 |
| def_ypa_allowed | 5.848704 | M89/M90 promoted rolling-8 |

The pricing rows also retained position/alignment allowed-efficiency fields:

- WR YPT allowed = `6.4`
- TE YPT allowed = `6.4`
- RB YPT allowed = `6.0`
- outside YPT allowed = `6.9`
- slot YPT allowed = `6.9`

Those YPT-allowed fields do not enter the generic player receiving-efficiency rule.

SEA offensive success rate was `0.410667`. KC defensive success rate was `0.384127`, but generic `project_game_script()` still writes `pass_share = 0.57`; the opponent success signal does not move SEA's generic pass/rush split.

## 3. Pressure / efficiency transmission

SEA pressure allowed = `0.190213`.
KC pressure generated = `0.149738`.

Pressure mismatch:

`0.149738 - 0.190213 = -0.040475`

The rule requires magnitude beyond `0.075` before changing pass efficiency.

Therefore for JSN, Kupp and Barner:

- `rules_pass_eff_mult = 1.000000`
- `rules_volatility_mult = 1.000000`
- Bayesian YPT passes to `rules_ypt` unchanged.

Exact YPT authority:

| Player | Bayes YPT | Rules YPT | KC-caused YPT movement |
|---|---:|---:|---:|
| Jaxon Smith-Njigba | 10.174822 | 10.174822 | 0 |
| Cooper Kupp | 7.940591 | 7.940591 | 0 |
| AJ Barner | 6.902557 | 6.902557 | 0 |

Thus KC's `def_pass_epa`, `def_ypa_allowed`, explosive suppression, WR/TE YPT allowed, and outside/slot YPT allowed do not directly change these players' generic receiving efficiency.

## 4. Direct generic target-share transmission

### Jaxon Smith-Njigba

Artifact state:

- alignment = `LWR`
- depth role = `WR1`
- generic rules role = `WR1`
- Bayes target share = `0.316891`
- rules target share = `0.316891`

KC man rate is `0.179`, below the generic `>= 0.50` man gate.

Zone and middle-open do not directly alter a WR1 target share.

**Direct KC target-share movement for JSN: 0.**

### Cooper Kupp

Artifact state:

- alignment = `SWR`
- depth role = `WR3`
- generic rules role = blank
- Bayes target share = `0.151325`
- rules target share = `0.151325`

Production therefore gives Kupp no man/zone/middle target multiplier.

**Direct KC target-share movement for Kupp: 0.**

Important anti-retest boundary: the fact that upstream `alignment_position = SWR` is dropped before generic SLOT labeling is a previously confirmed deterministic receiving-semantics defect. The frozen B1 slot-carry candidate was prospectively graded in Week 3 and did **not** clear; targeted SWR MAE worsened on all three scored metrics. No slot-carry repair is authorized from this Week-7 case.

### AJ Barner

Artifact state:

- alignment = `TE`
- depth role = `TE1`
- Bayes target share = `0.136722`

KC triggers two existing generic TE rules:

1. zone `0.789 >= 0.60` => `x 1.15`
2. middle-open `50.7` percentage points corresponds to `0.507`; intended threshold is `>= 0.50` => `x 1.10`

Combined:

`1.15 x 1.10 = 1.265`

Barner:

`0.136722 x 1.265 = 0.172954`

**Direct generic KC target-share movement for Barner: +26.5%.**

The historical middle-open unit mismatch is a real known production semantic defect, but it does **not** change the KC gate outcome in this specific case because `50.7% -> 0.507` still clears the intended `0.50` threshold. The frozen middle-open repair also did not clear its prospective Week-3 promotion gates, so this audit does not reopen it.

## 5. Conservation means opponent effects are not player-local

SEA's full football universe had:

- neutral/Bayesian raw target-share sum = `1.468634`
- as-run post-rule raw target-share sum = `1.601149`

Why the team total grew:
- all SEA TEs received the KC zone + middle multipliers;
- RB/FB receiving shares received the KC zone multiplier;
- WR shares in this room received no direct generic KC multiplier.

The explicit entitlement layer then normalizes modeled player target mass to `0.95`.

As-run team scale:

`0.95 / 1.601149 = 0.593324`

A rule-neutral team scale, holding player Bayesian shares but tripping none of the generic opponent target gates, would be:

`0.95 / 1.468634 = 0.646860`

So KC's direct boosts to TE/RB target shares mechanically compress the conserved target entitlement available to unchanged WRs.

This creates a real **indirect** opponent effect on JSN and Kupp even though their own generic matchup multipliers are 1.0.

## 6. Separate specialist allocation from opponent transmission

The promoted target-allocation specialists occur after generic rules and finite team entitlement.

### JSN

- rules target share = `0.316891`
- M38 post-hierarchy share = `0.409603`
- baseline conserved entitlement = `0.243027`
- WR-R15 route = `M38_WR1_ANCHOR`
- final target entitlement = `0.243027`

JSN's specialist effect is player/room allocation, not KC-specific matchup science.

### Kupp

- rules target share = `0.151325`
- M38 post-hierarchy share = `0.108976`
- baseline conserved entitlement = `0.064658`
- WR-R15 applies in the WR2+ room
- final target entitlement = `0.106951`
- WR-R15 delta = `+0.042293`

That restoration is based on WR room / snap-history science, not opponent coverage.

### Barner

- rules target share = `0.172954`
- baseline conserved entitlement = `0.102618`
- TE-R5P applies
- final target entitlement = `0.135849`
- TE-R5P delta = `+0.033231`

Again, TE-R5P is player/room allocation after generic opponent rules; it is not a KC defensive coefficient.

## 7. Neutral-opponent opportunity decomposition

Because WR-R15 and TE-R5P conserve their declared room pools, the rule-neutral counterfactual can be evaluated at the target-entitlement level without changing their learned player ordering.

### WRs: indirect compression only

The neutral/as-run scale ratio is:

`0.646860 / 0.593324 = 1.090230`

Therefore a generic opponent that trips none of the receiving target rules would imply approximately:

| Player | As-run final entitlement | Rule-neutral entitlement | KC effect |
|---|---:|---:|---:|
| JSN | 0.243027 | 0.264956 | -0.021929 (-8.28%) |
| Kupp | 0.106951 | 0.116601 | -0.009650 (-8.28%) |

KC does not directly suppress either WR. The suppression is a **finite-room consequence** of KC boosting SEA's TE/RB receiving entitlements.

### Barner: direct boost partly offset by conservation

Neutral TE pool:

`sum(Bayes TE target shares) x neutral team scale = 0.213683`

Barner's frozen TE-R5P room share = `0.547916`.

Rule-neutral Barner entitlement is therefore approximately:

`0.213683 x 0.547916 = 0.117080`

Compared with as-run `0.135849`:

- final opportunity movement = `+0.018769`
- final opportunity lift = approximately **+16.0%**

So Barner's direct +26.5% generic multiplier is damped by finite-team conservation; the final target entitlement increases about 16% versus a no-gate opponent.

## 8. Downstream mean materiality

SEA rules volume:

- plays = `58.804190`
- pass rate = `0.57`
- expected pass opportunities = `33.518388`

The simulator's receiving-yard expectation is approximately linear in target entitlement × YPT before finite Monte Carlo sampling.

As-run paid outputs:

| Player | MC rec yds | ML rec yds | State rec yds | Final model rec yds |
|---|---:|---:|---:|---:|
| JSN | 82.864647 | 94.459776 | 48.774518 | 84.629822 |
| Kupp | 28.614777 | 31.665101 | 48.774518 | 30.468071 |
| Barner | 31.435194 | 29.721104 | 24.821583 | 30.618585 |

`rec_yards` calibrated ensemble weights:
- MC = `0.659889`
- ML = `0.292427`
- State = `0.047684`

ML and State are opponent-blind here. Therefore generic opponent transmission reaches the final mean only through the MC arm.

Analytic rule-neutral MC expectations are approximately:

- JSN: `90.36` rec yds
- Kupp: `31.03` rec yds
- Barner: `27.09` rec yds

Using the frozen ensemble weights, this corresponds to approximate final-mean matchup materiality of:

- JSN: roughly **-4.95 receiving yards** versus rule-neutral
- Kupp: roughly **-1.60 receiving yards**
- Barner: roughly **+2.87 receiving yards**

These are deterministic/analytic counterfactual estimates around the exact as-run authority, not a re-simulated 25k-draw artifact and not a predictive promotion result.

## 9. What this case actually says

### JSN

- own KC target multiplier: none;
- own KC efficiency multiplier: none;
- opponent-specific YPT allowed / outside YPT / pass EPA do not reach his efficiency;
- final opportunity is nonetheless indirectly compressed because opponent rules boost other SEA position rooms.

### Kupp

- upstream player alignment correctly says `SWR`;
- generic production does not carry that into a SLOT rule label;
- direct KC target multiplier: none;
- own efficiency multiplier: none;
- same indirect finite-room compression as JSN;
- do **not** revive the previously failed slot-carry candidate from this case.

### Barner

- generic opponent coverage does materially reach opportunity;
- direct Bayes-to-rules target boost = +26.5%;
- finite team conservation reduces final entitlement lift to ~+16%;
- KC does not change his YPT or volatility in the live generic path;
- TE-R5P adds a separate player-specific allocation layer after matchup rules.

## 10. Architecture conclusion

The current receiving path is not literally opponent-blind, but its transmission is highly uneven:

1. **Opportunity:** coarse coverage threshold rules can move position/role target shares.
2. **Finite-room coupling:** one position's matchup boost can suppress unchanged teammates.
3. **Efficiency:** most rich defensive efficiency context remains dropped/stranded for generic WR/TE projection.
4. **Game script:** opponent success state still does not alter the generic 0.57 pass rate.
5. **Player identity/alignment:** upstream SWR can disappear before generic slot rules.
6. **Specialists:** TE-R5P / WR-R15 add player-specific allocation quality, but are not opponent-specific defensive models.

Disposition remains:

**OPPONENT / ENVIRONMENT TRANSMISSION = OPEN ARCHITECTURE / RESEARCH PROBLEM.**

This case does not authorize a new coefficient, a slot repair, a middle-open repair, or production mutation.

## 11. Next valid work

1. Preserve Walker + JSN/Kupp/Barner as descriptive architecture decompositions.
2. Finish the already-assigned frozen Stage-2 source-readiness reprobe before evaluating conditional allocation in production.
3. Correct Player Identity Card V1 and expose target-week role/participation state.
4. Future transmission research must test strictly-prior player × role × opponent features across independent seasons/windows, with no Week-7 fitting and no named-player carveout.
5. Keep betting-construction V2 separate from predictive science.
