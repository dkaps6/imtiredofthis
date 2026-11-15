# TNF 2026 week 10 (WAS 24, NYG 21) -- every card shipped, graded

Final per nflverse (WAS 24-21; a live-blog "21-17 NYG" quoted earlier in the
session was wrong). Rows: `data/research/tnf_2026_wk10_graded_cards_v1.csv`.

Key lines: Likely 9-127 (11 targets), Fields 3-62, Diggs 6-83, Nabers 5-44 (7 tgt),
Williams 2-42, Mooney 2-8, White 11 carries 47, Daniels 11 carries 28, Skattebo 18-78.

| card (time shipped) | built by | tickets won | legs hit |
|---|---|---|---|
| A nine-gate builder, cushion-ranked | validated gates | 2 of 2 | 5/5 distinct legs |
| B nine-gate builder, value-ranked (23:25Z) | validated gates | 2 of 2 | 5/5 distinct legs |
| C blended estimator (00:05Z) | ad hoc, invented weights | 1 of 4 | 2/6 distinct legs (33%) |
| D all-markets final card (00:20Z) | ad hoc, invented weights | **0 of 5** | 4/10 distinct legs (40%) |

Card B's SAFE (Diggs >29.5 + Fields >14.5) paid +115 and won. Each later
rebuild of that card was worse.

## Honest reading

* **One game.** Not evidence about any method. The gate-built cards are chalk
  with small payouts (-506, -466, +115, -305); B's SAFE does not clear the revised
  +150 floor.
* **What lost card D was the leg type, not bad luck.** Rachaad White under 38.5
  (47 rushing yards) was on four tickets and killed SAFE, MODERATE, MEDIUM and
  HIGH. `MODEL_BACKED_ALT_LINE_PARLAY_CONSTRUCTION_V1` section 9 excludes
  "RB rushing/rush+receiving under clusters solely because the model is low" and
  the validator now warns on exactly this. It was in the SAFE ticket anyway.
* **The Likely under was wrong and the operator's football read was right.** The
  model had him at 43.7 yards (UNDER 46.5 at 59%); he went 9-127. Diagnosis in
  `LIKELY_MISS_2026_WK10_DIAGNOSIS_V1.md`.
* **Overs on Nabers (44) and Mooney (8) missed**, both volume-dependent legs
  priced near fair. Diggs and Fields overs, the two legs in every version, hit.
* Third consecutive week where an ad hoc pick replaced a gate-certified one and
  did worse (V27 section 5 records week 9: "the gate existed and was not used for
  that pick").
