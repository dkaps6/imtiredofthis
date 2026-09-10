# QB Pass-Rate Anchor Semantic Recalibration A1 — Result

## Lineage

- Branch: `research-qb-pass-rate-anchor-semantic-recalibration-a1`
- Frozen plan commit: `8c12d223103406a14d7913da378c8d2d1b012276`
- Evaluator commit: `1d32b6454122c747901eccab8e5f5b4b30a994fb`
- Execution head: `65b2dc48693326ce837609a839842e107e80b4af`
- Run: `34539132735`
- Job: `103077537276`
- Artifact: `10176570118`
- Artifact name: `qb-pass-rate-anchor-semantic-a1`
- Digest: `sha256:cacf810f653d2f4c3bcb24a19359c03ab9e85ced2775b2a2c4fdb20e6a693ffc`

## Integrity

All integrity gates passed:

- exactly 444 corrected 2024 M89 QB-game rows;
- unique canonical keys;
- baseline `pred_D == pred_plays * 0.57` within numerical tolerance;
- baseline attempts identity reproduced within numerical tolerance;
- exact frozen candidate grid `0.57..0.63` by 0.01;
- zero sportsbook inputs;
- zero model fitting;
- zero production changes;
- M89/M90 synthesis was not changed and was not used to select the anchor;
- 2025 was not scored or summarized.

## Frozen winner

The preregistered lexicographic selection rule chose **`0.59`**.

### 0.57 baseline -> 0.59 winner

- Pass-rate MAE: `0.0882520861 -> 0.0862168982` (gain `0.0020351879`)
- Pass-rate bias: `-0.0204555495 -> -0.0004555495` (97.77% absolute-bias reduction)
- Pass-rate p90 abs error: `0.1830256410 -> 0.1738131313`

- Team pass-opportunity MAE: `7.4034391293 -> 7.0740335152` (gain `0.3294056141`)
- Team pass-opportunity bias: `-3.7101444250 -> -2.5466628480` (31.36% absolute-bias reduction)
- Team pass-opportunity p90 abs error: `14.9447697812 -> 14.4268282399`

- QB-attempt MAE: `7.3396804700 -> 7.0338535596` (gain `0.3058269104`)
- QB-attempt bias: `-4.3660223551 -> -3.3843970924` (22.48% absolute-bias reduction)
- QB-attempt p90 abs error: `15.0547287719 -> 14.8379380943`
- QB 8+ attempt miss rate: `39.1892% -> 35.5856%`
- QB 10+ attempt miss rate: `26.8018% -> 22.7477%`

- Deterministic mechanics passing-yard MAE: `65.1272276209 -> 63.0666399048` (gain `2.0605877161` yards)
- Mechanics passing-yard bias: `-38.0659621852 -> -31.1066834757`
- Mechanics passing-yard p90 abs error: `131.3688082697 -> 128.1591780458`

## Bootstrap evidence

Paired 5,000-draw bootstrap under the frozen seeds:

- Pass-rate MAE gain: mean `0.0020334`, `P(gain > 0)=0.9878`
- Team-D MAE gain: mean `0.3297006`, `P(gain > 0)=1.0000`
- QB-attempt MAE gain: mean `0.3065015`, `P(gain > 0)=1.0000`

## Gate disposition

Every scientific gate passed **except one**:

- required pass-rate MAE gain: `>= 0.0040`
- observed: `0.0020351879`

The preregistered gate is not relaxed after seeing the result.

Formal disposition:

`QB_PASS_RATE_ANCHOR_SEMANTIC_A1_FAIL_NO_CONFIRMATION`

Therefore:

- 2025 remains sealed for this candidate family;
- `0.59` is **not** authorized for production or confirmation;
- production remains at the protected `0.57` anchor;
- no retune of A1 is allowed.

## Scientific interpretation

A1 strongly supports a **level/centering issue** under corrected M89 semantics: moving the constant anchor from 0.57 to 0.59 nearly eliminated rate bias and materially improved team pass opportunities, QB attempts, attempt tails, and raw passing-yard mechanics.

However, the constant anchor shift did not improve per-game pass-rate MAE enough to clear the frozen threshold. That means the remaining opportunity problem is not primarily solved by a league-wide level correction. The next research question is **game-to-game pass-opportunity differentiation** around the league center.

This does not reopen M42's generic trailing pass-rate family. The next candidate must come from a genuinely different football mechanism or a more precise decomposition of why target-game rate departs from the league anchor.
