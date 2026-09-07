# RB-R1 Room Volume vs Individual Allocation — Result

## Canonical evidence
- Branch: `research-rb-r1-room-volume-vs-individual-allocation`
- Run: `34070566812`
- Job: `101587009981`
- Head SHA: `bba43f98a57eabcf44d31df03666bcf32e04369a`
- Artifact: `10000309225`
- Artifact digest: `sha256:daf7ace3bfc256b71d63930558322593bf0d88f046886848489f1877bf1bb116`
- Frozen plan commit: `9543967ef26a36c81db56b7f58526f23f04d4480`
- Mechanical filename correction before execution: `accc9657731461a5cfd0e203cbb50663bdb203c8`
- Evaluator commit: `5c7e5c07e9830a01da145e2422ee9086abc60f0a`

## Integrity
- Source rows: **1,393**
- Scoreable rows: **1,393**
- Qualifying players inherited from parent: **86**
- Parent CARRIES-dominant players: **35**
- Exact Shapley reconciliation max absolute error: **5.3290705182e-15**
- Sportsbook inputs used: no
- Model fitting used: no
- Production changed: no

## Official disposition
**`RB_CARRY_ERRORS_REMAIN_MIXED_ROOM_AND_ALLOCATION`**

Neither frozen routing gate passed. Thresholds are not changed.

## All player-games
- Carry MAE: **3.482576** attempts
- Actual-minus-projected carry bias: **+0.862725**
- Mean absolute RB-room-volume component: **2.200468** attempts
- Mean absolute individual-allocation component: **2.514077** attempts
- Allocation / room-volume ratio: **1.142519**

## CARRIES-dominant players — primary slice
473 player-games across 35 qualifying players:
- Carry MAE: **4.281844** attempts
- Actual-minus-projected carry bias: **+1.242116**
- Mean absolute RB-room-volume component: **2.540668**
- Mean absolute individual-allocation component: **2.903656**
- Allocation / room-volume ratio: **1.142871**

Frozen allocation route required ratio >=1.20; **FAIL**.

Player-level carry submechanisms among the 35 CARRIES-dominant RBs:
- **INDIVIDUAL_ALLOCATION: 17**
- **MIXED: 12**
- **ROOM_VOLUME: 6**

Allocation-dominant player rate: **48.57%**; frozen gate required >=50%; **FAIL**.
Room-volume-dominant player rate: **17.14%**; room-volume route also failed.

## Diagnostic contrasts
YPC-dominant players, 337 rows:
- room-volume component abs **2.178635**
- allocation component abs **2.229805**
- allocation/volume **1.023487**

Mixed parent players, 387 rows:
- room-volume component abs **2.307090**
- allocation component abs **2.338999**
- allocation/volume **1.013831**

## Individual heterogeneity is the main result
The global carry problem cannot be routed to one universal source, but the player-level split is substantial.

Examples classified `ROOM_VOLUME` within the parent CARRIES-dominant group include Christian McCaffrey, Kimani Vidal, Ashton Jeanty, Rico Dowdle, Alvin Kamara, and Kyle Monangai.

Examples classified `INDIVIDUAL_ALLOCATION` include Quinshon Judkins, Jaylen Wright, Ray Davis, Woody Marks, Tyler Allgeier, Josh Jacobs, Tank Bigsby, Chris Rodriguez Jr., Devin Neal, Samaje Perine, Chris Brooks, Brashard Smith, Ameer Abdullah, Trevor Etienne, LeQuint Allen Jr., Rasheen Ali, and Hunter Luepke.

This is exactly why one RB-wide correction is inappropriate. For some players the model's main carry problem is the amount of rushing volume available to the RB room. For others it is who receives that room volume.

## Next authorized research direction
RB-R1 itself authorizes no production change. It does, however, create genuinely new player mechanism classes that can be used in a separately frozen diagnostic:
- test current depth/role/injury/rotation information specifically inside the **INDIVIDUAL_ALLOCATION** subgroup;
- test team rush environment/game-script/pace/run-tendency information specifically inside the **ROOM_VOLUME** subgroup.

Those conditioned tests are new questions and must not be used to retroactively rescue the failed global RB-R1 routing gate.
