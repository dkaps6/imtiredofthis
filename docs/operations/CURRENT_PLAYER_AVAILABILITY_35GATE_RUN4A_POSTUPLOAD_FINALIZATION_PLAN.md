# Current Player Availability 35-Gate Certification V1 — Run4A Post-Upload Finalization Plan

Status: `FROZEN_BEFORE_EVIDENCE_ONLY_FINALIZATION`

## Why this supersedes rerunning football for the gate-35 plumbing repair

Run3 already produced the first valid immutable gates 1-34 evidence artifact:

- branch `ops-current-player-availability-35gate-cert-v1`
- head `76d01dd8e7b26ef8921cd70c18f27957da46c560`
- run `34461561636`
- job `102820358570`
- evidence artifact `10145975346`
- artifact name `current-player-availability-35gate-evidence-v1-run3`
- GitHub digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- gates 1-34: exactly 34/34 PASS

Recomputing the football stack and fixtures would create a second sample of evidence without scientific need. Because gate 35 is only immutable lineage capture, the more conservative repair is to consume the already-uploaded Run3 artifact and finalize Run3's exact lineage.

This plan is a separate frozen refinement of `CURRENT_PLAYER_AVAILABILITY_35GATE_RUN4_LINEAGE_FINALIZATION_REPAIR_PLAN.md`; that earlier plan remains preserved.

## Frozen implementation

The post-upload finalization workflow must:

1. verify the unchanged finalizer blob is exactly `a5302186eadb748863c70b175421d064885dde60`;
2. query GitHub artifact metadata for artifact ID `10145975346`;
3. require artifact name `current-player-availability-35gate-evidence-v1-run3`;
4. require digest exactly `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`;
5. download the exact Run3 evidence artifact by run/name;
6. require `current_player_availability_35gate_preliminary.json` to report `all_1_34_pass=true`, `passed_1_34=34`, `failed_1_34=0`, and `evaluated_gate_count=34`;
7. invoke the exact unchanged finalizer with Run3 lineage values: branch `ops-current-player-availability-35gate-cert-v1`, head `76d01dd8e7b26ef8921cd70c18f27957da46c560`, run `34461561636`, job `102820358570`, artifact ID `10145975346`, artifact digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`;
8. upload the resulting `current_player_availability_35gate_result.json` as a new immutable final-result artifact.

The new final-result artifact is merely a durable copy of the finalized Run3 result. Gate 35 evidence must point to the Run3 evidence artifact, not to the wrapper/final-result artifact.

## Prohibited changes

- no football stack rerun
- no fixture rerun
- no gate 1-34 reevaluation
- no gate definition or threshold change
- no candidate regeneration
- no scientific model change
- no R26/R22 change
- no availability semantic change
- no sportsbook input
- no production mutation

## Allowed disposition

Because gates 1-34 are already immutable PASS, the unchanged gate-35 finalizer decides the complete Run3 result solely from valid immutable lineage. A 35/35 result may be labeled `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`; anything else remains no-promotion.

A 35/35 PASS authorizes only a separately frozen promotion implementation and post-promotion Full Slate verification.
