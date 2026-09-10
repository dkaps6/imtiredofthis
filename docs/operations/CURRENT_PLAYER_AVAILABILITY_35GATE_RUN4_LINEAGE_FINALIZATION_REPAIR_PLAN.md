# Current Player Availability 35-Gate Certification V1 — Run4 Lineage Finalization Repair Plan

Status: `FROZEN_MECHANICAL_REPAIR_ONLY_BEFORE_RUN4`

## Parent evidence

Run3 is immutable:
- head `76d01dd8e7b26ef8921cd70c18f27957da46c560`
- run `34461561636`
- job `102820358570`
- evidence artifact `10145975346`
- evidence artifact digest `sha256:dd32b45f6746176911ca68aa1d73a8a75325f92cf7b0951c335ae5484752e0ed`
- gates 1-34: 34/34 PASS
- failure: gate-35 finalization step only

## Frozen diagnosis

The Run3 evidence artifact uploaded successfully and GitHub assigned both a stable artifact ID and SHA-256 digest. The workflow nevertheless failed when passing upload-step output metadata directly into the unchanged gate-35 finalizer. The frozen football evidence and gates 1-34 do not require rerunning to diagnose this plumbing issue.

## Frozen Run4 repair

Run4 may repeat the exact Run3 computation solely to preserve a single self-contained immutable 35-gate workflow, but the only implementation change versus Run3 is the lineage metadata resolution step after the gates 1-34 evidence upload.

Instead of passing `steps.evidence_upload.outputs.artifact-id` and `steps.evidence_upload.outputs.artifact-digest` directly to the finalizer, Run4 must:

1. query GitHub Actions artifact metadata for the current `GITHUB_RUN_ID` after upload;
2. select exactly one artifact named `current-player-availability-35gate-evidence-v1-run4`;
3. require a non-empty numeric artifact ID;
4. require a digest beginning `sha256:`;
5. pass those API-resolved values into the exact unchanged finalizer blob `a5302186eadb748863c70b175421d064885dde60`.

No fallback digest may be fabricated or locally calculated. Gate 35 remains exactly the frozen requirement: immutable execution lineage and disposition must capture branch, head, run, job, artifact ID and GitHub-reported artifact digest.

## Everything that must remain unchanged

- 35 frozen gate definitions and thresholds
- gates 1-34 evaluator blob `3bc2bb390ab6762f1c20e775152812d3f8e9729e`
- gate-35 finalizer blob `a5302186eadb748863c70b175421d064885dde60`
- football-stack runner blob `dc60ab10bc18777d6b13cb5f81f285e8c248f3ab`
- fixture builder blob `dbe9f41e4d175b503bf0cd8caf51c1dc8d0da95d`
- immutable candidate Run `34447900206`, Artifact `10140425929`, digest `sha256:6b31ae40d648780673b7953b57323d509b302ea7a82fe99801b024ede2603f37`
- eligible-team seams and QB C2 current-output seams
- R26, R22 and all scientific model parameters/assets
- no-sportsbook football boundary

## Disposition

If the unchanged finalizer receives valid GitHub API-resolved lineage and all 35 gates pass, Run4 may produce `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_PASS_READY_FOR_PROMOTION`.

If any gate fails, the result is immutable `CURRENT_PLAYER_AVAILABILITY_FULL_SLATE_INTEGRATION_FAIL_NO_PROMOTION`.

A PASS authorizes only a separately frozen promotion implementation; it does not itself mutate production.
