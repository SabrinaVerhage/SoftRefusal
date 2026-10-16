# Image LoRAs: installation and operation

A LoRA is a small set of changes applied to a base image model. It does not
replace that model. The website sends an approved adapter ID and strength; the
GPU server loads its local file for that image and unloads it afterward.

## Selected adapter

- Website name: **Body shape · portrait** (`body-shape-v1`).
- Creator: hejl1622; Civitai model 2558153, version 2711426, file 2597207.
- Source: https://civitai.red/models/2558153?modelVersionId=2711426
- File: `flux2_4B_ID_01B_DadBod_6.safetensors`.
- SHA256: `db64592e7c1213a6b1748b3489101b42702eaecfe1f801226954939c8d7f652a`.
- Catalog reports **Flux.2 Klein 4B-base**. The running checkpoint is the
  distilled **FLUX.2-klein-4B**. Matching family and size is promising, but does
  not prove compatibility or good results. The 16 October live test verified
  all 200 tensors against this exact running checkpoint and generated a clothed
  portrait successfully. This is a smoke test, not a general quality evaluation.
- Scope: non-explicit portraits of fully clothed adults. No trigger phrase or
  optimal strength has been confirmed. The initial 0.7 is a trial setting.

## Current state, 16 October 2026

Installed and enabled on the existing A40. After the browser download was blocked,
the user supplied a locally downloaded copy. Its 48,195,840 bytes matched the
published SHA256 above. PEFT 0.18.1 was installed with no dependency upgrades;
`pip check` passed. The six changed GPU source/catalog files are deployed.

This file exports bare PEFT keys ending in `.lora_A.default.weight` and
`.lora_B.default.weight`. The pipeline expects a `transformer.` prefix and no
saved adapter name. The loader now normalizes those names in memory, checks every
target's dimensions, then compares all loaded tensors against the source. It
does not alter the downloaded file or silently accept ignored weights.

The private test used a fully clothed adult man, seed 1729, 512x768, four steps,
strength 0.7. Original → LoRA → original succeeded. The LoRA changed the image;
the two original images had identical PNG hashes. No adapter modules remained
after each request. Images were visually reviewed before public enablement.
31 synthetic API/adapter tests passed on the Pod. The base remains BF16.

The running API environment has `SOFT_REFUSAL_ENABLE_LORAS=1`. Include this flag
when restarting the service; its safe default remains disabled. The standard
`SOFT_REFUSAL_ACCEPT_REQUESTS=1`, Hermes flag and existing bind/data/auth settings
must also be preserved. Do not restart or extend paid compute without checking
the current authorized session in DEPLOYMENT_STATUS.md.

## Installation sequence used

1. Verify the version, intended model and publisher's usage terms. Hash the
   local file and compare the complete SHA256 above. A matching hash verifies
   identity, not compatibility or output quality.
2. Recheck current deployment status, the existing Pod and remaining authorized
   budget. Back up its source and database, and wait for both queues to be idle.
3. Copy the verified file to the existing persistent volume at
   `/workspace/soft-refusal/loras/flux2_4B_ID_01B_DadBod_6.safetensors`.
   Recheck its SHA256 on the server. It belongs on RunPod because RunPod runs
   inference; completed images are copied to the VPS for the gallery.
4. Check `peft==0.18.1` against the installed packages. Install the optional
   requirements in the existing virtual environment without upgrading the
   pinned Torch, Transformers or Diffusers versions; check imports and package
   dependencies afterward. Do not replace the base weights or change precision.
5. Deploy the tested GPU source with admission paused. Keep
   `SOFT_REFUSAL_ENABLE_LORAS=0` until a private test can load the file. Use the
   existing worker's pipeline and a fully clothed adult portrait. Compare the
   same prompt and seed with/without the adapter, then run the original model
   again to verify cleanup. A filename match or successful import is insufficient.
6. Only after verification, enable `SOFT_REFUSAL_ENABLE_LORAS=1` in the API and
   worker environment and resume admission. The VPS refreshes availability from
   authenticated GPU health. Test selection through the website and confirm
   saved name, version, checksum and strength. Keep rollback and the existing
   weekend shutdown timer intact.

`image_adapter` uses safetensors, local files and checksum verification. The
adapter is never permanently fused. Cleanup also runs after load/generation
errors; an infrastructure failure causes the worker to be discarded. Visitors
cannot supply file paths or download URLs. New adapters need a reviewed catalog
entry, an allowed API ID and their own compatibility test.

Synthetic tests verify request handling and cleanup with fixtures. They do not
establish real tensor compatibility, GPU memory use or portrait quality.
