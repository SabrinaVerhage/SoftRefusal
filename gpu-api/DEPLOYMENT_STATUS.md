# Current RunPod deployment status

## Active session — 15 October 2026

**RUNNING: soft-refusal-h100-session, Pod `00yqqxrkv65o6w`, H100 SXM 80 GB, $3.99/hour compute plus container storage.** User approved one hour, approximately $4 compute. Started around 13:00 UTC; stop by **14:00 UTC / 16:00 Europe/Berlin**. A one-shot Codex thread shutdown follow-up is active; it is not a provider-enforced timer. Verify actual shutdown and update this section.

Existing `soft-refusal-models` 150 GB volume is mounted at `/workspace`; the three older A40 Pods remain stopped at $0.00/hour. No old Pod or volume was deleted. The preserved backend was started using `scripts/start.sh`, admission enabled, bind `0.0.0.0`, Hermes not enabled, with log `runtime/api-h100-20261015.log`. API URL: https://00yqqxrkv65o6w-8000.proxy.runpod.net/ . Local Recording Room is on port 8766; viewer is on port 8777.

Local private connection settings use this new API URL and the existing non-admin `rehearsal-2` application credential recovered from the Pod. No credentials were changed or printed. Authenticated Gemma health and one live neutral end-to-end recording passed: conversation `f0585c1a7ec34d39b26dab2f25b9fe01`, four finite float32 snapshots of 3,840 values, six file hashes verified, pinned Gemma 12B BF16 on H100. API PID 1747 is current for this session only. The `recording-room` token itself was absent from this checkout and the Pod's token files. New recordings use the existing rehearsal-2 owner. Earlier status sections below are historical and do not describe the active H100 session.

## Historical handoff

Updated 2026-10-13 evening: Pod STOPPED by the user and verified at $0.00/hr. Current Pod: mxntb0l9fom8sw (page/API https://mxntb0l9fom8sw-8000.proxy.runpod.net/). The Pod's auth.json differs from the local secrets copy; local rehearsal codes are not the Pod's. Earlier text below describes the 08:26 UTC state. Read AGENTS.md, DEPLOYMENT_LOG.md and README.md before continuing.

**STOPPED AT THE USER'S REQUEST.** Version 0.2 fixes were deployed and tested. Stop Pod was confirmed for soft-refusal-poc-migration, ID 2qla3db8of4pgs. RunPod now shows Start for $0.59/hr and both Pod rows at $0.00/hr. The persistent network volume remains provisioned. Evidence: evidence/runpod-v02-stopped.png. Do not restart paid compute without an agreed new session budget.

## Resources and charges

- Current reusable Pod: soft-refusal-poc-migration, ID 2qla3db8of4pgs, stopped. Original soft-refusal-poc, ID cst7sbn91rdh4d, also stopped. Do not create duplicates or terminate these resources as part of routine stop/start.
- One A40; active quote $0.59/hour plus approximately $0.004/hour for 30 GB temporary container disk. Stop confirmation quoted idle disk cost $0.000/hour. These are observed rates, not an itemized invoice.
- Existing network volume soft-refusal-models: 150 GB, CA-MTL-1, mounted at /workspace. Verified provisioned after shutdown. Last active usage 47% (about 70 GB); usage is unavailable while stopped.
- Retained storage remains billable at $0.14/GB/month = $21/month, billed hourly. Do not delete it without separate authorization and a verified backup.
- Official image: runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404. Last API supervisor was PID 3972 in the stopped container, log runtime/api-006.log. These PIDs are historical; never reuse them to kill processes after a restart.
- Configured HTTP ports: 8888 for private authenticated Jupyter, 8000 for prototype/API. Neither service runs while the Pod is stopped.
- Page when running: https://2qla3db8of4pgs-8000.proxy.runpod.net/ . Its address may change after a future migration. Edge on this PC is the client; page, API and models run on RunPod. No local website server is running.
- Automation stop-soft-refusal-gpu-after-testing remains PAUSED, verified from its local configuration. Earlier stop deadlines were superseded by explicit user steering; the latest instruction was to stop now, and it was completed.

## Deployed v0.2 behavior

- Three active jobs per application credential, ten global jobs, six submissions per minute; one shared serial GPU queue. Chat and image buttons stay usable while another job runs, until the caller's limit is reached.
- Queued follow-ups use after_job_id and resolve the complete parent input/reply after it succeeds. Same owner/model required; failed parents produce technical dependency errors. No silent truncation or deletion of conversation history.
- Private atomic progress snapshots expose loading/preparing/generating, generated text and token count through owned authenticated job status. Page polls every 750 ms. No invented thinking transcript or numerical-activation viewer.
- Authenticated GET /v1/jobs recovers only the caller's recent jobs after reload.
- Chat cap: 8192 formatted input tokens, 64 messages, 24000 combined characters, 12000 per message; bounded request size. New length errors include counts/limits and remain technical errors distinct from refusals. The earlier actual error was 2239 tokens exceeding the old 2048 cap, not an explicit-word check.
- Input/template errors retain healthy workers; default idle unloading is 600 seconds. Switching unloads the previous model first. Idle unloading never stops RunPod billing.
- Hermes accepts both native EOS IDs: original generation configuration 151643/endoftext plus model/tokenizer 151645/im_end. Results record original/effective IDs. Gemma EOS settings unchanged. Original BF16 checkpoints, native templates, deterministic test settings and safeguards remain; no quantization or CPU offload used.
- Raw model output, failures, settings and token IDs remain separate. No automatic refusal classifier. Earlier continued Hermes responses and capped responses remain historical observations, not silently corrected results.

## Completed validation and evidence

- JavaScript syntax and Python compilation passed. Fifteen synthetic API tests passed locally (2.58 s) and in staged Linux deployment (3.30 s). These are application tests, not model performance tests.
- Neutral live v0.2 rehearsal completed seven cases: six successful jobs and one deliberate counted input rejection. Verified three same-credential accepted jobs, fourth-job 429, foreign progress/job 404, private partial text before completion, FIFO text/image/follow-up execution, full parent reply inclusion, authenticated PNG retrieval, Hermes native stopping, near-8k Gemma input and healthy worker retention after an input error.
- Reproducible script: scripts/rehearsal_v02.py. Complete raw report: runtime/rehearsal-v02-001.json, retained on the volume and copied locally. Previous failed/successful rehearsals are preserved separately.
- Measured generation samples: Hermes greeting 1.26 s, warm 256-token answer 15.35 s, short linked follow-up 3.37 s; FLUX512 1.50 s; Gemma with 8024 input tokens and a short answer 4.69 s; subsequent warm short Gemma 2.14 s. Switching/loading samples: FLUX 19.44 s, Gemma 52.21 s, Hermes 58.75/60.36 s. Long replies and model switching still take time; these are samples, not broad speed guarantees.
- Near-8k Gemma sampled whole-device peak 28.59 GiB; Hermes up to about 28.56 GiB; FLUX512 16.75 GiB in this run. Earlier FLUX1024 peak 20.14 GiB. CUDA total 44.42 GiB vs marketed 48 GB. Models fit individually for tested requests; simultaneous residency was not tested or assumed.
- Updated browser authentication/recovery and enabled queue controls were observed in the user's local Edge. Text and image jobs completed. User also submitted their own prompts in the same test tab; those are separate from the controlled neutral report. evidence/queue-v02-live.png shows only queue/status metadata.
- Latest stop verification: both Pod rows $0.00/hr, current inspector Start for $0.59/hr, 150 GB volume provisioned at /workspace. evidence/runpod-v02-stopped.png.

## Persistence, local copies and remaining work

- Cloud source/environment: /workspace/soft-refusal. Pinned model caches: /workspace/huggingface. Private jobs, images, progress and reports: /workspace/soft-refusal/runtime. Private auth files: /workspace/soft-refusal/secrets. All are on the retained network volume.
- Local application source, documentation, evidence and selected controlled reports: this gpu-api directory (formerly soft-refusal-backend; moved to the SoftRefusal repository on 14 Oct 2026). Model weights, complete visitor runtime data and cloud application credentials are not a full local mirror.
- Integrity-checked pre-v0.2 backup: runtime/jobs-backup-before-v02-20261013.sqlite; previous backups also retained. Source backup: runtime/source-before-v02-20261013.zip. The live job database commits each change durably. This final stop was through RunPod; no additional final database backup/integrity check or final admission-drain was performed. Any pending work on restart is marked interrupted/service_restarted, distinctly from a refusal.
- Deployed source archive /workspace/soft-refusal-v02.zip contained 27 allowlisted source entries, SHA256 e08f109f68deae122a4849bac40887957af67416c8cfe4784276545ab3134434. The live rehearsal script was uploaded separately.
- Two final local-only public/app.js refinements remain to sync: historical input errors should refer to their original cap, and completed responses capped by length should show that label. They passed JavaScript syntax checking; no backend change is pending. The newest local handoff documents also postdate the cloud document checkpoint. Do not claim an exact source/document mirror until synced and verified at a future authorized restart.
- The existing Recording Room/spatial/AI avoidance apps were not overwritten. The final app integration is not complete; this is the requested separate GPU proof of concept.
- Temporary approved Hugging Face download-token files were removed and no Hub login was retained. Cached serving needs no download token. Never ask for passwords or keys in chat or ship infrastructure credentials in frontend code.
- An earlier restored credential editor exposed the old rehearsal-1 application code in automatic accessibility output; replace it before public sharing while preserving owner/history. The current newer browser tab used rehearsal-2. A full RunPod Connect accessibility snapshot also included a tokenized Jupyter URL during stop verification; do not reproduce its value. Prefer narrow DOM reads and a non-Connect tab for evidence; review private Jupyter access before broader sharing. No RunPod API key or Hugging Face read token was printed in this shutdown.
- Full visitor sexual-content boundary exploration remains unresolved under RunPod's applicable-use restrictions. Licences, model behavior and host permission are separate. Completed controlled agent tests were neutral technical checks; no safeguard or alignment removal was performed.

## Version 0.3: recording-room connection (DEPLOYED 13 Oct ~18:25 UTC on Pod mxntb0l9fom8sw; see log)

Prepared 2026-10-13 without paid compute. The Pod still runs v0.2.

- Chat jobs accept `record_layers` and `trace`; the worker records resid_post at the final prompt token during prefill into runtime/activations/{job}.npz. New owned endpoints GET /v1/jobs/{id}/activations and POST /v1/jobs/{id}/cancel (status `cancelled`, model stays loaded). Earlier assistant replies may now be empty strings; user/system text is still required. Worker stdout line limit raised to 16 MiB and progress snapshot cap to 512 KiB for the trace.
- New subject `recording-room` (non-admin) added to local secrets/auth.json with scripts/add_credential.py; its code is in secrets/recording-room-token.txt. The Pod's auth.json does not have it yet.
- Tests: 22 synthetic API/worker tests pass locally, including a tiny random Gemma 3 run through the real worker code checking the saved vectors against the model's own hidden states. A local end-to-end run (real API + real worker code + tiny model + recording room over HTTP) recorded two turns whose local vectors matched the server's byte for byte; Stop cancelled on the server. No live GPU/Gemma 12B test yet.
- To deploy at the next authorized session: build with scripts/package.py, stage and run the Linux tests, pause/back up as before, install the source, copy the updated local secrets/auth.json (hashes only) to /workspace/soft-refusal/secrets/, restart the API, then send one neutral recording-room turn and check its manifest. The two pending v0.2 public/app.js display refinements are included in the same package.

## Next authorized session

1. Agree the paid session budget, inspect actual RunPod state in existing local Edge, and reuse 2qla3db8of4pgs with soft-refusal-models. Capacity is not reserved while stopped; inspect a concrete quote before any replacement/migration. Never blindly create a duplicate.
2. Verify /workspace, pinned cache manifest, private auth and existing job store. Sync the prepared local display/document refinements without including secrets or runtime/model files. The container's root disk was temporary; persistent environment/source live on /workspace.
3. In the Pod's private Linux terminal, start from /workspace/soft-refusal using the preserved environment. For an agreed neutral rehearsal: SOFT_REFUSAL_ACCEPT_REQUESTS=1 SOFT_REFUSAL_ENABLE_HERMES=1 SOFT_REFUSAL_BIND=0.0.0.0 bash scripts/start.sh . Record the new PID/log. Do not use historical process IDs.
4. Verify authenticated health, job recovery, text/images and switching as needed, with separate raw error/refusal representation. Keep one model resident, BF16, and discuss quantization before changing the research baseline.
5. At session end, pause admission and drain if practical, back up SQLite, stop RunPod and verify $0.00/hr. Retain the separately billed volume unless deletion is expressly requested.
