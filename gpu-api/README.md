# Soft Refusal: first GPU proof of concept

**Current state, 13 October 2026 08:26 UTC / 10:26 Amsterdam: GPU stopped at the user's request. Both existing Pods show $0.00/hour compute. The 150 GB network volume remains provisioned at $21/month. The page/API are offline until an authorized restart.** Version 0.2 was deployed and tested before stopping; read `DEPLOYMENT_STATUS.md` for the current handoff. Source, models, saved jobs and images are on the network volume, and the application source and controlled test reports also have local copies. The newest handoff notes and two minor display refinements are local-only pending the next authorized sync; this is not an exact mirror of every cloud runtime file.

## 13 October usability update (deployed and tested)

The user requested continuing on the replacement A40 to resolve queue,
latency, live feedback and input-limit issues. The scheduled 09:37 Amsterdam stop
was explicitly cancelled; its automation is paused. Current Pod is
`soft-refusal-poc-migration`, ID `2qla3db8of4pgs`, using the original network volume.
Page address when running: https://2qla3db8of4pgs-8000.proxy.runpod.net/ . Read current status/log
before interpreting older stopped-state sections below.

Version 0.2 allows three active jobs per credential, bounded by ten total jobs and
six submissions/minute. Queued follow-ups use `after_job_id` and are resolved only
after the preceding successful reply, preserving full history and submission
order. Dependencies must belong to the same credential and chat model. Failed
parents produce explicit technical dependency errors. `GET /v1/jobs` returns only
the caller's recent jobs, enabling page recovery after a reload.

Authenticated job status includes private atomic progress snapshots during work:
loading/preparing/generating and generated text, polled by the page every 750 ms.
No thinking transcript is fabricated; emitted reasoning text is part of raw model
output, while numerical activations remain outside this prototype. API and model
logs do not print prompts or partial output. Progress files are on the private
volume, outside public assets, and ownership is checked before reading them.

Chat input cap is raised from the initial conservative 2048 to 8192 formatted
tokens, with 64 messages and 24000 combined characters. This changes request bounds,
not precision or checkpoint weights. A real 8024-token Gemma input completed in
BF16 with a sampled device peak of 28.59 GiB. Length errors now include count/limit and
remain technical errors; no silent truncation or automatic history deletion.
Input/template errors retain a healthy resident worker. Idle unloading moves from
120 to 600 seconds; model switching still unloads before loading another model.
Neither setting changes GPU billing while the Pod is running.

Hermes's pinned generation_config EOS (151643/endoftext) disagrees with its model
config/tokenizer ChatML turn-end EOS (151645/im_end). Version 0.2 explicitly accepts
both native stop markers for Hermes and records original/effective IDs with each
result. Gemma stop settings, native templates, temperature 0, BF16 weights and
safeguards stay as before. Earlier continued/repeated Hermes outputs remain saved
and must not be treated as clean single-turn baselines. No quantization/offload
change has been made. Local and Linux synthetic API suites: 15 passed each.
`runtime/rehearsal-v02-001.json`, backed up locally and retained on the volume,
completed all seven cases (six success and one deliberately oversized input
failure). It verified three same-credential queued jobs, fourth-job rejection,
ordered text/image execution, full parent reply inclusion, private partial output
before completion, native Hermes turn stopping, PNG retrieval, 8024-token BF16
memory fit, counted input rejection and healthy worker retention after that error.

Measured generation times: Hermes greeting 1.26 s (original continued greeting
15.4 s), 256-token Hermes answer 15.35 s, short Hermes follow-up 3.37 s; FLUX512
1.50 s; Gemma 8024-token input with short output 4.69 s; subsequent short warm
Gemma 2.14 s. Loads were 19.44 s for FLUX, 52.21 s for Gemma and 58.75/60.36 s
for Hermes. These are individual samples, not universal speed claims. Longer
answers still generate sequentially, and alternating text/images incurs reloads.
No simultaneous model-residency claim has been made.

Updated page authentication/recovery and enabled queue controls observed in local
Edge. A text job left image and text submission controls enabled; an image joined
the queue and completed. User also made their own requests directly while using
the updated page; these are separate from the controlled neutral test report.
Screenshot `evidence/queue-v02-live.png` shows only queue/status metadata, avoiding
user image/prompt content. Two final display refinements (historical length-error
wording and completed-response cap labels) are prepared locally but were not
uploaded before shutdown. Sync them at the next authorized restart.

## Historical 12 October setup and first rehearsal

Prepared 12 October 2026. **A RunPod A40 Pod and persistent volume have been created;
installation and the neutral live GPU rehearsal have succeeded. Compute is stopped
and verified at $0.00/hour. Browser sign-in/submission validation is pending
dismissal of an Edge extension popup.**
Read `DEPLOYMENT_LOG.md` and `DEPLOYMENT_STATUS.md` before continuing paid work.
The API has passed 12 local tests using synthetic model fixtures. The exact Gemma,
Hermes and FLUX loader classes import successfully in the isolated preparation
environment. Linux/CUDA operation, real outputs, memory fit and latency are
recorded in the live rehearsal below. Public visitor deployment remains outside
the approved neutral rehearsal scope.

The user clarified that the final “AI avoidance space” app is not finished and
requested a simple proof of concept on the GPU first. `public/` is that new test
page. Existing Recording Room, UMAP and spatial prototypes were not modified.

## What is prepared

- One FastAPI service, one durable job store, one bounded FIFO GPU queue.
- Gemma 3 12B IT and optional Hermes 4 14B chat; FLUX.2 klein 4B images.
- BF16 original weights and exact checkpoint revisions. No quantization, model
  alignment changes, safeguard-removal flags or “uncensored” system prompts.
- One resident model process at a time. Switching terminates the old process and
  frees its CUDA context before launching the next. Warm requests reuse it.
- GPU allocator peaks and independently sampled whole-device memory, plus model
  loading time, generation time, queue delay and text time to first generated
  token. Whole-device use can include other processes; it is named accordingly.
- Optional explicit BF16 FLUX model CPU offload. It changes placement and speed,
  not weight precision. Record it as a separate configuration if used.
- Raw model replies, rendered text prompts, token IDs, settings, package versions
  and errors saved separately. No automatic refusal classifier. Blank, filtered,
  failed or capped outputs must not become assumed refusals in a visualization.

## Observed live GPU rehearsal

`runtime/rehearsal-002.json` is a real, completed nine-job run on A40 Pod
`cst7sbn91rdh4d`, using Torch 2.8.0+cu128, Transformers 4.57.6 and Diffusers 0.37.1.
It passed text, 512/1024 PNG generation and retrieval, Gemma → FLUX → Hermes →
Gemma switching, a two-credential serialized burst, foreign-job rejection (404),
repeat active-job rejection (429), and unauthenticated submission rejection (401).
No quantization or CPU offload was used. Only one model was resident at a time.

| Model/test | Largest sampled device use | Inference time | Successful model-load time |
| --- | --- | --- | --- |
| Gemma 3 12B BF16 | 23.65 GiB | 3.41–4.08 s for 55–64 tokens | 22.22 / 46.86 s |
| Hermes 4 14B BF16 | 28.44 GiB | 3.72–4.46 s for 64 tokens | 92.88 s |
| FLUX klein 4B, 512 × 512 | 16.75 GiB | 1.58–1.75 s, four steps | 52.93 / 56.15 s |
| FLUX klein 4B, 1024 × 1024 | 20.14 GiB | 2.87 s, four steps | warm, already loaded |

These are individual neutral samples, not latency distributions or maximum-size
chat benchmarks. Model-load times exclude some worker startup overhead and depend
on filesystem caching; consult worker-roundtrip and queue metrics in the JSON.
The first Gemma load took 65.58 seconds but that request then failed due to an
instrumentation import error; `runtime/rehearsal-001.json` preserves it distinctly.
Whole-device memory was sampled through NVML every 100 ms and can include other
processes; allocator peaks are recorded separately. CUDA reports 44.42 GiB total
usable memory; nvidia-smi reports 46,060 MiB (44.98 GiB). These are different
reported totals, versus the advertised 48 GB.

All models fit individually for these requests. Simultaneous residency was not
tested or assumed. Switching introduces tens of seconds of loading, so the final
visitor flow should show that state. The 64-token test cap truncated some replies;
`finish_reason=length` and original token IDs preserve this without calling it a
refusal. Any Hermes ChatML end-token behavior should be reviewed against the pinned
generation configuration before changing the research baseline.

## Current scope and costs

RunPod account access was verified in the user's **local Microsoft Edge** session.
Hugging Face's Gemma 3 12B page reported that access was granted. Browser access does
not transfer a Hugging Face download credential onto a new Pod.

The full visitor sexual-boundary experiment remains unresolved under
[RunPod's current terms](https://www.runpod.io/legal/terms-of-service), updated
24 March 2026. Section 6 restricts pornography and graphic adult content;
Section 7 contains additional content restrictions. No research exception was
identified there. Lack of runtime filtering and Reddit success reports do not
establish permission. The proposed first paid rehearsal is neutral text and
non-explicit abstract images. The user approved this neutral rehearsal with a $3
initial spend ceiling and at most two GPU hours, followed by stopping compute.

[Gemma's policy](https://ai.google.dev/gemma/prohibited_use_policy) distinguishes
scientific, educational, documentary and artistic contexts from its restriction
on explicit content for pornography/sexual gratification; it separately prohibits
safety-filter circumvention. This is not an exemption from RunPod's rules.
[Hermes](https://huggingface.co/NousResearch/Hermes-4-14B) and
[FLUX klein 4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) show Apache
2.0 licences. The FLUX model card also describes safety training and filters in
its reference repository. The standard Diffusers pipeline is used with its
default components; no filter is removed. If the pipeline provides no filter
signal, results say `not_reported_by_pipeline`, not “safe” or “refused.”
Additional application moderation for a public visitor installation has not
been implemented; this prototype is an operator-controlled technical rehearsal.

Live catalogue prices observed on 12 October 2026 in
[the RunPod deployment console](https://console.runpod.io/deploy):

| GPU | VRAM | GPU price | Availability observed | Proposed use |
| --- | --- | --- | --- | --- |
| A40 | 48 GB | $0.59/hour | Available; 50 GB host RAM / 9 vCPU shown | Cheapest available starting rehearsal |
| RTX A6000 | 48 GB | $0.59/hour | Initially unavailable, then one allocation appeared | Similar memory-budget alternative when available |
| L40 | 48 GB | $0.82/hour | Initially unavailable, then available | Intermediate-priced option; speed unmeasured |
| L40S | 48 GB | $1.09/hour | Available; host resources varied with inventory | Candidate if measured latency justifies the higher cost |
| RTX 6000 Ada | 48 GB | $0.99/hour | Out of capacity | Alternative when available |
| PRO 6000 MIG 48 GB | 48 GB partition | $1.29/hour | Available | Not the cost-first choice; a GPU partition needs separate validation |

Availability and CPU/RAM allocations change. A volume's region must have compatible
GPU stock at launch; this quote is not a reservation or an established speed ranking.
Regional stock changed repeatedly during setup. The Canada volume was subsequently
created and mounted successfully on A40 Pod `cst7sbn91rdh4d` (soft-refusal-poc).
Inspect the existing resources before any restart; capacity is not reserved while stopped.

For the A40, the form showed a **30 GB container disk at $0.004/hour**.
The checkout total rounded to $0.59/hour; use the breakdown when budgeting:
**approximately $0.594/hour for GPU plus container disk**.

The A40 network-volume selector showed CA-MTL-1 (Canada, low GPU stock) and
US-MO-1. Selecting CA-MTL-1 enabled mandatory high-performance storage at
**$0.14/GB/month: 150 GB = $21/month**. With no region selected, the form reverted
to the generic $0.07/GB label and disabled creation. That generic label was not
the actual Canada quote. Standard network storage is advertised at $0.07/GB/month
(150 GB = $10.50/month), subject to compatible stock and a selectable standard tier.
Global-volume beta showed $0.09/GB stored/month **plus IOPS**; it is not a fixed
$13.50 total. Legacy Pod volume disk showed $0.10/GB/month while running and
$0.20/GB/month while stopped, and is erased on Pod termination.

Budget examples for the observed A40 + 30 GB root + 150 GB Canada network volume:

| Period | Compute + container | Persistent storage | Total before tax |
| --- | --- | --- | --- |
| 2 active hours, volume retained about 2 hours | $1.188 | about $0.058, prorated | about $1.25 |
| 2 active hours, volume retained a whole month | $1.188 | $21 | about $22.19 |
| 40 active hours, volume retained a whole month | $23.76 | $21 | about $44.76 |
| 730 active hours, volume retained a whole month | $433.62 | $21 | about $454.62 |

Storage proration above uses 730 hours/month as an estimate;
[RunPod pricing documentation](https://docs.runpod.io/pods/pricing) says network
volumes are billed hourly, while container/volume disks are billed per second.
Verify actual billing in the console. Taxes, exchange rates and any optional charges are additional.
Current balance was $19.49; it is not a spending limit or authorization.

150 GB is a starting disk budget, **not a VRAM requirement**. The model repositories
showed 24.4 GB for Gemma, 29.6 GB for Hermes and 23.7 GB for FLUX including duplicate
single-file/diffusers formats. The download script selects only the Diffusers
components for FLUX and skips its extra single-file transformer. Keep space for
the environment, cache overhead, image files and reports.

The original provider comparison was read, not changed:
[comparison sheet](https://docs.google.com/spreadsheets/d/1prAb6AFfu_HKwLaj8YbJt8G3ICCov8JGHlfQqhqTObk/edit).
Its TensorDock shortlist remains historical context; no provider search was restarted.

## Deploy after approval

1. Recheck the exact GPU and network-volume region quote in Edge. Create the
   approved 150 GB network volume and attach it during Pod deployment at
   `/workspace`. Use one A40 and the official template
   `runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404`. Do not deploy without the
   persistent mount. Choose one GPU, on-demand, not reserved or spot.
2. Upload the prepared source package into `/workspace/soft-refusal`. No local
   `.venv`, recordings, model caches or credential files belong in that package.
   Preserve the default container command until the application is installed.
3. Use the Pod's private terminal/SSH session to run `bash scripts/bootstrap.sh`.
   This installs the small application packages and verifies one BF16-capable GPU.
   It creates app-only access codes in private files, without printing them.
4. The pinned files are already cached; serving needs no Hugging Face token.
   If a future download is necessary, use an authorized, appropriately scoped
   read credential temporarily. `--temporary-token-file` reads it into memory and
   removes the file immediately; the official client does not save a login.
   Never paste a credential into chat, the public prototype, a URL, a command
   argument, or tracked source. The initial approval was temporary download access
   to this private Pod, not a persistent login or broader credential use.
5. With `HF_HOME=/workspace/huggingface`, run:

   ```bash
   .venv/bin/python scripts/download_models.py
   # Optional comparison, after deciding to test Hermes:
   .venv/bin/python scripts/download_models.py --include-hermes
   # For an explicitly authorized temporary credential file:
   # add --temporary-token-file /workspace/.hf-download-token
   ```

   GPU inference uses `local_files_only=True`, so it will not silently download
   missing checkpoints during a visitor job. Failed access/downloads must be
   resolved before paying for a long inference rehearsal.
6. Start locally first:

   ```bash
   SOFT_REFUSAL_ACCEPT_REQUESTS=1 bash scripts/start.sh
   # Add SOFT_REFUSAL_ENABLE_HERMES=1 only after downloading the comparison.
   ```

7. Run the live script from a second private session:

   ```bash
   .venv/bin/python scripts/rehearsal.py --out runtime/rehearsal-001.json
   # For Gemma → FLUX → Hermes → Gemma, include --include-hermes.
   ```

   Inspect the two generated PNGs visually. Keep the report and environment freeze.
   Record neutral refusal behaviour only if actually observed; these neutral
   prompts are a transport/performance rehearsal, not a boundary benchmark.
8. After authentication and local model tests pass, configure the Pod to expose
   **only port 8000 via RunPod HTTPS** for the proof of concept, and start with
   `SOFT_REFUSAL_BIND=0.0.0.0`. Bind private model workers to no network port. Do
   not publish unprotected Jupyter/admin/model routes. Verify the externally
   visible app, authenticated API, PNG results, unauthenticated rejection and
   two-client contention on the HTTPS address. Use individual rehearsal access
   codes to exercise the queue; sharing one code shares its one-active-job limit.

**Stop on OOM or poor latency; keep the failed result.** First consider sequential
loading (already the default), then explicitly measured BF16 FLUX CPU offload.
Quantization would be a separate research configuration, needs discussion, and is
not an automatic fallback. A 48 GB fit for all models together has not been claimed.

## API contract

Authenticate with `Authorization: Bearer <application credential>`. These credentials
are separate from RunPod/Hugging Face infrastructure keys. The proof-of-concept
page accepts an app access code interactively and keeps it only in page memory;
no key is shipped in its JavaScript. The final visitor app should normally use
its own session authentication and a server-side gateway to this API.

| Endpoint | Behaviour |
| --- | --- |
| `GET /healthz` | Public minimal process liveness; does not claim models are ready |
| `GET /v1/health` | Authenticated readiness, models, queue state and latest measured metrics |
| `POST /v1/chat/jobs` | Recommended async chat; model `gemma` or `hermes`; returns 202 and job ID |
| `POST /v1/chat/completions` | Non-streaming OpenAI-style chat reply on the same GPU queue; cold loads may exceed proxy timeouts |
| `POST /v1/images/jobs` | One queued image; returns 202 and job ID |
| `GET /v1/jobs/{id}` | Owned job status, queue position, raw result and metrics |
| `GET /v1/jobs/{id}/result` | 202 while pending, authenticated PNG for an image, otherwise result/error JSON |
| `GET /v1/jobs/{id}/activations` | v0.3: owned `.npz` of recorded `layer_N` float32 vectors; 404 when nothing was recorded |
| `POST /v1/jobs/{id}/cancel` | v0.3: removes a queued job, or stops a running one at its next token (status `cancelled`) |
| `GET /v1/openapi.json` | Authenticated schema |
| `POST /v1/admin/pause` | Admin-only admission pause; existing jobs drain; does not stop billing |

Chat example:

```json
{"model":"gemma","messages":[{"role":"user","content":"Explain rainbows in two sentences."}],"max_tokens":64,"temperature":0,"seed":0}
```

Image example:

```json
{"model":"flux","prompt":"An abstract sculpture made of paper.","width":512,"height":512,"steps":4,"seed":0}
```

Use `Idempotency-Key` on submissions; reusing a key and body returns the same job.
A different body with the same key returns 409. Model roles use their original
chat templates; requests are not rewritten or silently truncated.

Limits: one active job per app credential; at most 10 total queued/running jobs;
six new jobs/minute per credential; 64 KiB request body; 32 chat messages; 24,000
combined text characters; 2,048 formatted input tokens; 1,024 output tokens;
one 256–1024 pixel image per job, dimensions divisible by 64, four distilled steps;
900-second execution deadline; 1,800-second queue expiry; reject submissions at
10,000 retained jobs or under 2 GiB free disk. Completed jobs are retained until
explicitly exported/deleted; there is no silent research-data cleanup.

Jobs distinguish `queued`, `running`, `succeeded`, `failed` and `interrupted`.
Generated text/images start `outcome=unlabeled`, with no label source. Pipeline
filter signals use `pipeline_blocked`; model/transport failures use
`technical_error`. Response caps are `finish_reason=length`, not refusal.
Refusal/redirection/answer labels belong in independent annotations after review.
Version 0.3 (prepared locally, not yet deployed) can record residual activations:
chat jobs accept `record_layers` (unique zero-based decoder blocks, at most eight) and
`trace`. The worker saves `resid_post` (block output before the final RMSNorm) at the
final formatted-prompt token during generation prefill, the same site as the
recording room, and returns its checksum, token, module path, chat template and model
configuration. `trace` adds live token pieces, embedding/layer summaries and top-five
next-token candidates to progress and the final result. The recording room's cloud
mode uses this (`recorder/docs/cloud-gpu.md`). This backend does not calculate
UMAP coordinates.

## End a session and avoid compute charges

1. Stop submissions using the admin pause endpoint. Wait until `/v1/health`
   reports `running=false` and `queue_size=0`; keep the completed results.
2. Stop the application gracefully with Ctrl+C in its private session. This
   terminates the resident model process. A 120-second idle unload also releases
   GPU memory, **but neither action stops RunPod compute billing**.
3. Back up `runtime/`, the source and environment report to your computer or
   chosen private storage. SQLite uses a single writer and DELETE journaling.
   Copy the database only after stopping writes, or use SQLite's backup API.
   Keep access-code files separately and private. Verify the backup opens.
4. In Edge, open RunPod **Pods**, choose **Stop**, and verify the status and billed
   resource state. Stopping releases the GPU, erases the container root disk and
   retains `/workspace`. A network volume survives Pod termination as well.
5. To restart, select **Start**, recheck that one GPU was actually allocated, then
   run `scripts/start.sh` from the preserved volume. If host capacity is unavailable,
   deploy a replacement Pod attached to the same regional volume. Reinstall any
   base container dependency lost from root disk; retain the persistent environment.
   After an approved new session budget, the tested public rehearsal command is:

   ```bash
   cd /workspace/soft-refusal
   SOFT_REFUSAL_ACCEPT_REQUESTS=1 SOFT_REFUSAL_ENABLE_HERMES=1 SOFT_REFUSAL_BIND=0.0.0.0 bash scripts/start.sh
   ```

   The Pod now exposes HTTP 8888 (authenticated Jupyter) and 8000 (the prototype
   and authenticated API). Its prototype address is
   https://cst7sbn91rdh4d-8000.proxy.runpod.net/ and works only while compute and
   the API process are running. Sign in using an app code from the private
   `secrets/rehearsal-1-token.txt` file; never use an infrastructure credential.
6. To remove the Pod entirely, export first, verify `/workspace` really is the
   separate network volume, then terminate the Pod. Termination permanently erases
   non-network storage; it requires a separate destructive-action confirmation.
7. **Storage still bills while compute is off.** Keeping the quoted 150 GB Canada
   volume costs $21/month. Deleting it ends that storage charge and permanently
   deletes its data, so only do it after a verified backup and explicit confirmation.
   Do not rely on a depleted balance to preserve data indefinitely.

Sources: [Pod lifecycle](https://docs.runpod.io/pods/manage-pods),
[network-volume persistence/pricing](https://docs.runpod.io/storage/network-volumes).

## Where the website should live later

The first page and API can share the Pod: CPU, RAM and networking are included
alongside the GPU. Stopping the Pod makes both unavailable.

For the final installation, keep the lightweight website, visitor sessions,
database and server-side gateway on separate always-on hosting if the UI/history
must remain available while the GPU is stopped. The UI can say “models offline”
and reject/hold requests explicitly. The gateway retains the GPU application's
service credential; visitors receive only their normal website sessions. No
extra provider has been booked or Site published during this preparation.

## Local preparation checks

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
.\.venv\Scripts\python.exe -m pytest -q
```

The Windows sandbox blocks Python's internal loopback socket pair, so those tests
were run with the approved local-network permission. They use synthetic transport
fixtures, not learned-model inference. Do not mistake passing them for a live GPU
rehearsal. All new files are in this backend directory.
