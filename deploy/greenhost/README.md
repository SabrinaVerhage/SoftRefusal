# Soft Refusal on Greenhost

The phone recording room is at https://softrefusal.sabrinaverhage.com/ and the map at https://softrefusal.sabrinaverhage.com/map/. One private Python service owns visitor sessions, serializes GPU requests, saves recordings, and rebuilds the map. Caddy provides HTTPS. The selected language model runs on the separately managed RunPod GPU; pausing that GPU leaves the website and recordings available.

## Deploy an update

On Windows, commit your changes on `main`, then run this from the repository folder
on the computer with the VPS SSH key:

```powershell
.\deploy.ps1 -Push
```

This pushes your committed changes to GitHub, deploys that exact commit, and checks
the live website's version. It refuses an uncommitted working tree or a different
branch. A rejected push stops deployment; pull and resolve any differences first.
If visitors are recording, the server refuses the update and you can retry later.

Use `.\deploy.ps1` if you have already pushed, or `.\deploy.ps1 -Check` to compare
local, GitHub and live versions without changing anything. The script uses Windows
Git and OpenSSH, avoiding unrelated MSYS tools on your PATH. If PowerShell blocks
the script under its default execution policy, run
`powershell -ExecutionPolicy Bypass -File .\deploy.ps1 -Push`; this applies only to
that process and does not change your saved execution policy.

From another computer with the VPS SSH key, the direct command still works:

```powershell
ssh root@213.108.108.24 softrefusal-deploy
```

The server fetches GitHub, prepares a versioned release and dependencies, checks the configuration, and switches the service. It refuses to interrupt active conversations: retry when the queue is idle. A failed startup restores the previous release. The command prints both commit IDs. To roll back, pass the previous commit ID after `softrefusal-deploy`.

This is an explicit deployment command; pushing alone does not update the live website. The public repository needs no GitHub credential on the server.

## Chat style and live viewer — 17 October 2026

`hosted/chat_instruction.txt` is the backend chat-style instruction. It is combined
with the model's native instruction and sent as the system message for every new
turn, including resumed conversations. Earlier checkpoints retain their original
instructions. It requests final answers under 50 words in a natural chat style;
this is a model instruction, not a strict word-count guarantee.

The phone chat is a separate destination with no links to the image room or map.
New messages and streamed replies scroll to the bottom. The public viewer keeps
the pink donut and core visible. If refusal anchors are missing, the viewer labels
the core as illustrative and does not interpret UMAP distance as refusal.

The ten most recently active conversations keep their lines and directional arrows.
Older turns remain as faint, selectable dots. Turns with recorded completion times
have a five-minute halo (still under reduced-motion preferences); legacy timestamps
are not invented. Theme words summarize nearby prompt wording using a small local
vocabulary, not inferred meanings of model activation axes.

UMAP is still refitted after completed turns. The donut reference freezes after
the first eligible labelled export, so unchanged states keep their donut positions.
The browser switches to the measured donut when that reference becomes available.

Hermes automatic labels use the separate `reply-behaviour-v4-final-answer` rubric.
It excludes generated thinking from judgment and was checked against nine neutral
live classification fixtures before enabling it. This is a smoke validation, not
a general accuracy benchmark. Human corrections still win; failed judgments stay
unlabelled. `python -m hosted.label_recordings /var/lib/softrefusal/models/hermes
--backfill` validates before filling missing labels and temporarily holds admission
only when the visitor queue is idle. Its private report is `judge-validation.json`.

## Private server paths

- `/etc/softrefusal/gpu-connection.json`: `api_url` and `access_code_file` pointing to `/etc/softrefusal/gpu-token.txt`. Both are root-owned and readable only by the service group. Update the URL when replacing the GPU; restart `softrefusal` afterward. Never commit the token.
- `/var/lib/softrefusal/recordings`: complete recording checkpoints and automatic behavior tags.
- `/var/lib/softrefusal/visitors.sqlite`: visitor ownership and durable queue.
- `/var/lib/softrefusal/map`: private fitted reference and the public viewer payload.
- `/opt/softrefusal/current`: symlink to the deployed Git commit. Runtime data stays outside releases.

Useful checks: `systemctl status softrefusal caddy`, `journalctl -u softrefusal -n 60`, and the public `/healthz` endpoint, which returns the deployed commit.

## Choosing the model

The service uses `recorder/configs/gemma3-12b-remote.json` unless `SOFT_REFUSAL_MODEL_CONFIG` names another file in that folder. For visible thinking, add `Environment=SOFT_REFUSAL_MODEL_CONFIG=hermes4-14b-remote.json` to the service, enable Hermes on the GPU, and restart. The recording room and map then show the `<think>` text as the AI's thinking. Hermes automatic tagging is enabled after the 17 October validation described above. Gemma and Hermes recordings use different layer sets and cannot share one map. The gateway automatically keeps Hermes data in `models/hermes` beneath the data directory while preserving Gemma data at its original location; the earlier Gemma map remains available at `/map/?model=gemma`.

## Visitor and map behavior

Visitors explicitly agree that completed prompts and replies appear on the public map. An HTTP-only cookie owns their conversation; one visitor cannot send or stop another visitor's turn. GPU credentials and raw activation files are never public. The queue admits at most ten pending turns and one pending turn per visitor. Replies are capped at the smaller of 1024 and the model config's `max_new_tokens` (512 for Gemma, 1024 for Hermes, whose thinking counts toward it) and can be stopped. Automatic labels are model judgments, not ground truth.

The map starts with an empty pink donut until recordings load. Fewer than six distinct turns use a labeled PCA preview; UMAP starts once there is sufficient data. The fixed refusal reference requires suitable answered/refused labels; without those the core remains illustrative. A failed map rebuild retains the previous map and all recordings.

The public map draws only recorded points, with black dotted paths for the ten newest conversations and a pink illustrative donut and core. Backend prompt-theme analysis is reassessed on each export, using conversation context for brief follow-ups. Topic angle is an artistic text-based layout; radius retains the fixed refusal score. Contours show recorded-turn density in that display plane. It orbits automatically unless reduced motion is requested. The five newest conversations play in a continuous feature loop: each recorded turn shows its prompt and AI reply for ten seconds, with generated reasoning available to expand. **Pause features** stops the loop for reading; selecting or moving around the map also pauses it. Live updates preserve the current feature when it remains in the newest five. Add `?tools` to the map URL for the research controls: datasets, layers, filters, search and replay.

Phone chats end after 20 minutes without visitor activity and return to the consent/welcome screen. Typing, touch, keys and scrolling renew the session; background state polling does not. The backend also enforces expiry on resume and submission. Queued/running replies are protected, and completion gives the visitor a fresh idle period. Expiry clears the browser's draft/session selection; saved recordings remain intact. **Start over** and expiry persist across reloads without silently reopening the last chat. Existing visitor databases migrate in place on startup.

The GPU must be online for new replies. Hosting does not start, renew, or stop GPU billing. No periodic recording backup is configured by this deployment.

## Validation

From `recorder`, with hosted dependencies plus `pytest` and `httpx` installed, run:

```sh
PYTHONPATH=..:src python -m pytest ../hosted/tests tests/test_remote.py tests/test_judging.py tests/test_reference.py -q
```

These tests use a local fake model API and do not incur GPU costs.

## Weekend hosted setup — 16 October 2026

The service selects `hermes4-14b-remote.json` with `SOFT_REFUSAL_MODEL_CONFIG`.
Hermes recordings, visitor database and map live under `/var/lib/softrefusal/models/hermes`;
existing Gemma data stays untouched at `/var/lib/softrefusal`. The earlier Gemma map
is accessible at `/map/?model=gemma`; activation spaces and references never mix.
Hermes automatic judgment was initially disabled; the 17 October update enables it. Generated
`<think>` text is displayed separately while the full raw response is recorded.

The separate `/images/` room uses the same durable serial queue as chat. Image jobs
are owned by an HTTP-only visitor cookie and use the application token only on the
server. The server calls `/v1/images/jobs`, `/v1/jobs/{id}`, and the authenticated PNG
result endpoint. Completed PNGs are copied to private data storage and remain
available after GPU shutdown. Technical failures and missing images are never
automatically called refusals. Request size, per-owner concurrency, rate limits,
queue capacity and disk-space checks bound admission. Model loading takes time
when switching between chat and images. Reload recovers the owner's recent jobs;
restart interrupts pending jobs without automatically resubmitting them.

### Independent weekend shutdown

The existing VPS has `softrefusal-weekend-stop.timer` enabled for
2026-10-18 21:59 UTC (23:59 Europe/Amsterdam), with persistent catch-up after a VPS
restart. It calls `/usr/local/sbin/softrefusal-weekend-stop.py`; configuration is in
`/etc/softrefusal/weekend-stop.json` and its existing Pod-scoped key is root-only.
No account-wide key is needed. Check access without stopping compute:

```sh
python3 /usr/local/sbin/softrefusal-weekend-stop.py --check
systemctl list-timers softrefusal-weekend-stop.timer
```

The service refuses to stop before the configured deadline, stops only the specified
Pod, leaves the volume intact and retries temporary failures. After the deadline,
verify `journalctl -u softrefusal-weekend-stop.service` and the provider's stopped
state. A read-only check confirms access and scheduling, not execution of a future stop.
The installed units are saved beside this README for review; do not reuse this fixed
deadline for another paid session without new authorization.
