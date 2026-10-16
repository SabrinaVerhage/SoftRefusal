# Soft Refusal on Greenhost

The phone recording room is at https://softrefusal.sabrinaverhage.com/ and the map at https://softrefusal.sabrinaverhage.com/map/. One private Python service owns visitor sessions, serializes GPU requests, saves recordings, and rebuilds the map. Caddy provides HTTPS. Gemma runs on the separately managed RunPod GPU; pausing that GPU leaves the website and recordings available.

## Deploy an update

Commit and push the changes to GitHub's `main` branch. Then, from a computer with the VPS SSH key:

```powershell
ssh root@213.108.108.24 softrefusal-deploy
```

The server fetches GitHub, prepares a versioned release and dependencies, checks the configuration, and switches the service. It refuses to interrupt active conversations: retry when the queue is idle. A failed startup restores the previous release. The command prints both commit IDs. To roll back, pass the previous commit ID after `softrefusal-deploy`.

This is an explicit deployment command; pushing alone does not update the live website. The public repository needs no GitHub credential on the server.

## Private server configuration

- `/etc/softrefusal/gpu-connection.json`: `api_url` and `access_code_file` pointing to `/etc/softrefusal/gpu-token.txt`. Both are root-owned and readable only by the service group. Update the URL when replacing the GPU; restart `softrefusal` afterward. Never commit the token.
- `/var/lib/softrefusal/recordings`: complete recording checkpoints and automatic behavior tags.
- `/var/lib/softrefusal/visitors.sqlite`: visitor ownership and durable queue.
- `/var/lib/softrefusal/map`: private fitted reference and the public viewer payload.
- `/opt/softrefusal/current`: symlink to the deployed Git commit. Runtime data stays outside releases.

Useful checks: `systemctl status softrefusal caddy`, `journalctl -u softrefusal -n 60`, and the public `/healthz` endpoint, which returns the deployed commit.

## Visitor and map behavior

Visitors explicitly agree that completed prompts and replies appear on the public map. An HTTP-only cookie owns their conversation; one visitor cannot send or stop another visitor's turn. GPU credentials and raw activation files are never public. The queue admits at most ten pending turns and one pending turn per visitor. Replies are capped at 512 tokens and can be stopped. Automatic labels are model judgments, not ground truth.

The map starts with a clearly marked illustrative view. Actual recorded turns replace it automatically. Fewer than six distinct turns use a labeled PCA preview; UMAP starts once there is sufficient data. The fixed refusal-centered view also requires suitable answered/refused labels; it is unavailable until those exist. A failed map rebuild retains the previous map and all recordings.

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
Hermes automatic judgment is disabled pending model-specific validation. Generated
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

### Image gallery

`/gallery/` is the public image viewer, linked from the recording room and image
room. New submissions in the image room explicitly agree to publish the image and
prompt. Only completed jobs with that per-image consent appear in `/api/gallery`;
the migration defaults all existing jobs to private. Old clients can still submit
private jobs. Public image URLs serve the saved VPS PNG and never contact RunPod.
The gallery loads 24 images at a time, with full-image viewing and PNG downloads.
In the current Hermes setup, files live in
`/var/lib/softrefusal/models/hermes/images`, alongside their database metadata.

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
