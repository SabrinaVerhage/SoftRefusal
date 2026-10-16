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

## Choosing the model

The service uses `recorder/configs/gemma3-12b-remote.json` unless `SOFT_REFUSAL_MODEL_CONFIG` names another file in that folder. For visible thinking, add `Environment=SOFT_REFUSAL_MODEL_CONFIG=hermes4-14b-remote.json` to the service, enable Hermes on the GPU, and restart. The recording room and map then show the `<think>` text as the AI's thinking. Hermes automatic tagging is disabled in its config until validated, so new turns stay unlabelled and do not move the refusal reference. Gemma and Hermes recordings use different layer sets and cannot share one map, so start a fresh recordings folder when switching.

## Visitor and map behavior

Visitors explicitly agree that completed prompts and replies appear on the public map. An HTTP-only cookie owns their conversation; one visitor cannot send or stop another visitor's turn. GPU credentials and raw activation files are never public. The queue admits at most ten pending turns and one pending turn per visitor. Replies are capped at the model config's `max_new_tokens` (512 for Gemma, 1024 for Hermes, whose thinking counts toward it) and can be stopped. Automatic labels are model judgments, not ground truth.

The map starts with a clearly marked illustrative view. Actual recorded turns replace it automatically. Fewer than six distinct turns use a labeled PCA preview; UMAP starts once there is sufficient data. The fixed refusal-centered view also requires suitable answered/refused labels; it is unavailable until those exist. A failed map rebuild retains the previous map and all recordings.

The public map shows each conversation as a line, a legend and, for a selected line, the AI's thinking at its refusal moment (or its turn closest to the core). It does not show prompts or replies. Add `?tools` to the map URL for the research controls: datasets, layers, filters, search and replay.

The GPU must be online for new replies. Hosting does not start, renew, or stop GPU billing. No periodic recording backup is configured by this deployment.

## Validation

From `recorder`, with hosted dependencies plus `pytest` and `httpx` installed, run:

```sh
PYTHONPATH=..:src python -m pytest ../hosted/tests tests/test_remote.py tests/test_judging.py tests/test_reference.py -q
```

These tests use a local fake model API and do not incur GPU costs.
