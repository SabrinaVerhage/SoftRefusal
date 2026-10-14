# Recording with the cloud GPU

The recording room can use Gemma 3 **12B** on the RunPod GPU instead of Gemma 3 4B on this laptop.
The page, your conversations and all recordings stay on this computer. Only each message
(with its conversation history) goes to the GPU. The GPU sends back the reply and the four
snapshots, which are saved here in the usual files.

```
this computer                                RunPod (gpu-api)
recording room ── message + history ──────►  Gemma 3 12B, BF16
     ▲                                       snapshots at layers 12, 24, 31, 41
     └──── reply, live view, snapshots ────  (final prompt token, before the reply)
runs/gpu-chat/…  (steps.jsonl, activations.npz, manifest.json, …)
```

## Open and stop

1. Start the RunPod Pod and the API (see `gpu-api/DEPLOYMENT_STATUS.md`).
   The API must accept requests (`SOFT_REFUSAL_ACCEPT_REQUESTS=1`).
2. Double-click **Open Recording Room (cloud GPU).cmd**. It opens at <http://127.0.0.1:8766>.
   The laptop version keeps working separately at port 8765.
3. Your first message connects. If the GPU has been idle, the first reply waits about a minute
   while Gemma loads there.
4. Double-click **Stop Recording Room (cloud GPU).cmd** when finished. **This does not stop the
   GPU.** Stop the Pod in RunPod to end compute charges.

**Stop reply** also stops the reply on the GPU. Earlier complete turns stay saved.

## Connection settings

`secrets/gpu-connection.json` (not shared, ignored by git):

```json
{
  "api_url": "https://<pod-id>-8000.proxy.runpod.net",
  "access_code_file": "../../gpu-api/secrets/recording-room-token.txt"
}
```

- `api_url` changes if RunPod migrates the Pod to a new ID. Update it here; recordings are unaffected.
- The access code is the recording room's own app code (subject `recording-room`), created with
  `gpu-api/scripts/add_credential.py`. It is not a RunPod or Hugging Face key and is
  never written into recordings.

## What is different from the laptop version

| | Laptop | Cloud GPU |
|---|---|---|
| Model | Gemma 3 4B IT | Gemma 3 12B IT (`96b6f1ec…`) |
| Snapshot layers | 9, 17, 22, 29 of 34 | 12, 24, 31, 41 of 48 (same relative depth) |
| Numbers per snapshot | 2,560 | 3,840 |
| Attention implementation | eager | sdpa |
| Reply cap | up to 2,048 tokens | up to 1,024 tokens |
| Conversation cap | 2,048 input tokens | 8,192 input tokens |
| Recordings folder | `runs/chat/` | `runs/gpu-chat/` |

Keep the two sets apart in analysis: 4B and 12B snapshots have different sizes and cannot share
one UMAP. Each `manifest.json` records `"execution": "remote_gpu"`, the GPU name, the GPU's package
versions and the attention implementation. Each row records its cloud job ID.

The measurement site is unchanged: `resid_post` (block output, before the final RMSNorm) at the
final formatted-prompt token, captured during generation prefill. Replies use greedy decoding.

## When something goes wrong

| Message | Meaning |
|---|---|
| Can't reach the cloud GPU | Pod stopped, API not started, or the address changed. |
| Did not accept the access code | The Pod's `secrets/auth.json` lacks the `recording-room` entry. Copy the updated file and restart the API. |
| Running but paused | Restart the API with `SOFT_REFUSAL_ACCEPT_REQUESTS=1`. |
| Different Gemma checkpoint | The GPU serves another revision than `configs/gemma3-12b-remote.json`. Nothing was saved. |
| Recording length limit | Start a new conversation; nothing is silently removed. |

## Content on RunPod

Whether RunPod's terms allow sexual-content research prompts is still unresolved
(see `gpu-api/README.md`). Until it is, use the cloud room for neutral tests.
