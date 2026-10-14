# Mapping the Sexual Avoidance of Artificial Intelligence

Sabrina's first research pipeline: **Gemma 3 4B IT residual activations → 3D UMAP trajectories**, with independently annotated outcomes, a held-out refusal probe, and optional Gemma Scope 2 sparse features. No semantic embedding service is used.

**Start with [START_HERE.md](START_HERE.md)** for everyday use, start/stop instructions, storage and reply limits. The active project is now `recorder/` in the SoftRefusal repository (`C:\Users\sabri\Documents\_GITHUB\SoftRefusal`). See [model choice and policy](docs/model-choice-and-policy.md) and [the live-view explanation](docs/how-the-live-view-works.md).

## What works now

- Capture selected decoder block outputs before final normalization, one vector per turn per layer.
- Generate replies and feed them into subsequent turns; retain exact rendered prompts, token IDs, extraction positions and model settings.
- Save checksummed JSONL + float32 NumPy arrays, human annotation templates, UMAP coordinates and trajectory connections.
- Encode matching Gemma Scope 2 residual SAEs; retain all nonzero features and reconstruction diagnostics.
- Train a standardized linear refusal probe, calibrate on separate groups, evaluate on held-out groups, and score compatible future runs.

**Validation status:** access is now confirmed and the trained Gemma 3 4B IT model has completed a real two-turn conversation on this laptop. Responses and finite residual vectors were saved at all four selected layers; the layer-17 Scope SAE also ran successfully. Eighteen automated tests separately validate the mechanics and the added live observations. The full neutral fixture and a refusal study remain to be run on the trained model. See `VALIDATION.md`. No sexual-boundary dataset has been generated or run.

## Run on this computer

### Chat and record

Double-click **Open Gemma Recording Room.cmd** in this folder. It opens a local page at **http://127.0.0.1:8765**. If the server is already running, the launcher reuses it. Gemma loads on the first message and stays loaded for subsequent replies.

Type a message and press Enter. The right panel shows real preparation, snapshot, reply-generation and save events. Each of four strips displays 32 group averages from a full 2,560-value vector, with an independent scale. It is not a semantic or refusal visualization. The complete vectors, exact prompts, generated tokens and settings are saved; the strips are only previews.

Use **New conversation** to start independently, click a previous conversation to resume, or choose **Download recording** for a ZIP compatible with the analysis commands. **Stop reply** requests a stop at the next safe point; incomplete attempts stay separate from completed research data. Reloading/closing the page does not interrupt the server's active recording. Closing the browser tab does not unload the model; the local server remains running.

The chat preset allows 2,048 input tokens. The UI defaults to a 512-token reply ceiling, with choices of 256, 512, 1,024 or 2,048 recorded per turn; the config/API fallback remains 256 for older clients. It rejects overlong context without silently deleting messages and marks capped replies. A long reply can exceed the next turn's incoming-context budget. The UI does not classify refusal or automatically train a probe. Conversations are marked exploratory; related conversations must be assigned common prompt-family groups before a rigorous split-based study.

**Follow the computation** opens measured token pieces, compressed embedding/layer previews, and top-five next-token probabilities. The all-layer view is measured once during generation’s initial prompt pass, not during every generated token. Completed turns retain these summaries as `education_trace`; older recordings lack them. Use **Stop app & release memory** or **Stop Recording Room.cmd** to cancel active work safely and shut down the server. The launcher starts it again when needed; it is not registered for Windows startup.

Downloaded model/SAE files are in `models/huggingface/` (ignored by Git). The package uses that cache automatically when it exists, unless `HF_HUB_CACHE` is explicitly set. Authentication stays in the user's Hugging Face profile. The old Codex project and global model caches have been retained as backups; new writes occur only here.

Recordings live in `runs/chat/<conversation-id>/`. Each completed turn creates a new cumulative dataset in `checkpoints/`; `session.json` points to the last verified checkpoint. This preserves earlier complete recordings during a failed save. `attempts/` contains the event stream, including partial/failed attempts. Use the latest checkpoint or downloaded ZIP for analysis, not the conversation's parent directory. Data and browser assets stay local; the server binds only to loopback and uses no external chat API or tracking service.

To start from a terminal: `avoidance chat`. The launcher requires the `.venv` environment described below.

The setup environment already exists at `.venv`. In a PowerShell terminal in this directory:

```powershell
.\.venv\Scripts\Activate.ps1
avoidance doctor --check-access
python -m pytest -q
```

If activation is disabled by your shell, use `.\.venv\Scripts\python.exe -m avoidance.cli` instead of `avoidance`.

First accept the terms on [Google's Gemma 3 4B IT page](https://huggingface.co/google/gemma-3-4b-it), then sign in locally:

```powershell
hf auth login
avoidance doctor --check-access
avoidance collect --config configs/gemma3-4b-local.json --input data/neutral.jsonl --out runs/gemma-neutral
avoidance reduce --run runs/gemma-neutral --layer 17 --out runs/gemma-neutral-3d
avoidance features --run runs/gemma-neutral --layer 17 --out runs/gemma-neutral-sae
```

The local preset uses CPU bfloat16, batch size one, 512 input tokens and at most 32 generated tokens per turn. The laptop has 32 GB RAM and a 6 GB RTX 3060; only about 13 GB RAM was available when inspected. Close memory-heavy applications before the full run. CPU bfloat16 may be very slow on this processor. If necessary use float32 with more free RAM, recording it as a separate run. The GPU preset targets a GPU with enough free memory for unquantized weights and activations (plan for roughly 12–16 GB or more; sequence length matters). It is not intended for this 6 GB GPU. No quantization or CPU/GPU offloading is silently enabled.

Outputs must use **new directories**. Failed collection leaves an `incomplete` manifest; analysis refuses it. Restart into a new directory. Input sequences over the configured limit fail instead of being silently truncated.

## Test without model access

```powershell
avoidance collect --synthetic --out runs/synthetic-smoke
avoidance reduce --run runs/synthetic-smoke --layer 17 --out runs/synthetic-smoke-3d
```

This produces **random vectors and placeholder responses**, visibly marked `synthetic: true`. It cannot be used for SAE analysis or a research refusal probe. The test suite separately executes forward passes and generation through a tiny random Gemma 3 model, with no downloads.

## Fresh environment

Use Python 3.10–3.13. Create a virtual environment, upgrade pip, install the appropriate PyTorch build, then this package:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e ".[test]"
```

The existing setup reused this computer's PyTorch 2.6.0+cu124 and scientific libraries through a virtual environment with system packages enabled; it did not upgrade those global packages. Direct application dependencies are pinned in `pyproject.toml`; `requirements-tested.txt` records the tested dependency closure. For CUDA choose the matching official PyTorch wheel separately. Every run saves package versions. Model and SAE revisions are pinned to commit hashes, not moving branch names.

## Refusal measurement, once labeled data exists

Copy the run's `annotations.csv` and fill `outcome`, `annotator`, `split`, and `notes`. Read the rubric in [METHODOLOGY.md](METHODOLOGY.md) first. Related prompts and every turn of a conversation must stay in one split. Assign families before inspecting activations or UMAP. Use `train`, `calibration`, or `test`.

```powershell
avoidance probe --run runs/labeled-experiment --annotations labels.csv --layer 17 --out runs/probe-17
avoidance score --run runs/new-experiment --probe runs/probe-17/probe.joblib --layer 17 --out runs/new-scores
```

The neutral set is intentionally insufficient for a refusal probe. The command requires both classes in at least five independent training groups and three calibration/test groups each. These are software floors, not a scientifically adequate sample-size claim. Mixed/unclear responses are excluded from the binary model and remain available for analysis. Load `.joblib` files only from trusted sources.

## Files and extensions

| Artifact | Meaning |
|---|---|
| `steps.jsonl` | Prompt/history/response and point IDs; outcomes initially unlabeled |
| `activations.npz` | `layer_17`, etc.: `[point, residual dimension]`, indexed by `activation_row` |
| `manifest.json` | Model commit, dtype, settings, template hash, environment and data hashes |
| `annotations.csv` | Human outcomes and split assignment, joined by `point_id` |
| `points.jsonl` | 3D coordinates and `previous_point_id` for trajectories |
| `features.npz` + `features.jsonl` | Sparse SAE matrix and feature-row → point-ID mapping |
| `heldout_scores.jsonl` / `scores.jsonl` | Probe estimates joined by point ID; never UMAP-derived |

Point IDs are stable within matching conversation/turn inputs; join artifacts only when their source manifest hashes match. Separate runs can share point IDs while having different responses.

`HFAdapter` isolates model loading and hook paths. `hf_causal` accepts explicit `backbone_path` and `blocks_path` for compatible Hugging Face decoder models. Other architectures must be tested before use; there is no claim that all Qwen/Llama/OLMo/Kolibri versions already work. Base models need a separately specified prompt formatter because they may lack a chat template. Keep each model's residual space, UMAP, SAE and probe separate until a cross-model alignment method has been validated.

See [METHODOLOGY.md](METHODOLOGY.md) for measurement choices, limitations and official references, and [docs/gemma-scope-investigation.md](docs/gemma-scope-investigation.md) for the checkpoint audit.

For slow replies, cancellation and prompt instructions, see [speed and stopping](docs/speed-and-stopping.md).
