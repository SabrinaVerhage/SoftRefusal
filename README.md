# Mapping the Sexual Avoidance of Artificial Intelligence

How a language model refuses, redirects or answers, recorded from inside the model and drawn as paths through a "semantic donut" around a refusal core.

```
recorder  ──── message + history ────►  gpu-api (RunPod GPU)
(recording     ◄── reply + 4 snapshots ────  Gemma 3 12B, BF16
 room, laptop)
      │  labels: answered / redirected / refused
      ▼
  avoidance map  ──── data/recordings.js ────►  map (viewer)
```

| Folder | What it is | Start with |
|---|---|---|
| [`recorder/`](recorder/) | Chat page that records Gemma's internal state before every reply. Cloud mode keeps all recordings on this computer. | `START_HERE.md`, `docs/cloud-gpu.md` |
| [`gpu-api/`](gpu-api/) | GPU API on RunPod: queue, Gemma/Hermes/FLUX, activation recording. | `DEPLOYMENT_STATUS.md` before any paid session |
| [`map/`](map/) | 3D viewer: reference field, the semantic donut of your recordings, UMAP views. | `README.md` |

**Kept out of git:** conversations and recordings (`recorder/runs/`), model weights, access codes (`secrets/`), backend runtime data, RunPod screenshots, generated map data, and the `_design_thinking/` folder.

## Checks

```bash
cd recorder && .venv/Scripts/python.exe -m pytest -q --ignore=tests/check_instruction_ui.py --ignore=tests/check_live_browser.py
cd gpu-api && .venv/Scripts/python.exe -m pytest -q tests
cd map && node verify.cjs
```
