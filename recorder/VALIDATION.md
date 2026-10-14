# Validation record — updated 4 October 2026

## Persistent conversation instructions

Five relevant tests passed (existing chat tests plus instruction persistence/history isolation). An isolated browser check verified enabling short replies on an older chat, editing, saving, reload persistence, clearing, and the new-chat default, with no page errors. The actual cached Gemma tokenizer confirmed that the logical system instruction is folded into the first user turn without a system-turn token. No claim is made that the model always respects the word limit. The user's active generation was not interrupted to restart the server; this update loads on the next stop/reopen. Older running servers show a restart hint and disable unsupported instruction controls.

## Active-stop fix and single prompt pass — evening update

Fifteen existing tests passed after the prefill change; three additional tests passed for equivalence to the dedicated capture pass and one-prompt-pass counting, cancellation during prefill with hook cleanup, and busy HTTP shutdown waiting for its stopped-attempt log. A real neutral Gemma turn completed through the browser with all 34 layer summaries, four output decisions and restored recordings. No browser errors were reported.

The actual `Stop Recording Room.cmd` was then invoked while Gemma was processing a second neutral turn. It cancelled the active attempt, printed progress, closed the server successfully, and retained the first complete checkpoint. The interrupted attempt's result was verified as `stopped`. This specifically tests the formerly failing busy-server case, not just idle shutdown.

For the same 17-token neutral prompt in the earlier and new browser checks, time from capture start to the first token-decision event was 7.17 seconds with the old duplicate-pass version and 2.86 seconds with generation-prefill capture. These are two observational timings from separate runs, excluding model loading; they are not a controlled benchmark or a promised speedup for long chats. No model, precision or hardware configuration was changed. See [speed and stopping](docs/speed-and-stopping.md).

## Relocation and live computation view — 4 October

The active app is now in `C:\Users\sabri\Documents\Sabrina\WORK\_SXTCH\SEXUAL_AVOIDANCE_AI\recording-room`. All 130 existing chat files matched their source SHA256 checksums after copying. The model and selected SAE caches were copied into `models/huggingface/`; the relocated environment successfully loaded the actual pinned Gemma model. Old source, recordings and global caches remain as backups.

The 13 existing tests passed after the live-view implementation, followed by two new passing regression tests: observation does not change generated token IDs or full vectors; measured trace shape/order and top candidates match a tiny model's actual output; per-turn caps persist across restart; invalid caps and closing-state requests are rejected. Tiny/random fixture results say nothing about trained model refusal behavior.

A real browser check recorded a neutral “hello” turn through trained Gemma, capturing all 34 layer summaries and four next-token decisions. The default UI cap is 512, the selected test cap 256 was persisted. The token pieces, matrices and five candidate bars rendered without JavaScript/CSP errors. The slider switched from live following to a selected decision. Reload restored the saved turn. Two page refreshes caused zero downloads; one explicit export produced exactly one `gemma-conversation-ddd00129.zip` without navigating away. Desktop and 390-pixel mobile screenshots were inspected, with no horizontal overflow.

The integration check is reproducible with `python tests/check_live_browser.py` against the running app and adds one neutral test conversation. Its report and screenshots are in `runs/ui-validation/`. These are validation sessions, not sexual-boundary study data. Longer generation limits are covered by the API/control tests, not a costly real 2,048-token benchmark. Full attention matrices and per-generated-token residual trajectories are not implemented.

The inspected `Projects.zip` family contained sticker SVG artwork, not this app's exports. No originating external page/process was identified and no unrelated files were removed.

The stop launcher shut down the real idle server and released port 8765. The open launcher then started the relocated app successfully and opened the browser; all seven sessions, including the neutral validation session, remained available. The new trace survived the process restart, its latest checkpoint passed checksum verification, and its four full arrays each had shape `[1, 2560]`. The actual generated token IDs matched the observed leading candidates in this run. The app's cache resolved to the new project's model directory. `pip check` reported no broken requirements. The server was left running, with Gemma loading on the next message.

## Chat-and-record interface

The local recording room is implemented and tested at `http://127.0.0.1:8765`. Thirteen automated tests pass, including live event ordering, streamed generation, immutable checkpoint saving, reopening sessions, ZIP export, cancellation, concurrent-request rejection, failed-save recovery, and local-origin request checks.

A real browser test with the trained Gemma model completed two plant-related turns. The exported dataset passes checksum verification and contains four `[2, 2560]` activation arrays. Its second prompt history contains the exact previous model reply. The browser displayed the streamed reply, all four captured layers, and the final save confirmation. Reload restored the conversation. Stopping a third reply preserved the two completed turns and excluded the unfinished reply from the research dataset. The separate activity record retained the stopped attempt.

Desktop and narrow-screen layouts were inspected. The narrow layout exposes new-conversation, history and explanation controls; there was no horizontal overflow. No browser JavaScript errors occurred in the end-to-end checks. The app and server use no external web assets or chat provider. Browser-testing software is an optional development dependency, not needed to run the app.

The live strips are 32 signed group averages with independent scales. The saved vectors remain full precision float32 representations of the model's bfloat16 outputs. Neither strips nor activity logs are labels of mental states, semantic concepts or refusal.

The chat preset uses a longer context/reply cap than the initial two-turn script. These settings are recorded per conversation. Existing conversations require matching settings before they can be continued. The model stays loaded in the local server until that process exits.

**Updated result: the trained Gemma 3 4B IT model now runs locally and has completed the first two-turn check.**

After the user authenticated locally, gated access was confirmed. The pinned model downloaded successfully (about 8.6 GB of weights). The CPU bfloat16 preset completed `data/first-contact.jsonl` and saved `runs/gemma-first-contact` with a complete, checksummed manifest. Both replies ended normally, without hitting the response-length limit. Each of layers 9, 17, 22 and 29 contains two finite 2,560-dimensional residual vectors. The saved readout positions are the final newline of each formatted prompt. Conversation history was retained for the second turn.

First response: “Plants need sunlight, water, nutrients, and air to grow.” Second response: “A snake plant is a popular choice for indoor growing due to its low-maintenance nature.” These are actual trained-model outputs, not fixtures; outcomes remain unannotated in the research dataset.

The pinned layer-17 residual SAE also completed on both real snapshots, saving sparse features and reconstruction diagnostics in `runs/gemma-first-contact-sae`. This confirms end-to-end execution, not semantic validity of feature interpretations. Two points are insufficient for the configured UMAP analysis and establish no refusal pattern. The full neutral fixture, larger labeled experiment and peak-memory/speed benchmark remain outstanding.

## Earlier setup checks (before authentication)

- `python -m pytest -q`: **9 passed**, 34.10 seconds on the final test run.
- `python -m pip check`: no broken requirements.
- CLI synthetic collection: all 12 turns from the neutral fixture saved with `synthetic: true`.
- CLI 3D UMAP: completed on those synthetic arrays; points include conversation IDs and previous-point links. This is not a scientific map.
- Downloaded actual pinned Scope layer-17 16k/small parameters; confirmed encoder `[2560, 16384]`, decoder `[16384, 2560]`, biases and thresholds. A zero-vector plumbing input produced finite `[1, 16384]` features. This is not a validation of interpretability or SAE fidelity on Gemma conversations.
- Trained Gemma access check returned `GatedRepoError` / HTTP 401. No trained 4B weights, responses or residual activations were obtained. The full 4B runtime, memory use and end-to-end pretrained SAE reconstruction quality remain unverified.

The tests exercise a locally instantiated, tiny **random-weight Gemma 3 conditional-generation architecture**, including generation, last-prompt and last-response positions, multi-turn collection, and hook outputs compared with the model's hidden states. The final block hook is checked before RMSNorm. These tests download no model and have no evidential value about learned refusal behavior.

Other tests check checksum tampering, synthetic-data rejection by scientific analyses, deterministic UMAP coordinates, correct conversation edges, JumpReLU threshold/bias conventions, train/calibration/test family leakage prevention, insufficient label rejection, and probe scoring/signature compatibility. Probe tests use fabricated separable vectors in temporary fixtures; their metrics are not research results.

Hardware inspected: RTX 3060 Laptop GPU, 6 GB VRAM; approximately 32 GB installed RAM and 13 GB available during setup. Python 3.10.6, PyTorch 2.6.0+cu124 and Transformers 4.57.6. Exact tested dependency versions are in `requirements-tested.txt` and run manifests. The CPU bfloat16 preset was selected to reduce memory pressure; its full-model speed and peak memory have not been measured.

Next empirical validation: authenticate locally, run the neutral fixture, inspect saved token positions and responses, confirm finite residuals at each layer, examine SAE sparsity/reconstruction, and manually annotate outcomes. Only then design the larger, independently grouped and annotated boundary experiment.
