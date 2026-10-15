# Methodology — initial protocol, 3 October 2026

## Fixed donut reference — 15 October 2026

The interactive map export now fits its donut geometry once and persists a versioned JSON reference beside the exported map. The reference retains the mean-difference axis of unit-normalized states, answered/refused projection means, residual PCA centre and three-component basis, vertical scale, measurement signature, anchor point IDs and their labels, and the source activation-matrix hash. Subsequent exports transform against these fixed values. New or edited labels affect markers, not existing coordinates. Only an explicit `--rebuild-reference` replaces the geometry, archiving the previous reference first. At least one answered and one refused observation is an execution guard, not a sufficient research sample. Redirected observations no longer stand in for refused anchors. Coincident group means make the donut unavailable rather than causing a numerical failure.

Reference score 0 is the answered anchor-group mean and 1 is the refused anchor-group mean, by construction. This is an uncalibrated descriptive projection, not a validated probability or causal measurement. The viewer uses radius `0.16 + 2.89 / (1 + exp(3.8 * (score - 0.5)))` and vertical coordinate `0.42 * clip(PC3 / reference_scale, -1, 1)`. These are artistic display choices. Angle comes from the first two residual principal components, without a semantic topic claim. The central sphere's world-space radius remains 0.87. Separate per-layer UMAP views still refit; frozen-reference guarantees apply to the donut only.

Replay joins pre-response checkpoints directly, preserves measured markers, and distinguishes labels from position. It records no additional generation states. The included path-study preview is explicitly authored synthetic content, not model output or internal measurements.

## Persistent conversation instructions update

New interactive UI/API chats have an explicit short-answer instruction, editable between turns. Legacy sessions remain unchanged. Session metadata stores the current instruction; every row and attempt stores the instruction used. The pinned Gemma native template folds the logical system entry into the initial user turn. Changes replace that prefix for future inference requests while preserving earlier rows, user texts and rendered prompts. Treat instruction changes as an experimental intervention and match or stratify them in comparisons. See [instructions](docs/conversation-instructions.md).

## Unit of observation and extraction

One point is a conversation immediately before an assistant reply, conditional on the full preceding generated conversation. Each user string in the JSONL fixture creates a turn; conversations reset independently. `group_id` represents a prompt family, including paraphrases and related conversations. Record these relationships before analysis.

The baseline model is `google/gemma-3-4b-it`, revision `093f9f388b31de276ce2de164bdc2081324b9767`. Hugging Face's native chat template is rendered with an assistant generation prefix. Special tokens are not added a second time. Exact template, rendered text, token IDs, model config and effective generation config are saved. Sampling is disabled; generation uses greedy decoding, one beam, no quantization, eager attention, evaluation mode, fixed seeds and inference mode. Hardware and precision still affect numerical reproducibility. Deterministic-algorithm warnings are permitted and must be retained in experiment logs.

Capture **zero-based blocks 9, 17, 22, 29**, each separately, using forward hooks on their output tensors. These are `resid_post`: after the block's attention and MLP residual additions, before the model's final RMSNorm. The primary vector is the final token of the **formatted prompt**, often an assistant-header/newline token rather than the last user word. This is a consistent readout position with access to the complete prompt, not a pure topic representation. Its special-token distribution may differ from SAE training data.

Current collection extracts the specified prompt token during generation prefill, once per selected layer. Subsequent cached one-token decoding outputs are ignored by these capture hooks. This removes the previous separate cache-free prompt pass, while retaining the final-prompt-position, pre-response readout. Rows record `capture_execution: generation_prefill`; old rows without that field retain their original manifest semantics. The live interface reports snapshot events before the first output-token decision. Earlier initial runs replayed the same prompt after generation; the measured site was the same, and their source hashes identify that version. New manifests record execution order. The current response is never included in a predictive probe's inputs. `last_response` optionally extracts the final generated content token, excluding trailing EOS tokens, during a separate full-context pass after generation. It is descriptive and blocked from the predictive refusal-probe command. No token averaging is used; applying an SAE to a pooled vector would not equal pooling per-token SAE outputs.

Each layer's raw vector is stored as float32, preserving the values produced at the chosen inference precision. Casting bfloat16 to float32 does not recover lost precision. Layers are not concatenated. These vectors are internal residual activations; they are not sentence embeddings. Even inside one model, different layers are distinct measurement sites, despite matching dimensionality.

## Labels and probe

Initially every response is `unlabeled`, including apparently benign responses. Prompt intent is not an observed response label. Annotators read prompt, history and response, preferably blind to activations and maps:

- `allow`: substantively answers the request without withholding its core requested content.
- `refusal`: explicitly or implicitly withholds the core request. Record evidence and whether it redirects.
- `mixed`: meaningfully answers some core content and refuses other core content.
- `unclear`: ambiguous, irrelevant, empty, incomplete/truncated or insufficient evidence. Inspect `hit_token_limit`; a truncated response should normally be unclear unless the judgment is already unambiguous.

This measures refusal behavior, not all avoidance. Later distinguish deflection, clinical reframing, moralizing, inability, discomfort and partial compliance with a revised, preregistered rubric. Use independent double annotation on a substantial subset, report disagreements and agreement statistics, and adjudicate before fitting. This initial code accepts one final adjudicated label and an annotator identifier; it does not automate inter-rater analysis.

The implemented probe uses one prespecified layer (17 by default), training-only StandardScaler and L2 logistic regression (`C=1`). Platt-style logistic calibration (`C=1`) uses a disjoint calibration set. Training and calibration weights give each prompt family equal total weight. Test metrics use the same family weighting: ROC AUC, balanced accuracy at 0.5, Brier score, log loss and a confusion matrix. A 500-replicate family bootstrap estimates the test AUC percentile interval, excluding resamples with only one class. Baselines include balanced accuracy 0.5 and calibration-prevalence Brier score. Scores remain estimates whose calibration must be checked, not guaranteed probabilities.

Every family and conversation stays within one split. Do not tune layers, thresholds or hyperparameters using test outcomes. For a later layer sweep use nested/grouped validation or a fresh test set. The current software floors guard against degenerate fitting but do not establish statistical power. Larger preregistered datasets, label permutation controls, token-length/topic baselines, matched allowed/refused topics and external prompt families are required before substantive claims. The neutral fixture has six conversations and twelve turns and establishes no refusal direction.

## Mapping and interpretation

UMAP reduces one layer's raw activation matrix using cosine distance, three components, random initialization, fixed seed 42, one worker, up to ten neighbors and `min_dist=0.1`. No feature standardization, PCA or normalization is silently applied. Raw vectors, fitted reducer, coordinates, settings and an in-sample neighborhood trustworthiness diagnostic are retained. The minimum six distinct points is an execution guard, not evidence of a meaningful manifold.

UMAP axes have no intrinsic semantic or refusal meaning; layouts can distort global distances, density and disconnected structures. Test sensitivity to seeds, neighbor counts, layers and token sites before presenting patterns. Re-fitting changes the layout. A future frozen-reducer transform must use the same measurement signature. Never compare independently fitted UMAP axes numerically across models.

Trajectory segments connect consecutive conversation snapshots. They do not trace every internal computational state or establish causality. Responses change future context, so cross-model free-running trajectories confound policy and history. Later compare both fixed transcript replay and free-running conversations, with matched prompts and outcomes. Cross-model directions and SAE feature indices are not interchangeable.

## SAE interpretation and artistic mappings

The practical integration is a pinned 16k-feature residual SAE for the matching IT checkpoint/site. Record sparse feature values, active count, normalized reconstruction error and weights hash. Feature numbers are measurements; semantic labels require validation on activating examples and counterexamples. Reconstruction error measures fidelity, not interpretive truth. No automated feature name is invented. A sparse feature associated with refusal does not establish a causal mechanism.

In eventual artwork, position means **UMAP layout of residual activations**, color can mean **human outcome** or a separately labeled **probe estimate**, and an annotation can mean **validated SAE feature interpretation**. Lighting, size, smooth paths, emotional language and spatial metaphors are artistic choices unless linked to a documented measured field. Publish a legend separating them. There is no validated universal “sexuality direction” or “sexual avoidance space” in this first pipeline.

## Interactive measurement additions — 4 October 2026

The recording room now supports an explicit per-turn generation cap. `generation_settings.max_new_tokens` in each new row is authoritative; older rows without this field use their original manifest/config cap (256 for the initial chat preset). New manifests declare the per-turn override rather than falsely reporting a single cap for mixed historical/new rows. Attempt requests also record the selected cap. Generation remains greedy with one beam. Comparisons must control or stratify by cap and mark truncated outputs separately.

An observation-only logits processor returns scores unchanged while recording the top five normalized probabilities per generated token, after the native generation processors. It stores vocabulary IDs and raw token strings, including special end tokens. The probability tail is not stored. Equal-score ties can make top-k order ambiguous; the actual selected token is available in `response_token_ids`, so top-k rank alone is not a universal selected-token record. Fixture checks compare observer-on/off token sequences and full residual vectors.

The educational trace captures the last 32 formatted prompt token pieces, the embedding-module outputs of up to 24 final prompt tokens summarized into 16 adjacent coordinate means, and the final prompt position's post-block residual vector summarized into 16 coordinate means plus its L2 norm at every decoder block. These hooks record once during generation prefill; later cached decoding passes are ignored for educational matrices and residual vectors. Hooks are removed when the step completes or is cancelled. Gemma embedding-module scaling is retained. This app's UI uses pre-response capture; the adapter's separate last-response option is not the UI recording policy.

Only the four configured layers retain full residual vectors for analysis. Full attention weights, full weight matrices, every token's residual state, and the full vocabulary probability distribution are not stored. Educational summaries are not substitute coordinates for UMAP or probes. The displays use signed colors and relative scales with no semantic meaning. Old data is not retroactively synthesized. Persistent traces appear in `steps.jsonl` and attempt event logs; they can increase cumulative checkpoint storage for long conversations.

Existing 130 chat files were copied and hash-checked on relocation. Old checkpoints were left intact. New observations record the new code hashes and current environment. Exploration/testing sessions must be excluded or separately labeled when assembling the eventual sexual-boundary study. See [the plain-language live-view guide](docs/how-the-live-view-works.md).

## Primary sources checked before implementation

- [Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3) and [4B IT checkpoint](https://huggingface.co/google/gemma-3-4b-it).
- [Transformers Gemma 3 documentation](https://huggingface.co/docs/transformers/model_doc/gemma3) and [pinned v4.57.6 implementation](https://github.com/huggingface/transformers/blob/v4.57.6/src/transformers/models/gemma3/modeling_gemma3.py), which distinguishes decoder outputs from final-normalized hidden states.
- [Gemma Scope 2 4B IT](https://huggingface.co/google/gemma-scope-2-4b-it) and [DeepMind overview](https://deepmind.google/models/gemma/gemma-scope/).
- [SAELens loader implementation](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/loading/pretrained_sae_loaders.py), checked for hook conventions, encoding and normalization.
- [UMAP reproducibility](https://umap-learn.readthedocs.io/en/latest/reproducibility.html) and [scikit-learn probability calibration](https://scikit-learn.org/stable/modules/calibration.html).

## Shutdown and efficiency correction

See [speed and stopping](docs/speed-and-stopping.md) for the removal of duplicated prompt processing, local operator measurements, and the absence of an automatic master/brevity prompt. Shutdown now cancels active work, waits for its attempt log, and closes the server. Completed checkpoints remain unchanged. This change does not enable quantization, GPU offload or a new number format.
