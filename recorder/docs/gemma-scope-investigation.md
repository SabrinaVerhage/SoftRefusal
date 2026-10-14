# Gemma Scope 2 checkpoint audit

Checked 3 October 2026 against Google model repositories and the SAELens loader source. The base model and Scope repositories are distinct: model access is gated; the examined Scope files are publicly accessible.

| Item | Verified selection |
|---|---|
| Target | `google/gemma-3-4b-it` |
| Scope repo | `google/gemma-scope-2-4b-it` |
| Scope commit | `3e94b68be95290aada5b7525cf431d3040f81bb1` |
| Practical first SAE | `resid_post/layer_17_width_16k_l0_small` |
| Checkpoint shape | 2,560 residual dimensions → 16,384 features |
| Config | JumpReLU, target L0 20, no affine connection, SAE |
| Config input/output hook | `model.layers.17.output` (text backbone convention) |
| This adapter's equivalent | `model.language_model.layers.17` output, before final RMSNorm |
| Weights | `params.safetensors`, about 336 MB |

The repository file tree confirms residual subset layers **9, 17, 22, 29**. Its model-card example mentions layer 12 and elsewhere a 270M model; those appear to be copied examples and were not used as checkpoint identifiers. Actual per-checkpoint configs and the file tree determine compatibility.

The integration downloads only config and parameters, with fixed revision and SHA256 recording. It implements the small inference operation directly with PyTorch/safetensors, rather than requiring the larger SAELens/TransformerLens dependency stack. The [SAELens Gemma 3 loader](https://github.com/jbloomAus/SAELens/blob/main/sae_lens/loading/pretrained_sae_loaders.py) specifies no input normalization and no decoder-bias subtraction. For `pre = x @ w_enc + b_enc`, features are `relu(pre) * (pre > threshold)` and reconstruction is `features @ w_dec + b_dec`.

Google also provides residual/attention/MLP SAEs and single-layer transcoders, including all-layer collections. Transcoders predict MLP outputs from their specific MLP input site, potentially with a skip connection. Feeding a residual-post vector into a transcoder would be a site mismatch. Circuit tracing therefore needs additional input/output hooks and reconstruction/intervention checks. It is investigated but deliberately not presented as a working circuit-analysis feature here. The general model card describes cross-layer transcoders for 270M/1B; do not infer that a 4B CLT exists from shared folder descriptions.

The 16k/small SAE is a low-memory mechanics choice, not an optimal scientific feature dictionary. Larger dictionaries and other sparsities should be compared later. Features are saved without semantic names. Use [Neuronpedia's Scope 2 explorer](https://www.neuronpedia.org/gemma-scope-2) to form hypotheses, then independently inspect activating examples and counterexamples. Do not equate feature IDs across checkpoints.

References: [Google's model card](https://huggingface.co/google/gemma-scope-2-4b-it), [pinned checkpoint config](https://huggingface.co/google/gemma-scope-2-4b-it/blob/3e94b68be95290aada5b7525cf431d3040f81bb1/resid_post/layer_17_width_16k_l0_small/config.json), [file tree](https://huggingface.co/google/gemma-scope-2-4b-it/tree/3e94b68be95290aada5b7525cf431d3040f81bb1/resid_post), [DeepMind overview](https://deepmind.google/models/gemma/gemma-scope/).
