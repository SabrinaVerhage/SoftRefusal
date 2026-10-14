# What you are seeing inside Gemma

The short version is: **text pieces → number vectors → repeated transformations → scores for the next text piece → repeat**.

Gemma learned numerical weights from training data. During this conversation those weights stay fixed. Your conversation changes the activations flowing through the model, not the learned weights. Saving a chat does not train Gemma or give it lasting memory; earlier messages are supplied again as context on later turns.

## 1. Text becomes tokens

The tokenizer splits the formatted conversation into vocabulary pieces. Some are words, some are word fragments or punctuation. Special pieces mark message boundaries and speakers. The view shows the final 32 prompt tokens with their real IDs, including formatting. Hover for positions. These token IDs are lookup addresses, not meanings or probabilities.

## 2. Tokens become vectors

Each token looks up a learned vector. The embedding module produces a matrix: one row per token, many numbers per row. The live panel samples the last 24 rows of that actual output and averages adjacent coordinates into 16 columns to fit on screen. Gemma's embedding scaling is included in this measured output.

These are starting representations. They are not the contextual residual-stream coordinates used for our research map. A colored cell is an average of numerical coordinates, not a named idea such as intimacy or refusal.

## 3. Layers repeatedly update the representation

Inside each decoder layer, attention combines information from the available context, and feed-forward transformations further update the running representation. Learned matrix multiplications, nonlinear functions and normalization participate in those updates. This explanation describes the architecture; the grid does not display every operation or the weight matrices themselves.

The grid shows a **real measurement of the final formatted prompt position after each of Gemma's 34 decoder layers**. Each row is summarized into 16 coordinate-group averages. It fills during the first reading used to generate the reply. Earlier versions did a separate extra pass; that duplicate work has been removed. It is not an attention heatmap and does not animate the full model again for each generated token. The full research vectors are retained only at layers 9, 17, 22 and 29, numbering from zero.

Teal means positive and clay negative. Color intensity is relative to the largest absolute displayed group mean in that grid; scaling can change as new rows appear. This is a display choice. Neither sign nor brightness measures importance, sexuality or avoidance.

## 4. The model scores the next token

The final representation is converted into scores over the vocabulary. The panel displays the actual top five next-token probabilities after generation processing. The other tokens hold the remaining probability. Probabilities concern the next text piece; they are not confidence that an answer is true.

This experiment uses greedy decoding: choose the highest score, append that token, then predict again with the growing context. A slider lets you review earlier decisions. Special end tokens may appear and may not print as ordinary words. A reply ends when a stop token is selected or our length cap is reached. The observer returns generation scores unchanged.

## What gets saved

Each completed turn saves the text, exact token IDs, chosen reply cap, model settings, and four complete 2,560-value residual vectors. It additionally saves the live view's token preview, compressed embedding/layer matrices, per-layer vector norms, and top-five candidates for each generated token in `education_trace` in `steps.jsonl`. Attempt logs contain the corresponding events. These summaries add visibility without attempting to save every tensor in the model.

Older recordings remain valid but lack the new educational trace. They are not filled with invented measurements. Interrupted attempts stay separate from completed points. The interface does not yet assign human refusal labels, run a refusal probe or display a 3D UMAP; those remain separate analysis steps.

The planned 3D map reduces the **full residual vectors at a fixed layer**, one point per conversation step. It does not use the little starting-embedding grid as its coordinates. Refusal is assessed separately from the response and, when enough labeled data exists, a held-out probe. The live visualizations help explain the computation; they do not reveal readable thoughts or establish what caused a refusal.

For the architecture, see the [official Gemma 3 model card](https://ai.google.dev/gemma/docs/core/model_card_3). For output-length and greedy-generation controls, see [Transformers generation documentation](https://huggingface.co/docs/transformers/v4.57.1/en/main_classes/text_generation).
