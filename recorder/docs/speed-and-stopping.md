# Speed, stopping and instructions

**Later update:** saved conversation instructions now enable short replies for new chats. The no-master-prompt description below documents the earlier version. See [current behavior](conversation-instructions.md).

## What happened on 4 October 2026

The app was responsive and Gemma was generating a long answer; it had reached 371 of its allowed 512 output tokens when inspected. The preparation/snapshot phase had taken over eight minutes. This was slow computation, not evidence that a content filter froze the program. The user-requested stop was sent and the attempt stopped without adding an incomplete turn to the research dataset.

The old shutdown endpoint rejected shutdown while a reply was active. The stop launcher also hid the request behind a background window. Now both app shutdown and the stop launcher request cancellation, wait for the attempt log to finish, then close the server. Cancellation is checked between decoder layers during the initial reading and during generation. A currently executing numerical operation must still finish; stopping is not instantaneous. The stop launcher displays progress in a small console window. It does not terminate unrelated Python processes.

## Improvement implemented

The app previously ran a separate prompt pass for snapshots, then repeated the prompt pass for generation. It now captures the same final-prompt-position block outputs during generation's initial prompt pass, once per layer. Later one-token decoding passes do not overwrite the measurements. No second full prompt pass is needed. The model, precision, prompt, selected layers and decoding rule are unchanged.

This removes duplicated prompt computation, not all computation. It should especially help long histories; it does not promise a fixed end-to-end speedup or faster processing of every generated token. Each new row records `capture_execution: generation_prefill`; historical rows and checkpoints remain intact. Tiny-model tests verify a single prompt pass, compare against the old separate-pass vectors, and verify observer-on/off output equality. Numerical equivalence on every hardware/backend is not assumed; keep provenance when comparing runs.

## Why this computer is slow

The inspected machine has a Ryzen 9 5900HS (8 cores/16 threads), 32 GB RAM and a 6 GB RTX 3060 Laptop GPU. The current unquantized model runs on the CPU; all of its approximately 8.6 GB of weights do not fit in that GPU's 6 GB, even before working memory. It is capable of running the research app, but this setup favors inspectable recordings and modest RAM use over chat speed.

The CPU exposes AVX2. A local matrix test compared the current bfloat16 format with float32 using a `[4096,2560]` projection, three timed repetitions after warmup. At eight threads and 128 input rows, bfloat16 took about 46.6 ms versus float32's 7.1 ms. With one input row, bfloat16 was faster (0.42 vs 1.39 ms). This is a small operator benchmark, not a full-model throughput prediction. Float32 would roughly double weight memory and could cause paging with the other applications open, so it was not enabled. [PyTorch's documentation](https://pytorch.org/blog/empowering-pytorch-on-intel-xeon-scalable-processors-with-bfloat16/) explains the hardware-specific acceleration behind bfloat16 performance.

Longer chats supply all previous turns again. Processing that growing input costs more; predicting each next token also uses a growing context. The current app does not reuse a generation cache across chat turns or secretly remove history. The new implementation removes the second reading within each turn, but does not remove these context costs.

For now, request concise answers and use separate conversations for independent questions. Plugging in the laptop, avoiding a battery-saving mode, and freeing memory can help when those are bottlenecks. None was changed automatically. Future options include partial GPU offloading or a compressed model on the GPU; these need a separately identified and validated recording configuration because hardware/precision changes can affect measured values. No compressed checkpoint or new runtime was installed in this fix.

## Is there a master prompt?

No custom system/master instruction is inserted. The exact supplied content is previous user messages, previous complete Gemma replies, then the new user message. Gemma's native template adds speaker/end markers and the start of the next model turn. Its pretrained/instruction-tuned behavior remains present without a system prompt.

For a naturally short answer, write: **“Keep your reply under 50 words, in two or three sentences.”** That is a request to the model; a token cap is a mechanical limit and can cut a sentence off. A short answer also leaves less history to process next time. The request is stored as part of the prompt, so it is visible in the research record. No brevity instruction has been added automatically. For a controlled experiment, keep any such instruction consistent across compared prompts/models.
