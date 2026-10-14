# Model choice and sexual-content policy

Official sources checked 4 October 2026. Policies and model availability can change; recheck before distribution or a commercial deployment.

## Gemma's terms and its behavior are different things

Google's [Gemma Prohibited Use Policy](https://ai.google.dev/gemma/prohibited_use_policy), section 4, restricts sexually explicit content created for sexual gratification, including sexual chatbots, but expressly excludes scientific, educational, documentary and artistic purposes from that category. Section 2.4 separately restricts bypassing safety filters. The [Gemma terms](https://ai.google.dev/gemma/terms), section 3.2, incorporate the policy; running downloaded weights locally does not remove it. This research/art project has a relevant stated exception, but that is not a blanket assurance about every future use.

These are Gemma's terms, not the terms for the hosted Gemini chat service. The Gemma 3 model card describes its safety work. Permission for a particular research use does not make the instruction model willing to answer: learned refusal behavior can still occur. The app adds no separate content filter, and does not modify or bypass Gemma's safety training.

## Recommendation

Keep the existing, unmodified **Gemma 3 4B IT** as the first baseline. Its refusal behavior is part of the subject being studied, and its matching Gemma Scope tools make it especially useful for inspection. Add a second, separately recorded model after the mechanics and labeling rubric are stable. Do not replace existing data or treat different models' numerical coordinates as a shared space.

| Candidate | Why consider it | What remains uncertain or limiting |
|---|---|---|
| [Gemma 3 4B IT](https://ai.google.dev/gemma/docs/core/model_card_3) | Already running here; matched residual SAEs in Gemma Scope 2 | May refuse sexual material; allowed-purpose exception does not override trained behavior |
| [Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Similar scale; Apache 2.0; practical next unmodified instruction-model comparison | Its official card does not establish greater sexual permissiveness; must test, and its activations need separate validation |
| [Dolphin3.0-Qwen2.5-3b](https://huggingface.co/cognitivecomputations/Dolphin3.0-Qwen2.5-3b) | Creators emphasize user-steerable alignment; plausible contrast in response behavior | Claims are not evidence of sexual-content performance. This particular 3B checkpoint uses the Qwen Research license, not Apache 2.0; its [base license](https://huggingface.co/Qwen/Qwen2.5-3B/blob/main/LICENSE) limits use to noncommercial research/evaluation absent another license |
| [OLMo 3 7B Instruct](https://huggingface.co/allenai/Olmo-3-7B-Instruct) | Apache 2.0 and unusually documented training stages; useful later research comparison | Larger and heavier for this laptop; includes safety training and is not established as sexually permissive |

No alternative has been downloaded, tested or added to the interface yet. Marketing words such as “uncensored” do not establish suitability, a compatible license, or stable behavior.

## How to compare fairly

First define what “better on sexual content” means for this artwork: answering factual sexual-health questions, discussing intimacy, discussing sensuality in art, and producing sexual-gratification material are different tasks. Use permitted research prompts and record these categories separately. A single refusal percentage hides that distinction.

Give each model the same prompt families and conversation structure, using its native chat format. Record all versions and generation limits; keep capped replies separate from completed refusals. Human reviewers label the actual responses using the same rubric. Keep related conversations together when splitting probe training and evaluation data. Fit separate maps/probes per model, then compare measured outcomes and validated relationships—not raw UMAP positions across unrelated models.

Selecting a different model is a scientific comparison, not a promise that a particular prompt will be answered. Hardware fit and the ability to measure matching internal sites also need validation before adding a model.
