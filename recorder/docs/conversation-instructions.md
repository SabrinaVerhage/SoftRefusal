# Automatic conversation instructions

New UI/API conversations start with: Keep your reply under 50 words, in two or three sentences.

Open Conversation instructions above the composer to edit and save it. For an existing chat, click Use short replies. Clear and save to remove extra instructions. Saved settings are automatically included in each future request; no need to type them with every message. They are requests, not a guaranteed word limit.

Existing conversations keep their original behavior until you change the setting. Every completed row and attempt request records the instruction used. Changing it does not alter earlier recordings. Keep this setting consistent for comparisons, or label it as an experimental variable.

The app sends the instruction through Gemma's pinned native chat template, which folds it into the first user turn. Gemma 3 does not have a separate privileged system turn. See [Google's formatting guidance](https://ai.google.dev/gemma/docs/core/prompt-structure).

Practical suggestions: specify length and format, supply relevant context, avoid contradictory requests, and give an example when formatting matters. Check whether the answer follows the instruction. These are suggestions, not guarantees.

Use version-specific documentation: some generic Gemma tutorials now describe Gemma 4, and Gemini is a separate product. The [Gemma 3 developer guide](https://developers.googleblog.com/en/introducing-gemma3/) is a suitable starting point. No sampling or model settings changed for this feature.
