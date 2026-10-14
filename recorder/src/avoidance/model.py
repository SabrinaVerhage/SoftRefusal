from __future__ import annotations

import random
from contextlib import contextmanager

import numpy as np
import torch


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True, warn_only=True)


@contextmanager
def residual_hooks(blocks, layers, token_index, on_capture=None, once=False):
    """Capture block outputs before final RMSNorm; never index HF hidden_states blindly."""
    captured = {}
    handles = []
    if not layers or len(set(layers)) != len(layers) or any(i < 0 or i >= len(blocks) for i in layers):
        raise ValueError("Extraction layers must be unique, zero-based decoder block indices")
    def hook_for(layer):
        def hook(module, inputs, output):
            if once and layer in captured:
                return
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1:
                raise ValueError("Expected unpadded single-sequence [1, tokens, hidden] activations")
            captured[layer] = hidden[0, token_index].detach().float().cpu().numpy().copy()
            if on_capture:
                on_capture(layer, captured[layer])
        return hook
    try:
        for layer in layers:
            handles.append(blocks[layer].register_forward_hook(hook_for(layer)))
        yield captured
    finally:
        for handle in handles:
            handle.remove()


class GenerationCancelled(Exception):
    pass


class HFAdapter:
    """Thin HF adapter. A new architecture must explicitly identify its decoder blocks."""
    def __init__(self, config):
        from transformers import AutoModelForCausalLM, AutoTokenizer, Gemma3ForConditionalGeneration
        self.config = config
        seed_everything(config["seed"])
        if config["device"] == "cuda" and not torch.cuda.is_available():
            raise ValueError("CUDA requested but unavailable; select CPU explicitly")
        if config["dtype"] not in {"float32", "bfloat16", "float16"}:
            raise ValueError("Unsupported dtype; quantization is deliberately excluded from baseline")
        self.tokenizer = AutoTokenizer.from_pretrained(config["model_id"], revision=config["revision"], trust_remote_code=False)
        cls = Gemma3ForConditionalGeneration if config["adapter"] == "gemma3" else AutoModelForCausalLM
        self.model = cls.from_pretrained(
            config["model_id"], revision=config["revision"], torch_dtype=getattr(torch, config["dtype"]),
            device_map=config["device"], attn_implementation="eager", trust_remote_code=False,
        ).eval()
        if config["adapter"] == "gemma3":
            self.backbone = self.model.model.language_model
            self.blocks = self.backbone.layers
            self.module_path = "model.language_model.layers"
        elif config["adapter"] == "hf_causal":
            self.module_path = config["blocks_path"]
            self.blocks = self.model.get_submodule(self.module_path)
            self.backbone = self.model.get_submodule(config["backbone_path"])
        else:
            raise ValueError("adapter must be gemma3 or hf_causal")

    @torch.inference_mode()
    def step(self, messages, on_event=None, cancel_event=None, max_new_tokens=None):
        c = dict(self.config)
        if max_new_tokens is not None:
            c["max_new_tokens"] = max_new_tokens
        education = {"layers": [], "candidates": []}
        emit = on_event or (lambda kind, **data: None)
        def check_cancelled():
            if cancel_event is not None and cancel_event.is_set():
                raise GenerationCancelled("Stopped; this unfinished turn was not added to the recorded conversation.")
        check_cancelled()
        rendered = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = self.tokenizer(rendered, add_special_tokens=False, return_tensors="pt")
        inputs = {k: v.to(self.model.get_input_embeddings().weight.device)
                  for k, v in inputs.items() if k in {"input_ids", "attention_mask"}}
        n = inputs["input_ids"].shape[1]
        if n > c["max_input_tokens"]:
            raise ValueError(f"{n} input tokens exceeds limit; no silent truncation is permitted")
        emit("prepared", input_tokens=n, message="Read the conversation and prepared the next reply.")
        if on_event:
            ids = inputs["input_ids"][0].tolist()
            pieces = [{"position": i, "id": ids[i], "piece": self.tokenizer.convert_ids_to_tokens(ids[i])}
                      for i in range(max(0, n - 32), n)]
            education["tokens"] = {"total": n, "shown": pieces, "scope": "last_32_formatted_prompt_tokens"}
            emit("tokenized", **education["tokens"], message=f"Split the formatted conversation into {n} tokens (text pieces).")
        def captured_layer(layer, vector):
            if not np.isfinite(vector).all():
                raise ValueError("Non-finite activation encountered; no valid snapshot can be saved")
            emit("snapshot", layer=layer, dimensions=len(vector),
                 preview=[float(a.mean()) for a in np.array_split(vector, min(32, len(vector)))],
                 message=f"Captured layer {layer}: {len(vector):,} numbers.")
        generated = None
        def capture(capture_inputs, index, runner=None):
            nonlocal generated
            check_cancelled()
            emit("capturing", message="Taking snapshots of the conversation at the selected layers.")
            handles = []
            def embedding_hook(module, inputs, output):
                check_cancelled()
                if "embeddings" in education:
                    return
                matrix = output[0, -24:].detach().float().cpu().numpy()
                preview = np.stack([part.mean(axis=1) for part in np.array_split(matrix, 16, axis=1)], axis=1)
                education["embeddings"] = {"values": preview.tolist(), "dimensions": matrix.shape[1],
                    "first_position": capture_inputs["input_ids"].shape[1] - len(matrix),
                    "scope": "actual_embedding_module_output_last_24_tokens_16_column_means"}
                emit("embeddings", **education["embeddings"], message="Looked up the starting number vector for each token.")
            def layer_hook(layer):
                def hook(module, inputs, output):
                    check_cancelled()
                    if layer < len(education["layers"]):
                        return
                    hidden = output[0] if isinstance(output, tuple) else output
                    v = hidden[0, index].detach().float().cpu().numpy()
                    entry = {"layer": layer, "magnitude": float(np.linalg.norm(v)),
                             "preview": [float(part.mean()) for part in np.array_split(v, 16)], "total_layers": len(self.blocks)}
                    education["layers"].append(entry)
                    emit("layer_pass", **entry)
                return hook
            try:
                if on_event:
                    handles.append(self.model.get_input_embeddings().register_forward_hook(embedding_hook))
                    handles.extend(block.register_forward_hook(layer_hook(i)) for i, block in enumerate(self.blocks))
                if cancel_event is not None and not on_event:
                    handles.extend(block.register_forward_pre_hook(lambda *args: check_cancelled()) for block in self.blocks)
                with residual_hooks(self.blocks, c["layers"], index, captured_layer, once=True) as vectors:
                    if runner:
                        generated = runner()
                    else:
                        self.backbone(**capture_inputs, use_cache=False, return_dict=True)
            finally:
                for handle in handles:
                    handle.remove()
            if set(vectors) != set(c["layers"]):
                raise RuntimeError("Not all extraction hooks fired")
            check_cancelled()
            return vectors
        kwargs = {}
        if on_event:
            from transformers import TextStreamer, LogitsProcessor, LogitsProcessorList
            tokenizer = self.tokenizer
            class CandidateObserver(LogitsProcessor):
                def __call__(self, input_ids, scores):
                    if not education["candidates"]:
                        emit("generating", message="Snapshots captured during Gemma's first reading of the conversation. Gemma is now writing." if c["token_selection"] == "last_prompt" else "Gemma is writing a reply.")
                    probs = torch.softmax(scores[0].float(), dim=-1)
                    values, indices = torch.topk(probs, min(5, len(probs)))
                    items = [{"id": int(i), "piece": tokenizer.convert_ids_to_tokens(int(i)), "probability": float(v)}
                             for v, i in zip(values, indices)]
                    entry = {"step": len(education["candidates"]), "top": items,
                             "selection": "greedy_argmax_after_generation_processors"}
                    education["candidates"].append(entry)
                    emit("candidates", **entry)
                    return scores  # Observation only: never modify the generation distribution.
            kwargs["logits_processor"] = LogitsProcessorList([CandidateObserver()])
            class EventStreamer(TextStreamer):
                def __init__(self, tokenizer):
                    super().__init__(tokenizer, skip_prompt=True, skip_special_tokens=True)
                    self.first = True
                    self.count = 0
                def put(self, value):
                    if self.first:
                        self.first = False
                    else:
                        self.count += value.numel()
                        emit("progress", generated_tokens=self.count)
                    super().put(value)
                def on_finalized_text(self, text, stream_end=False):
                    if text:
                        emit("text", text=text)
            kwargs["streamer"] = EventStreamer(self.tokenizer)
        if cancel_event is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList
            class StopRequested(StoppingCriteria):
                def __call__(self, input_ids, scores, **kwargs):
                    return cancel_event.is_set()
            kwargs["stopping_criteria"] = StoppingCriteriaList([StopRequested()])
        def generate_reply():
            return self.model.generate(**inputs, do_sample=False, num_beams=1,
                                       max_new_tokens=c["max_new_tokens"], use_cache=True, **kwargs)
        if c["token_selection"] == "last_prompt":
            vectors = capture(inputs, n - 1, generate_reply)
        else:
            emit("generating", message="Gemma is writing a reply.")
            generated = generate_reply()
        check_cancelled()
        continuation = generated[0, n:].tolist()
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos])
        if c["token_selection"] == "last_prompt":
            capture_inputs, index, phase = inputs, n - 1, "pre_response"
        elif c["token_selection"] == "last_response":
            kept = len(continuation)
            while kept and continuation[kept - 1] in eos:
                kept -= 1
            if not kept:
                raise ValueError("No response content token to extract")
            ids = generated[:, :n + kept]
            capture_inputs = {"input_ids": ids, "attention_mask": torch.ones_like(ids)}
            index, phase = n + kept - 1, "post_response"
            vectors = capture(capture_inputs, index)
        else:
            raise ValueError("token_selection must be last_prompt or last_response")
        ids = capture_inputs["input_ids"][0].tolist()
        return {
            "rendered_prompt": rendered, "prompt_token_ids": inputs["input_ids"][0].tolist(),
            "response": self.tokenizer.decode(continuation, skip_special_tokens=True),
            "response_token_ids": continuation, "phase": phase,
            "token_index": index, "token_id": ids[index],
            "token_text": self.tokenizer.decode([ids[index]]),
            "input_token_count": n, "generated_token_count": len(continuation),
            "ended_with_eos": bool(continuation and continuation[-1] in eos),
            "hit_token_limit": len(continuation) >= c["max_new_tokens"] and continuation[-1] not in eos,
            "generation_settings": {"max_new_tokens": c["max_new_tokens"], "do_sample": False, "num_beams": 1},
            "capture_execution": "generation_prefill" if c["token_selection"] == "last_prompt" else "after_generation",
            "education_trace": education if on_event else None,
        }, vectors
