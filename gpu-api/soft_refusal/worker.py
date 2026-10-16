"""Private model subprocess. stdout is a JSON protocol; never serves HTTP."""
import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import os
import sys
import threading
import time
from pathlib import Path

from .config import MODELS
from .progress import cancel_marker, write_progress
from .loras import image_adapter

# Progress snapshots are re-read by the page/recording room every poll; keep a
# window of recent candidates there. The final result holds every decision.
PROGRESS_CANDIDATES = 64


class ModelFailure(Exception):
    def __init__(self, code, **details):
        super().__init__(code)
        self.code = code
        self.details = details


def chat_stop_ids(model_key, generation_eos, tokenizer_eos):
    ids = list(generation_eos if isinstance(generation_eos, list) else [generation_eos])
    # Hermes's pinned generation_config names endoftext, but its native ChatML
    # tokenizer/model config names im_end. Respect both original stop markers.
    if model_key == "hermes" and tokenizer_eos is not None and tokenizer_eos not in ids:
        ids.append(tokenizer_eos)
    return [value for value in ids if value is not None]


class MemorySampler:
    """Allocator peaks plus independent, sampled whole-device NVML use."""
    def __init__(self, torch):
        self.torch = torch
        self.stop_event = threading.Event()
        self.thread = None
        self.peak = None
        self.handle = None
        self.nvml = None
        try:
            import pynvml
            pynvml.nvmlInit()
            # Container PIDs can differ from host PIDs reported by NVML. Match
            # the CUDA-visible GPU UUID before falling back to process lookup.
            uuid = getattr(torch.cuda.get_device_properties(0), "uuid", None)
            if uuid is not None:
                gpu_uuid = str(uuid)
                if not gpu_uuid.startswith("GPU-"):
                    gpu_uuid = "GPU-" + gpu_uuid
                self.handle = pynvml.nvmlDeviceGetHandleByUUID(gpu_uuid)
                self.nvml = pynvml
                return
            for index in range(pynvml.nvmlDeviceGetCount()):
                handle = pynvml.nvmlDeviceGetHandleByIndex(index)
                if any(p.pid == os.getpid() for p in pynvml.nvmlDeviceGetComputeRunningProcesses(handle)):
                    self.handle, self.nvml = handle, pynvml
                    break
        except Exception:
            pass

    def sample(self):
        if self.handle is not None:
            try:
                used = self.nvml.nvmlDeviceGetMemoryInfo(self.handle).used
                self.peak = max(self.peak or 0, used)
            except Exception:
                pass

    def start(self):
        def loop():
            while not self.stop_event.wait(0.1):
                self.sample()
        self.sample()
        self.thread = threading.Thread(target=loop, daemon=True)
        self.thread.start()

    def finish(self):
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        self.sample()
        torch = self.torch
        free, total = torch.cuda.mem_get_info(0)
        props = torch.cuda.get_device_properties(0)
        return {
            "gpu_name": props.name, "gpu_total_bytes": total,
            "cuda_compute_capability": list(torch.cuda.get_device_capability(0)),
            "allocated_bytes": torch.cuda.memory_allocated(0),
            "reserved_bytes": torch.cuda.memory_reserved(0),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(0),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(0),
            "device_used_bytes_at_end": total - free,
            "sampled_device_peak_bytes": self.peak,
            "device_peak_source": "NVML, sampled every 100 ms" if self.peak is not None else "unavailable",
        }


def error_code(exc):
    if isinstance(exc, ModelFailure):
        return exc.code
    name = type(exc).__name__
    if "OutOfMemory" in name:
        return "gpu_out_of_memory"
    if name in {"GatedRepoError", "RepositoryNotFoundError", "HfHubHTTPError"}:
        return "model_access_or_download_error"
    if isinstance(exc, ImportError):
        return "dependency_error"
    if isinstance(exc, ValueError):
        return "model_configuration_error"
    return "model_execution_error"


class Runner:
    def __init__(self, model_key):
        self.model_key = model_key
        self.spec = MODELS[model_key]
        self.model = self.processor = self.pipe = None
        self.torch = None
        self.offload = os.getenv("SOFT_REFUSAL_FLUX_OFFLOAD", "0") == "1"
        self.device = "cuda:0"

    def on_gpu(self):
        return self.device.startswith("cuda")

    def synchronize(self):
        if self.on_gpu():
            self.torch.cuda.synchronize()

    def cancelled(self):
        return cancel_marker(os.environ["SOFT_REFUSAL_DATA"], self.job_id).exists()

    def decoder_blocks(self):
        """Explicit per-architecture block paths; never index hidden_states blindly."""
        if self.model_key == "gemma":
            return self.model.model.language_model.layers, "model.language_model.layers"
        return self.model.model.layers, "model.layers"

    def recording_metadata(self, tokenizer):
        template = getattr(self.processor, "chat_template", None) or tokenizer.chat_template
        return {"chat_template": template,
                "model_config": json.loads(self.model.config.to_json_string(use_diff=False)),
                "generation_config": json.loads(self.model.generation_config.to_json_string(use_diff=False)),
                "attn_implementation": getattr(self.model.config, "_attn_implementation", None)}

    def progress(self, phase, **details):
        try:
            write_progress(self.job_id, phase, model=self.model_key, **details)
        except OSError:
            pass  # Progress display failure must not change a model outcome.

    def load(self):
        import torch
        self.torch = torch
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ModelFailure("exactly_one_cuda_gpu_required")
        if not torch.cuda.is_bf16_supported():
            raise ModelFailure("bf16_not_supported")
        torch.cuda.init()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        memory = MemorySampler(torch)
        memory.start()
        start = time.monotonic()
        try:
            common = {"revision": self.spec["revision"], "torch_dtype": torch.bfloat16,
                      "local_files_only": True}
            if self.model_key == "flux":
                from diffusers import Flux2KleinPipeline
                self.pipe = Flux2KleinPipeline.from_pretrained(self.spec["repo"], **common)
                if self.offload:
                    self.pipe.enable_model_cpu_offload()
                else:
                    self.pipe.to("cuda:0")
                self.pipe.set_progress_bar_config(disable=True)
            elif self.model_key == "gemma":
                from transformers import AutoProcessor, Gemma3ForConditionalGeneration
                self.processor = AutoProcessor.from_pretrained(
                    self.spec["repo"], revision=self.spec["revision"], local_files_only=True,
                    trust_remote_code=False)
                self.model = Gemma3ForConditionalGeneration.from_pretrained(
                    self.spec["repo"], device_map={"": "cuda:0"}, trust_remote_code=False,
                    use_safetensors=True, attn_implementation="sdpa", **common).eval()
            else:
                from transformers import AutoModelForCausalLM, AutoTokenizer
                self.processor = AutoTokenizer.from_pretrained(
                    self.spec["repo"], revision=self.spec["revision"], local_files_only=True,
                    trust_remote_code=False)
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.spec["repo"], device_map={"": "cuda:0"}, trust_remote_code=False,
                    use_safetensors=True, attn_implementation="sdpa", **common).eval()
            torch.cuda.synchronize()
        finally:
            self.load_memory = memory.finish()
        return time.monotonic() - start

    def chat(self, payload):
        torch = self.torch
        import numpy as np
        from transformers import LogitsProcessor, LogitsProcessorList, StoppingCriteria, StoppingCriteriaList
        from transformers.generation.streamers import BaseStreamer
        started = time.monotonic()
        runner = self
        record_layers = payload.get("record_layers") or []
        tracing = bool(payload.get("trace"))
        # Measured summaries only; the full recorded vectors go to the .npz file.
        trace = {"snapshots": [], "layers": [], "candidates": []}
        def live_trace():
            if not (tracing or record_layers):
                return {}
            return {"trace": {**trace, "candidates": trace["candidates"][-PROGRESS_CANDIDATES:]}}
        class FirstToken(BaseStreamer):
            def __init__(self):
                self.prompt_seen = False
                self.first = None
                self.tokens = []
                self.last_update = 0
            def put(self, value):
                if not self.prompt_seen:
                    self.prompt_seen = True
                elif self.first is None:
                    self.first = time.monotonic() - started
                if self.prompt_seen and self.first is not None:
                    self.tokens.extend(value.reshape(-1).tolist())
                    if time.monotonic() - self.last_update >= 0.25:
                        self.publish()
            def publish(self):
                self.last_update = time.monotonic()
                runner.progress("generating", text=tokenizer.decode(self.tokens, skip_special_tokens=True),
                                output_tokens=len(self.tokens), elapsed_seconds=time.monotonic() - started,
                                input_tokens=input_length, **live_trace())
            def end(self):
                self.publish()
        class CandidateObserver(LogitsProcessor):
            def __call__(self, input_ids, scores):
                probs = torch.softmax(scores[0].float(), dim=-1)
                values, indices = torch.topk(probs, min(5, len(probs)))
                trace["candidates"].append({
                    "step": len(trace["candidates"]),
                    "top": [{"id": int(i), "piece": tokenizer.convert_ids_to_tokens(int(i)), "probability": float(v)}
                            for v, i in zip(values, indices)],
                    "selection": "greedy_argmax_after_generation_processors" if not kwargs["do_sample"]
                                 else "sampled_after_generation_processors"})
                return scores  # Observation only: never modify the generation distribution.
        class StopRequested(StoppingCriteria):
            def __init__(self):
                self.checked = 0
            def __call__(self, input_ids, scores, **kw):
                # A file check per token is cheap but the volume is networked; throttle it.
                if time.monotonic() - self.checked < 0.25:
                    return False
                self.checked = time.monotonic()
                return runner.cancelled()
        messages = payload["messages"]
        if self.model_key == "gemma":
            messages = [{"role": m["role"], "content": [{"type": "text", "text": m["content"]}]}
                        for m in messages]
        try:
            inputs = self.processor.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True, return_dict=True,
                return_tensors="pt", truncation=False)
        except Exception as exc:
            raise ModelFailure("invalid_chat_template") from exc
        input_length = inputs["input_ids"].shape[-1]
        input_limit = int(os.getenv("SOFT_REFUSAL_CHAT_INPUT_LIMIT", "8192"))
        if input_length > input_limit:
            raise ModelFailure("input_too_long", input_tokens=input_length, limit_tokens=input_limit,
                               message_count=len(payload["messages"]))
        blocks, module_path = self.decoder_blocks()
        if any(layer >= len(blocks) for layer in record_layers):
            raise ModelFailure("invalid_record_layers", num_layers=len(blocks))
        inputs = inputs.to(self.device)
        torch.manual_seed(payload["seed"])
        if self.on_gpu():
            torch.cuda.manual_seed_all(payload["seed"])
        tracker = FirstToken()
        kwargs = {"max_new_tokens": payload["max_tokens"], "do_sample": payload["temperature"] > 0,
                  "streamer": tracker, "stopping_criteria": StoppingCriteriaList([StopRequested()])}
        if kwargs["do_sample"]:
            kwargs.update(temperature=payload["temperature"], top_p=payload["top_p"])
        if tracing:
            kwargs["logits_processor"] = LogitsProcessorList([CandidateObserver()])
        tokenizer = getattr(self.processor, "tokenizer", self.processor)
        original_eos = self.model.generation_config.eos_token_id
        effective_eos = chat_stop_ids(self.model_key, original_eos, tokenizer.eos_token_id)
        if self.model_key == "hermes":
            kwargs["eos_token_id"] = effective_eos
        prompt_ids = inputs["input_ids"][0].tolist()
        rendered_prompt = tokenizer.decode(prompt_ids, skip_special_tokens=False)
        index = input_length - 1  # Final formatted-prompt token: the pre-response readout.
        if tracing:
            trace["tokens"] = {"total": input_length, "scope": "last_32_formatted_prompt_tokens", "shown": [
                {"position": i, "id": prompt_ids[i], "piece": tokenizer.convert_ids_to_tokens(prompt_ids[i])}
                for i in range(max(0, input_length - 32), input_length)]}
        captured = {}
        def prefill_hidden(output):
            hidden = output[0] if isinstance(output, tuple) else output
            if hidden.ndim != 3 or hidden.shape[0] != 1 or hidden.shape[1] != input_length:
                raise ModelFailure("activation_capture_failed")
            return hidden
        def record_hook(layer):
            def hook(module, args, output):
                if layer in captured:
                    return  # Later cached one-token decoding passes are not the readout.
                vector = prefill_hidden(output)[0, index].detach().float().cpu().numpy().copy()
                if not np.isfinite(vector).all():
                    raise ModelFailure("non_finite_activation", layer=layer)
                captured[layer] = vector
                trace["snapshots"].append({"layer": layer, "dimensions": len(vector), "preview": [
                    float(part.mean()) for part in np.array_split(vector, min(32, len(vector)))]})
            return hook
        def layer_pass_hook(layer):
            def hook(module, args, output):
                if len(trace["layers"]) > layer:
                    return
                vector = prefill_hidden(output)[0, index].detach().float().cpu().numpy()
                trace["layers"].append({"layer": layer, "magnitude": float(np.linalg.norm(vector)),
                                        "preview": [float(part.mean()) for part in np.array_split(vector, 16)],
                                        "total_layers": len(blocks)})
            return hook
        def embedding_hook(module, args, output):
            if "embeddings" in trace:
                return
            matrix = output[0, -24:].detach().float().cpu().numpy()
            preview = np.stack([part.mean(axis=1) for part in np.array_split(matrix, 16, axis=1)], axis=1)
            trace["embeddings"] = {"values": preview.tolist(), "dimensions": matrix.shape[1],
                                   "first_position": input_length - len(matrix),
                                   "scope": "actual_embedding_module_output_last_24_tokens_16_column_means"}
        handles = [blocks[layer].register_forward_hook(record_hook(layer)) for layer in record_layers]
        if tracing:
            handles.append(self.model.get_input_embeddings().register_forward_hook(embedding_hook))
            handles.extend(block.register_forward_hook(layer_pass_hook(i)) for i, block in enumerate(blocks))
        self.progress("generating", text="", input_tokens=input_length, limit_tokens=input_limit, output_tokens=0)
        try:
            if self.cancelled():
                raise ModelFailure("cancelled_by_user")
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **kwargs)
        finally:
            for handle in handles:
                handle.remove()
        self.synchronize()
        if self.cancelled():
            raise ModelFailure("cancelled_by_user")
        tokens = outputs[0, input_length:].tolist()
        text = tokenizer.decode(tokens, skip_special_tokens=True)
        eos_ids = set(effective_eos)
        ended = bool(tokens and tokens[-1] in eos_ids)
        elapsed = time.monotonic() - started
        recording = {}
        if record_layers:
            if set(captured) != set(record_layers):
                raise ModelFailure("activation_capture_failed")
            path = Path(os.environ["SOFT_REFUSAL_DATA"]) / "activations" / f"{self.job_id}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_name(path.name + ".tmp")
            with temporary.open("wb") as stream:
                np.savez_compressed(stream, **{f"layer_{layer}": captured[layer] for layer in record_layers})
            temporary.replace(path)
            recording["activations"] = {
                "available": True, "url": f"/v1/jobs/{self.job_id}/activations",
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "layers": record_layers, "layer_numbering": "zero_based", "num_layers": len(blocks),
                "dimensions": len(captured[record_layers[0]]), "storage_dtype": "float32",
                "site": "resid_post", "module_path": module_path, "token_selection": "last_prompt",
                "token_index": index, "token_id": prompt_ids[index],
                "token_text": tokenizer.decode([prompt_ids[index]]),
                "capture_execution": "generation_prefill"}
            recording["snapshot_preview"] = trace["snapshots"]
            recording["recording_metadata"] = self.recording_metadata(tokenizer)
        if tracing:
            recording["education_trace"] = {k: trace[k] for k in ("tokens", "embeddings", "layers", "candidates")
                                            if k in trace}
        return {**recording, "text": text, "outcome": "unlabeled", "label_source": None,
                "finish_reason": "stop" if ended else "length",
                "ended_with_eos": ended, "hit_token_limit": len(tokens) >= payload["max_tokens"] and not ended,
                "input_token_ids": inputs["input_ids"][0].tolist(), "output_token_ids": tokens,
                "rendered_prompt": rendered_prompt,
                "usage": {"prompt_tokens": input_length, "completion_tokens": len(tokens),
                          "total_tokens": input_length + len(tokens)},
                "generation_seconds": elapsed, "first_token_seconds": tracker.first,
                "tokens_per_second": len(tokens) / elapsed if elapsed else None,
                "stop_token_settings": {"original_generation_eos": original_eos,
                                        "effective_eos": effective_eos,
                                        "reason": "native_chatml_turn_end" if self.model_key == "hermes" else "original_generation_config"},
                "filter_signal": "not_reported_by_model"}

    def image(self, payload, job_id):
        torch = self.torch
        # Validate the SAME formatted text the baseline pipeline encodes before
        # permitting its fixed context limit to truncate anything.
        tokenizer = self.pipe.tokenizer
        messages = [{"role": "user", "content": payload["prompt"]}]
        # Matches reviewed Diffusers 0.37.1 _get_qwen3_prompt_embeds exactly.
        formatted = tokenizer.apply_chat_template(messages, tokenize=False,
                                                 add_generation_prompt=True, enable_thinking=False)
        count = len(tokenizer(formatted, truncation=False)["input_ids"])
        if count > 512:
            raise ModelFailure("input_too_long", input_tokens=count, limit_tokens=512)
        with image_adapter(self.pipe, payload.get("lora", "none"), payload.get("lora_strength", 0.7)) as adapter:
            with torch.inference_mode():
                output = self.pipe(prompt=payload["prompt"], height=payload["height"], width=payload["width"],
                                   guidance_scale=1.0, num_inference_steps=4, max_sequence_length=512,
                                   generator=torch.Generator(device="cuda").manual_seed(payload["seed"]))
        torch.cuda.synchronize()
        flags = getattr(output, "nsfw_content_detected", None)
        flags = [bool(flag) for flag in flags] if flags is not None else None
        flagged = flags is not None and any(flags)
        if flagged:
            return {"image_available": False, "outcome": "pipeline_blocked",
                    "label_source": "pipeline_filter", "filter_signal": flags}
        if not output.images or output.images[0] is None:
            raise ModelFailure("image_output_missing")
        path = Path(os.environ["SOFT_REFUSAL_DATA"]) / "images" / f"{job_id}.png"
        temporary = path.with_suffix(".tmp")
        output.images[0].save(temporary, format="PNG")
        temporary.replace(path)
        return {"image_available": True, "outcome": "unlabeled", "label_source": None,
                "lora": adapter,
                "width": payload["width"], "height": payload["height"],
                "image_sha256": __import__("hashlib").sha256(path.read_bytes()).hexdigest(),
                "filter_signal": flags if flags is not None else "not_reported_by_pipeline",
                "result_url": f"/v1/jobs/{job_id}/result"}

    def run(self, job_id, payload):
        self.job_id = job_id
        result = {"job_id": job_id, "model": self.spec["repo"], "revision": self.spec["revision"],
                  "precision": "bfloat16", "quantization": None,
                  "offload": "model_cpu_offload" if self.model_key == "flux" and self.offload else None,
                  "generation_settings": payload,
                  "packages": {name: importlib.metadata.version(name) for name in
                               ("torch", "transformers", "diffusers", "accelerate", "huggingface-hub")},
                  "metrics": {}}
        memory = None
        try:
            cold = self.model is None and self.pipe is None
            self.progress("loading" if cold else "preparing", cold_load=cold)
            result["metrics"]["cold_load"] = cold
            result["metrics"]["model_load_seconds"] = self.load() if cold else 0
            result["metrics"]["load_memory"] = self.load_memory if cold else None
            torch = self.torch
            if self.on_gpu():
                torch.cuda.reset_peak_memory_stats()
                memory = MemorySampler(torch)
                memory.start()
            start = time.monotonic()
            self.progress("generating", text="", output_tokens=0)
            result.update(self.image(payload, job_id) if self.model_key == "flux" else self.chat(payload))
            result["metrics"]["inference_seconds"] = time.monotonic() - start
        except Exception as exc:
            code = error_code(exc)
            cancelled = code == "cancelled_by_user"
            result.update(outcome="cancelled" if cancelled else "technical_error",
                          label_source="user" if cancelled else "infrastructure",
                          error={"code": code, "type": type(exc).__name__,
                                 **getattr(exc, "details", {})})
        finally:
            if memory:
                result["metrics"]["inference_memory"] = memory.finish()
            elif hasattr(self, "load_memory"):
                result["metrics"]["load_memory"] = self.load_memory
        return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=list(MODELS), required=True)
    args = parser.parse_args()
    runner = Runner(args.model)
    for line in sys.stdin:
        command = json.loads(line)
        # Third-party library stdout cannot corrupt the supervisor protocol.
        with contextlib.redirect_stdout(sys.stderr):
            result = runner.run(command["id"], command["payload"])
        sys.stdout.write(json.dumps(result, allow_nan=False) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
