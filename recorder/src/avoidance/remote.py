"""Cloud GPU adapter. The soft-refusal backend runs Gemma and takes the snapshots
during its prefill; this computer keeps the conversation and the recording.

Same step() contract as model.HFAdapter, so ChatService saves identical files."""
from __future__ import annotations

import hashlib
import io
import json
import os
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from urllib.parse import urlsplit

import numpy as np

from .io import read_json
from .exceptions import GenerationCancelled

ROOT = Path(__file__).resolve().parents[2]
CONNECTION_FILE = ROOT / "secrets" / "gpu-connection.json"
MAX_REPLY_TOKENS = 1024  # The backend's per-reply cap.
POLL_SECONDS = 0.75


class RemoteError(RuntimeError):
    pass


def load_connection(path=CONNECTION_FILE):
    """{"api_url": ..., "access_code_file": ...}; the code itself never enters recordings."""
    path = Path(path)
    if not path.is_file():
        raise RemoteError(f"Cloud GPU connection is not set up. Create {path} (see docs/cloud-gpu.md).")
    connection = read_json(path)
    url = os.getenv("SOFT_REFUSAL_API_URL") or connection.get("api_url", "")
    if not url.startswith("https://") and not url.startswith("http://127.0.0.1"):
        raise RemoteError("The cloud API address in secrets/gpu-connection.json must start with https://")
    code_file = connection.get("access_code_file")
    if code_file:
        code = (path.parent / code_file).read_text(encoding="utf-8").strip()
    else:
        code = connection.get("access_code", "").strip()
    if not code:
        raise RemoteError("No app access code found for the cloud API. See docs/cloud-gpu.md.")
    return {"api_url": url.rstrip("/"), "access_code": code}


class RemoteAdapter:
    def __init__(self, config, connection=None):
        self.config = config
        connection = connection or load_connection()
        self.base, self._code = connection["api_url"], connection["access_code"]
        self.host = urlsplit(self.base).netloc
        self.module_path = None
        self.metadata = None
        self.hardware = None
        self.packages = None
        self.attn_implementation = None
        status, health = self._request("GET", "/v1/health")
        if status != 200:
            raise RemoteError(self._explain(status, health))
        if not health.get("models", {}).get(config["remote_model"], {}).get("enabled"):
            raise RemoteError(f"The cloud API does not offer the model '{config['remote_model']}'.")
        if health.get("status") != "accepting":
            raise RemoteError("The cloud API is running but paused. Restart it with SOFT_REFUSAL_ACCEPT_REQUESTS=1.")

    # -- HTTP ---------------------------------------------------------------
    def _request(self, method, path, body=None, headers=None, raw=False, timeout=30):
        data = json.dumps(body).encode() if body is not None else None
        request = urllib.request.Request(self.base + path, data=data, method=method, headers={
            "Authorization": f"Bearer {self._code}",
            # RunPod's proxy rejects Python's default "Python-urllib" agent with 403.
            "User-Agent": "avoidance-recording-room/0.3",
            **({"Content-Type": "application/json"} if data is not None else {}), **(headers or {})})
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                payload = response.read()
                return response.status, payload if raw and response.status == 200 else json.loads(payload)
        except urllib.error.HTTPError as exc:
            try:
                return exc.code, json.loads(exc.read())
            except ValueError:
                # RunPod's proxy answers with an HTML page while the Pod or API is down.
                raise RemoteError(self._unreachable()) from None
        except (urllib.error.URLError, OSError, ValueError):
            raise RemoteError(self._unreachable()) from None

    def _unreachable(self):
        return (f"Can't reach the cloud GPU at {self.host}. Check that the RunPod Pod is running and the API "
                "was started. Your earlier recordings are safe.")

    @staticmethod
    def _code_of(payload):
        if not isinstance(payload, dict):
            return None
        for key in ("detail", "error"):
            if isinstance(payload.get(key), dict):
                return payload[key].get("code")
        return None

    def _explain(self, status, payload):
        code = self._code_of(payload)
        messages = {
            401: "The cloud API did not accept the access code in secrets/gpu-connection.json.",
            413: "This conversation is too large for one cloud request. Start a new conversation.",
            422: "The cloud API rejected this conversation (each message must be under 12,000 characters, "
                 "at most 64 messages and 24,000 characters in total). Start a new conversation.",
        }
        if code == "submissions_paused":
            return "The cloud API is running but paused. Restart it with SOFT_REFUSAL_ACCEPT_REQUESTS=1."
        if status == 429:
            return "The cloud GPU queue is full right now. Wait a moment and send again."
        return messages.get(status, f"The cloud API answered {status} ({code or 'no detail'}).")

    # -- One recorded turn --------------------------------------------------
    def step(self, messages, on_event=None, cancel_event=None, max_new_tokens=None):
        limit = max_new_tokens or self.config["max_new_tokens"]
        if limit > MAX_REPLY_TOKENS:
            raise ValueError("The cloud GPU caps replies at 1,024 tokens. Choose a shorter reply length.")
        emit = on_event or (lambda kind, **data: None)
        body = {"model": self.config["remote_model"], "messages": messages, "max_tokens": limit,
                "temperature": 0, "seed": self.config["seed"], "record_layers": self.config["layers"],
                "trace": on_event is not None}
        status, job = self._request("POST", "/v1/chat/jobs", body, {"Idempotency-Key": uuid.uuid4().hex})
        if status != 202:
            raise RemoteError(self._explain(status, job))
        relay = Relay(emit)
        while job["status"] in ("queued", "running"):
            relay.live(job)
            if cancel_event is not None and cancel_event.wait(POLL_SECONDS):
                self._request("POST", f"/v1/jobs/{job['job_id']}/cancel")
                raise GenerationCancelled("Stopped; this unfinished turn was not added to the recorded conversation.")
            if cancel_event is None:
                time.sleep(POLL_SECONDS)
            status, job = self._request("GET", f"/v1/jobs/{job['job_id']}")
            if status != 200:
                raise RemoteError(self._explain(status, job))
        result = job["result"] or {}
        if job["status"] != "succeeded":
            raise RemoteError(self._failure(job["status"], result))
        if (result.get("model"), result.get("revision")) != (self.config["model_id"], self.config["revision"]):
            raise RemoteError("The cloud GPU is serving a different Gemma checkpoint than this recording "
                              "configuration. Nothing was saved.")
        relay.final(result)
        vectors = self._download(job["job_id"], result["activations"])
        self._remember(result)
        return self._row(job["job_id"], result, limit), vectors

    def _failure(self, status, result):
        error = result.get("error") or {}
        if error.get("code") == "input_too_long":
            # ChatService turns this wording into its friendly start-a-new-conversation message.
            return (f"{error.get('input_tokens')} input tokens exceeds limit {error.get('limit_tokens')}; "
                    "no silent truncation is permitted")
        if status == "cancelled":
            return "The reply was stopped on the cloud GPU before it finished."
        return f"The cloud GPU could not finish this reply ({error.get('code', status)}). Nothing was saved."

    def _download(self, job_id, meta):
        status, payload = self._request("GET", f"/v1/jobs/{job_id}/activations", raw=True, timeout=60)
        if status != 200:
            raise RemoteError(f"The reply finished but its snapshots could not be downloaded ({status}).")
        if hashlib.sha256(payload).hexdigest() != meta["sha256"]:
            raise RemoteError("Downloaded snapshots failed their checksum. Nothing was saved.")
        with np.load(io.BytesIO(payload), allow_pickle=False) as data:
            vectors = {layer: data[f"layer_{layer}"].astype(np.float32) for layer in self.config["layers"]}
        if any(v.ndim != 1 or len(v) != meta["dimensions"] or not np.isfinite(v).all() for v in vectors.values()):
            raise RemoteError("Downloaded snapshots are invalid. Nothing was saved.")
        return vectors

    def _remember(self, result):
        """Details ChatService writes beside the checkpoint, as it does for a local model."""
        meta = result["recording_metadata"]
        self.metadata = (meta["chat_template"], meta["model_config"], meta["generation_config"])
        self.module_path = result["activations"]["module_path"]
        memory = (result.get("metrics") or {}).get("inference_memory") or {}
        self.hardware = {"device": "remote_gpu", "gpu": memory.get("gpu_name"), "api_host": self.host}
        self.packages = result.get("packages")
        self.attn_implementation = meta.get("attn_implementation")

    def recording_metadata(self):
        return self.metadata

    def manifest_extra(self):
        return {"hardware": self.hardware, "execution": "remote_gpu", "remote_packages": self.packages,
                "remote_attn_implementation": self.attn_implementation}

    def _row(self, job_id, result, limit):
        act, usage = result["activations"], result["usage"]
        return {
            "rendered_prompt": result["rendered_prompt"], "prompt_token_ids": result["input_token_ids"],
            "response": result["text"], "response_token_ids": result["output_token_ids"], "phase": "pre_response",
            "token_index": act["token_index"], "token_id": act["token_id"], "token_text": act["token_text"],
            "input_token_count": usage["prompt_tokens"], "generated_token_count": usage["completion_tokens"],
            "ended_with_eos": result["ended_with_eos"], "hit_token_limit": result["hit_token_limit"],
            "generation_settings": {"max_new_tokens": limit, "do_sample": False, "num_beams": 1},
            "capture_execution": act["capture_execution"],
            "education_trace": result.get("education_trace"),
            "remote": {"api_host": self.host, "job_id": job_id, "precision": result.get("precision"),
                       "quantization": result.get("quantization"), "activation_sha256": act["sha256"],
                       "metrics": result.get("metrics")},
        }


class Relay:
    """Turns the backend's polled progress into the recording room's live events, once each and in order."""

    def __init__(self, emit):
        self.emit = emit
        self.queue_position = None
        self.done = set()
        self.layers = 0
        self.snapshots = set()
        self.candidates = 0
        self.text = ""
        self.tokens = 0

    def once(self, key, kind, **data):
        if key not in self.done:
            self.done.add(key)
            self.emit(kind, **data)

    def live(self, job):
        if job["status"] == "queued":
            position = job.get("queue_position")
            if position != self.queue_position:
                self.queue_position = position
                self.emit("queued", message=f"Waiting for the cloud GPU · position {position} in its queue.")
            return
        progress = job.get("progress") or {}
        if progress.get("phase") == "loading":
            self.once("loading", "loading", message="The cloud GPU is loading Gemma 3 12B. This takes about a "
                                                    "minute after a pause or a model switch.")
        self.relay(progress, final=False)

    def final(self, result):
        trace = dict(result.get("education_trace") or {}, snapshots=result.get("snapshot_preview") or [])
        self.relay({"input_tokens": result["usage"]["prompt_tokens"], "trace": trace, "text": result["text"],
                    "output_tokens": result["usage"]["completion_tokens"]}, final=True)

    def relay(self, progress, final):
        if progress.get("input_tokens"):
            self.once("prepared", "prepared", input_tokens=progress["input_tokens"],
                      message="Read the conversation and prepared the next reply.")
        trace = progress.get("trace") or {}
        if trace.get("tokens"):
            total = trace["tokens"]["total"]
            self.once("tokenized", "tokenized", **trace["tokens"],
                      message=f"Split the formatted conversation into {total} tokens (text pieces).")
        if trace.get("embeddings"):
            self.once("embeddings", "embeddings", **trace["embeddings"],
                      message="Looked up the starting number vector for each token.")
        if trace.get("layers") or trace.get("snapshots"):
            self.once("capturing", "capturing", message="Taking snapshots of the conversation at the selected layers.")
        for entry in trace.get("layers", [])[self.layers:]:
            self.emit("layer_pass", **entry)
        self.layers = max(self.layers, len(trace.get("layers", [])))
        for snap in trace.get("snapshots", []):
            if snap["layer"] not in self.snapshots:
                self.snapshots.add(snap["layer"])
                self.emit("snapshot", **snap, message=f"Captured layer {snap['layer']}: {snap['dimensions']:,} numbers.")
        candidates = trace.get("candidates", [])
        if candidates or progress.get("output_tokens"):
            self.once("generating", "generating", message="Snapshots captured during Gemma's first reading of the "
                                                          "conversation. Gemma is now writing.")
        for entry in candidates:
            # Live progress shows a recent window. Keep decisions in order; the final result fills any gap.
            if entry["step"] == self.candidates:
                self.emit("candidates", **entry)
                self.candidates += 1
        text = progress.get("text") or ""
        if text.startswith(self.text) and len(text) > len(self.text):
            self.emit("text", text=text[len(self.text):])
            self.text = text
        tokens = progress.get("output_tokens") or 0
        if tokens > self.tokens:
            self.tokens = tokens
            self.emit("progress", generated_tokens=tokens)
