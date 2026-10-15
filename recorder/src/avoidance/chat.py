"""Loopback-only chat and live recording. Immutable checkpoints, atomic session pointers."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import os
import re
import threading
import time
import uuid
import zipfile
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import numpy as np

from .io import digest, environment, load_dataset, read_json, read_jsonl, write_json, write_jsonl
from .labels import OUTCOMES, label_records

DEFAULT_INSTRUCTION = "Keep your reply under 50 words, in two or three sentences."


def now():
    return datetime.now(timezone.utc).isoformat()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    write_json(temp, value)
    os.replace(temp, path)


class ChatService:
    def __init__(self, config, root, adapter_factory=None, auto_label=True):
        self.config = dict(config)
        if self.config["token_selection"] != "last_prompt":
            raise ValueError("Live chat currently requires last_prompt snapshots")
        self.root = Path(root).resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.adapter_factory = adapter_factory
        self.adapter = None
        self.lock = threading.RLock()
        self.cancel = threading.Event()
        self.job = None
        self.closing = False
        self.sessions = {}
        self.remote = self.config.get("adapter") == "remote"
        self.auto_label_enabled = auto_label and self.remote
        self.label_task = None
        for path in self.root.glob("*/session.json"):
            meta = read_json(path)
            self.sessions[meta["id"]] = meta

    def create(self, instruction=""):
        self.validate_instruction(instruction)
        with self.lock:
            if self.closing:
                raise ValueError("The app is stopping. Reopen it with the launcher.")
            sid = uuid.uuid4().hex
            meta = {"id": sid, "title": "New conversation", "created_utc": now(), "latest": None,
                    "config": self.config, "recording_policy": "selected_layers_last_prompt", "instruction": instruction.strip()}
            (self.root / sid).mkdir()
            atomic_json(self.root / sid / "session.json", meta)
            self.sessions[sid] = meta
            return sid

    @staticmethod
    def validate_instruction(instruction):
        if not isinstance(instruction, str) or len(instruction) > 4000:
            raise ValueError("Conversation instructions must be text, up to 4,000 characters")

    def set_instruction(self, sid, instruction):
        self.validate_instruction(instruction)
        with self.lock:
            if self.closing or (self.job and self.job["status"] == "running"):
                raise ValueError("Wait for the current reply to finish before changing instructions")
            meta = dict(self._meta(sid), instruction=instruction.strip())
            atomic_json(self.root / sid / "session.json", meta)
            self.sessions[sid] = meta

    def _meta(self, sid):
        if sid not in self.sessions:
            raise ValueError("Conversation not found")
        return self.sessions[sid]

    def _rows(self, sid):
        meta = self._meta(sid)
        if meta["latest"] is None:
            return [], {}
        _, rows, arrays = load_dataset(self.root / sid / meta["latest"])
        return rows, arrays

    def labels(self, sid):
        return {pid: row["outcome"] for pid, row in label_records(self.root / sid).items()}

    def label(self, sid, point_id, outcome):
        if outcome not in (*OUTCOMES, "auto"):
            raise ValueError("Choose answered, redirected, refused, unclear or unlabeled")
        with self.lock:
            if point_id not in {row["point_id"] for row in self._rows(sid)[0]}:
                raise ValueError("That turn is not in this conversation")
            with (self.root / sid / "labels.jsonl").open("a", encoding="utf-8") as f:
                f.write(json.dumps({"point_id": point_id, "outcome": outcome, "label_source": "human",
                                    "annotator": "local", "time": now()}) + "\n")

    def state(self, sid=None):
        with self.lock:
            rows = self._rows(sid)[0] if sid else []
            labels = label_records(self.root / sid) if sid else {}
            job = self.job if self.job and self.job["session_id"] == sid else None
            summaries = [{**{k: row.get(k) for k in ["point_id", "turn_id", "prompt", "response", "hit_token_limit",
                                                     "input_token_count", "generated_token_count", "snapshot_preview", "education_trace", "generation_settings"]},
                          "label": labels.get(row["point_id"], {}).get("outcome", "unlabeled"),
                          "label_source": labels.get(row["point_id"], {}).get("label_source"),
                          "label_reason": labels.get(row["point_id"], {}).get("reason", ""),
                          "label_status": labels.get(row["point_id"], {}).get("status", "complete")} for row in rows]
            if job:
                job = dict(job, events=[e for e in job["events"] if e.get("message")][-80:])
            return json.loads(json.dumps({
                "app": "gemma-recording-room", "mode": "remote" if self.remote else "local",
                "project_root": str(Path(__file__).resolve().parents[2]),
                "model_ready": self.adapter is not None, "config": self.config, "closing": self.closing,
                "sessions": sorted(self.sessions.values(), key=lambda s: s["created_utc"], reverse=True),
                "rows": summaries, "job": job, "busy": self.busy(),
                "auto_label_enabled": self.auto_label_enabled, "label_task": self.label_task,
                "instruction": self._meta(sid).get("instruction", "") if sid else "",
                "recording_path": str(self.root / sid) if sid else str(self.root),
            }))

    def submit(self, sid, prompt, reply_limit=None):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 16000:
            raise ValueError("Please enter a message of 1–16,000 characters")
        with self.lock:
            if reply_limit is None:
                reply_limit = self.config["max_new_tokens"]
            if type(reply_limit) is not int or not 1 <= reply_limit <= 2048:
                raise ValueError("Reply limit must be between 1 and 2,048 tokens")
            if self.closing:
                raise ValueError("The app is stopping. Reopen it with the launcher.")
            meta = self._meta(sid)
            if meta["config"] != self.config:
                raise ValueError("This conversation uses different recording settings. Start a new conversation.")
            if self.busy():
                raise ValueError("Gemma is already working. Wait for it to finish or press Stop.")
            rows, _ = self._rows(sid)
            messages = []
            for row in rows:
                messages.extend([{"role": "user", "content": row["prompt"]}, {"role": "assistant", "content": row["response"]}])
            messages.append({"role": "user", "content": prompt.strip()})
            instruction = meta.get("instruction", "")
            if instruction:
                # Gemma's native template folds this into the first user message.
                messages.insert(0, {"role": "system", "content": instruction})
            jid = uuid.uuid4().hex
            self.job = {"id": jid, "session_id": sid, "status": "running", "stage": "starting",
                        "prompt": prompt.strip(), "response": "", "events": [], "snapshots": [],
                        "started_utc": now(), "started_seconds": time.time(), "generated_tokens": 0,
                        "reply_limit": reply_limit, "education": {"layers": [], "candidates": []},
                        "instruction": instruction,
                        "saved_turns_before": len(rows)}
            self.cancel.clear()
            attempt = self.root / sid / "attempts" / jid
            attempt.mkdir(parents=True)
            write_json(attempt / "request.json", {"messages": messages, "instruction": instruction, "reply_limit": reply_limit, "started_utc": now(), "status": "attempt"})
            worker = threading.Thread(target=self._run, args=(sid, messages, attempt), daemon=True)
            worker.start()
            return jid

    def _emit(self, kind, **data):
        with self.lock:
            event = {"kind": kind, "time": now(), **data}
            self.job["events"].append(event)
            if kind == "text":
                self.job["response"] += data["text"]
            elif kind == "progress":
                self.job["generated_tokens"] = data["generated_tokens"]
            elif kind == "snapshot":
                self.job["snapshots"].append(data)
            elif kind == "tokenized":
                self.job["education"]["tokens"] = data
            elif kind == "embeddings":
                self.job["education"]["embeddings"] = data
            elif kind == "layer_pass":
                self.job["education"]["layers"].append(data)
            elif kind == "candidates":
                self.job["education"]["candidates"].append(data)
            else:
                self.job["stage"] = kind
            path = self.root / self.job["session_id"] / "attempts" / self.job["id"] / "events.jsonl"
            with path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def stop(self):
        with self.lock:
            if self.busy():
                self.cancel.set()

    def busy(self):
        return bool((self.job and self.job["status"] == "running") or
                    (self.label_task and self.label_task["status"] == "running"))

    def _label_saved(self, sid, row):
        from .judging import judge_reply
        verdict = judge_reply(self.adapter, row, self.cancel)
        with self.lock:
            with (self.root / sid / "auto_labels.jsonl").open("a", encoding="utf-8") as stream:
                stream.write(json.dumps({**verdict, "point_id": row["point_id"],
                                        "label_source": "model_judge", "time": now()}) + "\n")
        return verdict

    def auto_label_saved(self, sid):
        """Tag missing/failed labels in an existing conversation, preserving corrections."""
        with self.lock:
            if not self.auto_label_enabled or self.closing or self.busy():
                raise ValueError("Automatic tagging needs an idle cloud Recording Room.")
            labels = label_records(self.root / sid)
            rows = [r for r in self._rows(sid)[0]
                    if r["point_id"] not in labels or labels[r["point_id"]].get("status") == "error"]
            self.cancel.clear()
            self.label_task = {"session_id": sid, "status": "running", "total": len(rows), "done": 0}
            def run():
                try:
                    if self.adapter is None:
                        from .remote import RemoteAdapter
                        self.adapter = (self.adapter_factory or RemoteAdapter)(self.config)
                    for row in rows:
                        if self.cancel.is_set():
                            break
                        verdict = self._label_saved(sid, row)
                        with self.lock:
                            self.label_task["done"] += 1
                        if verdict["status"] == "error":
                            raise ValueError(verdict["reason"])
                except Exception as exc:
                    with self.lock:
                        self.label_task["error"] = str(exc)
                finally:
                    with self.lock:
                        self.label_task["status"] = "complete"
            threading.Thread(target=run, daemon=True).start()

    def _run(self, sid, messages, attempt):
        from .model import HFAdapter, GenerationCancelled
        outcome = "error"
        try:
            if self.adapter is None:
                if self.remote:
                    from .remote import RemoteAdapter
                    self._emit("loading", message="Connecting to the cloud GPU.")
                else:
                    self._emit("loading", message="Loading Gemma from this computer. This happens once when the app starts.")
                self.adapter = (self.adapter_factory or (RemoteAdapter if self.remote else HFAdapter))(self.config)
            if self.cancel.is_set():
                raise GenerationCancelled("Stopped before the reply began.")
            result, vectors = self.adapter.step(messages, on_event=self._emit, cancel_event=self.cancel,
                                                max_new_tokens=self.job["reply_limit"])
            if self.cancel.is_set():
                raise GenerationCancelled("Stopped before saving this turn.")
            self._emit("saving", message="Saving the reply, conversation and all four full snapshots to disk.")
            with self.lock:
                self._save(sid, messages, result, vectors)
                self.job["response"] = result["response"]
                self._emit("saved", message="Recording saved. This turn is ready for later analysis.")
                outcome = "complete"
            if self.auto_label_enabled:
                self._emit("labelling", message="Reply saved. Gemma is separately tagging its response behaviour.")
                try:
                    row = self._rows(sid)[0][-1]
                    verdict = self._label_saved(sid, row)
                    self._emit("saved", message="Automatic tag saved; you can correct it below." if verdict["status"] == "complete"
                               else "Reply saved; automatic tagging needs attention. You can retry or tag manually.")
                except Exception:
                    self._emit("saved", message="Reply saved; automatic tag could not be saved. Retry or tag manually.")
        except Exception as exc:
            with self.lock:
                cancelled = isinstance(exc, GenerationCancelled)
                message = str(exc)
                if "no silent truncation" in message:
                    message = "This conversation has reached the recording length limit. Your previous turns are safe. Start a new conversation; no earlier messages were silently removed."
                self._emit("stopped" if cancelled else "error", message=message)
                outcome = "stopped" if cancelled else "error"
                self.job["error"] = message
        finally:
            with self.lock:
                self.job["status"] = outcome
                write_json(attempt / "result.json", self.job)

    def _save(self, sid, messages, result, vectors):
        meta = self._meta(sid)
        rows, arrays = self._rows(sid)
        turn = len(rows)
        if set(vectors) != set(self.config["layers"]):
            raise ValueError("Recording incomplete: missing layer snapshot")
        for layer, vector in vectors.items():
            if vector.ndim != 1 or not np.isfinite(vector).all():
                raise ValueError("Recording failed: invalid snapshot")
            key = f"layer_{layer}"
            arrays[key] = np.vstack([arrays[key], vector]) if key in arrays else vector[np.newaxis, :]
        row = {"point_id": hashlib.sha256(f"{sid}\0{turn}".encode()).hexdigest()[:24],
               "activation_row": turn, "conversation_id": sid, "group_id": sid, "turn_id": turn,
               "prompt": messages[-1]["content"], "messages": messages, "outcome": "unlabeled", "label_source": None,
               "instruction": self.job["instruction"], "instruction_placement": "native_chat_template_system_prefix" if self.job["instruction"] else "none",
               "model_id": self.config["model_id"], "model_revision": self.config["revision"],
               "layers": self.config["layers"], "site": "resid_post", "token_selection": "last_prompt",
               "synthetic": False, "snapshot_preview": self.job["snapshots"],
               "recording_attempt_id": self.job["id"], **result}
        rows.append(row)
        relative = f"checkpoints/{turn + 1:06d}-{self.job['id'][:8]}"
        destination = self.root / sid / relative
        destination.mkdir(parents=True)
        write_jsonl(destination / "steps.jsonl", rows)
        np.savez_compressed(destination / "activations.npz", **arrays)
        write_jsonl(destination / "input.jsonl", [{"conversation_id": sid, "group_id": sid, "turns": [r["prompt"] for r in rows]}])
        if hasattr(self.adapter, "recording_metadata"):
            template, model_config, generation_config = self.adapter.recording_metadata()
        else:
            template = self.adapter.tokenizer.get_chat_template()
            model_config = self.adapter.model.config.to_dict()
            generation_config = self.adapter.model.generation_config.to_dict()
        (destination / "chat_template.txt").write_text(template, encoding="utf-8")
        write_json(destination / "model_config.json", model_config)
        write_json(destination / "generation_config.json", generation_config)
        with (destination / "annotations.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["point_id", "outcome", "annotator", "split", "notes"])
            writer.writeheader()
            writer.writerows({"point_id": r["point_id"], "outcome": "unlabeled"} for r in rows)
        import torch
        files = ["steps.jsonl", "activations.npz", "input.jsonl", "chat_template.txt", "model_config.json", "generation_config.json"]
        manifest = {"schema_version": 1, "status": "complete", "synthetic": False, "created_utc": now(),
                    "config": self.config, "environment": environment(), "site": "resid_post",
                    "layer_numbering": "zero_based", "storage_dtype": "float32", "point_count": len(rows),
                    "module_path": self.adapter.module_path, "input_sha256": digest(destination / "input.jsonl"),
                    "chat_template_sha256": hashlib.sha256(template.encode()).hexdigest(),
                    "hashes": {name: digest(destination / name) for name in files},
                    "source_sha256": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                    "generation_overrides": {"do_sample": False, "num_beams": 1, "max_new_tokens": "per_turn: see generation_settings in steps.jsonl", "use_cache": True},
                    "hardware": {"device": self.config["device"], "cuda": torch.version.cuda},
                    "collection_mode": "interactive_exploration", "capture_execution": "per_turn: see steps.jsonl; legacy rows use dedicated pre-response pass"}
        if hasattr(self.adapter, "manifest_extra"):
            manifest.update(self.adapter.manifest_extra())
        write_json(destination / "manifest.json", manifest)
        load_dataset(destination)  # Verify the complete checkpoint before advancing the session pointer.
        updated = dict(meta, latest=relative, title=rows[0]["prompt"][:60])
        atomic_json(self.root / sid / "session.json", updated)
        self.sessions[sid] = updated

    def export(self, sid):
        with self.lock:
            if self.closing:
                raise ValueError("The app is stopping. Reopen it with the launcher.")
            meta = self._meta(sid)
            if not meta["latest"]:
                raise ValueError("Record a complete turn before exporting")
            directory = self.root / sid / meta["latest"]
            load_dataset(directory)
            buffer = io.BytesIO()
            with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
                for p in directory.iterdir():
                    if p.is_file():
                        archive.write(p, p.name)
                for name in ("labels.jsonl", "auto_labels.jsonl"):
                    path = self.root / sid / name
                    if path.exists():
                        archive.write(path, name)
                for row in read_jsonl(directory / "steps.jsonl"):
                    attempt = self.root / sid / "attempts" / row["recording_attempt_id"]
                    for p in attempt.glob("*.json*"):
                        archive.write(p, f"recording-log/{attempt.name}/{p.name}")
            return buffer.getvalue()


def make_handler(service):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass  # Never put prompts or response contents in access logs.

        def respond(self, data, status=200, content_type="application/json; charset=utf-8", filename=None):
            raw = json.dumps(data, ensure_ascii=False).encode() if isinstance(data, (dict, list)) else data
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(raw)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'self'; script-src 'self'; style-src 'self'; connect-src 'self'; img-src 'self' data:; frame-ancestors 'none'")
            if filename:
                self.send_header("Content-Disposition", f'attachment; filename="{filename}"')
            self.end_headers()
            self.wfile.write(raw)

        def trusted(self):
            allowed = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
            host = self.headers.get("Host", "")
            origin = self.headers.get("Origin")
            return host in allowed and (not origin or origin in {"http://" + x for x in allowed})

        def do_GET(self):
            if not self.trusted():
                return self.respond({"error": "This app only accepts requests from its local page."}, 403)
            url = urlsplit(self.path)
            sid = parse_qs(url.query).get("session", [None])[0]
            try:
                if url.path == "/api/state":
                    return self.respond(service.state(sid))
                if url.path == "/api/export":
                    return self.respond(service.export(sid), content_type="application/zip", filename=f"gemma-conversation-{sid[:8]}.zip")
                assets = {"/": ("index.html", "text/html; charset=utf-8"), "/app.js": ("app.js", "application/javascript"), "/style.css": ("style.css", "text/css")}
                if url.path in assets:
                    name, mime = assets[url.path]
                    return self.respond((Path(__file__).parent / "web" / name).read_bytes(), content_type=mime)
                self.respond({"error": "Not found"}, 404)
            except ValueError as exc:
                self.respond({"error": str(exc)}, 400)

        def do_POST(self):
            if not self.trusted() or not self.headers.get("Content-Type", "").startswith("application/json"):
                return self.respond({"error": "Request rejected"}, 403)
            try:
                length = int(self.headers.get("Content-Length", "0"))
                if not 0 < length <= 70000:
                    raise ValueError("Invalid request size")
                data = json.loads(self.rfile.read(length))
                if self.path == "/api/new":
                    return self.respond({"session": service.create(data.get("instruction", DEFAULT_INSTRUCTION))})
                if self.path == "/api/instruction":
                    service.set_instruction(data["session"], data["instruction"])
                    return self.respond({"ok": True})
                if self.path == "/api/send":
                    return self.respond({"job": service.submit(data["session"], data["prompt"], data.get("reply_limit"))})
                if self.path == "/api/label":
                    service.label(data["session"], data["point_id"], data["outcome"])
                    return self.respond({"ok": True})
                if self.path == "/api/auto-label":
                    service.auto_label_saved(data["session"])
                    return self.respond({"ok": True})
                if self.path == "/api/stop":
                    service.stop()
                    return self.respond({"ok": True})
                if self.path == "/api/shutdown":
                    with service.lock:
                        service.closing = True
                        service.cancel.set()
                    self.respond({"ok": True, "message": "Stopping the current work and closing the app. Earlier recordings remain saved."})
                    def finish_shutdown():
                        while True:
                            with service.lock:
                                busy = service.busy()
                            if not busy:
                                break
                            time.sleep(.1)
                        self.server.shutdown()
                    threading.Thread(target=finish_shutdown, daemon=True).start()
                    return
                self.respond({"error": "Not found"}, 404)
            except (ValueError, KeyError, TypeError) as exc:
                self.respond({"error": str(exc)}, 400)
    return Handler


def serve(config_path, root, port=8765):
    service = ChatService(read_json(config_path), root)
    server = ThreadingHTTPServer(("127.0.0.1", port), make_handler(service))
    print(f"Gemma recording room: http://127.0.0.1:{port}", flush=True)
    try:
        server.serve_forever()
    finally:
        server.server_close()
