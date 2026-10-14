"""Neutral live checks for queueing, streaming, native stop IDs and context fit."""
import json
import time
from pathlib import Path

import httpx


def main():
    root = Path(__file__).resolve().parents[1]
    target = root / "runtime/rehearsal-v02-001.json"
    if target.exists():
        raise SystemExit("Use a new report name instead of overwriting evidence")
    token = (root / "secrets/rehearsal-2-token.txt").read_text().strip()
    other = (root / "secrets/rehearsal-1-token.txt").read_text().strip()
    report = {"synthetic": False, "version": "0.2", "started_at": time.time(), "jobs": [], "checks": {}}
    def save():
        target.write_text(json.dumps(report, indent=2))
    with httpx.Client(base_url="http://127.0.0.1:8000", headers={"Authorization": "Bearer " + token}, timeout=15) as client:
        health = client.get("/v1/health"); health.raise_for_status(); report["health"] = health.json()
        assert health.json()["limits"]["per_credential"] == 3
        assert health.json()["limits"]["chat_input_tokens"] == 8192
        def submit(path, payload):
            deadline = time.monotonic() + 120
            while True:
                response = client.post(path, json=payload)
                if response.status_code != 429 or response.json().get("detail", {}).get("code") != "submission_rate_limit":
                    response.raise_for_status(); return response.json()
                assert time.monotonic() < deadline
                time.sleep(1)
        def chat(model, content, max_tokens=64, **kwargs):
            return submit("/v1/chat/jobs", {"model": model, "messages": [{"role": "user", "content": content}],
                           "max_tokens": max_tokens, "temperature": 0, "seed": 0, **kwargs})
        def finish(job, label, expect="succeeded"):
            deadline = time.monotonic() + 900; phases = set(); samples = []
            while job["status"] in {"queued", "running"}:
                assert time.monotonic() < deadline, "Live job timeout"
                response = client.get(job["status_url"]); response.raise_for_status(); job = response.json()
                progress = job.get("progress") or {}
                if progress.get("phase"): phases.add(progress["phase"])
                if progress.get("text") and job["status"] == "running" and len(samples) < 12:
                    samples.append({"text": progress["text"], "output_tokens": progress.get("output_tokens")})
                save(); time.sleep(0.1)
            report["jobs"].append({"label": label, "job": job, "progress_phases": sorted(phases), "live_samples": samples})
            save(); assert job["status"] == expect, label + " status=" + job["status"]
            metrics = (job.get("result") or {}).get("metrics", {})
            print(label, job["status"], "load", metrics.get("model_load_seconds"), "inference", metrics.get("inference_seconds"), flush=True)
            return job, samples
        greeting, _ = finish(chat("hermes", "Hi."), "hermes_native_stop")
        result = greeting["result"]
        assert result["finish_reason"] == "stop" and result["ended_with_eos"]
        assert 151645 in result["stop_token_settings"]["effective_eos"]
        assert result["usage"]["completion_tokens"] < 64
        report["checks"]["hermes_native_chat_stop"] = True
        first = chat("hermes", "List twenty numbered, short facts about rainbows and light.", 256)
        image = submit("/v1/images/jobs", {"prompt": "An abstract folded paper flower, pink and violet, cream background.", "width": 512, "height": 512})
        follow = chat("hermes", "Summarize your preceding answer in one sentence.", 64, after_job_id=first["job_id"])
        fourth = client.post("/v1/chat/jobs", json={"model": "gemma", "messages": [{"role": "user", "content": "Hello"}]})
        assert fourth.status_code == 429 and fourth.json()["detail"]["code"] == "per_credential_queue_full"
        foreign = client.get(first["status_url"], headers={"Authorization": "Bearer " + other})
        assert foreign.status_code == 404
        a, samples = finish(first, "same_credential_streaming_chat")
        assert samples, "No partial generated text observed before completion"
        b, _ = finish(image, "same_credential_queued_image")
        png = client.get(b["result_url"]); png.raise_for_status()
        assert png.headers["content-type"].startswith("image/png") and png.content.startswith(b"\x89PNG\r\n\x1a\n")
        c, _ = finish(follow, "queued_chat_followup")
        assert c["result"]["generation_settings"]["messages"][1]["content"] == a["result"]["text"]
        assert a["finished_at"] <= b["started_at"] and b["finished_at"] <= c["started_at"]
        report["checks"].update(same_credential_three_jobs=True, fourth_rejected=True, shared_gpu_serialized=True,
                                live_text_before_completion=True, linked_chat_history=True, foreign_progress_404=True, png_retrieval=True)
        context = [{"role": "user", "content": "x " * 4000}, {"role": "assistant", "content": "x " * 4000},
                   {"role": "user", "content": "Respond with just OK."}]
        big = submit("/v1/chat/jobs", {"model": "gemma", "messages": context, "max_tokens": 16, "temperature": 0, "seed": 0})
        big, _ = finish(big, "gemma_near_8k_context")
        assert 7900 < big["result"]["usage"]["prompt_tokens"] <= 8192
        report["checks"]["near_8k_bf16_memory_fit"] = True
        oversized = [{"role": "user", "content": "x " * 5000}, {"role": "assistant", "content": "x " * 5000},
                     {"role": "user", "content": "Respond with just OK."}]
        bad = submit("/v1/chat/jobs", {"model": "gemma", "messages": oversized, "max_tokens": 16})
        bad, _ = finish(bad, "counted_input_limit", expect="failed")
        assert bad["result"]["error"]["code"] == "input_too_long"
        assert bad["result"]["error"]["input_tokens"] > bad["result"]["error"]["limit_tokens"] == 8192
        assert bad["result"]["outcome"] == "technical_error"
        warm, _ = finish(chat("gemma", "Explain rainbows in one sentence.", 64), "gemma_after_length_error_stays_warm")
        assert warm["result"]["metrics"]["model_load_seconds"] == 0
        report["checks"].update(counted_length_error=True, no_refusal_inferred=True, healthy_worker_retained=True)
        report["completed"] = True; report["finished_at"] = time.time(); save()
        print("V02_REHEARSAL_COMPLETE", flush=True)


if __name__ == "__main__":
    main()
