"""Neutral live end-to-end API rehearsal; reads an app token file without echoing it."""
import argparse
import json
import time
from pathlib import Path

import httpx


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--token-file", type=Path, default=Path("secrets/operator-token.txt"))
    parser.add_argument("--second-token-file", type=Path, default=Path("secrets/rehearsal-2-token.txt"))
    parser.add_argument("--out", type=Path, default=Path("runtime/rehearsal-report.json"))
    parser.add_argument("--include-hermes", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit("Use a new report path; previous GPU observations are preserved.")
    base = args.base_url.rstrip("/")
    if not (base.startswith("https://") or base.startswith("http://127.0.0.1:") or base.startswith("http://localhost:")):
        raise SystemExit("Use HTTPS or local loopback to protect the application credential.")
    token = args.token_file.read_text().strip()
    with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {token}"}, timeout=30) as client:
        health = client.get("/v1/health")
        health.raise_for_status()
        report = {"synthetic": False, "started_at": time.time(), "initial_health": health.json(), "jobs": []}
        cases = [("gemma-cold", "/v1/chat/jobs", {"model": "gemma", "messages": [
                    {"role": "user", "content": "Explain how a rainbow forms in two sentences."}],
                    "max_tokens": 64, "temperature": 0, "seed": 0}),
                 ("gemma-warm", "/v1/chat/jobs", {"model": "gemma", "messages": [
                    {"role": "user", "content": "Name three common geometric shapes."}],
                    "max_tokens": 64, "temperature": 0, "seed": 0}),
                 ("flux-512", "/v1/images/jobs", {"model": "flux", "prompt": "A folded paper flower, studio photograph, cream background.",
                    "width": 512, "height": 512, "steps": 4, "seed": 0}),
                 ("flux-1024", "/v1/images/jobs", {"model": "flux", "prompt": "An abstract sculpture made of coloured paper, soft studio lighting.",
                    "width": 1024, "height": 1024, "steps": 4, "seed": 0})]
        if args.include_hermes:
            cases.extend([(f"hermes-{mode}", "/v1/chat/jobs", {"model": "hermes", "messages": [
                {"role": "user", "content": "Explain how a rainbow forms in two sentences."}],
                "max_tokens": 64, "temperature": 0, "seed": 0}) for mode in ("cold", "warm")])
        cases.append(("gemma-reloaded", "/v1/chat/jobs", cases[0][2]))
        def submit_with_retry(endpoint, payload, active_client=client):
            key = str(time.time_ns())
            deadline = time.monotonic() + 180
            while True:
                response = active_client.post(endpoint, json=payload, headers={"Idempotency-Key": key})
                if response.status_code != 429 or time.monotonic() > deadline:
                    response.raise_for_status()
                    return response.json()
                # Respect the API's bounded admission rate, retaining the same
                # idempotency key if a retry is needed.
                time.sleep(min(60, max(1, int(response.headers.get("Retry-After", "5")))))

        def poll(job, active_client=client):
            deadline = time.monotonic() + 2800
            while job["status"] in ("queued", "running"):
                if time.monotonic() > deadline:
                    raise SystemExit("API polling deadline exceeded; inspect the saved job status before retrying.")
                time.sleep(2)
                response = active_client.get(job["status_url"])
                response.raise_for_status()
                job = response.json()
            return job

        for label, endpoint, payload in cases:
            job = poll(submit_with_retry(endpoint, payload))
            report["jobs"].append({"case": label, **job})
            args.out.parent.mkdir(parents=True, exist_ok=True)
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if job["status"] != "succeeded":
                raise SystemExit(f"{label} failed; exact error code saved in {args.out}. No quantization fallback was applied.")
            if job["kind"] == "image":
                image = client.get(job["result_url"])
                image.raise_for_status()
                if not image.headers.get("content-type", "").startswith("image/png") or not image.content.startswith(b"\x89PNG"):
                    raise SystemExit("Image endpoint did not return a PNG; inspect filter/error metadata.")
            metrics = job["result"].get("metrics", {})
            print(f"{label}: succeeded; load={metrics.get('model_load_seconds')}s; inference={metrics.get('inference_seconds')}s")
        second_token = args.second_token_file.read_text().strip()
        with httpx.Client(base_url=base, headers={"Authorization": f"Bearer {second_token}"}, timeout=30) as second:
            a = submit_with_retry("/v1/chat/jobs", cases[0][2])
            b = submit_with_retry("/v1/images/jobs", cases[2][2], second)
            denied = client.post("/v1/chat/jobs", json=cases[0][2])
            if denied.status_code != 429:
                # A very fast job may finish before the repeated submission.
                # Never leave a newly accepted extra test job unobserved.
                if denied.status_code == 202:
                    poll(denied.json())
            foreign = second.get(a["status_url"])
            report["cross_credential_job_status"] = foreign.status_code
            if foreign.status_code != 404:
                raise SystemExit("Cross-credential job isolation failed")
            a, b = poll(a), poll(b, second)
            report["jobs"].extend([{"case": "burst-chat", **a}, {"case": "burst-image", **b}])
            report["burst_repeat_submission_status"] = denied.status_code
            report["burst_serialized"] = a["finished_at"] <= b["started_at"]
            args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
            if a["status"] != "succeeded" or b["status"] != "succeeded" or not report["burst_serialized"]:
                raise SystemExit("Shared GPU burst failed; inspect the saved report")
        # Rejecting an unauthenticated submission is part of the live check.
        unauth = httpx.post(base + "/v1/chat/jobs", json=cases[0][2], timeout=30)
        report["unauthenticated_status"] = unauth.status_code
        if unauth.status_code != 401:
            raise SystemExit("Authentication check failed")
        report["completed_at"] = time.time()
        report["completed"] = True
        args.out.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"Live report saved to {args.out}. Review the generated images visually as well.")


if __name__ == "__main__":
    main()
