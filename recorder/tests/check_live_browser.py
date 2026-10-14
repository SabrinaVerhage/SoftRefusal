"""Manual integration check: requires the real local app; records one neutral turn.

Run explicitly with .venv/Scripts/python.exe tests/check_live_browser.py.
This is not part of the automatic fixture suite.
"""
import json
import sys
from pathlib import Path
import urllib.request

from playwright.sync_api import sync_playwright, expect

ROOT = Path(__file__).resolve().parents[1]
URL = "http://127.0.0.1:8765"


def request(path, data=None):
    req = urllib.request.Request(URL + path, data=json.dumps(data).encode() if data is not None else None,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=20) as response:
        return json.load(response)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sid = sys.argv[1]  # Resume only the explicitly named neutral validation session.
    else:
        assert not request("/api/state")["busy"], "Do not interrupt a user's active conversation"
        sid = request("/api/new", {})["session"]
    output = ROOT / "runs" / "ui-validation"
    output.mkdir(exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 1500, "height": 1100})
        errors, downloads = [], []
        page.on("pageerror", lambda e: errors.append(str(e)))
        page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
        page.on("download", lambda d: downloads.append(d.suggested_filename))
        page.add_init_script(f"localStorage.setItem('gemma-session', {json.dumps(sid)})")
        page.goto(URL)
        expect(page.locator("#recording-path")).to_contain_text(sid)
        assert page.locator("#reply-limit").input_value() == "512"
        if len(sys.argv) == 1:
            page.locator("#reply-limit").select_option("256")
            page.locator("#prompt").fill("Reply with exactly one word: hello.")
            page.locator("#send").click()
        page.locator("#inside-open").click()
        print("Neutral turn submitted; waiting for Gemma and actual measurements.", flush=True)
        expect(page.locator("#saved-count")).to_contain_text("1 turn recorded", timeout=300000)
        state = request("/api/state?session=" + sid)
        assert state["job"]["status"] == "complete"
        assert Path(state["project_root"]) == ROOT
        row = state["rows"][0]
        trace = row["education_trace"]
        assert len(trace["layers"]) == 34
        assert len(trace["candidates"]) == row["generated_token_count"]
        assert row["generation_settings"]["max_new_tokens"] == 256
        assert page.locator("#layer-matrix rect").count() == 34 * 16
        assert page.locator("#candidate-bars .candidate").count() == 5
        assert page.locator("#token-pieces .token-piece").count() > 0
        page.screenshot(path=str(output / "inside-model-desktop.png"))
        page.locator("#candidate-step").fill("0")
        assert not page.locator("#follow-candidates").is_checked()
        page.locator("#inside-close").click()
        page.reload()
        expect(page.locator("#saved-count")).to_contain_text("1 turn recorded")
        page.wait_for_timeout(2000)
        assert downloads == [], downloads
        with page.expect_download() as download:
            page.locator("#export").click()
        assert download.value.suggested_filename == f"gemma-conversation-{sid[:8]}.zip"
        assert page.url.rstrip("/") == URL
        page.reload()
        page.wait_for_timeout(2000)
        assert len(downloads) == 1
        page.screenshot(path=str(output / "recording-room-desktop.png"))
        page.set_viewport_size({"width": 390, "height": 844})
        page.locator("#inside-open").click()
        assert page.evaluate("document.documentElement.scrollWidth <= innerWidth")
        assert page.locator("#inside").evaluate("e => e.scrollWidth <= e.clientWidth")
        page.screenshot(path=str(output / "inside-model-mobile.png"))
        assert not errors, errors
        report = {"session_id": sid, "project_root": state["project_root"], "layer_rows": len(trace["layers"]),
                  "generated_tokens": row["generated_token_count"], "browser_errors": errors,
                  "downloads": downloads, "refresh_downloads": 0, "mobile_overflow": False}
        (output / "browser-check.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(json.dumps(report), flush=True)
        browser.close()
