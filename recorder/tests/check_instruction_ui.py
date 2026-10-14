"""Isolated browser/settings and real cached tokenizer check; no model generation."""
import json
import tempfile
import threading
from pathlib import Path
from http.server import ThreadingHTTPServer

from playwright.sync_api import sync_playwright, expect
from avoidance.chat import ChatService, make_handler, DEFAULT_INSTRUCTION

root = Path(__file__).resolve().parents[1]
config = json.loads((root / "configs/gemma3-4b-chat.json").read_text())
with tempfile.TemporaryDirectory() as temp:
    service = ChatService(config, temp)
    sid = service.create()
    server = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(service))
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            errors = []
            page.on("pageerror", lambda e: errors.append(str(e)))
            page.goto(f"http://127.0.0.1:{server.server_port}")
            page.locator('.instruction-box summary').click()
            expect(page.locator('#instruction')).to_have_value('')
            page.locator('#brief-instruction').click()
            expect(page.locator('#instruction-status')).to_have_text('· active')
            assert service.sessions[sid]['instruction'] == DEFAULT_INSTRUCTION
            page.locator('#instruction').fill('Use one sentence.')
            page.locator('#save-instruction').click()
            expect(page.locator('#instruction-status')).to_have_text('· active')
            page.reload()
            page.locator('.instruction-box summary').click()
            expect(page.locator('#instruction')).to_have_value('Use one sentence.')
            page.locator('#instruction').fill('')
            page.locator('#save-instruction').click()
            expect(page.locator('#instruction-status')).to_have_text('· none')
            page.locator('#new-chat').click()
            expect(page.locator('#instruction')).to_have_value(DEFAULT_INSTRUCTION)
            assert not errors, errors
            browser.close()
    finally:
        server.shutdown()
        server.server_close()

from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained(config['model_id'], revision=config['revision'], local_files_only=True)
rendered = tokenizer.apply_chat_template([{'role':'system','content':DEFAULT_INSTRUCTION},{'role':'user','content':'Hello'}],tokenize=False,add_generation_prompt=True)
assert '<start_of_turn>user\n' + DEFAULT_INSTRUCTION + '\n\nHello' in rendered
assert '<start_of_turn>system' not in rendered
print('Passed: browser default/edit/clear/reload and native Gemma instruction placement.')
