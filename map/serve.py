"""Serve only this prototype on the local machine."""
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import argparse
import json
import os
import threading
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parent


class Handler(SimpleHTTPRequestHandler):
    def json_response(self, payload):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if urlsplit(self.path).path == "/health":
            self.json_response({"app": "sxtech-umap-prototype", "project_root": str(ROOT), "pid": os.getpid()})
        else:
            super().do_GET()

    def do_POST(self):
        if urlsplit(self.path).path == "/shutdown" and self.headers.get("X-UMAP-Prototype") == "stop-local-prototype":
            self.json_response({"closing": True})
            threading.Thread(target=self.server.shutdown, daemon=True).start()
        else:
            self.send_error(404)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8777)
    args = parser.parse_args()
    print(f"UMAP prototype: http://127.0.0.1:{args.port}", flush=True)
    with ThreadingHTTPServer(("127.0.0.1", args.port), partial(Handler, directory=str(ROOT))) as server:
        server.serve_forever()
