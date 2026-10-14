"""Start/reuse this prototype's local server and open the browser.

The CMD wrapper leaves errors visible. The server itself runs without a console.
"""
import argparse
import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
PORTS = range(8777, 8787)
# Local requests must not be sent through a configured corporate proxy.
HTTP = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def health(port):
    try:
        with HTTP.open(f"http://127.0.0.1:{port}/health", timeout=0.4) as response:
            info = json.load(response)
        if info.get("app") == "sxtech-umap-prototype" and Path(info.get("project_root", "")).resolve() == ROOT:
            return info
    except (OSError, ValueError, urllib.error.URLError):
        pass
    return None


def free(port):
    try:
        with socket.socket() as probe:
            if os.name == "nt":
                probe.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
            probe.bind(("127.0.0.1", port))
        return True
    except OSError:
        return False


def start():
    for port in PORTS:
        if health(port):
            return port
    port = next((port for port in PORTS if free(port)), None)
    if port is None:
        raise RuntimeError("Ports 8777 to 8786 are already in use. Run serve.py --port with another port number.")
    logs = ROOT / "runtime"
    logs.mkdir(exist_ok=True)
    log_path = logs / "server.log"
    with log_path.open("a", encoding="utf-8") as log:
        process = subprocess.Popen(
            [sys.executable, str(ROOT / "serve.py"), "--port", str(port)],
            cwd=ROOT, stdin=subprocess.DEVNULL, stdout=log, stderr=log,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    for _ in range(40):
        if health(port):
            return port
        if process.poll() is not None:
            break
        time.sleep(0.15)
    raise RuntimeError(f"The local server could not start. Details are saved in:\n{log_path}")


def stop():
    found = False
    for port in PORTS:
        if not health(port):
            continue
        found = True
        request = urllib.request.Request(
            f"http://127.0.0.1:{port}/shutdown", data=b"",
            headers={"X-UMAP-Prototype": "stop-local-prototype"}, method="POST",
        )
        with HTTP.open(request, timeout=3) as response:
            response.read()
        for _ in range(20):
            if not health(port):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("The prototype received the stop request but has not finished closing.")
    print("Prototype server stopped." if found else "The prototype server is already stopped.", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stop", action="store_true")
    parser.add_argument("--no-browser", action="store_true", help="Start the server without opening a browser, for checking the launcher.")
    args = parser.parse_args()
    if args.stop:
        stop()
        return
    port = start()
    url = f"http://127.0.0.1:{port}/"
    print(f"Prototype ready: {url}", flush=True)
    if not args.no_browser and not webbrowser.open(url):
        raise RuntimeError(f"The server is running, but a browser could not open automatically.\nOpen this address in your browser: {url}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\n{exc}", file=sys.stderr, flush=True)
        sys.exit(1)
