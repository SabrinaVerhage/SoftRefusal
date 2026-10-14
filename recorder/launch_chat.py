"""Double-click launcher: reuse the local server or start it without a console window."""
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
import webbrowser

ROOT = Path(__file__).resolve().parent
GPU = "--gpu" in sys.argv
# The cloud-GPU room is a separate app: own port and own folder, so 4B laptop
# recordings and 12B cloud recordings never mix in one conversation list.
MODE = "remote" if GPU else "local"
URL = "http://127.0.0.1:8766" if GPU else "http://127.0.0.1:8765"
COMMAND = (["chat", "--config", "configs/gemma3-12b-remote.json", "--out", "runs/gpu-chat", "--port", "8766"]
           if GPU else ["chat"])
LOG = "gpu-chat-server.log" if GPU else "chat-server.log"


def ready():
    try:
        with urllib.request.urlopen(URL + "/api/state", timeout=2) as response:
            state = json.load(response)
            return (state.get("app") == "gemma-recording-room" and state.get("mode", "local") == MODE and
                    Path(state.get("project_root", "")).resolve() == ROOT)
    except Exception:
        return False


def main():
    if not ready():
        logs = ROOT / "runs"
        logs.mkdir(exist_ok=True)
        with (logs / LOG).open("a", encoding="utf-8") as log:
            subprocess.Popen([sys.executable, "-m", "avoidance.cli", *COMMAND], cwd=ROOT,
                             stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                             creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        for _ in range(45):
            if ready():
                break
            time.sleep(1)
        else:
            raise RuntimeError(f"The recording room could not start. See runs/{LOG}.")
    webbrowser.open(URL)


def stop():
    if not ready():
        print("The recording room is already stopped, or is not responding on its local address.", flush=True)
        return
    print("Stopping Gemma. Waiting for the current calculation to finish safely...", flush=True)
    request = urllib.request.Request(URL + "/api/shutdown", data=b"{}", headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=10) as response:
        json.load(response)
    for _ in range(120):
        if not ready():
            print("Gemma stopped. Your completed recordings are saved.", flush=True)
            return
        time.sleep(.5)
    raise RuntimeError("Gemma received the stop request but is still finishing a calculation. It will close automatically when that finishes. Your earlier completed recordings remain saved.")


if __name__ == "__main__":
    try:
        if "--stop" in sys.argv:
            stop()
        else:
            main()
    except Exception as exc:
        import ctypes
        ctypes.windll.user32.MessageBoxW(0, str(exc), "Gemma recording room", 0x10)
        sys.exit(1)
