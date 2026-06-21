"""
Serve the Deep Dyna-Q CamRest676 dashboard.

Usage:
    python dashboard_server.py          # opens browser at http://127.0.0.1:8000
    python dashboard_server.py --port 9000
    python dashboard_server.py --no-open
"""
from __future__ import annotations

import argparse
import json
import os
import threading
import webbrowser
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent
WEB_ENTRY = "/web/index.html"


def build_catalog() -> dict:
    ckpt_root = ROOT / "deep_dialog" / "checkpoints"
    img_root  = ROOT / "img"

    perf_files = []
    for path in sorted(ckpt_root.glob("**/*_performance_records.json")):
        label = path.parent.name
        perf_files.append({
            "label": label,
            "path":  path.relative_to(ROOT).as_posix(),
            "kind":  "performance",
        })

    img_files = []
    if img_root.exists():
        for path in sorted(img_root.glob("*")):
            if path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".gif", ".webp", ".json"}:
                continue
            img_files.append({
                "label": path.stem.replace("_", " ").title(),
                "path":  path.relative_to(ROOT).as_posix(),
                "kind":  "image" if path.suffix.lower() != ".json" else "dialogue",
            })

    return {
        "repo_root":    str(ROOT),
        "defaults": {
            "k0":       "deep_dialog/checkpoints/best/k0/agt_9_performance_records.json",
            "k5":       "deep_dialog/checkpoints/best/k5/agt_9_performance_records.json",
            "dialogue": "img/k0_vs_k5_dialogue.json",
            "k0_cases": "img/k0_cases.json",
        },
        "performance_files": perf_files,
        "extra_files":       img_files,
    }


class DashboardHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=str(ROOT), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/api/catalog":
            payload = json.dumps(build_catalog(), ensure_ascii=False, indent=2).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        if parsed.path == "/":
            self.send_response(302)
            self.send_header("Location", WEB_ENTRY)
            self.end_headers()
            return
        if parsed.path in ("/frames", "/frames/"):
            self.send_response(302)
            self.send_header("Location", "/web/frames.html")
            self.end_headers()
            return
        return super().do_GET()

    def log_message(self, format, *args):
        return


def find_port(start: int) -> tuple[ThreadingHTTPServer, int]:
    for port in range(start, start + 20):
        try:
            server = ThreadingHTTPServer(("127.0.0.1", port), DashboardHandler)
            return server, port
        except OSError:
            continue
    raise RuntimeError(f"No free port in [{start}, {start+19}]")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--port",    type=int, default=8000)
    parser.add_argument("--no-open", action="store_true")
    args = parser.parse_args()

    server, port = find_port(args.port)
    url = f"http://127.0.0.1:{port}{WEB_ENTRY}"
    print(f"Dashboard:  {url}")
    print(f"Press Ctrl+C to stop.")

    if not args.no_open:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
