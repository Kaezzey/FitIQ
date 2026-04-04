from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import urlopen


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from desktop.tk_demo import ApiClient, launch_desktop_demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the FitIQ Tkinter desktop demo.")
    parser.add_argument("--host", default="127.0.0.1", help="Local API host.")
    parser.add_argument("--port", type=int, default=8000, help="Local API port.")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=15.0,
        help="Seconds to wait for the local API to become healthy.",
    )
    parser.add_argument(
        "--reuse-running-api",
        action="store_true",
        help="Use an already-running local API if one is available.",
    )
    return parser.parse_args()


def healthcheck(base_url: str, timeout_seconds: float = 2.0) -> bool:
    try:
        with urlopen(f"{base_url}/health", timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return payload.get("status") == "ok"
    except URLError:
        return False
    except Exception:
        return False


def wait_for_api(base_url: str, timeout_seconds: float) -> bool:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        if healthcheck(base_url, timeout_seconds=1.5):
            return True
        time.sleep(0.25)
    return False


def terminate_process(process: subprocess.Popen | None) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def main() -> int:
    args = parse_args()
    base_url = f"http://{args.host}:{args.port}"
    api_process: subprocess.Popen | None = None

    os_command = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "run_demo.py"),
        "--host",
        args.host,
        "--port",
        str(args.port),
        "--hide-docs",
    ]

    if args.reuse_running_api and healthcheck(base_url):
        print(f"Reusing existing local API at {base_url}")
    else:
        print("Starting local FitIQ API for the desktop client...")
        api_process = subprocess.Popen(os_command, cwd=PROJECT_ROOT)
        if not wait_for_api(base_url, timeout_seconds=args.startup_timeout):
            terminate_process(api_process)
            print(
                f"Local API did not become healthy at {base_url} within {args.startup_timeout:.1f}s.",
                file=sys.stderr,
            )
            return 1

    client = ApiClient(base_url=base_url)
    try:
        return launch_desktop_demo(client=client, on_close=lambda: terminate_process(api_process))
    finally:
        terminate_process(api_process)


if __name__ == "__main__":
    raise SystemExit(main())
