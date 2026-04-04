from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the local FitIQ API server.")
    parser.add_argument("--host", default="127.0.0.1", help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind.")
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable uvicorn reload mode for local development.",
    )
    parser.add_argument(
        "--hide-docs",
        action="store_true",
        help="Disable Swagger/OpenAPI docs while serving the local demo.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

    try:
        import uvicorn
    except ImportError as exc:
        print(
            "uvicorn is required to run the FitIQ API server.\n"
            "Install it with: pip install -r requirements.txt\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    os.chdir(PROJECT_ROOT)
    os.environ["FITIQ_API_EXPOSE_DOCS"] = "0" if args.hide_docs else "1"

    try:
        from api.app import app
    except ImportError as exc:
        print(
            "Failed to import the FitIQ API app.\n"
            f"Import error: {exc}",
            file=sys.stderr,
        )
        return 1

    print("Starting FitIQ API server...")
    print(f"  URL: http://{args.host}:{args.port}")
    print(f"  Docs enabled: {not args.hide_docs}")
    print(f"  Reload: {args.reload}")

    uvicorn.run(app, host=args.host, port=args.port, reload=args.reload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
