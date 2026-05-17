"""Start the local web UI for the path-landscape emergence-analysis agent.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python serve_agent.py                # http://127.0.0.1:5174
    python serve_agent.py --port 8000    # custom port
    python serve_agent.py --out ./runs   # custom output directory
"""
from __future__ import annotations

# Force a non-interactive matplotlib backend before any plotting module
# imports pyplot — figures will be rendered from a background worker thread.
import matplotlib
matplotlib.use("Agg", force=True)

import argparse
from pathlib import Path

from path_landscape.webapp import run_server


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5174)
    p.add_argument("--out", default="./emergence_analysis_web",
                   help="output directory for per-job artifacts")
    args = p.parse_args()
    run_server(host=args.host, port=args.port,
               output_root=Path(args.out))


if __name__ == "__main__":
    main()
