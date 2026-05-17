"""Entry point: `python -m path_landscape.webapp`."""
from .server import run_server

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=5174)
    p.add_argument("--out", default="./emergence_analysis_web")
    args = p.parse_args()

    from pathlib import Path
    run_server(host=args.host, port=args.port, output_root=Path(args.out))
