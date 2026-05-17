"""End-to-end agentic emergence-analysis demo.

Pass a phenomenon description on the command line (or use the bundled examples)
and the pipeline will:

  1. Specify the phenomenon as a path-landscape System via Claude (tool use).
  2. Build the System and extract paths from inputs to outputs.
  3. Compute the four cross-system landscape metrics + persistent homology.
  4. Have Claude interpret what the metrics mean mechanistically.
  5. Save a multi-panel figure, a markdown report, and the spec JSON.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python demo_agent.py                                       # uses the default
    python demo_agent.py "Insulin signaling in a hepatocyte"   # custom phenomenon
    python demo_agent.py --all                                 # runs all examples
"""
from __future__ import annotations

import argparse
import os
import re
import sys

from path_landscape.agent import analyze_emergence

EXAMPLES = [
    "A flock of starlings turning as one",
    "Chain-of-thought reasoning emerging in a large language model",
    "Insight ('aha') moments arising from spreading activation in cortex",
    "Phase transition to ferromagnetism in an Ising model near criticality",
    "Stem-cell differentiation choosing a lineage from a pluripotent state",
]


def slug(text: str) -> str:
    s = re.sub(r"[^a-zA-Z0-9]+", "_", text).strip("_").lower()
    return s[:48]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("phenomenon", nargs="?", default=EXAMPLES[0],
                        help="natural-language description of the phenomenon")
    parser.add_argument("--out", default="./emergence_analysis",
                        help="output directory (per-phenomenon subdirs created)")
    parser.add_argument("--all", action="store_true",
                        help="run all bundled examples")
    parser.add_argument("--n-paths", type=int, default=1500,
                        help="paths to sample/enumerate (default 1500)")
    parser.add_argument("--eps", type=float, default=0.45,
                        help="DBSCAN clustering eps (default 0.45)")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("error: ANTHROPIC_API_KEY environment variable is not set.",
              file=sys.stderr)
        sys.exit(2)

    phenomena = EXAMPLES if args.all else [args.phenomenon]
    for ph in phenomena:
        out_dir = os.path.join(args.out, slug(ph))
        print("=" * 72)
        print(f"Phenomenon: {ph}")
        print(f"Output    : {out_dir}")
        print("=" * 72)
        try:
            analyze_emergence(
                ph,
                out_dir=out_dir,
                n_paths=args.n_paths,
                eps=args.eps,
            )
        except Exception as exc:
            print(f"  failed: {exc}")
            continue
        print()


if __name__ == "__main__":
    main()
