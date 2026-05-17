"""Flask web frontend for the path-landscape emergence-analysis agent.

Usage:
    export ANTHROPIC_API_KEY=sk-ant-...
    python serve_agent.py            # or:
    python -m path_landscape.webapp  # equivalent

Then open http://127.0.0.1:5174 in your browser.

The UI lets you type a phenomenon description, optionally tune `n_paths` and
`eps`, and submit. The pipeline runs in a background thread; progress events
stream live over Server-Sent Events; when finished, the page redirects to a
result view with the multi-panel figure, the four landscape metrics, and the
LLM's mechanistic interpretation.
"""
from .server import create_app, run_server

__all__ = ["create_app", "run_server"]
