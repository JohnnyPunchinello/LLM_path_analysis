"""Multi-panel visualization for the agent pipeline:
  1) original-system graph (with feedback/multiscale rendered),
  2) static path graph (after unrolling/coarsening),
  3) path-landscape MDS scatter colored by cluster,
  4) length-by-cluster bar plot,
  5) persistence diagram (H0 + H1).
"""
from __future__ import annotations

import warnings
from typing import Optional

# If pyplot hasn't been imported yet, pin a non-interactive backend so this
# module can be used from a worker thread (e.g. the Flask webapp). On macOS
# the default `MacOSX` backend hard-fails outside the main thread. Setting
# the backend after pyplot is already loaded is a no-op (and a warning).
import matplotlib
if "matplotlib.pyplot" not in __import__("sys").modules:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..system import System
from ..landscape import PathLandscape
from ..metrics import persistence_h0, persistence_h1
from .schemas import SystemSpec

warnings.filterwarnings(
    "ignore", category=RuntimeWarning, module="sklearn.manifold._mds"
)
warnings.filterwarnings(
    "ignore", category=FutureWarning, module="sklearn.manifold._mds"
)

INK = "#0c1220"
PAPER = "#fcfcfa"
AMBER = "#f0b65a"
CYAN = "#7dc8dc"
BRICK = "#c83c37"
GRAY = "#aab4c3"


# ----------------------------------------------------------------- helpers

def _node_color(role: str) -> str:
    return {"input": CYAN, "output": BRICK}.get(role, AMBER)


def _draw_system_graph(ax, sys: System, spec: SystemSpec) -> None:
    g = nx.DiGraph()
    for name in sys.units:
        g.add_node(name)
    for u, v, data in sys.graph.edges(data=True):
        g.add_edge(u, v, recurrent=data.get("recurrent", False))
    # role lookup
    roles = {u.name: u.role for u in spec.units}
    parents = {u.name: (u.parent or "_") for u in spec.units}

    # group nodes by parent for the layout
    try:
        pos = nx.spring_layout(g, seed=0, k=1.0 / max(1, len(g) ** 0.5))
    except Exception:
        pos = {n: (i, 0) for i, n in enumerate(g)}

    # edges
    rec_edges = [(u, v) for u, v, d in g.edges(data=True) if d.get("recurrent")]
    fwd_edges = [
        (u, v) for u, v, d in g.edges(data=True) if not d.get("recurrent")
    ]
    nx.draw_networkx_edges(
        g, pos, edgelist=fwd_edges, ax=ax, edge_color=INK,
        alpha=0.45, width=0.7, arrows=True, arrowsize=8,
    )
    nx.draw_networkx_edges(
        g, pos, edgelist=rec_edges, ax=ax, edge_color=BRICK,
        alpha=0.7, width=1.0, arrows=True, arrowsize=8,
        style="dashed",
    )
    # nodes
    node_colors = [_node_color(roles.get(n, "internal")) for n in g.nodes]
    # encode multiscale via node size — units with a parent are smaller
    sizes = [(180 if parents.get(n) == "_" else 110) for n in g.nodes]
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, node_size=sizes,
        edgecolors=INK, linewidths=0.6,
    )
    # short label
    labels = {n: (n if len(n) <= 8 else n[:7] + "…") for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax,
                            font_size=6, font_color=INK)
    ax.set_title(
        f"{spec.phenomenon_name} - system graph\n"
        f"({len(sys.units)} units; cyan=input, red=output; "
        f"dashed red = recurrent)",
        fontsize=10,
    )
    ax.set_axis_off()


def _draw_static_path_graph(ax, sys: System, T: int) -> None:
    g = sys.unroll(T)
    # too many nodes? thin out the labels
    n = g.number_of_nodes()
    pos = nx.kamada_kawai_layout(g) if n <= 200 else nx.spring_layout(g, seed=0)
    nx.draw_networkx_edges(g, pos, ax=ax, edge_color=INK, alpha=0.25, width=0.4,
                           arrows=False)
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=GRAY, node_size=14,
        edgecolors=INK, linewidths=0.3,
    )
    if n <= 50:
        nx.draw_networkx_labels(g, pos, ax=ax, font_size=5, font_color=INK)
    ax.set_title(
        f"static path graph after unroll(T={T}) - "
        f"{n} nodes, {g.number_of_edges()} edges",
        fontsize=10,
    )
    ax.set_axis_off()


def _draw_persistence(ax, L: PathLandscape) -> None:
    h0 = persistence_h0(L, max_features=30)
    h1 = persistence_h1(L)
    if h0:
        for h in h0[:25]:
            ax.plot([0.0, h], [h, h], color=INK, linewidth=0.7, alpha=0.55)
        ax.scatter(
            [0.0] * len(h0[:25]), h0[:25],
            color=CYAN, s=18, zorder=4,
            label=f"H0 (top {min(25, len(h0))} of {len(h0)})",
        )
    if h1:
        births = [b for b, _ in h1]
        deaths = [d for _, d in h1]
        ax.scatter(
            births, deaths, color=BRICK, s=24, zorder=5,
            edgecolors=INK, linewidths=0.4, label=f"H1 ({len(h1)})",
        )
    lim = 1.0
    if h0:
        lim = max(lim, max(h0))
    if h1:
        lim = max(lim, max((d for _, d in h1), default=0))
    ax.plot([0, lim], [0, lim], "k--", linewidth=0.4, alpha=0.4)
    ax.set_xlim(0, lim * 1.05)
    ax.set_ylim(0, lim * 1.05)
    ax.set_xlabel("birth")
    ax.set_ylabel("death")
    ax.set_title("persistence diagram", fontsize=10)
    ax.legend(loc="lower right", fontsize=8, frameon=False)


# ----------------------------------------------------------------- public

def _draw_ontology_panel(ax, spec: SystemSpec, sys: System,
                         L: Optional[PathLandscape] = None) -> None:
    """A header panel that names what the diagram is showing in framework terms.

    This is the user-facing answer to:
        - What is the System?
        - What is a Unit (basic computation)?
        - What is an Interaction (information flow)?
        - Is there multiscale structure?
        - What are the Observables / emergent phenomenon?
    """
    ax.axis("off")
    ax.set_facecolor(PAPER)
    # Title
    ax.text(0.0, 1.00, spec.phenomenon_name,
            ha="left", va="top", fontsize=15, fontweight="bold", color=INK)
    ax.text(0.0, 0.86,
            f"System  ·  path-landscape view of an emergent phenomenon",
            ha="left", va="top", fontsize=10, color=GRAY,
            fontstyle="italic")

    # Role counts
    n_in  = sum(1 for u in spec.units if u.role == "input")
    n_out = sum(1 for u in spec.units if u.role == "output")
    n_int = sum(1 for u in spec.units if u.role == "internal")
    n_rec = sum(1 for x in spec.interactions if x.recurrent)
    scales = sorted({u.scale for u in spec.units})
    multiscale = len(scales) > 1
    n_modes = (L.n_modes if L is not None else None)
    h1 = (persistence_h1(L) if L is not None else None)

    rows = [
        ("System",
         f"{len(sys.units)} units · {sys.graph.number_of_edges()} interactions · "
         f"T = {spec.time_steps}"),
        ("Unit (basic computation)",
         f"{len(spec.units)} total  ({n_in} input · {n_int} internal · {n_out} output)"),
        ("Interaction (information flow)",
         f"{len(spec.interactions)} directed edges, "
         f"{n_rec} recurrent (feedback loops in time)"),
        ("Multiscale structure",
         f"scales = {scales}"
         f" — {'hierarchical (parent/child grouping)' if multiscale else 'single scale'}"),
        ("Observable",
         "outputs: " + (", ".join(u.name for u in spec.units if u.role == "output")
                       or "(none specified)")),
        ("Emergent phenomenon",
         (f"path landscape with {n_modes} modes"
          + (f", {len(h1)} compositional loops (H1)" if h1 else "")
          + " — see right-side panels") if L is not None
         else "path landscape NOT computed (path extraction failed; see annotation)"),
    ]
    y = 0.70
    for label, value in rows:
        ax.text(0.0, y, f"{label}:", ha="left", va="top",
                fontsize=9.5, fontweight="bold", color=INK)
        ax.text(0.34, y, value, ha="left", va="top",
                fontsize=9.5, color=INK, wrap=True)
        y -= 0.10


def render_figure(
    spec: SystemSpec, sys: System, L: PathLandscape, out_path: str
) -> None:
    """Render the multi-panel analysis figure and save to `out_path`.

    Layout encodes the framework ontology:

        row 0 : ontology panel (left)     · original system graph (right)
        row 1 : unrolled path graph       · path landscape (MDS scatter)
        row 2 : paths-by-cluster bars     · persistence diagram

    Each panel title says explicitly what the panel shows in framework terms.
    """
    fig = plt.figure(figsize=(15, 12.5), facecolor=PAPER)
    gs = fig.add_gridspec(
        3, 2, height_ratios=[1.0, 1.0, 1.0],
        hspace=0.40, wspace=0.22,
        top=0.96, bottom=0.05, left=0.05, right=0.97,
    )

    # Row 0: ontology panel + system graph
    ax_onto = fig.add_subplot(gs[0, 0])
    _draw_ontology_panel(ax_onto, spec, sys, L)

    ax_sys = fig.add_subplot(gs[0, 1])
    _draw_system_graph(ax_sys, sys, spec)

    # Row 1: unrolled graph + landscape
    ax_static = fig.add_subplot(gs[1, 0])
    _draw_static_path_graph(ax_static, sys, spec.time_steps)

    ax_land = fig.add_subplot(gs[1, 1])
    L.plot(ax=ax_land, show_legend=False,
           title=f"Path landscape — observable / emergent structure\n"
                 f"({L.n_modes} modes · {len(L.paths)} paths · "
                 f"each point = one input→output path)")

    # Row 2: cluster bars + persistence
    ax_bars = fig.add_subplot(gs[2, 0])
    L.plot_length_by_cluster(
        ax=ax_bars, drop_noise=True, label_top_k=8,
        title="Modes (clusters) of paths · length on y-axis",
    )

    ax_pers = fig.add_subplot(gs[2, 1])
    _draw_persistence(ax_pers, L)

    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def render_diagnostic_figure(
    spec: SystemSpec, sys: System, error_message: str, out_path: str,
) -> None:
    """Render a 'path landscape not available' figure when path extraction fails.

    Shows the ontology panel + the system graph + the unrolled (static) graph
    + a clear annotation panel explaining why no paths were found and what to
    check. Always produces a usable PNG, so the frontend never has a missing
    image.
    """
    fig = plt.figure(figsize=(15, 9.5), facecolor=PAPER)
    gs = fig.add_gridspec(
        2, 2, height_ratios=[1.0, 1.0],
        hspace=0.40, wspace=0.22,
        top=0.96, bottom=0.05, left=0.05, right=0.97,
    )

    # Row 0: ontology + system graph
    ax_onto = fig.add_subplot(gs[0, 0])
    _draw_ontology_panel(ax_onto, spec, sys, L=None)

    ax_sys = fig.add_subplot(gs[0, 1])
    _draw_system_graph(ax_sys, sys, spec)

    # Row 1: unrolled graph + diagnostic annotation
    ax_static = fig.add_subplot(gs[1, 0])
    try:
        _draw_static_path_graph(ax_static, sys, spec.time_steps)
    except Exception as exc:
        ax_static.axis("off")
        ax_static.text(0.5, 0.5, f"(unrolled graph render failed: {exc})",
                       ha="center", va="center", fontsize=9, color=BRICK)

    ax_diag = fig.add_subplot(gs[1, 1])
    ax_diag.axis("off")
    ax_diag.set_facecolor("#fff0ee")
    ax_diag.text(0.02, 0.96,
                 "⚠  Path landscape not available",
                 ha="left", va="top", fontsize=13, fontweight="bold",
                 color=BRICK)
    ax_diag.text(0.02, 0.86,
                 "Path extraction from inputs to outputs returned no paths "
                 "after unrolling.\n\n"
                 f"Reported error:\n  {error_message}\n\n"
                 "Likely causes (check the System spec on the left):",
                 ha="left", va="top", fontsize=9.5, color=INK, wrap=True)
    causes = [
        "  ·  An output unit has no incoming feed-forward edge",
        "  ·  An input unit has no outgoing feed-forward edge",
        "  ·  All edges from internals to outputs are marked recurrent",
        "  ·  time_steps is too small to traverse from input to output",
        "  ·  The graph has a cut between the input cone and the output cone",
    ]
    for i, c in enumerate(causes):
        ax_diag.text(0.02, 0.42 - i * 0.07, c,
                     ha="left", va="top", fontsize=9, color=INK,
                     family="monospace")
    ax_diag.text(0.02, 0.04,
                 "The ontology panel and system graph above still describe what was specified.",
                 ha="left", va="top", fontsize=8.5, color=GRAY,
                 fontstyle="italic")

    # Border emphasising the diagnostic
    for spine in ax_diag.spines.values():
        spine.set_visible(True)
        spine.set_edgecolor(BRICK)
        spine.set_linewidth(1.0)

    plt.savefig(out_path, dpi=150)
    plt.close(fig)
