"""Figures for the 4-agent neural-circuit pipeline.

Two renderers:

  - `render_network_figure(spec, sys, out_path)` — Agent 3 output: the system
    as a directed network with a clear left-to-right input -> output flow,
    feedback loops drawn as dashed back-arcs, multiscale parents shown as
    bounding boxes.

  - `render_path_representation_figure(spec, sys, path_rep, out_path)` —
    Agent 4 output: time-unrolled DAG laid out in columns (one per time
    step), with each enumerated input->output path drawn as a coloured
    polyline. Feedforward paths and feedback-traversing paths use
    distinguishable colour palettes; the longest few paths are emphasised.
"""
from __future__ import annotations

import warnings
from typing import Optional

import matplotlib
if "matplotlib.pyplot" not in __import__("sys").modules:
    matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from ..paths import Path as PLPath
from ..landscape import PathLandscape
from ..similarity import composite_similarity
from ..system import System
from .neural_schemas import (
    ClassifiedPath,
    PathRepresentation,
    RouteCluster,
    UniqueRoute,
)
from .schemas import SystemSpec

warnings.filterwarnings("ignore", category=UserWarning)

INK = "#0c1220"
PAPER = "#fcfcfa"
AMBER = "#f0b65a"
CYAN = "#7dc8dc"
BRICK = "#c83c37"
LIME = "#94c47d"
GRAY = "#aab4c3"
GRAY_SOFT = "#d8d4c8"


# =============================================================== helpers


def _role_color(role: str) -> str:
    return {"input": CYAN, "output": BRICK}.get(role, AMBER)


def _build_left_right_layout(spec: SystemSpec) -> dict[str, tuple[float, float]]:
    """Place input units on the far left, outputs on the far right, internals
    arranged in between using a topological-ish layering of the spec's
    feedforward edges.

    Returns a mapping unit_name -> (x, y).
    """
    units = {u.name: u for u in spec.units}
    inputs = [u.name for u in spec.units if u.role == "input"]
    outputs = [u.name for u in spec.units if u.role == "output"]
    internals = [u.name for u in spec.units if u.role == "internal"]

    # Build a feedforward DAG (ignore recurrent edges) for layering.
    ff = nx.DiGraph()
    for u in spec.units:
        ff.add_node(u.name)
    for it in spec.interactions:
        if it.source in units and it.target in units and not it.recurrent:
            ff.add_edge(it.source, it.target)

    # Layer = longest path from any input (BFS depth on the feedforward DAG).
    layer: dict[str, int] = {n: 0 for n in inputs}
    # Topological order; if cycles remain (shouldn't, since recurrent is excluded),
    # fall back to BFS from inputs.
    try:
        order = list(nx.topological_sort(ff))
    except nx.NetworkXUnfeasible:
        order = list(nx.bfs_tree(ff, source=inputs[0] if inputs else
                                 next(iter(units)))) if units else []
    for n in order:
        preds = list(ff.predecessors(n))
        if preds:
            layer[n] = max(layer.get(p, 0) for p in preds) + 1
        else:
            layer.setdefault(n, 0)

    # Force inputs to layer 0 and outputs to the max layer.
    if inputs:
        for n in inputs:
            layer[n] = 0
    max_layer = max(layer.values()) if layer else 0
    if outputs:
        max_layer = max(max_layer, 1)
        for n in outputs:
            layer[n] = max_layer
    # Push internals between 0 and max_layer, never overlapping inputs/outputs.
    for n in internals:
        L = layer.get(n, 1)
        if L <= 0:
            L = 1
        if L >= max_layer:
            L = max_layer - 1 if max_layer >= 1 else 0
        layer[n] = max(0, L)

    # Group by layer, then place along y.
    by_layer: dict[int, list[str]] = {}
    for n, L in layer.items():
        by_layer.setdefault(L, []).append(n)

    pos: dict[str, tuple[float, float]] = {}
    # x scale: 0..1 across layers; y scale: spread within each layer.
    n_layers = max(by_layer) + 1 if by_layer else 1
    for L, names in by_layer.items():
        # group by parent so siblings within a module sit together
        names = sorted(names, key=lambda n: (units[n].parent or "", n))
        n_here = len(names)
        for i, name in enumerate(names):
            x = (L / max(1, n_layers - 1)) if n_layers > 1 else 0.5
            # center y around 0.5
            if n_here == 1:
                y = 0.5
            else:
                y = 0.05 + 0.9 * i / (n_here - 1)
            pos[name] = (x, y)
    return pos


def _draw_parent_boxes(ax, spec: SystemSpec,
                      pos: dict[str, tuple[float, float]]) -> None:
    """Draw a soft bounding box around each parent's children."""
    children_by_parent: dict[str, list[str]] = {}
    for u in spec.units:
        if u.parent:
            children_by_parent.setdefault(u.parent, []).append(u.name)
    pad = 0.04
    for parent, kids in children_by_parent.items():
        xs = [pos[k][0] for k in kids if k in pos]
        ys = [pos[k][1] for k in kids if k in pos]
        if not xs:
            continue
        x0, x1 = min(xs) - pad, max(xs) + pad
        y0, y1 = min(ys) - pad, max(ys) + pad
        rect = mpatches.FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle="round,pad=0.005,rounding_size=0.02",
            linewidth=0.9, edgecolor=GRAY, facecolor="#f5f2eb",
            alpha=0.55, zorder=1,
        )
        ax.add_patch(rect)
        ax.text(
            x0 + 0.005, y1 - 0.005, parent,
            ha="left", va="top", fontsize=7.5, color=GRAY,
            fontstyle="italic", zorder=2,
        )


# =============================================================== Agent 3


def _compute_edge_flows(
    path_rep: "PathRepresentation",
) -> dict[tuple[str, str], float]:
    """Aggregate information flow for every base-unit edge in the circuit.

    For each consecutive pair (src, dst) that appears in any unique route,
    sum the route's weight (= cumulative path weight through that route).
    The result maps (src, dst) -> total_flow and is used to scale edge
    thickness in the network figure: thicker edge = more information flows
    through that connection across all enumerated input→output paths.
    """
    flows: dict[tuple[str, str], float] = {}
    for route in path_rep.routes:
        chain = route.chain   # base-unit names, @t suffixes already stripped
        for a, b in zip(chain, chain[1:]):
            flows[(a, b)] = flows.get((a, b), 0.0) + route.weight
    return flows


def render_network_figure(
    spec: SystemSpec,
    sys: System,
    out_path: str,
    title_suffix: str = "",
    path_rep: Optional["PathRepresentation"] = None,
) -> None:
    """Network figure for Agent 3.

    Layout: input column on the left, output column on the right, internal
    units arranged into intermediate layers by feedforward depth. Feedforward
    edges drawn as solid arrows; recurrent edges drawn as red dashed back-arcs.
    Parent groupings (multiscale) drawn as rounded bounding boxes.

    When *path_rep* is supplied (Agent 4 output), **edge thickness encodes
    information flow**: for each directed connection the thickness is
    proportional to the total weight of all input→output routes that pass
    through it, so high-traffic edges are visually wider.  When path_rep is
    absent all edges are drawn with a uniform default width.
    """
    fig, ax = plt.subplots(figsize=(12, 7.5), facecolor=PAPER)
    ax.set_facecolor(PAPER)

    pos = _build_left_right_layout(spec)

    # 1) parent boxes (under everything)
    _draw_parent_boxes(ax, spec, pos)

    # 2) edges
    units = {u.name: u for u in spec.units}
    ff_edges = [
        (it.source, it.target)
        for it in spec.interactions
        if not it.recurrent and it.source in pos and it.target in pos
    ]
    rec_edges = [
        (it.source, it.target)
        for it in spec.interactions
        if it.recurrent and it.source in pos and it.target in pos
    ]

    # Per-edge widths from information flow (when path_rep is available).
    _FLOW_LW_MIN, _FLOW_LW_MAX = 0.5, 5.0
    if path_rep is not None and path_rep.routes:
        edge_flows = _compute_edge_flows(path_rep)
        max_flow = max(edge_flows.values()) if edge_flows else 1.0
        def _flow_lw(u: str, v: str) -> float:
            f = edge_flows.get((u, v), 0.0)
            return _FLOW_LW_MIN + (_FLOW_LW_MAX - _FLOW_LW_MIN) * (f / max_flow)
        ff_widths  = [_flow_lw(u, v) for u, v in ff_edges]  if ff_edges  else []
        rec_widths = [_flow_lw(u, v) for u, v in rec_edges] if rec_edges else []
        has_flow_data = True
    else:
        ff_widths  = [1.0] * len(ff_edges)
        rec_widths = [1.2] * len(rec_edges)
        has_flow_data = False

    # Build a tiny DiGraph just for nx drawing helpers
    g = nx.DiGraph()
    for n in pos:
        g.add_node(n)
    for u, v in ff_edges:
        g.add_edge(u, v, recurrent=False)
    for u, v in rec_edges:
        g.add_edge(u, v, recurrent=True)

    if ff_edges:
        nx.draw_networkx_edges(
            g, pos, edgelist=ff_edges, ax=ax,
            edge_color=INK, alpha=0.60, width=ff_widths,
            arrows=True, arrowsize=12, arrowstyle="-|>",
            connectionstyle="arc3,rad=0.05",
            node_size=900,
        )
    if rec_edges:
        nx.draw_networkx_edges(
            g, pos, edgelist=rec_edges, ax=ax,
            edge_color=BRICK, alpha=0.85, width=rec_widths,
            style="dashed",
            arrows=True, arrowsize=12, arrowstyle="-|>",
            connectionstyle="arc3,rad=-0.25",
            node_size=900,
        )

    # 3) nodes
    node_colors = [_role_color(units[n].role) if n in units else AMBER
                   for n in g.nodes]
    sizes = [950 if (n in units and units[n].role in ("input", "output")) else 700
             for n in g.nodes]
    nx.draw_networkx_nodes(
        g, pos, ax=ax, node_color=node_colors, node_size=sizes,
        edgecolors=INK, linewidths=0.9,
    )
    labels = {n: (n if len(n) <= 12 else n[:11] + "…") for n in g.nodes}
    nx.draw_networkx_labels(g, pos, labels=labels, ax=ax,
                            font_size=8, font_color=INK)

    # 4) input / output banners
    ax.text(-0.06, 1.04, "INPUT", ha="left", va="bottom",
            fontsize=12, fontweight="bold", color=CYAN)
    ax.text(1.06, 1.04, "OUTPUT", ha="right", va="bottom",
            fontsize=12, fontweight="bold", color=BRICK)
    ax.text(0.5, 1.04, "information flow →", ha="center", va="bottom",
            fontsize=10, color=GRAY, fontstyle="italic")

    # 5) legend
    legend_handles = [
        mpatches.Patch(color=CYAN, label="input unit"),
        mpatches.Patch(color=AMBER, label="internal unit"),
        mpatches.Patch(color=BRICK, label="output unit"),
        plt.Line2D([0], [0], color=INK, lw=1.2, label="feedforward edge"),
        plt.Line2D([0], [0], color=BRICK, lw=1.2, ls="--",
                   label="recurrent edge (feedback)"),
    ]
    if has_flow_data:
        legend_handles += [
            plt.Line2D([0], [0], color=INK, lw=_FLOW_LW_MIN + 0.3,
                       label=f"edge thickness = info flow (thin = low)"),
            plt.Line2D([0], [0], color=INK, lw=_FLOW_LW_MAX,
                       label=f"edge thickness = info flow (thick = high)"),
        ]
    ncol = min(7, len(legend_handles))
    ax.legend(handles=legend_handles, loc="lower center",
              bbox_to_anchor=(0.5, -0.08), ncol=ncol,
              frameon=False, fontsize=8.5)

    flow_note = "  ·  edge thickness ∝ information flow" if has_flow_data else ""
    suf = f" — {title_suffix}" if title_suffix else ""
    ax.set_title(
        f"Network representation: {spec.phenomenon_name}{suf}\n"
        f"{len(spec.units)} units · "
        f"{len(spec.interactions)} interactions "
        f"({sum(1 for i in spec.interactions if i.recurrent)} recurrent) · "
        f"T = {spec.time_steps}{flow_note}",
        fontsize=11, color=INK, pad=18,
    )

    ax.set_xlim(-0.12, 1.12)
    ax.set_ylim(-0.10, 1.10)
    ax.set_axis_off()
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PAPER)
    plt.close(fig)


# =============================================================== Agent 4


def classify_paths(
    sys: System,
    unrolled: nx.DiGraph,
    paths: list,
    spec: SystemSpec,
) -> list[ClassifiedPath]:
    """Tag each enumerated `Path` as feedforward vs. feedback-traversing.

    An edge `u@t -> v@(t+1)` is a feedback (recurrent) traversal iff the
    underlying spec interaction `(u, v)` is marked recurrent. Feedforward
    edges stay within a single time step.
    """
    rec_pairs: set[tuple[str, str]] = {
        (it.source, it.target) for it in spec.interactions if it.recurrent
    }

    classified: list[ClassifiedPath] = []
    for p in paths:
        nodes = tuple(p.nodes)
        # Strip "@t" suffixes to get the underlying base chain
        base_chain = tuple(n.split("@", 1)[0] for n in nodes)
        # Detect any recurrent edge traversal
        crosses = False
        ts: list[int] = []
        for i in range(len(nodes) - 1):
            src_base, dst_base = base_chain[i], base_chain[i + 1]
            try:
                t_src = int(nodes[i].split("@", 1)[1])
                t_dst = int(nodes[i + 1].split("@", 1)[1])
                ts.append(t_src)
            except (IndexError, ValueError):
                t_src = t_dst = 0
            # Recurrent if base pair is marked recurrent AND time advanced
            if (src_base, dst_base) in rec_pairs and t_dst > t_src:
                crosses = True
        if nodes:
            try:
                ts.append(int(nodes[-1].split("@", 1)[1]))
            except (IndexError, ValueError):
                ts.append(0)
        time_span = (max(ts) - min(ts)) if ts else 0
        classified.append(ClassifiedPath(
            nodes=nodes,
            base_chain=base_chain,
            length=len(nodes) - 1,
            crosses_feedback=crosses,
            weight=float(getattr(p, "weight", 1.0)),
            time_span=time_span,
        ))
    return classified


def estimate_time_scale(spec: SystemSpec, paths: list[ClassifiedPath]) -> int:
    """A simple, transparent estimate of the system's intrinsic time scale.

    Defined as the largest time_span observed among feedback-traversing paths,
    clamped to [1, time_steps]. If no feedback paths, the time scale is 1
    (purely feedforward).
    """
    feedback = [p for p in paths if p.crosses_feedback]
    if not feedback:
        return 1
    return max(1, min(int(spec.time_steps), max(p.time_span for p in feedback)))


def _build_unique_routes(classified: list[ClassifiedPath]) -> list[UniqueRoute]:
    """Collapse classified paths to unique routes by base-chain identity.

    Consecutive duplicates in the base chain are collapsed too — a path
    that lingers on a self-loop (e.g., u@0 -> u@1 -> u@2) folds onto the
    same chain as one that just visits u once.
    """
    bucket: dict[tuple[str, ...], dict] = {}
    for p in classified:
        chain: list[str] = []
        prev = None
        for n in p.base_chain:
            if n != prev:
                chain.append(n)
                prev = n
        key = tuple(chain)
        b = bucket.setdefault(key, {
            "count": 0,
            "weight": 0.0,
            "crosses_feedback": False,
        })
        b["count"] += 1
        b["weight"] += float(p.weight)
        b["crosses_feedback"] = b["crosses_feedback"] or p.crosses_feedback
    return [
        UniqueRoute(
            chain=k,
            count=v["count"],
            weight=v["weight"],
            crosses_feedback=v["crosses_feedback"],
        )
        for k, v in bucket.items()
    ]


def _auto_eps(
    D: "np.ndarray",
    min_samples: int = 2,
    max_noise_frac: float = 0.35,
    max_modes: int = 4,
) -> tuple[float, "np.ndarray"]:
    """Scan eps from tight → loose; return the eps + labels that maximise
    cluster count (up to max_modes) while keeping noise fraction ≤
    max_noise_frac.

    D is a precomputed distance matrix (D = 1 − similarity).
    Returns (best_eps, best_labels).

    max_modes caps the number of similarity clusters so the path-landscape
    figure remains legible. The algorithm picks the *tightest* eps that
    satisfies both constraints — more distinct modes → tighter grouping
    → clearer visual separation in the figure.
    """
    from sklearn.cluster import DBSCAN

    n = D.shape[0]
    # Scan from tight to loose.  LLM circuits with many unique-named intermediate
    # nodes (attention heads, MLP layers) tend to have higher pairwise distances
    # than bio circuits (even within-group distances can reach 0.60+), so we
    # extend the grid well past the old cap of 0.45.
    candidates = [
        0.05, 0.10, 0.15, 0.20, 0.25, 0.30,
        0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65,
    ]
    best_eps = 0.30
    best_n = 0
    best_labels: Optional["np.ndarray"] = None

    for eps_try in candidates:
        labels = DBSCAN(
            eps=eps_try, min_samples=min_samples, metric="precomputed"
        ).fit_predict(D)
        n_noise = int((labels == -1).sum())
        n_modes = len(set(labels.tolist()) - {-1})
        noise_frac = n_noise / n if n > 0 else 0.0
        # Accept: noise ≤ threshold AND at least 1 mode AND at most max_modes
        if noise_frac <= max_noise_frac and 1 <= n_modes <= max_modes:
            if n_modes > best_n:
                best_n = n_modes
                best_eps = eps_try
                best_labels = labels

    if best_labels is None:
        # Nothing satisfied both constraints — relax max_modes, try again
        for eps_try in candidates:
            labels = DBSCAN(
                eps=eps_try, min_samples=min_samples, metric="precomputed"
            ).fit_predict(D)
            n_noise = int((labels == -1).sum())
            n_modes = len(set(labels.tolist()) - {-1})
            noise_frac = n_noise / n if n > 0 else 0.0
            if noise_frac <= max_noise_frac and n_modes >= 1:
                # take loosest acceptable eps (keeps iterating, last survives)
                best_labels = labels
                best_eps = eps_try

    if best_labels is None:
        # Nothing satisfied the constraint — fall back to default
        best_labels = DBSCAN(
            eps=0.30, min_samples=min_samples, metric="precomputed"
        ).fit_predict(D)

    return best_eps, best_labels


def _cluster_routes_by_similarity(
    routes: list[UniqueRoute],
    eps: Optional[float] = None,
    min_samples: int = 2,
    max_noise_frac: float = 0.35,
) -> tuple[list[UniqueRoute], list[RouteCluster], int]:
    """Cluster unique routes by composite (edge-Jaccard + LCS) similarity.

    Routes that share many of the same nodes/edges in similar order end up
    in the same cluster. Returns (routes_with_cluster_ids, cluster_summaries,
    n_modes). Noise routes are tagged cluster_id=-1.

    When ``eps`` is None (default), the eps is *auto-tuned*: the algorithm
    scans from tight (eps=0.05, similarity≥0.95) to loose (eps=0.50,
    similarity≥0.50) and picks the threshold that maximises the number of
    distinct clusters while keeping noise routes ≤ ``max_noise_frac``
    of the total. This reliably produces more fine-grained modes than a
    fixed eps=0.30.
    """
    if not routes:
        return routes, [], 0
    if len(routes) == 1:
        routes[0].cluster_id = 0
        chain = routes[0].chain
        summary = RouteCluster(
            cluster_id=0, size=1, total_count=routes[0].count,
            representative_chain=chain,
            mean_length=float(len(chain) - 1),
            n_feedforward=0 if routes[0].crosses_feedback else 1,
            n_feedback=1 if routes[0].crosses_feedback else 0,
            shared_units=list(chain),
        )
        return routes, [summary], 1

    # Build PathLandscape on the unique routes (use route count as weight)
    paths = [PLPath(nodes=r.chain, weight=float(r.count)) for r in routes]
    L = PathLandscape(paths, kernel=composite_similarity)

    # Auto-tune eps or use the caller-supplied value.
    # max_modes=4 caps visual complexity: > 4 modes produce a crowded figure.
    if eps is None:
        _chosen_eps, labels = _auto_eps(L.D, min_samples=min_samples,
                                        max_noise_frac=max_noise_frac,
                                        max_modes=4)
        # Sync PathLandscape labels so downstream code (sub-matrix access) works
        L._labels = labels
    else:
        L.cluster(eps=eps, min_samples=min_samples)
        labels = L.labels

    for r, lab in zip(routes, labels):
        r.cluster_id = int(lab)

    # Build cluster summaries
    summaries: list[RouteCluster] = []
    for cid in sorted(set(int(l) for l in labels) - {-1}):
        members = [r for r, l in zip(routes, labels) if int(l) == cid]
        idx = [i for i, l in enumerate(labels) if int(l) == cid]
        if not members:
            continue
        sub = L.S[np.ix_(idx, idx)]
        mean_sim = sub.mean(axis=1)
        rep_local = idx[int(np.argmax(mean_sim))]
        rep_chain = routes[rep_local].chain
        lengths = [len(m.chain) - 1 for m in members]
        # Shared units = intersection of all chains in the cluster.
        # Preserve order in the representative and deduplicate (chains may
        # repeat a unit several times due to loops).
        common = set(members[0].chain)
        for m in members[1:]:
            common &= set(m.chain)
        shared: list[str] = []
        seen: set[str] = set()
        for u in rep_chain:
            if u in common and u not in seen:
                shared.append(u); seen.add(u)
        summaries.append(RouteCluster(
            cluster_id=int(cid),
            size=len(members),
            total_count=sum(m.count for m in members),
            representative_chain=rep_chain,
            mean_length=float(np.mean(lengths)),
            n_feedforward=sum(1 for m in members if not m.crosses_feedback),
            n_feedback=sum(1 for m in members if m.crosses_feedback),
            shared_units=shared,
        ))

    n_modes = len(summaries)
    return routes, summaries, n_modes


def build_path_representation(
    spec: SystemSpec,
    sys: System,
    unrolled: nx.DiGraph,
    raw_paths: list,
    cluster_eps: Optional[float] = None,
    cluster_min_samples: int = 2,
    cluster_max_noise_frac: float = 0.35,
) -> PathRepresentation:
    classified = classify_paths(sys, unrolled, raw_paths, spec)
    if not classified:
        return PathRepresentation(
            time_steps=int(spec.time_steps),
            intrinsic_time_scale=0,
            paths=[], routes=[], clusters=[], n_modes=0,
            n_feedforward_paths=0,
            n_feedback_paths=0,
            min_length=0, max_length=0, mean_length=0.0,
            notes="no input->output paths after unrolling",
        )
    lengths = [p.length for p in classified]
    n_fb = sum(1 for p in classified if p.crosses_feedback)

    routes = _build_unique_routes(classified)
    routes, clusters, n_modes = _cluster_routes_by_similarity(
        routes,
        eps=cluster_eps,
        min_samples=cluster_min_samples,
        max_noise_frac=cluster_max_noise_frac,
    )

    return PathRepresentation(
        time_steps=int(spec.time_steps),
        intrinsic_time_scale=estimate_time_scale(spec, classified),
        paths=classified,
        routes=routes,
        clusters=clusters,
        n_modes=n_modes,
        n_feedforward_paths=len(classified) - n_fb,
        n_feedback_paths=n_fb,
        min_length=min(lengths),
        max_length=max(lengths),
        mean_length=float(np.mean(lengths)),
    )


def render_paper_circuit_figure(
    paper_info,            # PaperInfo – avoid circular import at module level
    system_info,           # NeuralSystemInfo
    out_path: str,
) -> None:
    """Schematic circuit diagram sourced from the anchor paper (Agent 1/2).

    Renders the brain regions / computational modules identified by Agent 2
    as a left-to-right information-flow diagram, annotated with neuron types,
    circuit motifs, and up to three key findings from the anchor paper.
    This gives Agent 3's network figure a literature grounding by showing
    the canonical circuit as reported in the identified paper.

    Layout
    ------
    - Brain regions as coloured rounded boxes, left → right (input → output).
    - Solid arrows connect adjacent regions (feedforward flow).
    - A curved dashed red arc is added when motifs mention 'recurrent' or
      'feedback'.
    - Neuron types are listed beneath each matching region box.
    - Key findings (up to 3) appear in a panel at the bottom-left.
    - Circuit motifs appear in a panel at the bottom-right.
    - Title block cites the paper.
    """
    regions = list(system_info.brain_regions)[:6]
    motifs = list(system_info.key_circuit_motifs)[:4]
    neuron_types = list(system_info.neuron_types)[:8]
    key_findings = list(paper_info.key_findings)[:3]

    if not regions:
        # Fallback: draw a placeholder indicating no region data.
        fig, ax = plt.subplots(figsize=(12, 5), facecolor=PAPER)
        ax.set_facecolor(PAPER); ax.set_axis_off()
        ax.text(0.5, 0.55, system_info.system_name or "(circuit)",
                ha="center", va="center", fontsize=16, color=INK)
        ax.text(0.5, 0.38,
                f"{paper_info.title}\n{paper_info.authors} · {paper_info.year}",
                ha="center", va="center", fontsize=9, color=GRAY)
        plt.savefig(out_path, dpi=150, facecolor=PAPER)
        plt.close(fig)
        return

    n_regions = len(regions)

    fig, ax = plt.subplots(figsize=(13, 7), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(-0.08, 1.08)
    ax.set_ylim(-0.08, 1.08)
    ax.set_axis_off()

    # ---- palette for region boxes ----
    _REG_COLORS = [CYAN, AMBER, "#8c5db6", "#5b9a6b", BRICK, "#3f63b5"]

    # ---- horizontal positions of region box centres ----
    box_w, box_h = 0.13, 0.14
    y_center = 0.62
    if n_regions == 1:
        xs_center = [0.5]
    else:
        xs_center = [0.12 + i * (0.76 / (n_regions - 1))
                     for i in range(n_regions)]

    # ---- draw region boxes ----
    for i, (region, xc) in enumerate(zip(regions, xs_center)):
        col = _REG_COLORS[i % len(_REG_COLORS)]

        rect = mpatches.FancyBboxPatch(
            (xc - box_w / 2, y_center - box_h / 2), box_w, box_h,
            boxstyle="round,pad=0.012,rounding_size=0.018",
            linewidth=1.8, edgecolor=col,
            facecolor=col, alpha=0.18, zorder=2,
        )
        ax.add_patch(rect)

        # Region label (truncate long names)
        label = region if len(region) <= 16 else region[:14] + "…"
        ax.text(xc, y_center, label,
                ha="center", va="center",
                fontsize=8.5, fontweight="bold", color=col, zorder=3)

        # Neuron types that "belong" to this region (heuristic: first token
        # of the region name appears in the neuron-type string).
        region_key = region.lower().split()[0][:4]
        nt_here = [nt for nt in neuron_types
                   if region_key in nt.lower()][:2]
        for j, nt in enumerate(nt_here):
            nt_short = nt if len(nt) <= 20 else nt[:18] + "…"
            ax.text(xc, y_center - box_h / 2 - 0.04 - j * 0.055,
                    nt_short,
                    ha="center", va="top",
                    fontsize=7, color=col, fontstyle="italic", zorder=3)

    # ---- feedforward arrows between adjacent regions ----
    for i in range(n_regions - 1):
        x0 = xs_center[i] + box_w / 2 + 0.008
        x1 = xs_center[i + 1] - box_w / 2 - 0.008
        ax.annotate(
            "", xy=(x1, y_center), xytext=(x0, y_center),
            arrowprops=dict(
                arrowstyle="-|>", color=INK, lw=1.6,
                connectionstyle="arc3,rad=0.0",
            ),
            zorder=4,
        )

    # ---- recurrent back-arc (if any motif mentions feedback/recurrent) ----
    has_recurrent = any(
        any(kw in m.lower()
            for kw in ("recurrent", "feedback", "loop", "re-entrant"))
        for m in motifs
    )
    if has_recurrent and n_regions >= 2:
        arc_y = y_center + 0.21
        ax.annotate(
            "", xy=(xs_center[0] - box_w / 2, y_center + 0.06),
            xytext=(xs_center[-1] + box_w / 2, y_center + 0.06),
            arrowprops=dict(
                arrowstyle="-|>", color=BRICK, lw=1.5, ls="dashed",
                connectionstyle=f"arc3,rad=-0.38",
            ),
            zorder=4,
        )
        ax.text(0.5 * (xs_center[0] + xs_center[-1]), arc_y,
                "recurrent / feedback projection",
                ha="center", va="center",
                fontsize=7.5, color=BRICK, fontstyle="italic", zorder=3)

    # ---- INPUT / OUTPUT labels ----
    ax.text(xs_center[0] - box_w / 2 - 0.03, y_center,
            "INPUT\n→",
            ha="right", va="center",
            fontsize=9.5, fontweight="bold", color=CYAN, zorder=3)
    ax.text(xs_center[-1] + box_w / 2 + 0.03, y_center,
            "→\nOUTPUT",
            ha="left", va="center",
            fontsize=9.5, fontweight="bold", color=BRICK, zorder=3)

    # ---- Key findings panel (bottom-left) ----
    if key_findings:
        findings_text = "\n".join(f"• {f}" for f in key_findings)
        ax.text(0.01, 0.33,
                "Key findings:",
                ha="left", va="top",
                fontsize=8.5, fontweight="bold", color=INK)
        ax.text(0.01, 0.27,
                findings_text,
                ha="left", va="top",
                fontsize=7.5, color=INK,
                wrap=True,
                bbox=dict(facecolor="#f5f5ef", edgecolor=GRAY_SOFT,
                          boxstyle="round,pad=0.35", alpha=0.85),
                zorder=2)

    # ---- Motifs panel (bottom-right) ----
    if motifs:
        motif_text = "\n".join(f"◆ {m}" for m in motifs)
        ax.text(0.60, 0.33,
                "Circuit motifs:",
                ha="left", va="top",
                fontsize=8.5, fontweight="bold", color=INK)
        ax.text(0.60, 0.27,
                motif_text,
                ha="left", va="top",
                fontsize=7.5, color=INK,
                bbox=dict(facecolor="#f0f0f8", edgecolor=GRAY_SOFT,
                          boxstyle="round,pad=0.35", alpha=0.85),
                zorder=2)

    # ---- title: paper citation ----
    raw_title = paper_info.title or "(untitled)"
    short_title = raw_title if len(raw_title) <= 72 else raw_title[:70] + "…"
    authors_year = f"{paper_info.authors} · {paper_info.year}"
    if paper_info.venue:
        authors_year += f" · {paper_info.venue}"
    ax.set_title(
        f"Circuit schematic — {short_title}\n{authors_year}",
        fontsize=10, color=INK, pad=14,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, facecolor=PAPER)
    plt.close(fig)


# Categorical palette for cluster colors. Includes one neutral entry
# (last) used for noise routes (cluster_id = -1).
_CLUSTER_PALETTE = [
    "#4f8aa0",  # teal
    "#d49a2e",  # gold
    "#8c5db6",  # purple
    "#5b9a6b",  # green
    "#c8643a",  # rust
    "#3f63b5",  # blue
    "#b94c8a",  # magenta
    "#2f8a8a",  # cyan-dark
    "#a8a13a",  # olive
    "#7a5a3c",  # brown
]
_NOISE_COLOR = "#9aa3b3"


def _cluster_color(cluster_id: int) -> str:
    if cluster_id < 0:
        return _NOISE_COLOR
    return _CLUSTER_PALETTE[cluster_id % len(_CLUSTER_PALETTE)]


def render_path_representation_figure(
    spec: SystemSpec,
    sys: System,
    path_rep: PathRepresentation,
    out_path: str,
    max_routes_to_draw: int = 20,
) -> None:
    """Render Agent 4's output as a *path distribution* (no time axis).

    Each unique input→output route gets its own non-overlapping horizontal
    lane. All lanes start at a single shared INPUT node on the left, fan
    out to the lane's y-position, carry the route's intermediate units
    along a straight horizontal segment, then converge back at a shared
    OUTPUT node on the right.

      INPUT ●─╮ ╭─── ●A ─── ●B ─── ●C ─╮ ╭─● OUTPUT
              │ │                       │ │
              │ ╰─── ●A ─── ●D ────────╮│ │
              │                         ╰╮│
              ╰─── ●E ────── ●F ────────╯│
                                         ╯

    The y-coordinate carries *no* meaning beyond "different lane = different
    route" — there is no time dimension. Routes are grouped by similarity
    cluster (the *modes* of the path landscape): each cluster's lanes are
    stacked together with a small vertical gap separating clusters.
    Line color = cluster id (mode); line style = solid for feedforward-only,
    dashed for feedback-traversing. **Line thickness encodes information
    flow**: it scales with ``route.weight``, the cumulative path weight
    (product of edge weights × multiplicity) of all raw paths that collapsed
    onto that route.  When all edge weights are 1.0 this equals the raw count;
    in weighted circuits it properly reflects the strength of each route.
    """
    all_routes = list(path_rep.routes)

    # --- Empty case --------------------------------------------------
    if not all_routes:
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=PAPER)
        ax.set_facecolor(PAPER); ax.set_axis_off()
        ax.text(0.5, 0.5,
                "No input → output paths\n(re-specify the circuit "
                "or increase time_steps)",
                ha="center", va="center", fontsize=14, color=BRICK)
        plt.savefig(out_path, dpi=150, facecolor=PAPER)
        plt.close(fig)
        return

    # --- Choose which routes to render -------------------------------
    # Group by cluster, ordering clusters by size (largest first), and
    # within each cluster sort by popularity. Noise routes (cluster -1)
    # go to the bottom.
    cluster_sizes = {c.cluster_id: c.size for c in path_rep.clusters}
    cluster_order = sorted(
        set(r.cluster_id for r in all_routes),
        key=lambda cid: (cid == -1, -cluster_sizes.get(cid, 0), cid),
    )
    by_cluster: dict[int, list[UniqueRoute]] = {cid: [] for cid in cluster_order}
    for r in all_routes:
        by_cluster[r.cluster_id].append(r)
    for cid in by_cluster:
        by_cluster[cid].sort(key=lambda r: (-r.count, len(r.chain)))

    # If too many routes total, subsample within each cluster proportionally,
    # always keeping at least one route per cluster.
    total = sum(len(v) for v in by_cluster.values())
    if total > max_routes_to_draw:
        kept: dict[int, list[UniqueRoute]] = {}
        leftover = max_routes_to_draw - len(by_cluster)
        for cid, members in by_cluster.items():
            share = max(1, int(round(
                len(members) / total * (max_routes_to_draw - len(by_cluster))
            )) + 1)
            share = min(share, len(members))
            kept[cid] = members[:share]
        # If undershot, fill from the largest clusters
        kept_total = sum(len(v) for v in kept.values())
        if kept_total < max_routes_to_draw:
            for cid in cluster_order:
                while kept_total < max_routes_to_draw and len(kept[cid]) < len(by_cluster[cid]):
                    kept[cid].append(by_cluster[cid][len(kept[cid])])
                    kept_total += 1
        by_cluster = kept

    # Flat list of (cluster_id, route) in drawing order
    routes_in_order: list[tuple[int, UniqueRoute]] = []
    for cid in cluster_order:
        for r in by_cluster.get(cid, []):
            routes_in_order.append((cid, r))
    n = len(routes_in_order)
    if n == 0:
        # extremely defensive
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=PAPER)
        ax.set_axis_off()
        plt.savefig(out_path, dpi=150, facecolor=PAPER); plt.close(fig)
        return

    # Figure size: keep a landscape aspect.
    longest = max((len(r.chain) for _, r in routes_in_order), default=4)
    fig_w = max(15.0, 1.4 * longest + 8.0)
    fig_h = max(6.5, 0.55 * n + 2.5 + 0.4 * (len(cluster_order) - 1))
    fig_h = min(fig_h, 0.85 * fig_w)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_axis_off()

    # Use route.weight (cumulative path weight) as the information-flow proxy.
    # Falls back gracefully to count when all edge weights are 1.0.
    max_flow = max(r.weight for _, r in routes_in_order) or 1.0
    _ROUTE_LW_MIN, _ROUTE_LW_MAX = 0.8, 5.5   # wider range → more contrast

    # Shared input / output anchor positions
    x_in, y_in = 0.04, 0.5
    x_out, y_out = 0.96, 0.5
    lane_x0, lane_x1 = 0.22, 0.78  # the horizontal stretch carrying the route

    # Assign y-positions: lanes packed top-to-bottom, with a small gap
    # between clusters so groups are visually separated.
    lane_top = 0.95
    lane_bot = 0.05
    n_clusters_shown = sum(1 for cid in cluster_order if by_cluster.get(cid))
    # Wider gap between clusters so modes are visually distinct (especially
    # with ≤ 4 modes each cluster has plenty of vertical room).
    inter_cluster_gap = 0.06 if n_clusters_shown > 1 else 0.0
    total_gap = inter_cluster_gap * max(0, n_clusters_shown - 1)
    avail = (lane_top - lane_bot) - total_gap
    lane_step = avail / max(1, n)

    lane_ys: list[float] = []
    cluster_y_extents: dict[int, tuple[float, float]] = {}  # cluster_id -> (y_top, y_bottom)
    y_cursor = lane_top
    for cid in cluster_order:
        members = by_cluster.get(cid, [])
        if not members:
            continue
        y_top_here = y_cursor
        for _ in members:
            lane_ys.append(y_cursor)
            y_cursor -= lane_step
        cluster_y_extents[cid] = (y_top_here + 0.5 * lane_step,
                                  y_cursor + 0.5 * lane_step)
        y_cursor -= inter_cluster_gap

    # --- Draw each route ---------------------------------------------
    roles = {u.name: u.role for u in spec.units}
    for i, (cid, route) in enumerate(routes_in_order):
        chain = route.chain
        interior = list(chain[1:-1]) if len(chain) >= 2 else []

        # Interior x-positions evenly within [lane_x0, lane_x1]
        if len(interior) == 0:
            interior_xs: list[float] = []
        elif len(interior) == 1:
            interior_xs = [0.5 * (lane_x0 + lane_x1)]
        else:
            interior_xs = list(np.linspace(lane_x0, lane_x1, len(interior)))

        y_lane = float(lane_ys[i])

        # Build the polyline:
        #   (x_in, y_in) -> (lane_x0, y_lane) -> [interior nodes] -> (lane_x1, y_lane) -> (x_out, y_out)
        xs = [x_in, lane_x0] + interior_xs + [lane_x1, x_out]
        ys = [y_in, y_lane] + [y_lane] * len(interior) + [y_lane, y_out]

        color = _cluster_color(cid)
        # Solid = feedforward-only, dashed = feedback-traversing
        linestyle = "--" if route.crosses_feedback else "-"
        # line width encodes information flow (route.weight = cumulative path weight)
        lw = _ROUTE_LW_MIN + (_ROUTE_LW_MAX - _ROUTE_LW_MIN) * (route.weight / max_flow)
        ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.88,
                linestyle=linestyle,
                solid_capstyle="round", solid_joinstyle="round",
                dash_capstyle="round", zorder=3)

        # Interior nodes — labels rendered inline with the lane (over the
        # line itself, on a white-ish bbox), so they never collide with the
        # lane above or below.
        if interior:
            ax.scatter(interior_xs, [y_lane] * len(interior),
                       color=_route_node_color(interior, roles),
                       s=80, edgecolors=INK, linewidths=0.5, zorder=4)
            for x, name in zip(interior_xs, interior):
                ax.text(
                    x, y_lane, name,
                    ha="center", va="center",
                    fontsize=7.5, color=INK, zorder=5,
                    bbox=dict(facecolor=PAPER, edgecolor=GRAY_SOFT,
                              boxstyle="round,pad=0.18", linewidth=0.4,
                              alpha=0.95),
                )

        # Lane badge: raw path count + flow weight
        badge = (f"×{route.count}"
                 if abs(route.weight - route.count) < 0.01
                 else f"×{route.count}  w={route.weight:.2g}")
        ax.text(lane_x1 + 0.015, y_lane, badge,
                ha="left", va="center", fontsize=7.5, color=GRAY, zorder=5)

    # --- Cluster labels and brackets ---------------------------------
    # Left margin: vertical bracket spanning each cluster's lanes plus a
    # text label ("Mode 0", "Mode 1", or "noise").
    for cid in cluster_order:
        if cid not in cluster_y_extents:
            continue
        y_top, y_bot = cluster_y_extents[cid]
        x_b = 0.105   # x of the bracket
        ax.plot([x_b, x_b], [y_bot, y_top],
                color=_cluster_color(cid), linewidth=2.4, alpha=0.85,
                solid_capstyle="round", zorder=2)
        # short horizontal ticks at the top and bottom of the bracket
        ax.plot([x_b, x_b + 0.012], [y_top, y_top],
                color=_cluster_color(cid), linewidth=2.0, zorder=2)
        ax.plot([x_b, x_b + 0.012], [y_bot, y_bot],
                color=_cluster_color(cid), linewidth=2.0, zorder=2)
        label = ("noise" if cid == -1 else f"Mode {cid}")
        size = (sum(1 for _, r in routes_in_order if r.cluster_id == cid))
        ax.text(x_b - 0.008, 0.5 * (y_top + y_bot), f"{label}\n({size})",
                ha="right", va="center",
                fontsize=8.5, fontweight="bold",
                color=_cluster_color(cid), zorder=2)

    # --- Shared INPUT and OUTPUT nodes -------------------------------
    ax.scatter([x_in], [y_in], s=850, color=CYAN, edgecolors=INK,
               linewidths=1.6, zorder=8)
    ax.text(x_in, y_in, "INPUT", ha="center", va="center",
            fontsize=10, fontweight="bold", color=INK, zorder=9)
    ax.scatter([x_out], [y_out], s=850, color=BRICK, edgecolors=INK,
               linewidths=1.6, zorder=8)
    ax.text(x_out, y_out, "OUTPUT", ha="center", va="center",
            fontsize=10, fontweight="bold", color=PAPER, zorder=9)

    # --- Title and counts --------------------------------------------
    total_routes = len(all_routes)
    n_ff_total = sum(1 for r in all_routes if not r.crosses_feedback)
    n_fb_total = total_routes - n_ff_total
    n_modes = path_rep.n_modes
    suffix = (f"  (showing {n} of {total_routes})"
              if n < total_routes else "")
    ax.set_title(
        f"Path distribution: {spec.phenomenon_name}\n"
        f"{total_routes} distinct routes  ·  "
        f"{n_modes} similarity mode{'s' if n_modes != 1 else ''}  ·  "
        f"feedforward: {n_ff_total}  ·  "
        f"feedback-traversing: {n_fb_total}{suffix}",
        fontsize=13, color=INK, pad=14,
    )

    # --- Legend ------------------------------------------------------
    # Cluster colour swatches + line-style key + node + multiplicity hint
    legend_handles: list = []
    # cluster swatches (only modes that are actually shown)
    for cid in cluster_order:
        if cid not in cluster_y_extents:
            continue
        size = sum(1 for _, r in routes_in_order if r.cluster_id == cid)
        label = "noise" if cid == -1 else f"Mode {cid}"
        legend_handles.append(plt.Line2D(
            [0], [0], color=_cluster_color(cid), lw=3.0,
            label=f"{label} ({size})",
        ))
    # line-style key
    legend_handles.append(plt.Line2D(
        [0], [0], color=INK, lw=2.0, linestyle="-",
        label="feedforward-only"))
    legend_handles.append(plt.Line2D(
        [0], [0], color=INK, lw=2.0, linestyle="--",
        label="feedback-traversing"))
    legend_handles.append(plt.Line2D(
        [0], [0], color=GRAY, lw=0, marker="o",
        markerfacecolor=AMBER, markeredgecolor=INK,
        markersize=7, label="intermediate unit"))
    legend_handles.append(plt.Line2D(
        [0], [0], color=INK, lw=_ROUTE_LW_MIN + 0.3,
        label="thin = low information flow"))
    legend_handles.append(plt.Line2D(
        [0], [0], color=INK, lw=_ROUTE_LW_MAX,
        label="thick = high information flow"))
    legend_handles.append(plt.Line2D(
        [0], [0], color="none", marker="",
        label="× N = raw paths  ·  w = cumulative path weight"))

    ncol = min(5, max(2, len(legend_handles) // 2))
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=ncol, frameon=False, fontsize=9)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.06)
    plt.savefig(out_path, dpi=150, facecolor=PAPER)
    plt.close(fig)


def _route_node_color(interior: list[str], roles: dict[str, str]) -> list[str]:
    """Color interior nodes by their role (almost always internal,
    but be defensive if an input/output sneaks into the middle)."""
    out = []
    for n in interior:
        r = roles.get(n, "internal")
        out.append(_role_color(r))
    return out


# =============================================================== Agent 4 (cluster bundle view)
#
# Alternative renderer that makes the *similarity cluster* the primary
# visual object instead of the individual route. Each mode is drawn as
# one horizontal band containing:
#
#   - its representative route as a thick centerline,
#   - all other members of the cluster overlaid as faint translucent
#     curves with a slight y-jitter (the "bundle"),
#   - shared anchor nodes (units that appear in EVERY member route of
#     the cluster) drawn as large amber circles on the centerline —
#     these are the structural fingerprint of the mode.
#
# A user looking at this figure sees the path landscape *as* the
# clustering: a tight stack of bundles, each one anchored on the units
# that all its routes share.

def render_path_clusters_figure(
    spec: SystemSpec,
    sys: System,
    path_rep: PathRepresentation,
    out_path: str,
    max_routes_per_cluster: int = 8,
    include_noise: bool = True,
) -> None:
    """Render the path landscape as a stack of similarity-cluster bundles.

    Each row of the figure is one mode (cluster) of the path landscape:
    a bundle of similar routes drawn through their shared anchor units.
    The bundle's thickness conveys how many routes the mode contains;
    the anchor units (intersection of all member chains) are the
    structural signature of the mode.

    This view is complementary to ``render_path_representation_figure``
    (which draws one lane per route) — here the *cluster* is the
    primary object, not the individual path.
    """
    # ---------- gather data ---------------------------------------
    routes_by_cluster: dict[int, list[UniqueRoute]] = {}
    for r in path_rep.routes:
        routes_by_cluster.setdefault(r.cluster_id, []).append(r)

    # Real clusters (size >= 1, cluster_id >= 0), largest first
    clusters = sorted(
        [c for c in path_rep.clusters if c.cluster_id >= 0],
        key=lambda c: -c.size,
    )

    # Synthesise a pseudo-cluster for noise routes (cluster_id == -1) if
    # the user asked for it and any exist.
    noise_routes = routes_by_cluster.get(-1, [])
    if include_noise and noise_routes:
        # build a fake RouteCluster summary for the noise group
        rep = max(noise_routes, key=lambda r: r.count)
        common = set(noise_routes[0].chain)
        for r in noise_routes[1:]:
            common &= set(r.chain)
        shared = [u for u in rep.chain if u in common]
        lengths = [len(r.chain) - 1 for r in noise_routes]
        noise_summary = RouteCluster(
            cluster_id=-1,
            size=len(noise_routes),
            total_count=sum(r.count for r in noise_routes),
            representative_chain=rep.chain,
            mean_length=float(np.mean(lengths)) if lengths else 0.0,
            n_feedforward=sum(1 for r in noise_routes if not r.crosses_feedback),
            n_feedback=sum(1 for r in noise_routes if r.crosses_feedback),
            shared_units=shared,
        )
        clusters.append(noise_summary)

    # ---------- empty case ----------------------------------------
    if not clusters:
        fig, ax = plt.subplots(figsize=(14, 6), facecolor=PAPER)
        ax.set_facecolor(PAPER); ax.set_axis_off()
        ax.text(0.5, 0.5,
                "No similarity modes\n(no clusterable routes — see paths figure)",
                ha="center", va="center", fontsize=14, color=BRICK)
        plt.savefig(out_path, dpi=150, facecolor=PAPER)
        plt.close(fig)
        return

    # ---------- figure & layout -----------------------------------
    n_clusters = len(clusters)
    longest = max(
        (len(c.representative_chain) for c in clusters), default=4
    )
    fig_w = max(15.0, 1.4 * longest + 8.0)
    # taller bands when there are few clusters, shorter when many
    band_h_target = max(0.6, min(1.6, 4.5 / max(1, n_clusters)))
    fig_h = max(6.5, band_h_target * n_clusters + 2.0)
    fig_h = min(fig_h, 0.85 * fig_w)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h), facecolor=PAPER)
    ax.set_facecolor(PAPER)
    ax.set_xlim(-0.06, 1.06)
    ax.set_ylim(-0.06, 1.06)
    ax.set_axis_off()

    # x-anchors
    x_in, y_in = 0.04, 0.5
    x_out, y_out = 0.96, 0.5
    lane_x0, lane_x1 = 0.24, 0.78
    band_left, band_right = 0.14, 0.92

    # y-layout: stack bands top-to-bottom
    band_top, band_bot = 0.93, 0.07
    # Wider inter-band gap: with ≤ 4 modes each band gets generous height,
    # and the gap makes mode boundaries unmistakeable.
    inter_band = 0.06 if n_clusters > 1 else 0.0
    total_gap = inter_band * max(0, n_clusters - 1)
    band_height = ((band_top - band_bot) - total_gap) / n_clusters

    # Global scale for line width: use route.weight (information flow proxy).
    # When all edge weights are 1.0, weight == count — no visual change.
    all_weights = [r.weight for rs in routes_by_cluster.values() for r in rs]
    max_weight = max(all_weights) if all_weights else 1.0
    _CLUS_LW_MIN, _CLUS_LW_MAX = 0.6, 5.0

    roles = {u.name: u.role for u in spec.units}

    # ---------- draw each cluster band ----------------------------
    y_cursor = band_top
    for cluster in clusters:
        band_y_top = y_cursor
        band_y_bot = y_cursor - band_height
        band_center = 0.5 * (band_y_top + band_y_bot)
        y_cursor = band_y_bot - inter_band

        cid = cluster.cluster_id
        color = _cluster_color(cid)

        # member routes for this cluster, sorted by popularity
        members = sorted(routes_by_cluster.get(cid, []),
                         key=lambda r: -r.count)
        if not members:
            continue

        # pick the representative — match the cluster's stored
        # representative_chain to one of the member routes if possible
        rep_route = next(
            (m for m in members if tuple(m.chain) == tuple(cluster.representative_chain)),
            members[0],
        )
        rep_chain = rep_route.chain
        rep_interior = list(rep_chain[1:-1]) if len(rep_chain) >= 2 else []
        if len(rep_interior) == 0:
            rep_xs_interior: list[float] = []
        elif len(rep_interior) == 1:
            rep_xs_interior = [0.5 * (lane_x0 + lane_x1)]
        else:
            rep_xs_interior = list(
                np.linspace(lane_x0, lane_x1, len(rep_interior))
            )

        # ---- band background tint ----
        bg = mpatches.Rectangle(
            (band_left, band_y_bot),
            band_right - band_left, band_height,
            facecolor=color, alpha=0.05,
            edgecolor=color, linewidth=0.6, zorder=1,
        )
        ax.add_patch(bg)

        # ---- overlay non-rep member routes as the BUNDLE ----
        other_members = [m for m in members
                         if m is not rep_route][:max_routes_per_cluster - 1]
        if other_members:
            jitter_span = band_height * 0.35
            for mi, route in enumerate(other_members):
                chain = route.chain
                interior = list(chain[1:-1]) if len(chain) >= 2 else []
                if len(interior) == 0:
                    xs_interior: list[float] = []
                elif len(interior) == 1:
                    xs_interior = [0.5 * (lane_x0 + lane_x1)]
                else:
                    xs_interior = list(
                        np.linspace(lane_x0, lane_x1, len(interior))
                    )
                if len(other_members) == 1:
                    y_offset = 0.0
                else:
                    y_offset = ((mi / (len(other_members) - 1)) - 0.5) * jitter_span
                y_m = band_center + y_offset

                xs = [x_in, lane_x0] + xs_interior + [lane_x1, x_out]
                ys = ([y_in, y_m] + [y_m] * len(interior) + [y_m, y_out])

                ls = "--" if route.crosses_feedback else "-"
                lw = _CLUS_LW_MIN + (_CLUS_LW_MAX - _CLUS_LW_MIN) * (route.weight / max_weight)
                ax.plot(xs, ys, color=color, linewidth=lw, alpha=0.30,
                        linestyle=ls, zorder=3,
                        solid_capstyle="round", dash_capstyle="round")
                # very faint interior dots so the overlay is readable
                if xs_interior:
                    ax.scatter(xs_interior, [y_m] * len(xs_interior),
                               s=22, color=color, alpha=0.45,
                               edgecolors="none", zorder=3)

        # ---- representative route along the centerline ----
        xs_rep = [x_in, lane_x0] + rep_xs_interior + [lane_x1, x_out]
        ys_rep = ([y_in, band_center]
                  + [band_center] * len(rep_interior)
                  + [band_center, y_out])
        ls_rep = "--" if rep_route.crosses_feedback else "-"
        lw_rep = _CLUS_LW_MIN + (_CLUS_LW_MAX - _CLUS_LW_MIN) * (rep_route.weight / max_weight)
        lw_rep = max(lw_rep, 1.8)  # representative is always at least 1.8 wide
        ax.plot(xs_rep, ys_rep, color=color, linewidth=lw_rep, alpha=0.95,
                linestyle=ls_rep, zorder=5,
                solid_capstyle="round", dash_capstyle="round")

        # ---- anchor nodes (shared across all members) ----
        # Shared units get a prominent amber halo behind the text;
        # variable units get a small empty circle with a faint bbox.
        # The amber circle is drawn LARGER than the text bbox so the
        # halo is visible (the bbox is rendered without facecolor so the
        # amber shows through).
        shared_set = set(cluster.shared_units)
        for name, x in zip(rep_interior, rep_xs_interior):
            is_shared = name in shared_set
            if is_shared:
                # prominent amber halo (drawn first / below text)
                ax.scatter([x], [band_center],
                           s=620, color=AMBER, edgecolors=INK,
                           linewidths=1.3, zorder=6)
                # text directly on the halo, no opaque bbox
                ax.text(x, band_center, name,
                        ha="center", va="center",
                        fontsize=7.5, fontweight="bold", color=INK,
                        zorder=8)
            else:
                # rep-only / variable node — small empty circle
                ax.scatter([x], [band_center],
                           s=70, color=PAPER, edgecolors=color,
                           linewidths=1.0, zorder=6)
                ax.text(x, band_center, name,
                        ha="center", va="center",
                        fontsize=7.5, color=INK, zorder=7,
                        bbox=dict(facecolor=PAPER, edgecolor=GRAY_SOFT,
                                  boxstyle="round,pad=0.18",
                                  linewidth=0.4, alpha=0.95))

        # ---- left-margin cluster label ----
        truncated = len(members) - 1 - len(other_members)
        label = ("noise group" if cid == -1 else f"Mode {cid}")
        label_text = (f"{label}\n"
                      f"{cluster.size} routes  ·  ×{cluster.total_count}")
        if cluster.shared_units:
            label_text += f"\nshared: {len(cluster.shared_units)} unit(s)"
        if truncated > 0:
            label_text += f"\n(+{truncated} more in bundle)"
        ax.text(0.020, band_center, label_text,
                ha="left", va="center",
                fontsize=8.2, fontweight="bold",
                color=color, zorder=2)

        # ---- right-margin: bundle popularity badge ----
        ax.text(0.985, band_center,
                f"×{rep_route.count}", ha="right", va="center",
                fontsize=8, color=GRAY, zorder=2)

    # ---------- shared INPUT / OUTPUT anchors ---------------------
    # Larger circles + labels rendered just outside so they never clip
    # against the circle edge regardless of font metrics.
    ax.scatter([x_in], [y_in], s=1400, color=CYAN, edgecolors=INK,
               linewidths=1.8, zorder=10)
    ax.text(x_in, y_in, "in", ha="center", va="center",
            fontsize=11, fontweight="bold", color=INK, zorder=11)
    ax.text(x_in, y_in - 0.045, "INPUT", ha="center", va="top",
            fontsize=9, fontweight="bold", color=INK, zorder=11)
    ax.scatter([x_out], [y_out], s=1400, color=BRICK, edgecolors=INK,
               linewidths=1.8, zorder=10)
    ax.text(x_out, y_out, "out", ha="center", va="center",
            fontsize=11, fontweight="bold", color=PAPER, zorder=11)
    ax.text(x_out, y_out - 0.045, "OUTPUT", ha="center", va="top",
            fontsize=9, fontweight="bold", color=INK, zorder=11)

    # ---------- title ---------------------------------------------
    n_modes = path_rep.n_modes
    total_routes = len(path_rep.routes)
    total_paths = len(path_rep.paths)
    ax.set_title(
        f"Path landscape (cluster view): {spec.phenomenon_name}\n"
        f"{n_modes} similarity mode{'s' if n_modes != 1 else ''}  ·  "
        f"{total_routes} unique routes  ·  "
        f"{total_paths} raw paths",
        fontsize=13, color=INK, pad=14,
    )

    # ---------- legend --------------------------------------------
    legend_handles = [
        plt.Line2D([0], [0], color=INK, lw=2.4, linestyle="-",
                   label="representative route (mode centerline)"),
        plt.Line2D([0], [0], color=INK, lw=1.0, linestyle="-",
                   alpha=0.4, label="other member routes (bundle)"),
        plt.Line2D([0], [0], color=INK, lw=2.0, linestyle="--",
                   label="route traverses feedback edge"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=AMBER, markeredgecolor=INK,
                   markersize=10,
                   label="shared anchor unit (in every member route)"),
        plt.Line2D([0], [0], marker="o", color="none",
                   markerfacecolor=PAPER, markeredgecolor=GRAY,
                   markersize=7,
                   label="variable unit (rep-only / not in every member)"),
        plt.Line2D([0], [0], color=INK, lw=_CLUS_LW_MIN + 0.3,
                   label="thin = low information flow"),
        plt.Line2D([0], [0], color=INK, lw=_CLUS_LW_MAX,
                   label="thick = high information flow"),
    ]
    ax.legend(handles=legend_handles,
              loc="upper center", bbox_to_anchor=(0.5, -0.01),
              ncol=3, frameon=False, fontsize=9)

    plt.subplots_adjust(left=0.03, right=0.97, top=0.92, bottom=0.08)
    plt.savefig(out_path, dpi=150, facecolor=PAPER)
    plt.close(fig)
