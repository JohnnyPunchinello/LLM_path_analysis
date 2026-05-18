"""Generate the four illustrations for the biological vs. artificial
emergence paper.

  fig_path_landscape_definition.png
  fig_biological_landscape.png
  fig_artificial_landscape.png
  fig_comparison_metrics.png

Each figure is a stand-alone PNG written to the same directory.
"""
from __future__ import annotations

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, Rectangle, FancyBboxPatch
import networkx as nx


HERE = os.path.dirname(os.path.abspath(__file__))

# Palette (kept consistent with the companion paper's TikZ palette)
INK = "#0C1220"
PAPER = "#FCFCFA"
AMBER = "#C88A2A"
CYAN = "#2A88A0"
BRICK = "#B8332E"
SLATE = "#4A5468"
TEAL = "#4f8aa0"
GOLD = "#d49a2e"
PURPLE = "#8c5db6"
GREEN = "#5b9a6b"
RUST = "#c8643a"
GRAY = "#aab4c3"
GRAY_SOFT = "#d8d4c8"

CLUSTER_COLORS = [TEAL, GOLD, PURPLE, GREEN, RUST]


# ============================================================ Figure 1


def fig1_path_landscape_definition() -> None:
    """Three-panel conceptual figure:
       (A) the System (directed graph with recurrent edge)
       (B) the unrolled DAG over T=3
       (C) the path landscape: routes clustered into similarity modes."""
    fig = plt.figure(figsize=(15, 5.0), facecolor=PAPER)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.4, 1.4], wspace=0.20)

    # --- Panel A: the System -------------------------------------------------
    axA = fig.add_subplot(gs[0, 0]); axA.set_facecolor(PAPER); axA.set_axis_off()
    axA.set_xlim(-0.1, 1.1); axA.set_ylim(-0.15, 1.15)

    # Units
    nodes = {
        "in":   (0.05, 0.50),
        "u1":   (0.35, 0.78),
        "u2":   (0.35, 0.22),
        "u3":   (0.65, 0.50),
        "out":  (0.95, 0.50),
    }
    edges = [
        ("in",  "u1", False),
        ("in",  "u2", False),
        ("u1",  "u3", False),
        ("u2",  "u3", False),
        ("u3",  "out", False),
        ("u3",  "u1",  True),     # recurrent edge
    ]
    for (a, b, rec) in edges:
        x1, y1 = nodes[a]; x2, y2 = nodes[b]
        rad = -0.32 if rec else 0.0
        col = BRICK if rec else INK
        ls = "dashed" if rec else "solid"
        arrow = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle=f"arc3,rad={rad}",
            arrowstyle="-|>", mutation_scale=12,
            color=col, linewidth=1.4 if rec else 1.0,
            linestyle=ls, alpha=0.85,
        )
        axA.add_patch(arrow)
    for name, (x, y) in nodes.items():
        if name == "in":
            c = CYAN; lbl = "in"
        elif name == "out":
            c = BRICK; lbl = "out"
        else:
            c = AMBER; lbl = name
        axA.scatter([x], [y], s=750, c=c, edgecolors=INK, linewidths=1.3, zorder=5)
        axA.text(x, y, lbl, ha="center", va="center",
                 fontsize=10, fontweight="bold", color=INK, zorder=6)
    axA.text(0.5, 1.06, "(A) System $G$ with recurrent edge",
             ha="center", fontsize=11, fontweight="bold", color=INK)
    axA.text(0.5, -0.08, "directed graph; one edge $u_3\\to u_1$ is recurrent",
             ha="center", fontsize=8.5, color=SLATE, style="italic")

    # --- Panel B: the unrolled DAG -----------------------------------------
    axB = fig.add_subplot(gs[0, 1]); axB.set_facecolor(PAPER); axB.set_axis_off()
    axB.set_xlim(-0.05, 1.05); axB.set_ylim(-0.15, 1.15)

    T = 3
    layer_x = {0: 0.18, 1: 0.44, 2: 0.70, 3: 0.92}
    rowy   = {"in": 0.92, "u1": 0.72, "u2": 0.48, "u3": 0.24, "out": 0.06}
    base = ["in", "u1", "u2", "u3", "out"]
    # Build positions for each (unit, t) — only show units that can be on a path.
    pos: dict[str, tuple[float, float]] = {}
    for t in range(T):
        for u in base:
            # Inputs only at t=0; outputs at t=T-1; internals at all t
            if u == "in" and t != 0:    continue
            if u == "out" and t != T-1: continue
            pos[f"{u}@{t}"] = (layer_x[t], rowy[u])

    g = nx.DiGraph()
    for n in pos: g.add_node(n)
    # Feedforward edges
    ff_edges = [
        ("in@0", "u1@0"),
        ("in@0", "u2@0"),
        ("u1@0", "u3@0"),
        ("u2@0", "u3@0"),
        ("u1@1", "u3@1"),
        ("u2@1", "u3@1"),
        ("u1@2", "u3@2"),
        ("u2@2", "u3@2"),
        ("u3@2", "out@2"),
    ]
    # Time-edges that "carry over" non-recurrent state across t (optional, not drawn)
    # Recurrent edge u3@t -> u1@(t+1) shown in BRICK
    rec_edges = [
        ("u3@0", "u1@1"),
        ("u3@1", "u1@2"),
    ]
    for a, b in ff_edges + rec_edges:
        if a in pos and b in pos:
            g.add_edge(a, b, rec=(a, b) in rec_edges)

    # Time slab backgrounds
    for t in range(T):
        rect = Rectangle((layer_x[t]-0.10, -0.02), 0.20, 1.04,
                         facecolor="#f5f2eb", alpha=0.55, edgecolor=GRAY_SOFT,
                         linewidth=0.6, zorder=0)
        axB.add_patch(rect)
        axB.text(layer_x[t], 1.06, f"$t={t}$",
                 ha="center", fontsize=10, color=SLATE)

    for u, v in g.edges():
        rec = g.edges[u, v].get("rec", False)
        x1, y1 = pos[u]; x2, y2 = pos[v]
        col = BRICK if rec else INK
        ls = "dashed" if rec else "solid"
        ar = FancyArrowPatch(
            (x1, y1), (x2, y2),
            connectionstyle="arc3,rad=0.05",
            arrowstyle="-|>", mutation_scale=9,
            color=col, linewidth=1.4 if rec else 0.9,
            linestyle=ls, alpha=0.78, zorder=2)
        axB.add_patch(ar)

    for name, (x, y) in pos.items():
        base_u = name.split("@")[0]
        if base_u == "in":
            c = CYAN
        elif base_u == "out":
            c = BRICK
        else:
            c = AMBER
        axB.scatter([x], [y], s=240, c=c, edgecolors=INK, linewidths=0.9, zorder=5)
        axB.text(x, y - 0.04, name, ha="center", va="top",
                 fontsize=7, color=INK, zorder=6)

    axB.text(0.5, 1.13, "(B) Unrolled DAG $G^{(T)}$",
             ha="center", fontsize=11, fontweight="bold", color=INK)
    axB.text(0.5, -0.10, "every recurrent edge becomes a forward time-edge",
             ha="center", fontsize=8.5, color=SLATE, style="italic")

    # --- Panel C: the path landscape (modes) -----------------------------
    axC = fig.add_subplot(gs[0, 2]); axC.set_facecolor(PAPER); axC.set_axis_off()
    axC.set_xlim(-0.05, 1.05); axC.set_ylim(-0.15, 1.15)

    # Each lane = one route; group three routes per mode
    in_x, out_x = 0.08, 0.92
    in_y, out_y = 0.52, 0.52
    lane_x0, lane_x1 = 0.22, 0.78

    # Mode 0 (teal): top band — 2 short feedforward routes
    # Mode 1 (gold): middle band — 3 long feedback-traversing routes
    # Mode 2 (purple): bottom band — 2 medium routes
    band_centers = [0.86, 0.50, 0.14]
    band_colors  = [TEAL, GOLD, PURPLE]
    band_labels  = ["Mode 0 (short, FF)",
                    "Mode 1 (long, feedback)",
                    "Mode 2 (medium)"]
    # n_routes per mode
    nr = [2, 3, 2]

    for m in range(3):
        center = band_centers[m]
        color = band_colors[m]
        # background band
        bw = 0.14
        axC.add_patch(Rectangle((0.10, center - bw/2), 0.82, bw,
                                facecolor=color, alpha=0.07,
                                edgecolor=color, linewidth=0.6, zorder=0))
        axC.text(0.05, center, band_labels[m], ha="right", va="center",
                 fontsize=8.5, color=color, fontweight="bold")
        # individual route lanes
        offsets = np.linspace(-bw/3, bw/3, nr[m]) if nr[m] > 1 else [0]
        for k, off in enumerate(offsets):
            y = center + off
            # interior nodes — mode 1 (gold) has 3 interior nodes (long), mode 0 has 1, mode 2 has 2
            n_int = {0:1, 1:3, 2:2}[m]
            xs_int = np.linspace(lane_x0, lane_x1, n_int) if n_int > 1 else [0.5*(lane_x0+lane_x1)]
            xs = [in_x, lane_x0] + list(xs_int) + [lane_x1, out_x]
            ys = [in_y, y] + [y]*n_int + [y, out_y]
            # thicker line in the centre of the band (representative)
            is_rep = (k == nr[m]//2)
            lw = 3.0 if is_rep else 1.3
            alpha = 0.95 if is_rep else 0.55
            ls = "--" if m == 1 else "-"   # Mode 1 traverses feedback (dashed)
            axC.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                     linestyle=ls,
                     solid_capstyle="round", dash_capstyle="round", zorder=3)
            if is_rep:
                axC.scatter(xs_int, [y]*n_int, s=70, color=color,
                            edgecolors=INK, linewidths=0.6, zorder=4)

    # Input / output anchors
    axC.scatter([in_x], [in_y], s=550, color=CYAN, edgecolors=INK,
                linewidths=1.3, zorder=8)
    axC.text(in_x, in_y, "in", ha="center", va="center",
             fontsize=9, fontweight="bold", color=INK, zorder=9)
    axC.scatter([out_x], [out_y], s=550, color=BRICK, edgecolors=INK,
                linewidths=1.3, zorder=8)
    axC.text(out_x, out_y, "out", ha="center", va="center",
             fontsize=9, fontweight="bold", color=PAPER, zorder=9)

    axC.text(0.5, 1.10, "(C) Path landscape $\\mathcal{L}$",
             ha="center", fontsize=11, fontweight="bold", color=INK)
    axC.text(0.5, -0.10,
             "routes clustered by edge / order similarity into modes",
             ha="center", fontsize=8.5, color=SLATE, style="italic")

    out_path = os.path.join(HERE, "fig_path_landscape_definition.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out_path}")


# ============================================================ Figure 2


def fig2_biological_landscape() -> None:
    """Biological path landscape: hippocampal CA3 pattern completion.

    Layout: DG (input) -> CA3 ensemble (recurrent) -> CA1 -> output.
    Show 4 routes that all loop through the same CA3 ensemble many times
    -> few, long, anchor-rich modes with high concentration.
    """
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.0), facecolor=PAPER,
                                    gridspec_kw={"width_ratios": [1.2, 1.0]})
    # ----- Left panel: anatomy with recurrent CA3 -----
    axL.set_facecolor(PAPER); axL.set_axis_off()
    axL.set_xlim(-0.05, 1.05); axL.set_ylim(-0.10, 1.15)

    # Units
    DG    = (0.06, 0.50)
    CA3 = [
        ("CA3_p1", (0.30, 0.72)),
        ("CA3_p2", (0.32, 0.50)),
        ("CA3_p3", (0.30, 0.28)),
        ("CA3_p4", (0.42, 0.62)),
        ("CA3_p5", (0.42, 0.38)),
    ]
    CA1 = [
        ("CA1_p1", (0.62, 0.62)),
        ("CA1_p2", (0.62, 0.38)),
    ]
    OUT = (0.92, 0.50)

    # Multiscale parent boxes
    axL.add_patch(FancyBboxPatch((0.23, 0.20), 0.27, 0.62,
                                   boxstyle="round,pad=0.01,rounding_size=0.02",
                                   facecolor="#f5f2eb", edgecolor=GRAY,
                                   linewidth=0.9, alpha=0.6, zorder=0))
    axL.text(0.36, 0.85, "CA3 (recurrent ensemble)",
             ha="center", fontsize=9.5, color=SLATE, style="italic")
    axL.add_patch(FancyBboxPatch((0.55, 0.30), 0.16, 0.42,
                                   boxstyle="round,pad=0.01,rounding_size=0.02",
                                   facecolor="#f5f2eb", edgecolor=GRAY,
                                   linewidth=0.9, alpha=0.6, zorder=0))
    axL.text(0.63, 0.75, "CA1", ha="center", fontsize=9.5,
             color=SLATE, style="italic")

    # DG → CA3 (feedforward, mossy fiber)
    for name, (x, y) in CA3:
        ar = FancyArrowPatch(DG, (x, y),
                              arrowstyle="-|>", mutation_scale=10,
                              color=INK, alpha=0.55, linewidth=1.0,
                              connectionstyle="arc3,rad=0.05", zorder=2)
        axL.add_patch(ar)
    # CA3 recurrent collaterals — dense, drawn as red dashed arcs
    ca3_names = [n for n, _ in CA3]
    ca3_pos = {n: p for n, p in CA3}
    rec_pairs = [
        ("CA3_p1", "CA3_p2"), ("CA3_p2", "CA3_p1"),
        ("CA3_p1", "CA3_p4"), ("CA3_p4", "CA3_p1"),
        ("CA3_p2", "CA3_p3"), ("CA3_p3", "CA3_p2"),
        ("CA3_p2", "CA3_p4"), ("CA3_p4", "CA3_p5"),
        ("CA3_p5", "CA3_p3"), ("CA3_p3", "CA3_p1"),
        ("CA3_p2", "CA3_p5"),
    ]
    for a, b in rec_pairs:
        x1, y1 = ca3_pos[a]; x2, y2 = ca3_pos[b]
        ar = FancyArrowPatch((x1, y1), (x2, y2),
                              arrowstyle="-|>", mutation_scale=7,
                              color=BRICK, alpha=0.55, linewidth=0.85,
                              linestyle="dashed",
                              connectionstyle="arc3,rad=-0.25", zorder=1)
        axL.add_patch(ar)
    # CA3 → CA1 (Schaffer)
    for a, _ in CA3:
        for b, _ in CA1:
            x1, y1 = ca3_pos[a]; x2, y2 = CA1[0][1] if b == CA1[0][0] else CA1[1][1]
            ar = FancyArrowPatch((x1, y1), (x2, y2),
                                  arrowstyle="-|>", mutation_scale=8,
                                  color=INK, alpha=0.30, linewidth=0.8,
                                  connectionstyle="arc3,rad=0.05", zorder=2)
            axL.add_patch(ar)
    # CA1 → OUT
    for _, (x, y) in CA1:
        ar = FancyArrowPatch((x, y), OUT,
                              arrowstyle="-|>", mutation_scale=10,
                              color=INK, alpha=0.65, linewidth=1.1,
                              connectionstyle="arc3,rad=0.05", zorder=2)
        axL.add_patch(ar)

    # Nodes
    axL.scatter([DG[0]], [DG[1]], s=850, c=CYAN, edgecolors=INK,
                linewidths=1.4, zorder=5)
    axL.text(DG[0], DG[1], "DG", ha="center", va="center",
             fontsize=10, fontweight="bold", color=INK, zorder=6)
    axL.text(DG[0], DG[1]-0.07, "(input)", ha="center", va="top",
             fontsize=8, color=SLATE, style="italic", zorder=6)
    for name, (x, y) in CA3:
        axL.scatter([x], [y], s=460, c=AMBER, edgecolors=INK,
                    linewidths=0.9, zorder=5)
        axL.text(x, y, name.replace("CA3_", ""), ha="center", va="center",
                 fontsize=7.5, color=INK, zorder=6)
    for name, (x, y) in CA1:
        axL.scatter([x], [y], s=460, c=AMBER, edgecolors=INK,
                    linewidths=0.9, zorder=5)
        axL.text(x, y, name.replace("CA1_", ""), ha="center", va="center",
                 fontsize=7.5, color=INK, zorder=6)
    axL.scatter([OUT[0]], [OUT[1]], s=850, c=BRICK, edgecolors=INK,
                linewidths=1.4, zorder=5)
    axL.text(OUT[0], OUT[1], "out", ha="center", va="center",
             fontsize=10, fontweight="bold", color=PAPER, zorder=6)

    axL.text(0.5, 1.07,
             "(A) Hippocampal CA3 pattern completion: anatomy",
             ha="center", fontsize=11.5, fontweight="bold", color=INK)
    axL.text(0.5, -0.07,
             "feedforward DG$\\to$CA3$\\to$CA1$\\to$out; dense recurrent CA3 collaterals (red dashed)",
             ha="center", fontsize=8.5, color=SLATE, style="italic")

    # ----- Right panel: path landscape — few, long, overlapping routes -----
    axR.set_facecolor(PAPER); axR.set_axis_off()
    axR.set_xlim(-0.05, 1.05); axR.set_ylim(-0.10, 1.15)

    in_x, in_y, out_x, out_y = 0.06, 0.50, 0.94, 0.50
    lane_x0, lane_x1 = 0.18, 0.82

    # Two modes only (concentrated landscape). Mode 0 is dominant.
    # Mode 0 (teal): 4 routes, all long (8 interior nodes), recurrent-traversing,
    # all share 5 anchor units in the middle
    # Mode 1 (purple): 2 routes, medium length (5 interior nodes), fewer shared anchors
    band_centers = [0.66, 0.26]
    band_colors  = [TEAL, PURPLE]
    band_labels  = ["Mode 0\n(4 routes, dominant)",
                    "Mode 1\n(2 routes)"]
    n_routes     = [4, 2]
    n_interiors  = [7, 4]
    interior_names = [
        ["p1", "p2", "p4", "p2", "p3", "p2", "CA1_p1"],   # mode 0 chain (long, revisits CA3 pyramidals)
        ["p1", "p5", "p3", "CA1_p2"],                       # mode 1 chain (shorter)
    ]
    shared_anchors_idx = [[0, 1, 2, 3, 5], [0, 2]]   # which interior positions are "shared anchors"

    for m in range(2):
        center = band_centers[m]
        color = band_colors[m]
        bw = 0.30 if m == 0 else 0.18
        axR.add_patch(Rectangle((0.10, center - bw/2), 0.82, bw,
                                facecolor=color, alpha=0.07,
                                edgecolor=color, linewidth=0.6, zorder=0))
        axR.text(0.04, center, band_labels[m], ha="right", va="center",
                 fontsize=9, color=color, fontweight="bold")

        # offsets across the band for individual routes
        offsets = np.linspace(-bw*0.35, bw*0.35, n_routes[m]) if n_routes[m]>1 else [0]
        n_int = n_interiors[m]
        xs_int = np.linspace(lane_x0, lane_x1, n_int)

        for k, off in enumerate(offsets):
            y = center + off
            xs = [in_x, lane_x0] + list(xs_int) + [lane_x1, out_x]
            ys = [in_y, y] + [y]*n_int + [y, out_y]
            is_rep = (k == n_routes[m]//2)
            lw = 4.2 if is_rep else 1.6   # thicker — concentrated flow
            alpha = 0.95 if is_rep else 0.55
            ls = "--" if m == 0 else "-"
            axR.plot(xs, ys, color=color, linewidth=lw, alpha=alpha,
                     linestyle=ls, solid_capstyle="round",
                     dash_capstyle="round", zorder=3)
            if is_rep:
                for j, (xj, nm) in enumerate(zip(xs_int, interior_names[m])):
                    is_anchor = j in shared_anchors_idx[m]
                    if is_anchor:
                        axR.scatter([xj], [y], s=380, c=AMBER,
                                    edgecolors=INK, linewidths=1.0, zorder=4)
                    else:
                        axR.scatter([xj], [y], s=70, c=PAPER,
                                    edgecolors=color, linewidths=1.0, zorder=4)
                    axR.text(xj, y, nm, ha="center", va="center",
                             fontsize=6.5,
                             fontweight=("bold" if is_anchor else "normal"),
                             color=INK, zorder=5)

    axR.scatter([in_x], [in_y], s=750, color=CYAN, edgecolors=INK,
                linewidths=1.4, zorder=8)
    axR.text(in_x, in_y, "DG", ha="center", va="center",
             fontsize=9, fontweight="bold", color=INK, zorder=9)
    axR.scatter([out_x], [out_y], s=750, color=BRICK, edgecolors=INK,
                linewidths=1.4, zorder=8)
    axR.text(out_x, out_y, "out", ha="center", va="center",
             fontsize=9, fontweight="bold", color=PAPER, zorder=9)

    axR.text(0.5, 1.07,
             "(B) Biological path landscape: few, long, anchor-rich",
             ha="center", fontsize=11.5, fontweight="bold", color=INK)
    axR.text(0.5, -0.07,
             "thick lines $=$ concentrated flow; amber circles $=$ shared anchor units",
             ha="center", fontsize=8.5, color=SLATE, style="italic")

    plt.tight_layout()
    out_path = os.path.join(HERE, "fig_biological_landscape.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out_path}")


# ============================================================ Figure 3


def fig3_artificial_landscape() -> None:
    """Artificial path landscape: GPT-2 IOI circuit.

    Left: stylised transformer architecture (token embed -> attention heads
    at layers L7-L11 -> logit). Right: many short, depth-uniform routes
    grouped into many specialised modes with low anchor overlap."""
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(15, 6.0), facecolor=PAPER,
                                    gridspec_kw={"width_ratios": [1.2, 1.0]})
    # ----- Left panel: transformer architecture with attention heads -----
    axL.set_facecolor(PAPER); axL.set_axis_off()
    axL.set_xlim(-0.05, 1.05); axL.set_ylim(-0.10, 1.15)

    # Layout: columns = layers, rows = head positions
    # Tokens at far left, logits at far right
    tok = (0.04, 0.50)
    # Specialised heads (Wang et al. 2022 IOI circuit)
    heads = [
        # (name, layer_x, y, kind)
        ("duplicate_L0",  0.18, 0.85, "duplicate"),
        ("prev_token_L2", 0.18, 0.18, "duplicate"),
        ("induction_L5H5", 0.32, 0.85, "induction"),
        ("induction_L5H8", 0.32, 0.62, "induction"),
        ("induction_L6H9", 0.32, 0.18, "induction"),
        ("s_inhib_L7H3",   0.50, 0.85, "s_inhib"),
        ("s_inhib_L7H9",   0.50, 0.62, "s_inhib"),
        ("s_inhib_L8H6",   0.50, 0.38, "s_inhib"),
        ("s_inhib_L8H10",  0.50, 0.14, "s_inhib"),
        ("name_mover_L9H6",  0.70, 0.78, "name_mover"),
        ("name_mover_L9H9",  0.70, 0.55, "name_mover"),
        ("name_mover_L10H0", 0.70, 0.32, "name_mover"),
        ("backup_L10H7",     0.70, 0.10, "backup"),
        ("neg_name_L11H10",  0.84, 0.50, "neg"),
    ]
    logit = (0.96, 0.50)

    # Layer columns as faint backgrounds
    layer_xs = [0.18, 0.32, 0.50, 0.70, 0.84]
    layer_labels = ["L0–L2", "L5–L6", "L7–L8", "L9–L10", "L11"]
    for lx, ll in zip(layer_xs, layer_labels):
        axL.add_patch(Rectangle((lx-0.06, 0.02), 0.12, 0.98,
                                 facecolor="#f5f2eb", alpha=0.45,
                                 edgecolor=GRAY_SOFT, linewidth=0.5, zorder=0))
        axL.text(lx, 1.03, ll, ha="center", fontsize=9, color=SLATE)

    head_color = {
        "duplicate":  GREEN,
        "induction":  GOLD,
        "s_inhib":    BRICK,
        "name_mover": TEAL,
        "backup":     PURPLE,
        "neg":        RUST,
    }
    head_pos = {h[0]: (h[1], h[2]) for h in heads}

    # Tok → first-layer heads
    for name, x, y, kind in heads:
        if kind in ("duplicate",):
            ar = FancyArrowPatch(tok, (x, y),
                                  arrowstyle="-|>", mutation_scale=8,
                                  color=INK, alpha=0.45, linewidth=0.7,
                                  connectionstyle="arc3,rad=0.04", zorder=2)
            axL.add_patch(ar)
    # Induction heads receive from duplicate / prev-token via residual stream
    for src in ("duplicate_L0", "prev_token_L2"):
        for tgt in ("induction_L5H5", "induction_L5H8", "induction_L6H9"):
            ar = FancyArrowPatch(head_pos[src], head_pos[tgt],
                                  arrowstyle="-|>", mutation_scale=8,
                                  color=INK, alpha=0.30, linewidth=0.6,
                                  connectionstyle="arc3,rad=0.05", zorder=2)
            axL.add_patch(ar)
    # Induction → S-inhibition
    for src in ("induction_L5H5", "induction_L5H8", "induction_L6H9"):
        for tgt in ("s_inhib_L7H3", "s_inhib_L7H9", "s_inhib_L8H6", "s_inhib_L8H10"):
            ar = FancyArrowPatch(head_pos[src], head_pos[tgt],
                                  arrowstyle="-|>", mutation_scale=8,
                                  color=INK, alpha=0.25, linewidth=0.55,
                                  connectionstyle="arc3,rad=0.05", zorder=2)
            axL.add_patch(ar)
    # S-inhibition → Name movers
    for src in ("s_inhib_L7H3", "s_inhib_L7H9", "s_inhib_L8H6", "s_inhib_L8H10"):
        for tgt in ("name_mover_L9H6", "name_mover_L9H9", "name_mover_L10H0"):
            ar = FancyArrowPatch(head_pos[src], head_pos[tgt],
                                  arrowstyle="-|>", mutation_scale=8,
                                  color=INK, alpha=0.25, linewidth=0.55,
                                  connectionstyle="arc3,rad=0.05", zorder=2)
            axL.add_patch(ar)
    # Backup name mover (used when L9 ablated)
    for src in ("s_inhib_L7H3", "s_inhib_L7H9"):
        ar = FancyArrowPatch(head_pos[src], head_pos["backup_L10H7"],
                              arrowstyle="-|>", mutation_scale=8,
                              color=INK, alpha=0.20, linewidth=0.5,
                              connectionstyle="arc3,rad=0.06", zorder=2)
        axL.add_patch(ar)
    # Negative name mover
    for src in ("induction_L5H8", "s_inhib_L8H6"):
        ar = FancyArrowPatch(head_pos[src], head_pos["neg_name_L11H10"],
                              arrowstyle="-|>", mutation_scale=8,
                              color=INK, alpha=0.20, linewidth=0.5,
                              connectionstyle="arc3,rad=0.06", zorder=2)
        axL.add_patch(ar)
    # Name movers, backup, neg → logit
    for src in ("name_mover_L9H6", "name_mover_L9H9", "name_mover_L10H0",
                "backup_L10H7", "neg_name_L11H10"):
        ar = FancyArrowPatch(head_pos[src], logit,
                              arrowstyle="-|>", mutation_scale=10,
                              color=INK, alpha=0.55, linewidth=0.9,
                              connectionstyle="arc3,rad=0.04", zorder=2)
        axL.add_patch(ar)

    # Nodes
    axL.scatter([tok[0]], [tok[1]], s=750, c=CYAN, edgecolors=INK,
                linewidths=1.4, zorder=5)
    axL.text(tok[0], tok[1], "tok", ha="center", va="center",
             fontsize=9, fontweight="bold", color=INK, zorder=6)
    axL.text(tok[0], tok[1]-0.07, "embed", ha="center", va="top",
             fontsize=7.5, color=SLATE, style="italic", zorder=6)
    for name, x, y, kind in heads:
        c = head_color[kind]
        axL.scatter([x], [y], s=320, c=c, edgecolors=INK,
                    linewidths=0.7, zorder=5)
        # short label
        short = name.split("_")[-1]   # e.g. L9H6
        axL.text(x, y, short, ha="center", va="center",
                 fontsize=6.5, color=INK, zorder=6)
    axL.scatter([logit[0]], [logit[1]], s=750, c=BRICK,
                edgecolors=INK, linewidths=1.4, zorder=5)
    axL.text(logit[0], logit[1], "logit", ha="center", va="center",
             fontsize=8.5, fontweight="bold", color=PAPER, zorder=6)

    # Tiny legend for head kinds
    leg = [
        mpatches.Patch(color=GREEN,  label="duplicate / prev-token"),
        mpatches.Patch(color=GOLD,   label="induction head"),
        mpatches.Patch(color=BRICK,  label="S-inhibition head"),
        mpatches.Patch(color=TEAL,   label="name mover head"),
        mpatches.Patch(color=PURPLE, label="backup name mover"),
        mpatches.Patch(color=RUST,   label="negative name mover"),
    ]
    axL.legend(handles=leg, loc="lower center", bbox_to_anchor=(0.5, -0.12),
                ncol=3, frameon=False, fontsize=7.5)

    axL.text(0.5, 1.10,
             "(A) GPT-2 small IOI circuit: anatomy",
             ha="center", fontsize=11.5, fontweight="bold", color=INK)

    # ----- Right panel: many short, disjoint, depth-uniform routes -----
    axR.set_facecolor(PAPER); axR.set_axis_off()
    axR.set_xlim(-0.05, 1.05); axR.set_ylim(-0.10, 1.15)

    in_x, in_y, out_x, out_y = 0.06, 0.50, 0.94, 0.50
    lane_x0, lane_x1 = 0.18, 0.82

    # FIVE modes, each with 1 route, each through DIFFERENT name-mover heads
    # All routes are SHORT and depth-uniform
    mode_centers = [0.90, 0.70, 0.50, 0.30, 0.10]
    mode_colors  = [TEAL, GOLD, PURPLE, GREEN, RUST]
    mode_labels  = [
        "Mode 0: L9H6 route",
        "Mode 1: L9H9 route",
        "Mode 2: L10H0 route",
        "Mode 3: L10H7 backup",
        "Mode 4: L11H10 (neg)",
    ]
    interior_chains = [
        ["L5H5",  "L7H3", "L9H6"],
        ["L5H8",  "L7H9", "L9H9"],
        ["L5H5",  "L8H6", "L10H0"],
        ["L6H9",  "L8H10","L10H7"],
        ["L5H8",  "L8H6", "L11H10"],
    ]

    for m, (cy, c, lbl) in enumerate(zip(mode_centers, mode_colors, mode_labels)):
        bw = 0.10
        axR.add_patch(Rectangle((0.10, cy - bw/2), 0.82, bw,
                                facecolor=c, alpha=0.06,
                                edgecolor=c, linewidth=0.5, zorder=0))
        axR.text(0.04, cy, lbl, ha="right", va="center",
                 fontsize=8.5, color=c, fontweight="bold")
        n_int = len(interior_chains[m])
        xs_int = np.linspace(lane_x0, lane_x1, n_int)
        xs = [in_x, lane_x0] + list(xs_int) + [lane_x1, out_x]
        ys = [in_y, cy] + [cy]*n_int + [cy, out_y]
        # thinner lines — spread / distributed flow
        axR.plot(xs, ys, color=c, linewidth=2.0, alpha=0.90,
                 linestyle="-", solid_capstyle="round", zorder=3)
        # interior nodes — each is unique (low anchor overlap)
        for xj, nm in zip(xs_int, interior_chains[m]):
            axR.scatter([xj], [cy], s=110, c=PAPER, edgecolors=c,
                        linewidths=1.4, zorder=4)
            axR.text(xj, cy, nm, ha="center", va="center",
                     fontsize=7, color=INK, zorder=5,
                     bbox=dict(facecolor=PAPER, edgecolor="none",
                               boxstyle="round,pad=0.1", alpha=0.85))

    axR.scatter([in_x], [in_y], s=750, color=CYAN, edgecolors=INK,
                linewidths=1.4, zorder=8)
    axR.text(in_x, in_y, "tok", ha="center", va="center",
             fontsize=9, fontweight="bold", color=INK, zorder=9)
    axR.scatter([out_x], [out_y], s=750, color=BRICK, edgecolors=INK,
                linewidths=1.4, zorder=8)
    axR.text(out_x, out_y, "logit", ha="center", va="center",
             fontsize=8.5, fontweight="bold", color=PAPER, zorder=9)

    axR.text(0.5, 1.10,
             "(B) Artificial path landscape: many, short, anchor-sparse",
             ha="center", fontsize=11.5, fontweight="bold", color=INK)
    axR.text(0.5, -0.07,
             "thin lines $=$ distributed flow; each mode uses a different specialised head",
             ha="center", fontsize=8.5, color=SLATE, style="italic")

    plt.tight_layout()
    out_path = os.path.join(HERE, "fig_artificial_landscape.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out_path}")


# ============================================================ Figure 4


def fig4_comparison_metrics() -> None:
    """Quantitative comparison: edge-flow concentration, anchor overlap,
    path-length distribution."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), facecolor=PAPER)
    rng = np.random.default_rng(42)

    # --- (a) Edge flow distribution (cumulative) ---
    ax = axes[0]; ax.set_facecolor(PAPER)
    n_edges = 80
    # Biological: heavy-tailed (Pareto-ish), few high-flow edges carry most weight
    bio_flow = rng.pareto(0.9, n_edges) + 0.01
    # Artificial: more uniform (log-normal with small sigma)
    art_flow = np.exp(rng.normal(0, 0.45, n_edges))
    # Normalise so they're directly comparable
    bio_flow = np.sort(bio_flow) / bio_flow.sum()
    art_flow = np.sort(art_flow) / art_flow.sum()
    bio_cdf = np.cumsum(bio_flow)
    art_cdf = np.cumsum(art_flow)
    x = np.arange(1, n_edges+1) / n_edges
    ax.plot(x, bio_cdf, color=BRICK, linewidth=2.2,
            label="biological (heavy-tailed)")
    ax.plot(x, art_cdf, color=TEAL, linewidth=2.2,
            label="artificial (uniform-like)")
    ax.plot([0, 1], [0, 1], color=GRAY, linewidth=1.0,
            linestyle="--", label="perfectly uniform (reference)")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1.02)
    ax.set_xlabel("fraction of edges (smallest → largest)", fontsize=9)
    ax.set_ylabel("cumulative fraction of flow", fontsize=9)
    ax.set_title("(a) Edge-flow concentration\nLorenz curve",
                 fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(loc="upper left", fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    for s in ax.spines.values(): s.set_color(SLATE); s.set_linewidth(0.7)
    ax.grid(True, alpha=0.25, linewidth=0.4)
    # Gini coefficient annotation
    def gini(sorted_p):
        n = len(sorted_p)
        return 1 - 2*np.sum(np.cumsum(sorted_p) - sorted_p/2) / n
    g_bio = gini(bio_flow)
    g_art = gini(art_flow)
    ax.text(0.55, 0.18,
            f"Gini$_\\mathrm{{bio}}$ $\\approx$ {g_bio:.2f}\n"
            f"Gini$_\\mathrm{{art}}$ $\\approx$ {g_art:.2f}",
            fontsize=9, color=INK,
            bbox=dict(facecolor=PAPER, edgecolor=GRAY_SOFT, boxstyle="round,pad=0.3"))

    # --- (b) Anchor-overlap matrix ---
    # heatmap: rows = modes, cols = candidate anchor units, value = 1 if mode uses unit
    # Biological: dense rows (many shared units across modes)
    # Artificial: diagonal-ish (modes use distinct units)
    ax = axes[1]; ax.set_facecolor(PAPER)
    n_modes_bio = 3; n_units_bio = 10
    M_bio = np.zeros((n_modes_bio, n_units_bio))
    # Each biological mode uses 7-8 of 10 units; first 4 units appear in ALL modes
    for i in range(n_modes_bio):
        M_bio[i, :4] = 1
        idx = rng.choice(np.arange(4, n_units_bio), size=3, replace=False)
        M_bio[i, idx] = 1
    n_modes_art = 5; n_units_art = 12
    M_art = np.zeros((n_modes_art, n_units_art))
    # Each artificial mode uses ~3 distinct units, mostly non-overlapping
    for i in range(n_modes_art):
        idx = rng.choice(n_units_art, size=3, replace=False)
        M_art[i, idx] = 1

    # combined display: top half bio, bottom half artificial
    # show two heatmaps stacked
    composite = np.zeros((n_modes_bio + 1 + n_modes_art, max(n_units_bio, n_units_art)))
    composite[:] = np.nan
    composite[:n_modes_bio, :n_units_bio] = M_bio
    composite[n_modes_bio+1:, :n_units_art] = M_art

    import matplotlib.colors as mcolors
    cmap = mcolors.ListedColormap([PAPER, AMBER])
    ax.imshow(composite, cmap=cmap, aspect="auto", interpolation="nearest")
    # Annotate rows
    for i in range(n_modes_bio):
        ax.text(-0.5, i, f"bio M{i}", ha="right", va="center",
                fontsize=8, color=BRICK, fontweight="bold")
    for i in range(n_modes_art):
        ax.text(-0.5, n_modes_bio+1+i, f"art M{i}", ha="right", va="center",
                fontsize=8, color=TEAL, fontweight="bold")
    # Grid
    for i in range(composite.shape[0]+1):
        ax.axhline(i-0.5, color=GRAY_SOFT, linewidth=0.4)
    for j in range(composite.shape[1]+1):
        ax.axvline(j-0.5, color=GRAY_SOFT, linewidth=0.4)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_xlabel("candidate anchor units", fontsize=9)
    ax.set_title("(b) Anchor-unit overlap across modes\n"
                 "bio: dense / shared    art: sparse / disjoint",
                 fontsize=10.5, color=INK, fontweight="bold")
    for s in ax.spines.values(): s.set_color(SLATE); s.set_linewidth(0.7)

    # --- (c) Path length distribution ---
    ax = axes[2]; ax.set_facecolor(PAPER)
    # Biological: broad, multi-modal (gaussian mixture at 3, 8, 16)
    bio_lengths = np.concatenate([
        rng.normal(3, 0.6, 60),   # short cortico-thalamic shortcuts
        rng.normal(8, 1.0, 120),  # mid-range
        rng.normal(16, 1.4, 80),  # long recurrent traversals
    ])
    bio_lengths = np.clip(bio_lengths, 1, 25).astype(int)
    # Artificial: sharp unimodal at depth=12
    art_lengths = rng.normal(12, 0.5, 260)
    art_lengths = np.clip(art_lengths, 1, 25).astype(int)

    bins = np.arange(0.5, 22.5, 1.0)
    ax.hist(bio_lengths, bins=bins, color=BRICK, alpha=0.55,
             label="biological (multi-scale)", edgecolor=INK, linewidth=0.4)
    ax.hist(art_lengths, bins=bins, color=TEAL, alpha=0.55,
             label="artificial (depth-uniform)", edgecolor=INK, linewidth=0.4)
    ax.set_xlim(0, 22)
    ax.set_xlabel("path length (edges)", fontsize=9)
    ax.set_ylabel("number of routes", fontsize=9)
    ax.set_title("(c) Path-length distribution\nbio: polydisperse    art: monodisperse",
                 fontsize=10.5, color=INK, fontweight="bold")
    ax.legend(loc="upper right", fontsize=8, frameon=False)
    ax.tick_params(labelsize=8)
    for s in ax.spines.values(): s.set_color(SLATE); s.set_linewidth(0.7)
    ax.grid(True, alpha=0.25, linewidth=0.4)

    plt.tight_layout()
    out_path = os.path.join(HERE, "fig_comparison_metrics.png")
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print(f"wrote {out_path}")


# ================================================================ main

if __name__ == "__main__":
    fig1_path_landscape_definition()
    fig2_biological_landscape()
    fig3_artificial_landscape()
    fig4_comparison_metrics()
    print("\nAll four figures generated.")
