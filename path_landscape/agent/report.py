"""Render the final markdown report combining spec, metrics, and LLM analysis."""
from __future__ import annotations

import json
from typing import Optional

from ..landscape import PathLandscape
from ..metrics import summarize, persistence_h0, persistence_h1
from ..system import System
from .schemas import SystemSpec


def _format_spec(spec: SystemSpec) -> str:
    parts: list[str] = []
    parts.append(f"## System specification\n")
    parts.append(f"**Phenomenon:** {spec.phenomenon_name}\n")
    parts.append(f"{spec.phenomenon_summary}\n")
    parts.append(spec.summary() + "\n")
    if spec.parameters:
        parts.append("\n**External parameters:**\n")
        for p in spec.parameters:
            val = f" = {p.value}" if p.value else ""
            parts.append(f"- `{p.name}`{val}: {p.role}\n")
    parts.append("\n**Units:**\n")
    parts.append("| name | role | scale | parent | description |\n")
    parts.append("|---|---|---|---|---|\n")
    for u in spec.units:
        parent = u.parent or ""
        parts.append(f"| `{u.name}` | {u.role} | {u.scale} | {parent} | {u.description} |\n")
    parts.append("\n**Interactions:**\n")
    parts.append("| source | target | weight | recurrent | description |\n")
    parts.append("|---|---|---|---|---|\n")
    for it in spec.interactions:
        rec = "yes" if it.recurrent else ""
        parts.append(
            f"| `{it.source}` | `{it.target}` | {it.weight:.2f} | {rec} | {it.description} |\n"
        )
    if spec.notes:
        parts.append(f"\n**Notes:** {spec.notes}\n")
    return "".join(parts)


def _format_metrics(spec: SystemSpec, sys: System, L: PathLandscape) -> str:
    s = summarize(L)
    h0 = persistence_h0(L, max_features=10)
    h1 = persistence_h1(L)
    lengths = [p.length for p in L.paths]
    lines = [
        "## Path-landscape metrics\n",
        "",
        f"After unrolling feedback loops over T={spec.time_steps} and",
        f"coarsening to the lowest scale, the static path graph has",
        f"{sys.graph.number_of_nodes()} units and {sys.graph.number_of_edges()} edges.",
        "",
        f"- **Paths sampled:** {s['n_paths']}",
        f"- **Modes (clusters):** {s['n_modes']}",
        f"- **Path length range:** {min(lengths)} - {max(lengths)} (mean {sum(lengths)/len(lengths):.2f})",
        f"- **Cluster-size exponent (alpha):** {s['size_exponent']:.3f} "
        f"(R² = {s['size_exponent_r2']:.3f})",
        f"- **H0 max persistence:** {s['h0_max']:.3f}",
    ]
    if h1 is not None:
        lines.append(
            f"- **H1 features (compositional loops):** {s['h1_count']}, "
            f"max persistence {s['h1_max_persistence']:.3f}"
        )
    mg = s["meta"]
    lines.extend([
        f"- **Meta-graph:** {mg['n_clusters']} clusters, "
        f"giant-component fraction {mg['giant_fraction']:.3f}, "
        f"mean degree {mg['mean_degree']:.2f}, density {mg['density']:.3f}",
        "",
        "**Representative paths (top clusters):**",
        "",
    ])
    summaries = L.cluster_summary()
    summaries.sort(key=lambda c: -c.size)
    for c in summaries[:5]:
        rep = c.representative
        chain = " -> ".join(rep.nodes) if len(rep.nodes) <= 8 else (
            " -> ".join(rep.nodes[:3]) + " ... " + " -> ".join(rep.nodes[-2:])
        )
        lines.append(
            f"- mode {c.label}  (size {c.size}, total weight {c.total_weight:.2f}, "
            f"length {rep.length}):  `{chain}`"
        )
    lines.append("")
    return "\n".join(lines)


def write_report(
    spec: SystemSpec,
    sys: System,
    L: PathLandscape,
    interpretation: str,
    out_path: str,
    figure_filename: Optional[str] = "landscape.png",
) -> None:
    """Write the full markdown report to `out_path`."""
    body: list[str] = []
    body.append(f"# {spec.phenomenon_name}\n")
    body.append(f"*Path-landscape analysis of an emergent phenomenon.*\n\n")
    if figure_filename:
        body.append(f"![landscape figure]({figure_filename})\n\n")
    body.append(_format_spec(spec))
    body.append("\n")
    body.append(_format_metrics(spec, sys, L))
    body.append("\n## Mechanistic interpretation\n\n")
    body.append(interpretation.rstrip() + "\n")
    with open(out_path, "w") as f:
        f.write("".join(body))


def write_spec_json(spec: SystemSpec, out_path: str) -> None:
    with open(out_path, "w") as f:
        json.dump(spec.to_dict(), f, indent=2)
