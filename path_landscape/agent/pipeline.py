"""End-to-end agent pipeline.

Public entry point: `analyze_emergence(phenomenon, out_dir, ...)`.

The pipeline orchestrates five stages:

  1. specify_system  - LLM (tool use) -> SystemSpec
  2. build_system    - SystemSpec     -> System
  3. run_analysis    - System         -> PathLandscape + metrics
  4. interpret       - LLM (text)     -> mechanistic interpretation
  5. write_report    - everything     -> markdown + figure + spec.json

Defaults to `claude-opus-4-7`. The specifier uses `effort: medium`; the
interpreter uses adaptive thinking + `effort: high` so the analysis is
substantive.
"""
from __future__ import annotations

import json
import os
import textwrap
import time
from dataclasses import dataclass
from typing import Optional

try:
    import anthropic
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "path_landscape.agent requires the `anthropic` package. "
        "Install with `pip install anthropic`."
    ) from exc

from ..landscape import PathLandscape
from ..metrics import summarize, persistence_h0, persistence_h1
from ..paths import enumerate_paths, sample_paths
from ..system import System
from .builder import build_system_from_spec
from .prompts import (
    INTERPRETER_SYSTEM,
    SPECIFIER_SYSTEM,
    SPECIFY_SYSTEM_TOOL,
)
from .schemas import SystemSpec

DEFAULT_MODEL = "claude-opus-4-7"


# ============================================================ stage 1


def specify_system(
    phenomenon: str,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
) -> SystemSpec:
    """Run the LLM specifier and return a SystemSpec.

    Uses Anthropic tool use to force a structured output that conforms to
    `SPECIFY_SYSTEM_TOOL`'s schema.
    """
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": SPECIFIER_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[SPECIFY_SYSTEM_TOOL],
        tool_choice={"type": "tool", "name": "specify_system"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": phenomenon}],
    )
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == "specify_system":
            return SystemSpec.from_dict(block.input)
    raise RuntimeError(
        f"specifier did not call `specify_system` tool "
        f"(stop_reason={response.stop_reason})"
    )


# ============================================================ stage 2-3


@dataclass
class AnalysisResult:
    spec: SystemSpec
    system: System
    landscape: PathLandscape
    metrics: dict


def run_analysis(
    spec: SystemSpec,
    n_paths: int = 1500,
    eps: float = 0.45,
    min_samples: int = 3,
    max_length: int = 64,
) -> AnalysisResult:
    """Build the System, extract paths, compute the landscape and metrics."""
    sys = build_system_from_spec(spec)
    T = max(1, int(spec.time_steps))
    g = sys.unroll(T)
    # Sources: inputs at t=0 only (information enters at the start).
    # Sinks: outputs at *any* time step >= 1 (the system can reach the
    # observable at different latencies, which is informative for the
    # length-by-cluster picture).
    sources = sys.unroll_sources(T)
    if T == 1:
        sinks = sys.unroll_sinks(T)
    else:
        sinks = [f"{n}@{t}" for n in sys.outputs for t in range(T)]
    if not sources:
        raise RuntimeError("no source units after unrolling")
    if not sinks:
        raise RuntimeError("no sink units after unrolling")
    # Prefer enumeration when the graph is small enough; otherwise sample.
    n_edges = g.number_of_edges()
    if n_edges <= 200:
        paths = enumerate_paths(
            g, sources, sinks, max_paths=n_paths, max_length=max_length,
        )
    else:
        paths = sample_paths(
            g, sources, sinks,
            n_samples=n_paths, max_length=max_length,
        )
        if not paths:
            paths = enumerate_paths(
                g, sources, sinks, max_paths=n_paths, max_length=max_length,
            )
    if not paths:
        raise RuntimeError("no paths from inputs to outputs after unrolling")
    L = PathLandscape(paths)
    L.cluster(eps=eps, min_samples=min_samples)
    return AnalysisResult(spec=spec, system=sys, landscape=L,
                          metrics=summarize(L))


# ============================================================ stage 4


def _format_metrics_for_llm(result: AnalysisResult) -> str:
    spec = result.spec
    sys = result.system
    L = result.landscape
    s = result.metrics
    h0 = persistence_h0(L, max_features=10)
    h1 = persistence_h1(L)
    lengths = [p.length for p in L.paths]

    lines: list[str] = []
    lines.append(f"PHENOMENON: {spec.phenomenon_name}")
    lines.append(f"SUMMARY: {spec.phenomenon_summary}")
    if spec.notes:
        lines.append(f"MODELING NOTES: {spec.notes}")
    lines.append("")
    lines.append(
        f"SYSTEM SPEC: {len(spec.units)} units "
        f"(in={sum(1 for u in spec.units if u.role=='input')}, "
        f"int={sum(1 for u in spec.units if u.role=='internal')}, "
        f"out={sum(1 for u in spec.units if u.role=='output')}); "
        f"{len(spec.interactions)} interactions "
        f"({sum(1 for x in spec.interactions if x.recurrent)} recurrent); "
        f"time_steps={spec.time_steps}; "
        f"scales={sorted({u.scale for u in spec.units})}"
    )
    if spec.parameters:
        lines.append("EXTERNAL PARAMETERS: " + "; ".join(
            f"{p.name} ({p.role})" for p in spec.parameters
        ))
    lines.append("")
    lines.append("PATH-LANDSCAPE METRICS:")
    lines.append(f"  unrolled graph: {sys.graph.number_of_nodes()} units "
                 f"(unrolled: same, see T={spec.time_steps}), "
                 f"{sys.graph.number_of_edges()} edges")
    lines.append(f"  paths sampled / enumerated: {s['n_paths']}")
    lines.append(
        f"  path length range: {min(lengths)} - {max(lengths)} "
        f"(mean {sum(lengths) / len(lengths):.2f})"
    )
    lines.append(f"  n_modes (clusters): {s['n_modes']}")
    lines.append(
        f"  cluster-size exponent alpha: {s['size_exponent']:.3f} "
        f"(R² = {s['size_exponent_r2']:.3f})"
    )
    lines.append(f"  H0 max persistence: {s['h0_max']:.3f}")
    if h1 is not None:
        lines.append(
            f"  H1 features (compositional loops): {s['h1_count']}, "
            f"max persistence {s['h1_max_persistence']:.3f}"
        )
    mg = s["meta"]
    lines.append(
        f"  meta-graph: {mg['n_clusters']} clusters, "
        f"giant fraction {mg['giant_fraction']:.3f}, "
        f"mean degree {mg['mean_degree']:.2f}, density {mg['density']:.3f}"
    )
    lines.append("")
    lines.append("TOP CLUSTERS:")
    summaries = L.cluster_summary()
    summaries.sort(key=lambda c: -c.size)
    for c in summaries[:5]:
        rep = c.representative
        chain = " -> ".join(rep.nodes) if len(rep.nodes) <= 8 else (
            " -> ".join(rep.nodes[:3]) + " ... " + " -> ".join(rep.nodes[-2:])
        )
        lines.append(
            f"  mode {c.label}: size {c.size}, total_weight "
            f"{c.total_weight:.2f}, length {rep.length}: {chain}"
        )
    return "\n".join(lines)


def interpret(
    result: AnalysisResult,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> str:
    """Run the LLM interpreter on the metrics and return the explanation text."""
    client = client or anthropic.Anthropic()
    metrics_text = _format_metrics_for_llm(result)
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": INTERPRETER_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        thinking={"type": "adaptive"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": metrics_text}],
    )
    return "".join(
        b.text for b in response.content
        if getattr(b, "type", None) == "text"
    ).strip()


# ============================================================ stage 5 / orchestrator


def analyze_emergence(
    phenomenon: str,
    out_dir: str = "./emergence_analysis",
    n_paths: int = 1500,
    eps: float = 0.45,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    on_progress: Optional[callable] = None,
) -> dict:
    """Full pipeline: phenomenon (str) -> on-disk artifacts + result dict.

    Robust to path-extraction failure: when the unrolled graph has no
    input-to-output paths, a diagnostic figure is rendered instead of the
    landscape figure, and a short diagnostic interpretation is written.
    The frontend always receives a valid PNG, so the user never sees a
    broken image.

    `on_progress` is an optional callback `on_progress(step, percent, message)`
    invoked at each pipeline stage. The web frontend uses this to stream
    progress to the browser.
    """
    from .visualize import render_figure, render_diagnostic_figure
    from .report import write_report, write_spec_json
    from .builder import build_system_from_spec

    def _emit(step: str, percent: int, message: str) -> None:
        if verbose:
            print(f"  [{percent:3d}%] {step}: {message}")
        if on_progress:
            try:
                on_progress(step, percent, message)
            except Exception:
                pass  # don't let the callback crash the pipeline

    os.makedirs(out_dir, exist_ok=True)
    client = client or anthropic.Anthropic()

    _emit("specifying", 5,
          f"calling Claude to specify the system for {phenomenon!r}...")
    t0 = time.time()
    spec = specify_system(phenomenon, client=client, model=model)
    _emit("specified", 30,
          f"got spec in {time.time() - t0:.1f}s -- {spec.summary()}")

    figure_path = os.path.join(out_dir, "landscape.png")
    spec_path = os.path.join(out_dir, "spec.json")
    report_path = os.path.join(out_dir, "report.md")

    # Always write spec.json early so it survives any later failure
    write_spec_json(spec, spec_path)

    _emit("building", 35,
          f"building System and extracting paths (T={spec.time_steps})...")
    t0 = time.time()
    try:
        result = run_analysis(spec, n_paths=n_paths, eps=eps)
    except RuntimeError as exc:
        # Path extraction failed (e.g. no paths inputs→outputs after unrolling).
        # Render the diagnostic figure so the frontend has a usable image,
        # then return a partial result so the result page can still load.
        err_msg = str(exc)
        _emit("analysis_failed", 70,
              f"path extraction failed: {err_msg}; rendering diagnostic figure")
        sys_only = build_system_from_spec(spec)
        render_diagnostic_figure(spec, sys_only, err_msg, figure_path)

        diagnostic_interpretation = _diagnostic_interpretation(spec, err_msg)
        # Minimal report
        from .report import _format_spec
        body = [
            f"# {spec.phenomenon_name}\n",
            f"*Path-landscape analysis — partial (no path landscape).*\n\n",
            f"![diagnostic figure](landscape.png)\n\n",
            f"## ⚠ Path landscape not computed\n\n",
            f"Reason: `{err_msg}`\n\n",
            f"The System was specified successfully, but no paths from "
            f"input units to output units exist after unrolling over "
            f"T = {spec.time_steps} time steps. The diagnostic figure "
            f"shows the specified system and the unrolled graph so the "
            f"connectivity gap is visible.\n\n",
            _format_spec(spec),
            "\n## Mechanistic interpretation (diagnostic)\n\n",
            diagnostic_interpretation,
            "\n",
        ]
        with open(report_path, "w") as f:
            f.write("".join(body))
        _emit("done", 100,
              "analysis complete (partial — landscape unavailable).")

        # Return a partial result; landscape/metrics are None
        return {
            "spec": spec,
            "system": sys_only,
            "landscape": None,
            "metrics": None,
            "interpretation": diagnostic_interpretation,
            "out_dir": out_dir,
            "report_path": report_path,
            "figure_path": figure_path,
            "spec_path": spec_path,
            "partial": True,
            "error": err_msg,
        }

    _emit("analyzed", 60,
          f"landscape ready in {time.time() - t0:.1f}s -- "
          f"{result.landscape.describe()}")

    _emit("interpreting", 65,
          "asking Claude to interpret the metrics mechanistically...")
    t0 = time.time()
    interpretation = interpret(result, client=client, model=model)
    _emit("interpreted", 90,
          f"got {len(interpretation)} chars of analysis "
          f"in {time.time() - t0:.1f}s")

    _emit("rendering", 92, f"writing figure and report to {out_dir!r}...")
    render_figure(spec, result.system, result.landscape, figure_path)
    write_report(spec, result.system, result.landscape, interpretation,
                 report_path, figure_filename="landscape.png")
    _emit("done", 100, "analysis complete.")

    return {
        "spec": spec,
        "system": result.system,
        "landscape": result.landscape,
        "metrics": result.metrics,
        "interpretation": interpretation,
        "out_dir": out_dir,
        "report_path": report_path,
        "figure_path": figure_path,
        "spec_path": spec_path,
        "partial": False,
    }


def _diagnostic_interpretation(spec: SystemSpec, error_msg: str) -> str:
    """Path-structure diagnostic when no path landscape can be computed."""
    n_in  = sum(1 for u in spec.units if u.role == "input")
    n_out = sum(1 for u in spec.units if u.role == "output")
    n_rec = sum(1 for x in spec.interactions if x.recurrent)
    n_ff  = len(spec.interactions) - n_rec

    # Heuristic: identify obvious connectivity holes
    ff_inputs = {x.source for x in spec.interactions
                 if not x.recurrent and any(u.name == x.source and u.role == "input"
                                            for u in spec.units)}
    ff_into_outputs = {x.target for x in spec.interactions
                       if not x.recurrent and any(u.name == x.target and u.role == "output"
                                                  for u in spec.units)}
    output_names = {u.name for u in spec.units if u.role == "output"}
    input_names  = {u.name for u in spec.units if u.role == "input"}
    missing_out_edges = output_names - ff_into_outputs
    missing_in_edges  = input_names - ff_inputs

    parts = [
        "**Path-structural mechanism.**",
        f"No path landscape was computed because the specified path "
        f"graph has a structural cut: no directed path connects any "
        f"input unit to any output unit after unrolling over "
        f"T = {spec.time_steps}.",
        "",
        "**Interpretation.**",
        f"The specification provides {len(spec.units)} units "
        f"({n_in} input, {n_out} output, "
        f"{len(spec.units) - n_in - n_out} internal) and "
        f"{len(spec.interactions)} interactions "
        f"({n_ff} feed-forward, {n_rec} recurrent). ",
    ]
    if missing_out_edges:
        parts.append(
            f"The output unit(s) `{', '.join(sorted(missing_out_edges))}` "
            f"have no incoming feed-forward edge — all incoming edges are "
            f"recurrent (firing at the *next* time step), so they can "
            f"never receive signal at the unrolling horizon. "
        )
    if missing_in_edges:
        parts.append(
            f"The input unit(s) `{', '.join(sorted(missing_in_edges))}` "
            f"have no outgoing feed-forward edge — signal cannot leave "
            f"the source. "
        )
    if not missing_out_edges and not missing_in_edges:
        parts.append(
            "Inputs and outputs are individually wired, but no chain "
            "of feed-forward edges links the two cones within "
            f"T = {spec.time_steps} steps. "
        )
    parts.append("")
    parts.append("**Key bottleneck.**")
    parts.append(
        "The graph has a cut between the input cone and the output "
        "cone in the unrolled (time-expanded) graph. The path "
        "landscape is empty by construction; no path-structural "
        "feature (clusters, hubs, loops, hierarchy, recurrence) can "
        "be observed until the cut is closed."
    )
    parts.append("")
    parts.append("**Falsifiable prediction.**")
    if missing_out_edges:
        target = sorted(missing_out_edges)[0]
        parts.append(
            f"Adding a single feed-forward edge from any internal unit "
            f"into `{target}` would create at least one cluster "
            f"(n_modes ≥ 1) and produce a non-degenerate path landscape."
        )
    elif missing_in_edges:
        source = sorted(missing_in_edges)[0]
        parts.append(
            f"Adding a single feed-forward edge from `{source}` to any "
            f"internal unit would propagate signal and produce a "
            f"non-degenerate landscape (n_modes ≥ 1)."
        )
    else:
        parts.append(
            f"Increasing time_steps from {spec.time_steps} would not help: "
            "the cut is structural, not temporal. Re-specifying with at "
            "least one input→internal→…→output feed-forward chain is "
            "required."
        )
    return "\n".join(parts)
