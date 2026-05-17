"""Four-agent neural-circuit analysis pipeline.

Public entry point: `analyze_neural_circuit(phenomenon, out_dir, ...)`.

Stages:

  Agent 1  find_paper                 -> PaperInfo
  Agent 2  identify_neural_system     -> NeuralSystemInfo  (uses paper)
  Agent 3  build_network              -> SystemSpec + network figure
                                           (uses paper + system)
  Agent 4  extract_path_representation
             - unrolls feedback loops over T time steps
             - enumerates input -> output paths
             - separates feedforward vs. feedback-traversing paths
             - writes the path-representation figure

Artefacts written to `out_dir`:
  network.png        -- Agent 3's figure
  paths.png          -- Agent 4's figure (this is the main UI output)
  spec.json          -- the SystemSpec (re-uses existing format)
  paper.json         -- Agent 1's output
  neural_system.json -- Agent 2's output
  path_rep.json      -- Agent 4's output (paths + statistics)
  report.md          -- combined markdown report
"""
from __future__ import annotations

import json
import os
import time
from typing import Callable, Optional

try:
    import anthropic
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "path_landscape.agent.neural_pipeline requires the `anthropic` package. "
        "Install with `pip install anthropic`."
    ) from exc

from ..paths import enumerate_paths, sample_paths
from ..system import System
from .builder import build_system_from_spec, validate_spec
from .neural_prompts import (
    BUILD_NETWORK_SYSTEM,
    DESCRIBE_PATHS_SYSTEM,
    FIND_PAPER_SYSTEM,
    FIND_PAPER_TOOL,
    IDENTIFY_SYSTEM_SYSTEM,
    IDENTIFY_SYSTEM_TOOL,
    INTERPRET_LANDSCAPE_SYSTEM,
    INTERPRET_LANDSCAPE_TOOL,
)
from .neural_schemas import (
    LandscapeInterpretation,
    NeuralAnalysisResult,
    NeuralSystemInfo,
    PaperInfo,
    PathRepresentation,
)
from .neural_visualize import (
    build_path_representation,
    render_network_figure,
    render_path_clusters_figure,
    render_path_representation_figure,
)
from .prompts import SPECIFY_SYSTEM_TOOL
from .schemas import SystemSpec

DEFAULT_MODEL = "claude-opus-4-7"


# ================================================================ utils


def _emit(on_progress: Optional[Callable],
          step: str, percent: int, message: str,
          verbose: bool = True) -> None:
    if verbose:
        print(f"  [{percent:3d}%] {step}: {message}")
    if on_progress is not None:
        try:
            on_progress(step, percent, message)
        except Exception:
            pass  # never let a progress callback crash the pipeline


def _extract_tool_use(response, tool_name: str) -> Optional[dict]:
    for block in response.content:
        if getattr(block, "type", None) == "tool_use" and block.name == tool_name:
            return block.input
    return None


# ================================================================ Agent 1


def find_paper(
    phenomenon: str,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> PaperInfo:
    """Agent 1: identify a research paper about the responsible circuit."""
    client = client or anthropic.Anthropic()
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": FIND_PAPER_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[FIND_PAPER_TOOL],
        tool_choice={"type": "tool", "name": "report_research_paper"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": phenomenon}],
    )
    payload = _extract_tool_use(response, "report_research_paper")
    if payload is None:
        raise RuntimeError(
            f"find_paper: model did not call the tool "
            f"(stop_reason={response.stop_reason})"
        )
    return PaperInfo.from_dict(payload)


# ================================================================ Agent 2


def identify_neural_system(
    phenomenon: str,
    paper: PaperInfo,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 2048,
) -> NeuralSystemInfo:
    """Agent 2: identify the neural circuit/system responsible."""
    client = client or anthropic.Anthropic()
    user_msg = (
        f"PHENOMENON:\n{phenomenon}\n\n"
        f"ANCHOR PAPER (Agent 1):\n"
        f"  Title: {paper.title}\n"
        f"  Authors: {paper.authors}\n"
        f"  Year: {paper.year}\n"
        f"  Summary: {paper.summary}\n"
        f"  Key findings: "
        f"{'; '.join(paper.key_findings) if paper.key_findings else '(none)'}\n"
    )
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": IDENTIFY_SYSTEM_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[IDENTIFY_SYSTEM_TOOL],
        tool_choice={"type": "tool", "name": "report_neural_system"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": user_msg}],
    )
    payload = _extract_tool_use(response, "report_neural_system")
    if payload is None:
        raise RuntimeError(
            f"identify_neural_system: model did not call the tool "
            f"(stop_reason={response.stop_reason})"
        )
    return NeuralSystemInfo.from_dict(payload)


# ================================================================ Agent 3


def build_network(
    phenomenon: str,
    paper: PaperInfo,
    system_info: NeuralSystemInfo,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 8192,
    max_retries: int = 2,
) -> SystemSpec:
    """Agent 3: produce the SystemSpec for the identified circuit.

    Retries up to `max_retries` times when the returned spec fails structural
    validation (missing inputs/outputs, disconnected graph, dangling
    references, or time_steps inconsistent with recurrent edges).
    """
    client = client or anthropic.Anthropic()

    base_user_msg = (
        f"PHENOMENON:\n{phenomenon}\n\n"
        f"ANCHOR PAPER (Agent 1):\n"
        f"  Title: {paper.title}\n"
        f"  Year: {paper.year}\n"
        f"  Summary: {paper.summary}\n\n"
        f"NEURAL SYSTEM (Agent 2):\n"
        f"  Name: {system_info.system_name}\n"
        f"  Brain regions: {', '.join(system_info.brain_regions) or '(none)'}\n"
        f"  Neuron types: {', '.join(system_info.neuron_types) or '(none)'}\n"
        f"  Circuit motifs: "
        f"{', '.join(system_info.key_circuit_motifs) or '(none)'}\n"
        f"  Description: {system_info.description}\n\n"
        f"Build the System for this circuit. Use cell-type / region names "
        f"as unit names; mark all real feedback loops as recurrent=True; "
        f"include multiscale parents where region groupings apply."
    )

    user_msg = base_user_msg
    last_spec: Optional[SystemSpec] = None

    for attempt in range(max_retries + 1):
        response = client.messages.create(
            model=model,
            max_tokens=max_tokens,
            system=[{
                "type": "text",
                "text": BUILD_NETWORK_SYSTEM,
                "cache_control": {"type": "ephemeral"},
            }],
            tools=[SPECIFY_SYSTEM_TOOL],
            tool_choice={"type": "tool", "name": "specify_system"},
            output_config={"effort": "medium"},
            messages=[{"role": "user", "content": user_msg}],
        )
        payload = _extract_tool_use(response, "specify_system")
        if payload is None:
            raise RuntimeError(
                f"build_network: model did not call the tool "
                f"(stop_reason={response.stop_reason})"
            )
        spec = SystemSpec.from_dict(payload)
        last_spec = spec
        problems = validate_spec(spec)
        if not problems:
            return spec  # spec passed all checks

        if attempt < max_retries:
            # Feed the problems back so the next attempt can fix them
            feedback = (
                f"\n\n⚠ PREVIOUS ATTEMPT FAILED VALIDATION "
                f"(attempt {attempt + 1}/{max_retries + 1}).\n"
                "The spec you produced has the following structural problems "
                "that MUST be fixed:\n"
                + "\n".join(f"  • {p}" for p in problems)
                + "\n\nPlease call specify_system again and fix ALL of the "
                "issues listed above. Pay special attention to:\n"
                "  - Every interaction source/target must match a unit name exactly.\n"
                "  - There must be a directed path from an input unit to an output unit.\n"
                "  - time_steps must be ≥ 2 if any interaction has recurrent=True."
            )
            user_msg = base_user_msg + feedback
        else:
            # All retries exhausted — return the best we got, note the issues
            warning = "Validation warnings (after retries): " + "; ".join(problems)
            spec.notes = (spec.notes + " | " + warning).lstrip(" | ") if spec.notes else warning
            return spec

    # Unreachable, but satisfy type checker
    assert last_spec is not None
    return last_spec


# ================================================================ Agent 4


def extract_path_representation(
    spec: SystemSpec,
    sys: System,
    n_paths: int = 1500,
    max_length: int = 64,
) -> PathRepresentation:
    """Agent 4: unroll feedback loops, enumerate paths, classify them.

    The procedural core of Agent 4. The unrolled DAG is the "no feedback"
    static graph required by the framework; paths that traverse a recurrent
    edge are tagged crosses_feedback=True and are typically longer than the
    pure-feedforward paths.
    """
    T = max(1, int(spec.time_steps))
    unrolled = sys.unroll(T)
    sources = sys.unroll_sources(T)
    # Outputs at any time step >= 1 (so feedback-only paths can land at
    # output@T-1 after going through the loop)
    if T == 1:
        sinks = sys.unroll_sinks(T)
    else:
        sinks = [f"{n}@{t}" for n in sys.outputs for t in range(T)]
    if not sources:
        raise RuntimeError("no source units after unrolling")
    if not sinks:
        raise RuntimeError("no sink units after unrolling")

    n_edges = unrolled.number_of_edges()
    if n_edges <= 200:
        raw = enumerate_paths(
            unrolled, sources, sinks,
            max_paths=n_paths, max_length=max_length,
        )
    else:
        raw = sample_paths(
            unrolled, sources, sinks,
            n_samples=n_paths, max_length=max_length,
        )
        if not raw:
            raw = enumerate_paths(
                unrolled, sources, sinks,
                max_paths=n_paths, max_length=max_length,
            )
    if not raw:
        raise RuntimeError("no paths from inputs to outputs after unrolling")

    return build_path_representation(spec, sys, unrolled, raw)


def describe_path_representation(
    spec: SystemSpec,
    system_info: NeuralSystemInfo,
    path_rep: PathRepresentation,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 1024,
) -> str:
    """Agent 4 (text companion): short prose describing what the path
    representation reveals. Kept under 120 words by the system prompt."""
    if not path_rep.paths:
        return (
            "No input-to-output paths were enumerable after unrolling the "
            "feedback loops; the circuit has a structural cut. Increase "
            "time_steps or re-specify the connectivity to obtain a non-empty "
            "path representation."
        )

    client = client or anthropic.Anthropic()
    # A couple of representative paths from each class
    ff = [p for p in path_rep.paths if not p.crosses_feedback]
    fb = [p for p in path_rep.paths if p.crosses_feedback]
    ff_sample = sorted(ff, key=lambda p: -p.weight)[:3]
    fb_sample = sorted(fb, key=lambda p: -p.weight)[:3]

    def _chain(p):
        return " -> ".join(p.nodes if len(p.nodes) <= 10
                           else list(p.nodes[:4])
                                + ["…"] + list(p.nodes[-2:]))

    msg_parts = [
        f"NEURAL SYSTEM: {system_info.system_name}",
        f"  Motifs: {', '.join(system_info.key_circuit_motifs) or '(none)'}",
        "",
        f"PATH REPRESENTATION SUMMARY:",
        f"  T (unroll steps): {path_rep.time_steps}",
        f"  intrinsic time scale: {path_rep.intrinsic_time_scale}",
        f"  paths: {len(path_rep.paths)} "
        f"(feedforward: {path_rep.n_feedforward_paths}, "
        f"feedback-traversing: {path_rep.n_feedback_paths})",
        f"  path length range: {path_rep.min_length}-{path_rep.max_length} "
        f"(mean {path_rep.mean_length:.2f})",
        "",
        "FEEDFORWARD-ONLY EXAMPLES:",
    ]
    for p in ff_sample:
        msg_parts.append(f"  L={p.length}: {_chain(p)}")
    msg_parts.append("")
    msg_parts.append("FEEDBACK-TRAVERSING EXAMPLES:")
    for p in fb_sample:
        msg_parts.append(f"  L={p.length} (span={p.time_span}): {_chain(p)}")
    user_msg = "\n".join(msg_parts)

    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": DESCRIBE_PATHS_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        messages=[{"role": "user", "content": user_msg}],
    )
    return "".join(
        b.text for b in response.content
        if getattr(b, "type", None) == "text"
    ).strip()


# ================================================================ Agent 5


def _format_landscape_for_llm(
    phenomenon: str,
    system_info: NeuralSystemInfo,
    spec: SystemSpec,
    path_rep: PathRepresentation,
) -> str:
    """Render the path landscape into a compact text brief for Agent 5."""
    lines: list[str] = []
    lines.append(f"PHENOMENON: {phenomenon}")
    lines.append("")
    lines.append(f"NEURAL SYSTEM: {system_info.system_name}")
    lines.append(f"  Brain regions: "
                 f"{', '.join(system_info.brain_regions) or '(none)'}")
    lines.append(f"  Neuron types: "
                 f"{', '.join(system_info.neuron_types) or '(none)'}")
    lines.append(f"  Circuit motifs: "
                 f"{', '.join(system_info.key_circuit_motifs) or '(none)'}")
    lines.append("")
    lines.append(f"SPEC: {len(spec.units)} units, "
                 f"{len(spec.interactions)} interactions "
                 f"({sum(1 for i in spec.interactions if i.recurrent)} "
                 f"recurrent), T = {spec.time_steps}")
    lines.append("")
    lines.append("PATH LANDSCAPE:")
    lines.append(f"  raw paths enumerated: {len(path_rep.paths)}")
    lines.append(f"  unique routes (deduplicated): {len(path_rep.routes)}")
    lines.append(f"  similarity modes (clusters): {path_rep.n_modes}")
    lines.append(f"  feedforward routes: "
                 f"{sum(1 for r in path_rep.routes if not r.crosses_feedback)}"
                 f" · feedback-traversing routes: "
                 f"{sum(1 for r in path_rep.routes if r.crosses_feedback)}")
    lines.append(f"  intrinsic time scale: {path_rep.intrinsic_time_scale}")
    lines.append(f"  path length range: {path_rep.min_length}-"
                 f"{path_rep.max_length} (mean {path_rep.mean_length:.2f})")
    lines.append("")
    lines.append("MODES (clusters in the path landscape):")
    for c in sorted(path_rep.clusters, key=lambda c: -c.size):
        rep_chain = c.representative_chain
        if len(rep_chain) > 10:
            rep_str = " -> ".join(list(rep_chain[:5]) + ["..."]
                                  + list(rep_chain[-2:]))
        else:
            rep_str = " -> ".join(rep_chain)
        lines.append(
            f"  Mode {c.cluster_id}: size = {c.size} routes "
            f"(collapsed from {c.total_count} raw paths); "
            f"mean length = {c.mean_length:.1f}; "
            f"feedforward = {c.n_feedforward}, "
            f"feedback = {c.n_feedback}"
        )
        lines.append(f"    shared units (intersection of all routes): "
                     f"{c.shared_units}")
        lines.append(f"    representative route: {rep_str}")
    n_noise = sum(1 for r in path_rep.routes if r.cluster_id == -1)
    if n_noise:
        lines.append(f"  noise: {n_noise} unique routes did not join any mode")
    lines.append("")
    # A few sample noise routes — useful for the LLM to see outliers.
    noise_routes = [r for r in path_rep.routes if r.cluster_id == -1]
    for r in sorted(noise_routes, key=lambda r: -r.count)[:3]:
        chain = " -> ".join(r.chain[:8]) + (
            " ..." if len(r.chain) > 8 else ""
        )
        lines.append(f"    noise route x{r.count}: {chain}")
    return "\n".join(lines)


def interpret_landscape(
    phenomenon: str,
    system_info: NeuralSystemInfo,
    spec: SystemSpec,
    path_rep: PathRepresentation,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 4096,
) -> LandscapeInterpretation:
    """Agent 5: read the path landscape and explain the type of emergence."""
    if not path_rep.routes:
        return LandscapeInterpretation(
            emergence_type="(none)",
            type_rationale=(
                "no input-to-output routes exist after unrolling; the "
                "circuit specification has a structural cut between input "
                "and output cones, so no emergence can be read off."
            ),
            dominant_features=[],
            mechanism=(
                "Path landscape is empty by construction. Re-specify the "
                "system or increase time_steps to obtain non-trivial paths."
            ),
            prediction=(
                "Adding even one feed-forward edge into an output unit "
                "would create at least one mode and produce a non-degenerate "
                "landscape."
            ),
        )

    client = client or anthropic.Anthropic()
    user_msg = _format_landscape_for_llm(
        phenomenon, system_info, spec, path_rep,
    )
    # NOTE: `thinking` cannot be combined with a forced `tool_choice` on the
    # Anthropic API ("Thinking may not be enabled when tool_choice forces
    # tool use."). We need the structured tool output here, so we drop
    # thinking and let `output_config.effort` handle deliberation budget.
    response = client.messages.create(
        model=model,
        max_tokens=max_tokens,
        system=[{
            "type": "text",
            "text": INTERPRET_LANDSCAPE_SYSTEM,
            "cache_control": {"type": "ephemeral"},
        }],
        tools=[INTERPRET_LANDSCAPE_TOOL],
        tool_choice={"type": "tool", "name": "interpret_landscape"},
        output_config={"effort": "high"},
        messages=[{"role": "user", "content": user_msg}],
    )
    payload = _extract_tool_use(response, "interpret_landscape")
    if payload is None:
        raise RuntimeError(
            f"interpret_landscape: model did not call the tool "
            f"(stop_reason={response.stop_reason})"
        )
    return LandscapeInterpretation.from_dict(payload)


# ================================================================ orchestrator


def analyze_neural_circuit(
    phenomenon: str,
    out_dir: str = "./neural_circuit_analysis",
    n_paths: int = 1500,
    client: Optional["anthropic.Anthropic"] = None,
    model: str = DEFAULT_MODEL,
    verbose: bool = True,
    on_progress: Optional[Callable] = None,
) -> dict:
    """Run the 4-agent neural-circuit pipeline end-to-end.

    Returns a dict with all artefacts and on-disk file paths. The path
    representation figure (`paths.png`) is the primary user-facing output.
    """
    os.makedirs(out_dir, exist_ok=True)
    client = client or anthropic.Anthropic()

    paper_path = os.path.join(out_dir, "paper.json")
    system_path = os.path.join(out_dir, "neural_system.json")
    spec_path = os.path.join(out_dir, "spec.json")
    path_rep_path = os.path.join(out_dir, "path_rep.json")
    network_fig = os.path.join(out_dir, "network.png")
    paths_fig = os.path.join(out_dir, "paths.png")        # cluster-bundle view (primary)
    routes_fig = os.path.join(out_dir, "routes.png")      # per-route lane view (secondary)
    report_path = os.path.join(out_dir, "report.md")

    result = NeuralAnalysisResult(phenomenon=phenomenon)

    # ----------------------- Agent 1
    _emit(on_progress, "agent1_paper", 5,
          f"Agent 1: finding a research paper for {phenomenon!r}...",
          verbose)
    t0 = time.time()
    paper = find_paper(phenomenon, client=client, model=model)
    result.paper = paper
    with open(paper_path, "w") as f:
        json.dump(paper.to_dict(), f, indent=2)
    _emit(on_progress, "agent1_done", 25,
          f"Agent 1: {paper.title!r} ({paper.authors}, {paper.year}) "
          f"in {time.time() - t0:.1f}s",
          verbose)

    # ----------------------- Agent 2
    _emit(on_progress, "agent2_system", 30,
          "Agent 2: identifying the neural system responsible...",
          verbose)
    t0 = time.time()
    system_info = identify_neural_system(phenomenon, paper,
                                         client=client, model=model)
    result.system_info = system_info
    with open(system_path, "w") as f:
        json.dump(system_info.to_dict(), f, indent=2)
    _emit(on_progress, "agent2_done", 45,
          f"Agent 2: {system_info.system_name} "
          f"({len(system_info.brain_regions)} regions, "
          f"{len(system_info.neuron_types)} cell types) "
          f"in {time.time() - t0:.1f}s",
          verbose)

    # ----------------------- Agent 3
    _emit(on_progress, "agent3_network", 50,
          "Agent 3: building the network representation (units, edges, "
          "feedback, multiscale)...",
          verbose)
    t0 = time.time()
    spec = build_network(phenomenon, paper, system_info,
                         client=client, model=model)
    _validation_problems = validate_spec(spec)
    if _validation_problems:
        _emit(on_progress, "agent3_warn", 70,
              f"Agent 3 validation warnings: "
              + "; ".join(_validation_problems),
              verbose)
    result.spec = spec
    with open(spec_path, "w") as f:
        json.dump(spec.to_dict(), f, indent=2)
    sys = build_system_from_spec(spec)
    render_network_figure(spec, sys, network_fig,
                          title_suffix=system_info.system_name)
    _emit(on_progress, "agent3_done", 70,
          f"Agent 3: {spec.summary()} - "
          f"network figure -> {os.path.basename(network_fig)} "
          f"in {time.time() - t0:.1f}s",
          verbose)

    # ----------------------- Agent 4 (procedural + figure + short prose)
    _emit(on_progress, "agent4_paths", 75,
          "Agent 4: unrolling feedback loops and enumerating "
          "input → output paths...",
          verbose)
    t0 = time.time()
    try:
        path_rep = extract_path_representation(spec, sys, n_paths=n_paths)
    except RuntimeError as exc:
        # Render a still-useful diagnostic (empty paths) so the UI has an image.
        path_rep = PathRepresentation(
            time_steps=int(spec.time_steps),
            intrinsic_time_scale=0,
            paths=[],
            notes=str(exc),
        )
    result.path_rep = path_rep

    with open(path_rep_path, "w") as f:
        json.dump(path_rep.to_dict(), f, indent=2)
    # Primary figure: the path landscape AS the cluster bundles.
    # Each similarity mode is one band; member routes overlay as a
    # translucent bundle through shared anchor units.
    render_path_clusters_figure(spec, sys, path_rep, paths_fig)
    # Secondary figure: one lane per individual route, grouped by cluster.
    # Useful for inspecting specific routes; kept as a download artefact.
    render_path_representation_figure(spec, sys, path_rep, routes_fig)
    _emit(on_progress, "agent4_figure", 88,
          f"Agent 4: {path_rep.summary()} - "
          f"cluster figure -> {os.path.basename(paths_fig)}, "
          f"routes figure -> {os.path.basename(routes_fig)} "
          f"in {time.time() - t0:.1f}s",
          verbose)

    _emit(on_progress, "agent4_describe", 88,
          "Agent 4: asking Claude for a short qualitative reading of the "
          "path representation...",
          verbose)
    t0 = time.time()
    description = describe_path_representation(
        spec, system_info, path_rep, client=client, model=model,
    )
    _emit(on_progress, "agent4_done", 90,
          f"Agent 4: got {len(description)} chars of prose "
          f"in {time.time() - t0:.1f}s",
          verbose)

    # ----------------------- Agent 5 (landscape interpreter)
    interpretation_path = os.path.join(out_dir, "interpretation.json")
    _emit(on_progress, "agent5_interpret", 92,
          "Agent 5: interpreting the path landscape — identifying "
          "features and emergence type...",
          verbose)
    t0 = time.time()
    interpretation = None
    try:
        interpretation = interpret_landscape(
            phenomenon, system_info, spec, path_rep,
            client=client, model=model,
        )
        with open(interpretation_path, "w") as f:
            json.dump(interpretation.to_dict(), f, indent=2)
        _emit(on_progress, "agent5_done", 96,
              f"Agent 5: emergence_type = "
              f"{interpretation.emergence_type!r}, "
              f"{len(interpretation.dominant_features)} features "
              f"in {time.time() - t0:.1f}s",
              verbose)
    except Exception as exc:
        _emit(on_progress, "agent5_warn", 96,
              f"Agent 5 skipped (will show partial result): {exc}",
              verbose)

    # ----------------------- combined report
    _emit(on_progress, "report", 98, "writing combined markdown report...",
          verbose)
    _write_report(
        report_path, phenomenon, paper, system_info, spec, path_rep,
        description, interpretation,
        network_filename="network.png",
        paths_filename="paths.png",
    )
    _emit(on_progress, "done", 100,
          f"5-agent analysis complete -> {out_dir!r}", verbose)

    return {
        "phenomenon": phenomenon,
        "paper": paper.to_dict(),
        "neural_system": system_info.to_dict(),
        "spec": spec.to_dict(),
        "path_representation": path_rep.to_dict(),
        "description": description,
        "interpretation": interpretation.to_dict() if interpretation else None,
        "out_dir": out_dir,
        "files": {
            "paper": paper_path,
            "neural_system": system_path,
            "spec": spec_path,
            "path_rep": path_rep_path,
            "interpretation": interpretation_path,
            "network_figure": network_fig,
            "paths_figure": paths_fig,
            "routes_figure": routes_fig,
            "report": report_path,
        },
    }


# ---------------------------------------------------------------- report


def _write_report(
    out_path: str,
    phenomenon: str,
    paper: PaperInfo,
    system_info: NeuralSystemInfo,
    spec: SystemSpec,
    path_rep: PathRepresentation,
    description: str,
    interpretation: Optional[LandscapeInterpretation],
    network_filename: str = "network.png",
    paths_filename: str = "paths.png",
) -> None:
    lines: list[str] = []
    lines.append(f"# Neural circuit analysis: {spec.phenomenon_name}\n")
    lines.append(f"*Four-agent path-representation pipeline.*\n\n")
    lines.append(f"**Phenomenon (user input):** {phenomenon}\n\n")

    lines.append("## Agent 1 — Research paper\n")
    lines.append(f"**{paper.title}**\n\n")
    lines.append(f"_{paper.authors}_ ({paper.year})"
                 + (f" — {paper.venue}" if paper.venue else "")
                 + "\n\n")
    if paper.summary:
        lines.append(paper.summary + "\n\n")
    if paper.key_findings:
        lines.append("Key findings:\n")
        for k in paper.key_findings:
            lines.append(f"- {k}\n")
        lines.append("\n")
    if paper.doi_or_url:
        lines.append(f"Reference: `{paper.doi_or_url}`\n\n")

    lines.append("## Agent 2 — Neural system\n")
    lines.append(f"**{system_info.system_name}**\n\n")
    if system_info.brain_regions:
        lines.append("Brain regions: " + ", ".join(
            f"`{r}`" for r in system_info.brain_regions) + "\n\n")
    if system_info.neuron_types:
        lines.append("Neuron types: " + ", ".join(
            f"`{r}`" for r in system_info.neuron_types) + "\n\n")
    if system_info.key_circuit_motifs:
        lines.append("Circuit motifs: " + ", ".join(
            f"`{r}`" for r in system_info.key_circuit_motifs) + "\n\n")
    if system_info.description:
        lines.append(system_info.description + "\n\n")

    lines.append("## Agent 3 — Network representation\n")
    lines.append(f"![network figure]({network_filename})\n\n")
    n_in = sum(1 for u in spec.units if u.role == "input")
    n_out = sum(1 for u in spec.units if u.role == "output")
    n_int = sum(1 for u in spec.units if u.role == "internal")
    n_rec = sum(1 for it in spec.interactions if it.recurrent)
    lines.append(
        f"{len(spec.units)} units ({n_in} input, {n_int} internal, "
        f"{n_out} output) · {len(spec.interactions)} interactions "
        f"({n_rec} recurrent) · T = {spec.time_steps}\n\n"
    )
    lines.append("### Units\n\n")
    lines.append("| name | role | scale | parent | description |\n")
    lines.append("|---|---|---|---|---|\n")
    for u in spec.units:
        lines.append(
            f"| `{u.name}` | {u.role} | {u.scale} | "
            f"{u.parent or ''} | {u.description} |\n"
        )
    lines.append("\n### Interactions\n\n")
    lines.append("| source | target | weight | recurrent | description |\n")
    lines.append("|---|---|---|---|---|\n")
    for it in spec.interactions:
        rec = "yes" if it.recurrent else ""
        lines.append(
            f"| `{it.source}` | `{it.target}` | {it.weight:.2f} | "
            f"{rec} | {it.description} |\n"
        )
    lines.append("\n")

    lines.append("## Agent 4 — Path representation\n")
    lines.append(f"![path representation]({paths_filename})\n\n")
    lines.append(
        f"- Unroll steps (T): **{path_rep.time_steps}**\n"
        f"- Intrinsic time scale: **{path_rep.intrinsic_time_scale}**\n"
        f"- Raw paths: **{len(path_rep.paths)}** "
        f"(feedforward-only: {path_rep.n_feedforward_paths}, "
        f"feedback-traversing: {path_rep.n_feedback_paths})\n"
        f"- Unique routes (deduplicated): **{len(path_rep.routes)}**\n"
        f"- Similarity modes (clusters): **{path_rep.n_modes}**\n"
        f"- Path length range: {path_rep.min_length} – {path_rep.max_length} "
        f"(mean {path_rep.mean_length:.2f})\n\n"
    )
    if path_rep.clusters:
        lines.append("### Similarity modes\n\n")
        lines.append("| mode | size | total count | mean length | "
                     "feedforward | feedback | shared units | "
                     "representative |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for c in sorted(path_rep.clusters, key=lambda c: -c.size):
            rep = " → ".join(c.representative_chain[:8]) + (
                " …" if len(c.representative_chain) > 8 else "")
            shared = ", ".join(f"`{u}`" for u in c.shared_units)
            lines.append(
                f"| Mode {c.cluster_id} | {c.size} | {c.total_count} | "
                f"{c.mean_length:.1f} | {c.n_feedforward} | "
                f"{c.n_feedback} | {shared} | {rep} |\n"
            )
        lines.append("\n")
    if path_rep.notes:
        lines.append(f"_Note:_ {path_rep.notes}\n\n")
    lines.append("### Description\n\n")
    lines.append(description.rstrip() + "\n\n")

    lines.append("## Agent 5 — Landscape interpretation\n")
    if interpretation is None:
        lines.append("_(Agent 5 was skipped or failed — no interpretation available.)_\n")
    else:
        lines.append(f"**Emergence type:** {interpretation.emergence_type}\n\n")
        if interpretation.type_rationale:
            lines.append(f"{interpretation.type_rationale}\n\n")
        if interpretation.dominant_features:
            lines.append("### Dominant path-structural features\n\n")
            for f in interpretation.dominant_features:
                metric_suffix = f"  _( {f.metric} )_" if f.metric else ""
                lines.append(f"- **{f.name}**{metric_suffix}: {f.description}\n")
            lines.append("\n")
        if interpretation.mechanism:
            lines.append("### Mechanism\n\n")
            lines.append(interpretation.mechanism.rstrip() + "\n\n")
        if interpretation.prediction:
            lines.append("### Falsifiable prediction\n\n")
            lines.append(interpretation.prediction.rstrip() + "\n")

    with open(out_path, "w") as f:
        f.write("".join(lines))
