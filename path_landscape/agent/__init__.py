"""Agentic pipeline: phenomenon → path-landscape analysis → mechanistic report.

Given a natural-language description of an emergent phenomenon, this module
runs the full thesis-framework pipeline:

  1. **specify** — call Claude with a structured-output tool to encode the
     phenomenon as a System: basic computing units, interactions (including
     feedback loops and multi-scale parent relationships), time-unrolling
     parameters.
  2. **build**   — convert the spec into a `path_landscape.System` object.
  3. **extract** — unroll feedback loops in time, coarsen scales, sample
     paths from inputs to outputs through the resulting static graph.
  4. **analyze** — cluster the paths into modes and compute the four
     cross-system landscape metrics (modes, size exponent, persistence,
     meta-graph connectivity).
  5. **interpret** — call Claude again with the spec + metrics to produce
     a focused mechanistic explanation of how the path structure produces
     (or fails to produce) emergence here.
  6. **report**  — save a markdown report, a multi-panel figure, and the
     raw spec JSON to a chosen output directory.

Entry point:

    from path_landscape.agent import analyze_emergence
    result = analyze_emergence("A flock of starlings turning as one")
"""
from .schemas import (
    SpecUnit,
    SpecInteraction,
    SpecParameter,
    SystemSpec,
)
from .pipeline import analyze_emergence, specify_system, run_analysis, interpret
from .builder import build_system_from_spec
from .neural_schemas import (
    ClassifiedPath,
    LandscapeFeature,
    LandscapeInterpretation,
    NeuralAnalysisResult,
    NeuralSystemInfo,
    PaperInfo,
    PathRepresentation,
    RouteCluster,
    UniqueRoute,
)
from .neural_pipeline import (
    analyze_neural_circuit,
    build_network,
    describe_path_representation,
    extract_path_representation,
    find_paper,
    identify_neural_system,
    interpret_landscape,
)

__all__ = [
    "SpecUnit",
    "SpecInteraction",
    "SpecParameter",
    "SystemSpec",
    "analyze_emergence",
    "specify_system",
    "run_analysis",
    "interpret",
    "build_system_from_spec",
    # 5-agent neural-circuit pipeline
    "PaperInfo",
    "NeuralSystemInfo",
    "ClassifiedPath",
    "UniqueRoute",
    "RouteCluster",
    "PathRepresentation",
    "LandscapeFeature",
    "LandscapeInterpretation",
    "NeuralAnalysisResult",
    "analyze_neural_circuit",
    "find_paper",
    "identify_neural_system",
    "build_network",
    "extract_path_representation",
    "describe_path_representation",
    "interpret_landscape",
]
