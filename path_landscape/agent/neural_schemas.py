"""Dataclasses for the 5-agent neural-circuit analysis pipeline.

The neural-circuit pipeline produces five artefacts, one per agent:

    Agent 1  ->  PaperInfo               (research paper identifying the circuit)
    Agent 2  ->  NeuralSystemInfo        (brain regions / cell types / motifs)
    Agent 3  ->  SystemSpec + figure     (network representation, input/output)
    Agent 4  ->  PathRepresentation      (unique routes, clustered by similarity)
    Agent 5  ->  LandscapeInterpretation (emergence type + dominant features)

`SystemSpec` is re-used from `schemas.py`. The rest are defined here.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Optional


@dataclass
class PaperInfo:
    """A research paper that grounds the circuit identification."""

    title: str
    authors: str          # comma-separated; we don't need a list here
    year: int
    venue: str = ""       # journal / conference, optional
    summary: str = ""     # 2-4 sentences in plain language
    key_findings: list[str] = field(default_factory=list)
    doi_or_url: str = ""  # optional pointer; the model may leave blank

    @classmethod
    def from_dict(cls, d: dict) -> "PaperInfo":
        return cls(
            title=d["title"],
            authors=d.get("authors", ""),
            year=int(d.get("year", 0) or 0),
            venue=d.get("venue", ""),
            summary=d.get("summary", ""),
            key_findings=list(d.get("key_findings", []) or []),
            doi_or_url=d.get("doi_or_url", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class NeuralSystemInfo:
    """Identification of the neural system / circuit responsible."""

    system_name: str                 # e.g., "Hippocampal CA3-CA1 recurrent circuit"
    brain_regions: list[str] = field(default_factory=list)
    neuron_types: list[str] = field(default_factory=list)
    key_circuit_motifs: list[str] = field(default_factory=list)
    description: str = ""            # 2-4 sentences

    @classmethod
    def from_dict(cls, d: dict) -> "NeuralSystemInfo":
        return cls(
            system_name=d["system_name"],
            brain_regions=list(d.get("brain_regions", []) or []),
            neuron_types=list(d.get("neuron_types", []) or []),
            key_circuit_motifs=list(d.get("key_circuit_motifs", []) or []),
            description=d.get("description", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class ClassifiedPath:
    """One enumerated input->output path, tagged by feedback content."""

    nodes: tuple[str, ...]            # ordered chain of unrolled node names ("u@t")
    base_chain: tuple[str, ...]       # the same chain stripped of "@t" suffixes
    length: int                       # number of edges = len(nodes) - 1
    crosses_feedback: bool            # True iff this path traverses any recurrent edge
    weight: float = 1.0
    time_span: int = 0                # max(t) - min(t) over the path

    def to_dict(self) -> dict:
        return {
            "nodes": list(self.nodes),
            "base_chain": list(self.base_chain),
            "length": self.length,
            "crosses_feedback": self.crosses_feedback,
            "weight": self.weight,
            "time_span": self.time_span,
        }


@dataclass
class UniqueRoute:
    """A deduplicated input->output route: a sequence of unit names with no
    consecutive duplicates and no time-step suffix.

    `count` is the number of raw enumerated paths that collapsed onto this
    route (paths through the same units at different time steps, or paths
    that lingered on a self-loop).
    """

    chain: tuple[str, ...]
    count: int
    weight: float
    crosses_feedback: bool
    cluster_id: int = -1     # -1 = noise / unclustered

    def to_dict(self) -> dict:
        return {
            "chain": list(self.chain),
            "count": int(self.count),
            "weight": float(self.weight),
            "crosses_feedback": bool(self.crosses_feedback),
            "cluster_id": int(self.cluster_id),
        }


@dataclass
class RouteCluster:
    """A similarity-based cluster of unique routes (a *mode* of the
    path landscape)."""

    cluster_id: int                      # -1 = noise
    size: int                            # number of unique routes in cluster
    total_count: int                     # sum of route counts (popularity)
    representative_chain: tuple[str, ...]
    mean_length: float
    n_feedforward: int = 0
    n_feedback: int = 0
    shared_units: list[str] = field(default_factory=list)  # units present in every route in the cluster

    def to_dict(self) -> dict:
        return {
            "cluster_id": int(self.cluster_id),
            "size": int(self.size),
            "total_count": int(self.total_count),
            "representative_chain": list(self.representative_chain),
            "mean_length": float(self.mean_length),
            "n_feedforward": int(self.n_feedforward),
            "n_feedback": int(self.n_feedback),
            "shared_units": list(self.shared_units),
        }


@dataclass
class PathRepresentation:
    """Output of Agent 4: the path representation of the circuit."""

    time_steps: int                     # T used for unrolling
    intrinsic_time_scale: int           # estimated time-scale of the circuit
    paths: list[ClassifiedPath] = field(default_factory=list)
    routes: list[UniqueRoute] = field(default_factory=list)
    clusters: list[RouteCluster] = field(default_factory=list)
    n_modes: int = 0
    n_feedforward_paths: int = 0
    n_feedback_paths: int = 0
    min_length: int = 0
    max_length: int = 0
    mean_length: float = 0.0
    notes: str = ""

    def to_dict(self) -> dict:
        return {
            "time_steps": self.time_steps,
            "intrinsic_time_scale": self.intrinsic_time_scale,
            "n_paths": len(self.paths),
            "n_routes": len(self.routes),
            "n_modes": self.n_modes,
            "n_feedforward_paths": self.n_feedforward_paths,
            "n_feedback_paths": self.n_feedback_paths,
            "min_length": self.min_length,
            "max_length": self.max_length,
            "mean_length": self.mean_length,
            "notes": self.notes,
            "paths": [p.to_dict() for p in self.paths],
            "routes": [r.to_dict() for r in self.routes],
            "clusters": [c.to_dict() for c in self.clusters],
        }

    def summary(self) -> str:
        return (
            f"PathRepresentation(T={self.time_steps}, "
            f"intrinsic_time_scale={self.intrinsic_time_scale}, "
            f"n_paths={len(self.paths)}, "
            f"n_routes={len(self.routes)}, "
            f"n_modes={self.n_modes} "
            f"[feedforward={self.n_feedforward_paths}, "
            f"feedback={self.n_feedback_paths}], "
            f"length={self.min_length}-{self.max_length} "
            f"(mean {self.mean_length:.2f}))"
        )


# ============================================================ Agent 5


@dataclass
class LandscapeFeature:
    """One named, prominent feature of the path landscape."""

    name: str            # e.g., "dominant cluster", "shared hub", "compositional loop"
    description: str     # one-sentence explanation referencing concrete units
    metric: str = ""     # optional metric anchor (e.g., "mode 0 = 12/24 routes")

    @classmethod
    def from_dict(cls, d: dict) -> "LandscapeFeature":
        return cls(
            name=d["name"],
            description=d["description"],
            metric=d.get("metric", ""),
        )

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class LandscapeInterpretation:
    """Agent 5's structured interpretation of the path landscape."""

    emergence_type: str
    type_rationale: str
    dominant_features: list[LandscapeFeature] = field(default_factory=list)
    mechanism: str = ""
    prediction: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "LandscapeInterpretation":
        return cls(
            emergence_type=d["emergence_type"],
            type_rationale=d.get("type_rationale", ""),
            dominant_features=[
                LandscapeFeature.from_dict(f)
                for f in d.get("dominant_features", []) or []
            ],
            mechanism=d.get("mechanism", ""),
            prediction=d.get("prediction", ""),
        )

    def to_dict(self) -> dict:
        return {
            "emergence_type": self.emergence_type,
            "type_rationale": self.type_rationale,
            "dominant_features": [f.to_dict() for f in self.dominant_features],
            "mechanism": self.mechanism,
            "prediction": self.prediction,
        }


@dataclass
class NeuralAnalysisResult:
    """Bundle of the five agent outputs, kept together for reporting."""

    phenomenon: str
    paper: Optional[PaperInfo] = None
    system_info: Optional[NeuralSystemInfo] = None
    spec: Optional[object] = None              # SystemSpec (avoid circular import)
    path_rep: Optional[PathRepresentation] = None
    interpretation: Optional[LandscapeInterpretation] = None
