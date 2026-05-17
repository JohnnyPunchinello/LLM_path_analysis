"""Dataclasses for the system specification produced by the specifier LLM."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class SpecUnit:
    """A basic computing unit in the system."""

    name: str
    role: str                # "input" | "internal" | "output"
    scale: int = 0
    parent: Optional[str] = None
    description: str = ""


@dataclass
class SpecInteraction:
    """A directed interaction between two units."""

    source: str
    target: str
    weight: float = 1.0
    recurrent: bool = False
    description: str = ""


@dataclass
class SpecParameter:
    """An external parameter that shapes the system (size, temperature, ...)."""

    name: str
    role: str
    value: str = ""


@dataclass
class SystemSpec:
    """Full system spec for one emergent phenomenon."""

    phenomenon_name: str
    phenomenon_summary: str
    units: list[SpecUnit]
    interactions: list[SpecInteraction]
    time_steps: int = 1
    parameters: list[SpecParameter] = field(default_factory=list)
    notes: str = ""

    @classmethod
    def from_dict(cls, d: dict) -> "SystemSpec":
        def _unit(u: dict) -> "SpecUnit":
            return SpecUnit(
                name=str(u["name"]),
                role=str(u.get("role", "internal")),
                scale=int(u.get("scale", 0) or 0),
                parent=u.get("parent") or None,
                description=str(u.get("description", "")),
            )

        def _interaction(i: dict) -> "SpecInteraction":
            return SpecInteraction(
                source=str(i["source"]),
                target=str(i["target"]),
                weight=float(i.get("weight", 1.0) or 1.0),
                recurrent=bool(i.get("recurrent", False)),
                description=str(i.get("description", "")),
            )

        def _parameter(p: dict) -> "SpecParameter":
            return SpecParameter(
                name=str(p["name"]),
                role=str(p.get("role", "")),
                value=str(p.get("value", "")),
            )

        return cls(
            phenomenon_name=str(d.get("phenomenon_name", "unknown")),
            phenomenon_summary=str(d.get("phenomenon_summary", "")),
            units=[_unit(u) for u in d.get("units", []) or []],
            interactions=[_interaction(i) for i in d.get("interactions", []) or []],
            time_steps=max(1, int(d.get("time_steps", 1) or 1)),
            parameters=[_parameter(p) for p in d.get("parameters", []) or []],
            notes=str(d.get("notes", "")),
        )

    def to_dict(self) -> dict:
        return asdict(self)

    # ------------------------------------------------------------- summary
    def summary(self) -> str:
        ni = sum(1 for u in self.units if u.role == "input")
        no = sum(1 for u in self.units if u.role == "output")
        nint = sum(1 for u in self.units if u.role == "internal")
        nrec = sum(1 for x in self.interactions if x.recurrent)
        scales = sorted({u.scale for u in self.units})
        return (
            f"SystemSpec({self.phenomenon_name!r}: "
            f"{len(self.units)} units [in={ni} int={nint} out={no}], "
            f"{len(self.interactions)} interactions [recurrent={nrec}], "
            f"time_steps={self.time_steps}, scales={scales})"
        )
