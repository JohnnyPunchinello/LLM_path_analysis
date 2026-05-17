"""Build a `path_landscape.System` from a `SystemSpec`."""
from __future__ import annotations

from ..system import System
from .schemas import SystemSpec


def build_system_from_spec(spec: SystemSpec) -> System:
    """Construct a System from the LLM-generated spec.

    - Each `SpecUnit` becomes a `Unit` (scale and parent preserved).
    - Each `SpecInteraction` becomes an edge; recurrent edges are flagged so
      `System.unroll(T)` materializes them as forward edges across time.
    - Inputs and outputs are taken from unit `role`.
    """
    sys = System()
    # de-dup names defensively
    seen: set[str] = set()
    for u in spec.units:
        if u.name in seen:
            continue
        seen.add(u.name)
        sys.add_unit(
            u.name,
            scale=int(u.scale),
            parent=(u.parent if u.parent else None),
        )
    for it in spec.interactions:
        if it.source not in sys.units or it.target not in sys.units:
            # silently skip dangling references rather than erroring out
            continue
        sys.add_edge(
            it.source,
            it.target,
            weight=float(it.weight),
            recurrent=bool(it.recurrent),
        )

    inputs = [u.name for u in spec.units if u.role == "input" and u.name in sys.units]
    outputs = [u.name for u in spec.units if u.role == "output" and u.name in sys.units]
    if not inputs:
        raise ValueError(
            "spec has no units with role='input'; can't define path sources"
        )
    if not outputs:
        raise ValueError(
            "spec has no units with role='output'; can't define path sinks"
        )
    sys.set_input(*inputs)
    sys.set_output(*outputs)
    return sys


def validate_spec(spec: "SystemSpec") -> list[str]:
    """Return a (possibly empty) list of structural problems with the spec.

    Each string is a human-readable description of a problem that must be
    fixed before the spec can be used for path enumeration.  An empty list
    means the spec passed all checks.
    """
    problems: list[str] = []
    unit_names = {u.name for u in spec.units}

    inputs  = [u for u in spec.units if u.role == "input"]
    outputs = [u for u in spec.units if u.role == "output"]

    if not inputs:
        problems.append("No units with role='input'.")
    if not outputs:
        problems.append("No units with role='output'.")
    if len(spec.units) < 3:
        problems.append(
            f"Only {len(spec.units)} unit(s) total — need at least 3 "
            f"(1 input + 1+ internal + 1 output) for a meaningful path landscape."
        )

    # Dangling references
    dangling: list[str] = []
    for it in spec.interactions:
        if it.source not in unit_names:
            dangling.append(f"source {it.source!r}")
        if it.target not in unit_names:
            dangling.append(f"target {it.target!r}")
    if dangling:
        problems.append(
            "Interactions reference unit names not in the units list: "
            + ", ".join(dict.fromkeys(dangling))
        )

    # Connectivity: is there any directed path from an input to an output?
    if inputs and outputs and not dangling:
        try:
            import networkx as _nx
            g = _nx.DiGraph()
            for u in spec.units:
                g.add_node(u.name)
            for it in spec.interactions:
                if it.source in unit_names and it.target in unit_names:
                    g.add_edge(it.source, it.target)
            input_names  = {u.name for u in inputs}
            output_names = {u.name for u in outputs}
            reachable: set[str] = set()
            for inp in input_names:
                reachable |= _nx.descendants(g, inp) | {inp}
            if not (reachable & output_names):
                problems.append(
                    f"No directed path from inputs "
                    f"({', '.join(sorted(input_names))}) to outputs "
                    f"({', '.join(sorted(output_names))}). "
                    f"Ensure that interactions form a connected chain "
                    f"from at least one input unit to at least one output unit."
                )
        except Exception:
            pass  # if networkx unavailable, skip graph check

    # time_steps vs. recurrence consistency
    has_recurrent = any(it.recurrent for it in spec.interactions)
    ts = int(spec.time_steps)
    if ts < 1:
        problems.append(f"time_steps must be ≥ 1 (got {ts}).")
    elif has_recurrent and ts < 2:
        problems.append(
            f"Spec contains recurrent edges but time_steps={ts}. "
            f"Set time_steps ≥ 2 so feedback loops can be unrolled."
        )

    return problems
