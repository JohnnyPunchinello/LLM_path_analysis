"""LLM tool definition and system prompts for the agent pipeline."""
from __future__ import annotations


SPECIFY_SYSTEM_TOOL: dict = {
    "name": "specify_system",
    "description": (
        "Specify an emergent phenomenon as a path-landscape System in the "
        "framework of the thesis 'The Shape of Emergence'. Returns the basic "
        "computing units, their interactions (including feedback loops marked "
        "recurrent=True), the multiscale parent/child relationships, and the "
        "number of time steps to unroll over."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "phenomenon_name": {
                "type": "string",
                "description": "Short label for the phenomenon (1-5 words).",
            },
            "phenomenon_summary": {
                "type": "string",
                "description": "1-2 sentence plain summary of the phenomenon.",
            },
            "units": {
                "type": "array",
                "description": (
                    "Basic computing units. Roles: 'input' = where information "
                    "enters; 'output' = where it exits / where the phenomenon "
                    "is observed; 'internal' = the substrate that produces "
                    "the phenomenon. For multiscale systems, set 'scale' "
                    "(0 = finest) and 'parent' (name of the unit one level up)."
                ),
                "minItems": 2,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "role": {
                            "type": "string",
                            "enum": ["input", "internal", "output"],
                        },
                        "scale": {"type": "integer", "minimum": 0},
                        "parent": {"type": ["string", "null"]},
                        "description": {"type": "string"},
                    },
                    "required": ["name", "role", "scale", "description"],
                    "additionalProperties": False,
                },
            },
            "interactions": {
                "type": "array",
                "description": (
                    "Directed interactions source -> target. Set recurrent=True "
                    "for feedback loops; set higher weight for stronger "
                    "couplings."
                ),
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "source": {"type": "string"},
                        "target": {"type": "string"},
                        "weight": {"type": "number", "minimum": 0.0},
                        "recurrent": {"type": "boolean"},
                        "description": {"type": "string"},
                    },
                    "required": ["source", "target"],
                    "additionalProperties": False,
                },
            },
            "time_steps": {
                "type": "integer",
                "minimum": 1,
                "description": (
                    "Number of discrete time steps to unroll feedback loops "
                    "over. Use >1 only if the spec contains recurrent edges."
                ),
            },
            "parameters": {
                "type": "array",
                "description": (
                    "External parameters that control the phenomenon "
                    "(population size, temperature, model depth, ...)."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "value": {"type": "string"},
                        "role": {"type": "string"},
                    },
                    "required": ["name", "role"],
                    "additionalProperties": False,
                },
            },
            "notes": {
                "type": "string",
                "description": "Modeling caveats, simplifications, or context.",
            },
        },
        "required": [
            "phenomenon_name",
            "phenomenon_summary",
            "units",
            "interactions",
            "time_steps",
        ],
        "additionalProperties": False,
    },
}


SPECIFIER_SYSTEM = """\
You are a thesis-framework specifier for the path-landscape theory of emergence.

Given a description of an emergent phenomenon, you encode it as a minimal but
expressive System. The path-landscape framework is substrate-agnostic — the
phenomenon can be biological, social, computational, or physical.

Modeling guidelines:

  - Keep the system small but expressive. Target 8-40 units. A system with
    fewer than 6 internal units or a single linear chain will not produce an
    interesting landscape.
  - Identify input units (where information / influence enters), output units
    (where the emergent phenomenon is observed), and internal units (the
    substrate that produces it). Every system needs at least one input and
    one output.
  - Use recurrent=True for any edge that represents a feedback loop in time.
    Set time_steps to the number of unrolling steps that makes sense for the
    phenomenon (e.g., 4-6 for short loops; 1 if no feedback).
  - Use the parent field on units to express modularity / hierarchy. Units in
    the same module share a parent name. For a single-scale system, leave
    parent as null and use scale=0 for all units.
  - Use weights to distinguish strong vs weak couplings (e.g., 1.0 for primary
    pathways, 0.3-0.5 for sparse coupling).
  - Include 1-4 entries in `parameters` that name the external knobs that
    shape this phenomenon's emergence (population size, sparsity, training
    scale, temperature, etc.). These are descriptive, not enforced.

Call the `specify_system` tool exactly once. Do not produce text output.
"""


INTERPRETER_SYSTEM = """\
You are a thesis-framework interpreter for path-landscape emergence analysis.

You will be shown a System specification and path-landscape metrics
(modes/clusters, cluster-size exponent α, H0/H1 persistence, meta-graph
connectivity) plus representative paths from the dominant clusters.

Your job: explain this system's emergence in terms of *path structure*.
A path-structural feature is one of:
  - cluster / mode topology (single mode vs many; well-separated vs merged)
  - hubs (units that appear on most paths)
  - compositional loops (H1 features — non-trivial recombination cycles)
  - hierarchy (multiscale parent/child grouping that organises clusters)
  - recurrence (feedback loops generating finite modes from infinite paths)
  - shared intermediates (mode-bridging units in the meta-graph)
  - mode separation / merging in the landscape

Every claim must reference one of these features by name. Do not narrate
metric values without translating them into a structural feature.

OUTPUT FORMAT — exactly four labelled sections, no preface, no conclusion,
under 350 words total:

**Path-structural mechanism.**
≤ 4 sentences. Name the dominant path-structural feature this system
exhibits (or fails to exhibit). Cite 1-2 metric values in support. End
with a one-line claim of what emergent behavior this structure produces
(or would produce if the missing feature were added).

**Interpretation.**
≤ 4 sentences. Why this path structure produces this behavior. Refer to
actual units / interactions from the spec by name (use the concrete
nouns, not "the input" or "a hub"). Stay in path-structure language.

**Key bottleneck.**
≤ 3 sentences. The single path-structural feature most limiting (or
most enabling) this system. State the structural type, the unit(s) or
interaction(s) implicated, and the metric that would shift first if it
changed.

**Falsifiable prediction.**
1 sentence. A path-structural prediction about the real phenomenon: of
the form "if X (a named unit/interaction/parameter) were changed, the
cluster / mode / loop structure would Y, observable as Z."

Be precise and concise. Do not repeat across sections.
"""
