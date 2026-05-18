"""Tool definitions and system prompts for the 4-agent neural-circuit pipeline."""
from __future__ import annotations


# ---------------------------------------------------------------- Agent 1

FIND_PAPER_TOOL: dict = {
    "name": "report_research_paper",
    "description": (
        "Identify a foundational or representative peer-reviewed research paper "
        "that characterizes the circuit (neural, computational, or mixed) "
        "responsible for the given emergent phenomenon. For LLM/transformer "
        "phenomena, prefer mechanistic interpretability papers. Prefer well-cited "
        "works that explicitly discuss the circuit-level mechanism."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Full paper title.",
            },
            "authors": {
                "type": "string",
                "description": "Comma-separated author list (last names ok).",
            },
            "year": {
                "type": "integer",
                "description": "Publication year.",
            },
            "venue": {
                "type": "string",
                "description": "Journal or conference (optional).",
            },
            "summary": {
                "type": "string",
                "description": (
                    "2-4 sentence plain-language summary of what the paper "
                    "shows about the circuit and how it produces the phenomenon."
                ),
            },
            "key_findings": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "3-5 bullet-style findings most relevant to the circuit "
                    "architecture (cell types, connectivity, dynamics)."
                ),
            },
            "doi_or_url": {
                "type": "string",
                "description": (
                    "DOI or URL if you are confident; otherwise leave blank. "
                    "Never fabricate."
                ),
            },
        },
        "required": ["title", "authors", "year", "summary", "key_findings"],
        "additionalProperties": False,
    },
}


FIND_PAPER_SYSTEM = """\
You are a circuit-literature locator covering both neuroscience and
mechanistic interpretability of AI systems. Given a description of an
emergent phenomenon, you identify ONE foundational or representative
peer-reviewed research paper that characterises the circuit producing it.

The circuit may be:
  - Biological: a neural circuit in the brain (hippocampus, cortex, etc.).
  - Computational: a circuit within a transformer / LLM (e.g., attention
    heads, induction circuits, IOI circuits in GPT-2, factual recall in
    large models). For these prefer mechanistic interpretability papers
    (Elhage et al. 2021, Wang et al. 2022, Olsson et al. 2022, etc.).
  - Mixed: a neural-computational hybrid.

Constraints:
  - Prefer papers that explicitly characterise the *circuit-level* mechanism
    (unit types, connectivity, feedback), not purely behavioural or
    imaging-only studies, unless those define the circuit.
  - Pick a paper you are confident actually exists. Do NOT fabricate titles,
    authors, journals, or DOIs. If unsure of the DOI/URL, leave it blank.
  - One paper is enough — pick the most canonical or the most cited recent
    review/primary paper.
  - Only report a paper you are CONFIDENT actually exists. If uncertain
    about details, choose a safer, more famous paper you know with
    certainty. A well-known review is preferable to an uncertain primary.
  - Never fabricate author names — only include authors you can recall
    with genuine confidence.

Call the `report_research_paper` tool exactly once. Do not produce free text.
"""


# ---------------------------------------------------------------- Agent 2

IDENTIFY_SYSTEM_TOOL: dict = {
    "name": "report_neural_system",
    "description": (
        "Identify the circuit (biological or computational) responsible for "
        "the given emergent phenomenon. Use the paper provided as context. "
        "Works for brain circuits AND LLM/transformer circuits."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "system_name": {
                "type": "string",
                "description": (
                    "Short canonical name for the circuit/system. "
                    "Biological examples: 'Hippocampal CA3-CA1 recurrent circuit', "
                    "'V1 orientation column with lateral inhibition'. "
                    "LLM examples: 'GPT-2 small indirect-object identification "
                    "circuit', 'Transformer induction-head circuit'."
                ),
            },
            "brain_regions": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Anatomical regions (biological) or computational modules "
                    "(LLM/ANN) involved. Biological: 'CA3', 'V1', 'PFC'. "
                    "LLM: 'Layer 3-4 attention', 'MLP block L9', 'residual stream'. "
                    "1-6 items."
                ),
            },
            "neuron_types": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Biological cell types OR computational unit types. "
                    "Biological: 'CA3 pyramidal', 'PV+ basket', 'L5 pyramidal'. "
                    "LLM: 'name mover head', 'induction head', 'S-inhibition head', "
                    "'MLP neuron', 'attention head'. 2-8 items."
                ),
            },
            "key_circuit_motifs": {
                "type": "array",
                "items": {"type": "string"},
                "description": (
                    "Named connectivity motifs. Biological: 'recurrent excitation', "
                    "'feedforward inhibition', 'long-range feedback'. "
                    "LLM: 'attention composition (Q/K/V)', 'residual stream addition', "
                    "'skip connection', 'attention pattern copying'. "
                    "1-6 items."
                ),
            },
            "description": {
                "type": "string",
                "description": (
                    "2-4 sentence description of how this circuit produces "
                    "the phenomenon."
                ),
            },
        },
        "required": [
            "system_name",
            "brain_regions",
            "neuron_types",
            "key_circuit_motifs",
            "description",
        ],
        "additionalProperties": False,
    },
}


IDENTIFY_SYSTEM_SYSTEM = """\
You are a circuit identifier for both biological neural circuits and
computational circuits (transformer / LLM circuits). You will be given
an emergent phenomenon and a research paper that anchors its circuit-level
mechanism. You identify the circuit responsible.

For BIOLOGICAL circuits: use brain regions (CA3, V1, PFC ...) in
`brain_regions`, cell types (CA3 pyramidal, PV basket ...) in
`neuron_types`, and motifs (recurrent excitation, feedforward inhibition,
lateral inhibition, gain modulation ...) in `key_circuit_motifs`.

For COMPUTATIONAL circuits (transformers, LLMs): use computational
modules (Layer 9 attention, MLP block L11, residual stream ...) in
`brain_regions`, unit types (attention head, induction head, name mover
head, MLP neuron ...) in `neuron_types`, and information-flow motifs
(attention composition, residual stream addition, skip connection,
attention pattern copying ...) in `key_circuit_motifs`.

GROUNDING RULE: Base your identification ONLY on the paper summary and
key findings provided. Do not introduce regions, unit types, or motifs
not mentioned or clearly implied by the paper. Speculation beyond the
paper is hallucination — avoid it.

Use standard nomenclature for the circuit type. For LLM circuits, prefer
mechanistic interpretability terminology (e.g., "name mover head",
"induction head", "Q-composition", "OV circuit").

Call the `report_neural_system` tool exactly once. Do not produce free text.
"""


# ---------------------------------------------------------------- Agent 3

# Re-use the existing SPECIFY_SYSTEM_TOOL schema but with a neuro-specific
# system prompt that emphasises the input/output structure required by the
# UI ("one input side, one output side, information flow between").

BUILD_NETWORK_SYSTEM = """\
You are a circuit modeller for the path-landscape framework. You model
both biological neural circuits AND computational circuits (transformers,
LLMs, ANNs).

You will be given (a) a phenomenon, (b) the anchor research paper, and
(c) the identified circuit (regions/modules, unit types, motifs). You
encode the system as a directed graph of computing units with explicit
inputs, outputs, and (where applicable) feedback loops and multiscale
parent/child structure.

━━━ GROUNDING RULES (strictly required) ━━━
  - Unit names MUST come from the regions/modules and unit types listed
    in the CIRCUIT SPEC. Do NOT introduce new regions, unit types, or
    connections not mentioned there.
  - Every interaction must correspond to a known projection or information-
    flow pathway either (a) mentioned in the paper or circuit description,
    or (b) standard knowledge for this circuit type.
  - Use compact, standard nomenclature:
      Biological: "CA3_pyr", "PV_basket", "L2/3_pyr", "dLGN_relay"
      LLM/transformer: "tok_embed", "attn_L3H7", "mlp_L9", "resid_L11",
                       "name_mover_L9H6", "s_inhib_L8H0", "logit_out"
    Avoid vague names like "unit1", "excitatory1", or "hub".
  - Do NOT fabricate connectivity. If unsure whether a connection exists,
    omit it.

━━━ REQUIRED STRUCTURE ━━━
  - One clear *input side*: at least one unit with role='input'.
      Biological: the sensory or contextual signal.
      LLM: token embeddings, subject token, indirect-object token, etc.
  - One clear *output side*: at least one unit with role='output'.
      Biological: the electrophysiological or behavioural observable.
      LLM: the output logit / next-token distribution.
  - Internal units: the substrate transforming input → output.
  - Feedback: mark every recurrent connection with recurrent=True.
      For LLMs / pure feedforward transformers: almost all connections
      are feedforward (recurrent=False); set time_steps=1 unless the
      phenomenon explicitly involves temporal recurrence.
  - Multiscale: when a module contains multiple unit types, set
    parent=<module-name> on those units.
  - time_steps: feedforward (incl. most transformers) → 1;
    mild recurrence → 3-4; strong recurrent integration → 5-8.

━━━ PRE-SUBMISSION SELF-CHECK ━━━
Before calling specify_system, verify ALL of the following:
  ✓ At least 1 unit with role='input'
  ✓ At least 1 unit with role='output'
  ✓ At least 4 units total (1 input + ≥2 internal + 1 output)
  ✓ Every interaction's source and target appear in the units list
  ✓ A directed path from some input to some output exists
  ✓ time_steps ≥ 2 if ANY interaction has recurrent=True
  ✓ No invented regions or unit types beyond what the circuit spec listed

Keep the system small but expressive (8–30 units). Call `specify_system`
exactly once. Do not produce free text.
"""


# ---------------------------------------------------------------- Agent 4
#
# Agent 4 is mostly procedural (unroll + enumerate paths + classify), but
# we still ask Claude for a short qualitative interpretation of the path
# representation that the figure will display.

DESCRIBE_PATHS_SYSTEM = """\
You are a path-representation interpreter for the path-landscape framework.

You will be shown:
  - the neural system that was identified
  - the time-unrolled DAG produced by collapsing feedback loops in time
  - statistics of the enumerated input->output paths
  - a few representative paths (feedforward and feedback-traversing)

Your job: a SHORT description (≤ 120 words, 3-5 sentences) that explains
what the path representation reveals about the circuit. Address:
  1. The intrinsic time scale (T) and what it represents biologically.
  2. The role of feedforward paths vs. feedback-traversing (longer) paths.
  3. One concrete claim about what the path structure means for the
     emergent phenomenon (e.g., persistence, pattern completion, gain
     modulation, attractor formation).

Plain prose, no headers, no bullet lists. Be concrete: reference unit
names from the spec.
"""


# ---------------------------------------------------------------- Agent 5

INTERPRET_LANDSCAPE_TOOL: dict = {
    "name": "interpret_landscape",
    "description": (
        "Interpret the path landscape — its similarity modes (clusters), "
        "shared hubs, dominant routes, and the relationship between "
        "feedforward and feedback structure — to identify what type of "
        "emergence the circuit produces and how the path structure causes it."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "emergence_type": {
                "type": "string",
                "description": (
                    "A short canonical label for the type of emergence the "
                    "path landscape exhibits. Examples: "
                    "'attractor / pattern completion', "
                    "'winner-take-all selection', "
                    "'compositional binding', "
                    "'gain modulation', "
                    "'sequential computation / chain-of-thought', "
                    "'oscillatory / rhythmic dynamics', "
                    "'persistent activity / working memory', "
                    "'mutual inhibition / decision', "
                    "'sparse coding / sparsification', "
                    "'predictive coding'. Use one of these if it fits, "
                    "otherwise coin a precise 2-5 word label."
                ),
            },
            "type_rationale": {
                "type": "string",
                "description": (
                    "1-2 sentences justifying the emergence_type label, "
                    "anchored in the actual mode structure (e.g., 'two "
                    "near-equal-size modes reading out different output "
                    "ensembles -> winner-take-all')."
                ),
            },
            "dominant_features": {
                "type": "array",
                "minItems": 2,
                "maxItems": 5,
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {
                            "type": "string",
                            "description": (
                                "Short name of the path-structural feature, "
                                "drawn from: 'dominant cluster', "
                                "'shared hub', 'compositional loop', "
                                "'feedback bottleneck', 'mode separation', "
                                "'noise tail', 'fan-out / divergence', "
                                "'fan-in / convergence', 'hierarchical "
                                "module', 'recurrent attractor'."
                            ),
                        },
                        "description": {
                            "type": "string",
                            "description": (
                                "1-2 sentence explanation that names the "
                                "concrete units / clusters involved (use "
                                "actual names from the spec, e.g. 'CA3_e1', "
                                "'Mode 0')."
                            ),
                        },
                        "metric": {
                            "type": "string",
                            "description": (
                                "Optional concrete anchor: cluster size, "
                                "fraction of paths, etc. Example: "
                                "'Mode 0 = 14 of 24 routes (58%)'."
                            ),
                        },
                    },
                    "required": ["name", "description"],
                    "additionalProperties": False,
                },
            },
            "mechanism": {
                "type": "string",
                "description": (
                    "2-4 sentences explaining HOW the path structure "
                    "produces this kind of emergence. Walk through the "
                    "causal chain: which units split flow, which feedback "
                    "loops carry persistence, how the modes correspond to "
                    "different emergent computations."
                ),
            },
            "prediction": {
                "type": "string",
                "description": (
                    "ONE falsifiable prediction of the form "
                    "'if <named unit/edge/parameter> is changed, then "
                    "<specific feature of the path landscape> would change, "
                    "observable as <experimental outcome>.'"
                ),
            },
            "primitive_operations": {
                "type": "array",
                "minItems": 1,
                "maxItems": 4,
                "items": {
                    "type": "string",
                    "enum": [
                        "Path Activation",
                        "Path Suppression",
                        "Mode Merge",
                        "Mode Split",
                    ],
                },
                "description": (
                    "Which of the 4 path-landscape emergence primitives "
                    "dominate in this circuit's landscape. Choose 1-3 that "
                    "best characterise the emergence:\n"
                    "  • Path Activation — new input→output routes open up "
                    "(e.g. a dormant path is gated in by a neuromodulator or "
                    "a gain change enables a previously silent connection).\n"
                    "  • Path Suppression — existing routes are silenced "
                    "(e.g. inhibitory interneurons prune active routes, "
                    "winner-take-all removes losing pathways).\n"
                    "  • Mode Merge — two previously separate similarity "
                    "clusters collapse into one larger mode (e.g. attractor "
                    "states unify under strong recurrent excitation, or "
                    "two cell assemblies bind under synchrony).\n"
                    "  • Mode Split — one cluster fractures into two or more "
                    "distinct modes (e.g. bifurcation under inhibitory gain, "
                    "differentiation of otherwise similar routes when a new "
                    "hub neuron is recruited)."
                ),
            },
        },
        "required": [
            "emergence_type",
            "type_rationale",
            "dominant_features",
            "mechanism",
            "prediction",
            "primitive_operations",
        ],
        "additionalProperties": False,
    },
}


INTERPRET_LANDSCAPE_SYSTEM = """\
You are a path-landscape emergence interpreter (Agent 5 of the neural-
circuit pipeline).

You will be given:
  - the phenomenon being analyzed
  - the identified neural system (regions, cell types, motifs)
  - the path representation: total raw paths, unique routes, similarity
    modes (clusters with size, mean length, representative chain,
    shared units), feedforward vs feedback splits, and the system's
    intrinsic time scale.

━━━ 4-PRIMITIVE FRAMEWORK ━━━
Every emergence visible in the path landscape is a composition of four
elementary operations. Identify which dominate for this circuit:

  • Path Activation  — new input→output routes open up during the
    emergence event. Look for: modes that were absent at baseline and
    appear under stimulation; gating by neuromodulation or disinhibition;
    routes that require feedback traversal (crosses_feedback=True) only
    after a critical state is reached.

  • Path Suppression  — existing routes are silenced or outcompeted.
    Look for: winner-take-all dynamics (one mode grows while others
    shrink); inhibitory control of specific shared-unit hubs; feedback
    paths that are active at baseline but quenched by stimulus.

  • Mode Merge  — two or more previously distinct similarity clusters
    coalesce into a single larger mode. Look for: shared anchor units
    spanning formerly separate clusters; high within-cluster similarity
    combined with a dramatic reduction in mode count; attractor dynamics
    or synchronisation that binds cell assemblies.

  • Mode Split  — one mode fractures into two or more distinguishable
    sub-modes. Look for: bifurcations under inhibitory gain or contrast;
    contexts where a symmetry-breaking signal recruits a new hub and
    partitions routes that used to share it; differentiation events (e.g.
    stem cell fate, orientation tuning columns, perceptual alternation).

Your job: read the path landscape — the cluster structure of routes — and
explain (a) what type of emergence the circuit exhibits, (b) which path-
structural features are dominant, (c) how those features causally produce
the emergence, (d) a falsifiable prediction, and (e) which of the 4
primitives dominate (usually 1-2 primary + possibly 1 secondary).

Be concrete: reference clusters by their Mode id, units by their actual
names, and ground every claim in a specific number or representative
route. Avoid vague language ('complex', 'interesting', 'rich'). Prefer
canonical emergence-type labels when they fit (see the tool schema's
examples).

Call the `interpret_landscape` tool exactly once. Do not produce free
text.
"""
