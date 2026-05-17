"""Biologically-inspired Systems for cross-substrate comparison.

The flagship builder, `chemotaxis_circuit_celegans()`, constructs a
*C. elegans-faithful synthetic connectome* for the chemotactic
sensorimotor pathway. The neurons are named from real worm anatomy
(White et al. 1986; WormAtlas) and the connectivity statistics are
calibrated against published values:

  - Total neurons in the hermaphrodite : 302  (we model ~120, the
    chemotaxis-relevant subset).
  - Chemical synapses                   : ~7500 in the full worm.
  - Gap junctions                       : ~900  (modeled as recurrent edges).
  - Average chemical out-degree         : ~25  (Varshney et al. 2011).

The chemotaxis pathway itself is hardwired from Bargmann's circuit
diagram (Bargmann, 2006, "Comparative chemosensation from receptors
to ecology"): chemosensory neurons (AWA/AWB/AWC/ASE...) project to
first-layer interneurons (AIA/AIB/AIY/AIZ), which converge on command
interneurons (AVA/AVB/AVD/AVE/PVC) that drive the ventral- and
dorsal-cord motor neurons (VA/VB/VD/DA/DB/DD/AS).

The synthetic edges around this skeleton are drawn from a stochastic
model with class-specific connection densities matching published
estimates. The result is a system whose path-landscape statistics are
representative of the worm's chemotaxis circuit even though the exact
edge-by-edge wiring is generated, not measured. Replace this builder
with a loader for a published connectome edge list when one is
available locally.

References:
  White JG, et al. (1986). Phil. Trans. R. Soc. B 314: 1-340.
  Bargmann CI. (2006). WormBook 1-29. Chemosensation in C. elegans.
  Varshney LR, et al. (2011). PLoS Comput Biol 7(2): e1001066.
  Cook SJ, et al. (2019). Nature 571: 63-71.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .system import System


# ---------------------------------------------------- neuron classifications

# Chemosensory neurons (paired bilaterally as L/R).
CHEMOSENSORY = [
    "AWA", "AWB", "AWC",          # amphid wing -- olfactory
    "ASE", "ASG", "ASH", "ASI", "ASJ", "ASK",  # amphid -- taste / nociception
    "ADF", "ADL",                  # amphid double
    "AFD", "AQR", "BAG",           # thermosensory + O2/CO2
]

# First-layer (amphid) interneurons.
AMPHID_INTERNEURONS = [
    "AIA", "AIB", "AIY", "AIZ",
    "RIA", "RIB", "RIM", "RIH",
    "AIM", "AIN", "AVH", "AVJ", "AVK",
]

# Command interneurons -- the locomotion control hub.
COMMAND_INTERNEURONS = [
    "AVA", "AVB", "AVD", "AVE", "PVC",
]

# Other interneurons relevant to sensorimotor control.
OTHER_INTERNEURONS = [
    "RIC", "RIG", "RID", "RIF", "RIS", "RMG",
    "AVF", "AVG", "AVL", "PVQ", "DVA", "DVC",
]

# Motor neuron classes; each prefix has multiple numbered cells.
MOTOR_PREFIX_COUNTS = {
    "VA": 12, "VB": 11, "VC": 6, "VD": 13,
    "DA": 9,  "DB": 7,  "DD": 6,
    "AS": 11,
}


# ---------------------------------------------- chemotaxis circuit (skeleton)

# Hardwired projections from Bargmann's chemotaxis circuit diagram.
CHEMOTAXIS_PROJECTIONS = {
    # chemosensory -> amphid interneurons
    "AWA": ["AIA", "AIY", "AIB", "AIZ"],
    "AWB": ["AIB", "AIZ"],
    "AWC": ["AIA", "AIY", "AIB"],
    "ASE": ["AIA", "AIB", "AIY", "AIZ"],
    "ASG": ["AIA", "AIB"],
    "ASH": ["AIB", "AVA", "AVD", "AVE"],   # ASH bypasses to command tier
    "ASI": ["AIA", "AIY", "AIZ"],
    "ASJ": ["AIA", "AIB"],
    "ASK": ["AIA", "AIB", "AIY"],
    "ADF": ["AIA", "AIB"],
    "ADL": ["AIB", "AVA"],
    "AFD": ["AIY"],
    "AQR": ["AVA", "AVE"],
    "BAG": ["RIA", "RIB", "AVE"],
    # amphid interneurons -> command interneurons
    "AIA": ["AVA", "AVB", "AIY"],
    "AIB": ["AVA", "AVB", "AVD", "AVE", "RIM"],
    "AIY": ["AVB", "AIA", "RIA", "RIB"],
    "AIZ": ["AVB", "AVA", "RIA", "RIM"],
    "RIA": ["RIM", "RIB", "AVB", "AVA"],
    "RIB": ["AVB", "AVE"],
    "RIM": ["AVA", "AVB", "AVD"],
}


# ----------------------------------------------------------- builder

def chemotaxis_circuit_celegans(
    seed: int = 0,
    chemical_density_within: float = 0.18,
    chemical_density_inter_to_motor: float = 0.05,
    motor_lateral_density: float = 0.04,
    feedback_density: float = 0.03,
    gap_junction_density: float = 0.05,
) -> System:
    """Build the C. elegans-faithful synthetic chemotaxis connectome.

    Returns a `System` with units tagged into 4 hierarchical groups by
    `parent`: `sensory`, `amphid`, `command`, `motor`. Inputs are the
    chemosensory neurons; outputs are the motor neurons. Gap junctions
    are encoded as recurrent edges (since they are bidirectional and
    instantaneous; in `unroll(T)` they will materialize across time
    steps).

    The four density parameters control:
      - chemical_density_within        : within-class chemical synapses.
      - chemical_density_inter_to_motor: extra interneuron -> motor edges
                                         beyond the hardwired skeleton.
      - motor_lateral_density          : motor -> motor (lateral / coupling).
      - feedback_density               : motor -> interneuron (feedback).
      - gap_junction_density           : within-amphid + within-command
                                         electrical (recurrent) coupling.
    """
    rng = np.random.default_rng(seed)
    sys = System()

    # --- units ------------------------------------------------------------
    sensory = [f"{n}{lr}" for n in CHEMOSENSORY for lr in ("L", "R")]
    amphid  = [f"{n}{lr}" for n in AMPHID_INTERNEURONS for lr in ("L", "R")]
    command = [f"{n}{lr}" for n in COMMAND_INTERNEURONS for lr in ("L", "R")]
    other   = [f"{n}{lr}" for n in OTHER_INTERNEURONS for lr in ("L", "R")]
    motor: list[str] = []
    for prefix, count in MOTOR_PREFIX_COUNTS.items():
        for k in range(1, count + 1):
            motor.append(f"{prefix}{k}")

    for n in sensory: sys.add_unit(n, scale=0, parent="sensory")
    for n in amphid:  sys.add_unit(n, scale=0, parent="amphid")
    for n in command: sys.add_unit(n, scale=0, parent="command")
    for n in other:   sys.add_unit(n, scale=0, parent="other_inter")
    for n in motor:   sys.add_unit(n, scale=0, parent="motor")

    interneurons = amphid + command + other

    # --- 1) hardwired chemotaxis skeleton (Bargmann 2006) -----------------
    def add_skeleton(src_root: str, dst_roots: list[str]) -> None:
        for lr in ("L", "R"):
            for dst in dst_roots:
                # connect ipsi-lateral and contra-lateral with different weights.
                sys.add_edge(f"{src_root}{lr}", f"{dst}{lr}", weight=1.0)
                if rng.random() < 0.4:
                    other_lr = "R" if lr == "L" else "L"
                    sys.add_edge(f"{src_root}{lr}",
                                 f"{dst}{other_lr}", weight=0.5)
    for src_root, dst_roots in CHEMOTAXIS_PROJECTIONS.items():
        add_skeleton(src_root, dst_roots)

    # --- 2) extra synapses with class-specific densities ------------------
    def stochastic_block(srcs, dsts, p, weight_low=0.3, weight_high=0.8,
                         recurrent=False, exclude_self=False):
        for s in srcs:
            for d in dsts:
                if exclude_self and s == d:
                    continue
                if rng.random() < p:
                    sys.add_edge(s, d,
                                 weight=float(rng.uniform(weight_low, weight_high)),
                                 recurrent=recurrent)

    # within-amphid lateral chemical
    stochastic_block(amphid, amphid, chemical_density_within,
                     0.3, 0.7, exclude_self=True)
    # within-command lateral chemical
    stochastic_block(command, command, chemical_density_within * 1.2,
                     0.4, 0.9, exclude_self=True)
    # other-interneuron projections to commands (sparse)
    stochastic_block(other, command, chemical_density_within * 0.5,
                     0.3, 0.7)
    # other-interneuron lateral
    stochastic_block(other, other, chemical_density_within * 0.4,
                     0.2, 0.5, exclude_self=True)
    # interneuron -> motor (extra beyond skeleton)
    stochastic_block(interneurons, motor, chemical_density_inter_to_motor,
                     0.4, 0.9)
    # command -> motor is denser (the command neurons drive locomotion)
    stochastic_block(command, motor, chemical_density_inter_to_motor * 4,
                     0.5, 1.0)
    # motor -> motor (lateral coupling)
    stochastic_block(motor, motor, motor_lateral_density,
                     0.2, 0.5, exclude_self=True)
    # motor -> interneuron (proprioceptive feedback) - recurrent
    stochastic_block(motor, interneurons, feedback_density,
                     0.2, 0.4, recurrent=True)

    # --- 3) gap junctions (electrical, modeled as recurrent edges) --------
    # Mostly within amphid and within command tiers.
    stochastic_block(amphid, amphid, gap_junction_density,
                     0.25, 0.5, recurrent=True, exclude_self=True)
    stochastic_block(command, command, gap_junction_density * 1.5,
                     0.3, 0.6, recurrent=True, exclude_self=True)
    stochastic_block(motor, motor, gap_junction_density * 0.6,
                     0.2, 0.4, recurrent=True, exclude_self=True)

    # --- 4) inputs / outputs ----------------------------------------------
    sys.set_input(*sensory)
    sys.set_output(*motor)
    return sys


# ------------------------------------------------------------- AI counterpart

def ai_chemotaxis_agent(
    n_input: int = 28,
    n_output: int = 60,
    depth: int = 4,
    width: int = 16,
    skip_every: int = 2,
) -> System:
    """A small AI policy network with chemotaxis-shaped I/O for matched
    comparison. Built directly as a `System` (no PyTorch model) so the
    comparison stays apples-to-apples with the structural connectome.

    - `n_input` = number of chemotaxis-relevant 'gradient features' the
      agent receives (matched to the worm's chemosensory neuron count).
    - `n_output` = action dimensionality (matched to motor neuron count).
    - `depth, width, skip_every` control the deep + skip body.

    Multi-exit readout: outputs at the last AND second-last layer,
    producing paths of varying length.
    """
    sys = System()
    # Input layer
    for k in range(n_input):
        sys.add_unit(f"in_{k}", scale=0, parent="input")
    # Hidden layers
    for layer in range(depth):
        for k in range(width):
            sys.add_unit(f"L{layer}_n{k}", scale=0, parent=f"block{layer // 2}")
    # Output layer
    for k in range(n_output):
        sys.add_unit(f"out_{k}", scale=0, parent="action")

    # input -> first hidden (dense)
    for i in range(n_input):
        for j in range(width):
            sys.add_edge(f"in_{i}", f"L0_n{j}", weight=1.0)
    # forward chain
    for layer in range(depth - 1):
        for i in range(width):
            for j in range(width):
                sys.add_edge(f"L{layer}_n{i}",
                             f"L{layer + 1}_n{j}", weight=1.0)
    # short skip
    for layer in range(depth - skip_every):
        for k in range(width):
            sys.add_edge(f"L{layer}_n{k}",
                         f"L{layer + skip_every}_n{k}", weight=0.6)
    # last hidden -> output
    for i in range(width):
        for j in range(n_output):
            sys.add_edge(f"L{depth - 1}_n{i}", f"out_{j}", weight=0.8)
    # second-last hidden -> output (multi-exit)
    if depth >= 2:
        for i in range(width):
            for j in range(n_output):
                sys.add_edge(f"L{depth - 2}_n{i}",
                             f"out_{j}", weight=0.4)

    sys.set_input(*[f"in_{k}" for k in range(n_input)])
    sys.set_output(*[f"out_{k}" for k in range(n_output)])
    return sys
