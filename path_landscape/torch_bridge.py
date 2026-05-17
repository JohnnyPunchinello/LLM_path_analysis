"""Bridge between PyTorch models and the path-landscape `System`.

Two kinds of conversion are supported:

  - **MLPs** (any `nn.Sequential` whose linear blocks are `nn.Linear`):
    `mlp_to_system` builds one neuron per scalar unit and one edge per
    nonzero weight. Pointwise nonlinearities (ReLU, GELU, ...) do not add
    edges; they live on the units themselves.

  - **Recurrent cells** (`nn.RNNCell`, or any cell with `weight_ih`/`weight_hh`):
    `rnncell_to_system` builds an input unit per input dim, a hidden unit per
    hidden dim, with feedback (recurrent) edges among hidden units. Calling
    `system.unroll(T)` then materializes the time-unrolled DAG.

Edge re-weighting:

  - `reweight_by_weight(sys, model)`         : |W|
  - `reweight_by_grad(sys, model, x, ...)`   : |dL/dW| at input batch x
  - `reweight_by_pathways(sys, model, x)`    : |W * a_in| -- the actual
                                                magnitude of signal flowing
                                                along each edge for input x.

These give *functional* edge weights instead of structural ones; the
PathLandscape then reflects the trained network's actual computation
under the chosen input distribution.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

try:
    import torch
    import torch.nn as nn
except Exception as exc:  # pragma: no cover
    raise ImportError(
        "torch_bridge requires PyTorch. Install with `pip install torch`."
    ) from exc

from .system import System


# ---------------------------------------------------------------- MLP bridge

def _linear_layers(model: nn.Module) -> list[nn.Linear]:
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def mlp_to_system(
    model: nn.Module,
    input_size: Optional[int] = None,
    prefix: str = "L",
) -> System:
    """Build a `System` from an MLP-like `nn.Module`.

    `model` should contain `nn.Linear` layers in forward order; pointwise
    activations between them are ignored at the structural level.

    Each scalar neuron becomes a unit named `f"{prefix}{layer}_n{k}"`. Each
    nonzero weight `W[j, i]` of layer `l` becomes a directed edge
    `L{l-1}_n{i} -> L{l}_n{j}` with weight `|W[j, i]|`.
    """
    layers = _linear_layers(model)
    if not layers:
        raise ValueError("no nn.Linear modules found in model")
    if input_size is None:
        input_size = layers[0].in_features

    sys = System()
    # Input layer (L0).
    for k in range(input_size):
        sys.add_unit(f"{prefix}0_n{k}", scale=0)
    sys.set_input(*[f"{prefix}0_n{k}" for k in range(input_size)])

    prev = input_size
    for li, layer in enumerate(layers, start=1):
        out = layer.out_features
        for k in range(out):
            sys.add_unit(f"{prefix}{li}_n{k}", scale=0)
        W = layer.weight.detach().cpu().abs().numpy()  # (out, in)
        for j in range(out):
            for i in range(prev):
                w = float(W[j, i])
                if w > 0.0:
                    sys.add_edge(f"{prefix}{li - 1}_n{i}", f"{prefix}{li}_n{j}", weight=w)
        prev = out

    # Last linear layer's outputs are the system outputs.
    last = len(layers)
    sys.set_output(*[f"{prefix}{last}_n{k}" for k in range(layers[-1].out_features)])
    return sys


def _set_edge_weights_from_layers(
    sys: System, weight_per_layer: list[np.ndarray], prefix: str
) -> None:
    """Overwrite the edge weights of `sys` from a per-layer list of (out, in)
    weight magnitude arrays. Edges that don't exist in `sys` are added.
    Edges that exist but get a zero weight are removed.
    """
    # Drop existing edges (we'll rebuild).
    sys.graph.clear_edges()
    prev = weight_per_layer[0].shape[1]
    for li, W in enumerate(weight_per_layer, start=1):
        out_size, in_size = W.shape
        for j in range(out_size):
            for i in range(in_size):
                w = float(W[j, i])
                if w > 0.0:
                    sys.add_edge(
                        f"{prefix}{li - 1}_n{i}",
                        f"{prefix}{li}_n{j}",
                        weight=w,
                    )
        prev = out_size


def reweight_by_weight(sys: System, model: nn.Module, prefix: str = "L") -> None:
    """In-place: edge weight = |W|. (Same as default `mlp_to_system`.)"""
    Ws = [layer.weight.detach().cpu().abs().numpy() for layer in _linear_layers(model)]
    _set_edge_weights_from_layers(sys, Ws, prefix)


def reweight_by_grad(
    sys: System,
    model: nn.Module,
    x: torch.Tensor,
    target: Optional[torch.Tensor] = None,
    target_idx: Optional[int] = None,
    loss_fn: Optional[nn.Module] = None,
    prefix: str = "L",
) -> None:
    """In-place: edge weight = |dL/dW| evaluated on input batch `x`.

    Choose ONE of:
      - `target` and `loss_fn`     : standard supervised gradient.
      - `target_idx`               : gradient of `out[..., target_idx].sum()`.
      - neither                    : gradient of `out.sum()`.

    Picks up which weights actually carry training signal under `x`.
    """
    layers = _linear_layers(model)
    model.zero_grad()
    out = model(x)
    if target is not None:
        if loss_fn is None:
            loss_fn = nn.MSELoss()
        loss = loss_fn(out, target)
    elif target_idx is not None:
        loss = out[..., target_idx].sum()
    else:
        loss = out.sum()
    loss.backward()
    Gs = [layer.weight.grad.detach().cpu().abs().numpy() for layer in layers]
    _set_edge_weights_from_layers(sys, Gs, prefix)


def reweight_by_pathways(
    sys: System,
    model: nn.Module,
    x: torch.Tensor,
    prefix: str = "L",
    aggregate: str = "mean",
) -> None:
    """In-place: edge weight = average |W[j,i] * a_in[i]| over the batch.

    This is the actual magnitude of signal flowing along each edge under input
    distribution `x`. `aggregate` is 'mean' or 'max' across the batch.
    """
    layers = _linear_layers(model)
    activations: list[torch.Tensor] = [x]
    handles = []

    # Hook every Linear's input.
    def _hook(_mod, inputs, _output):
        activations.append(inputs[0].detach())

    for layer in layers:
        handles.append(layer.register_forward_hook(_hook))
    try:
        model.eval()
        with torch.no_grad():
            model(x)
    finally:
        for h in handles:
            h.remove()
    # activations is now [x, input_to_L1, input_to_L2, ...]; drop the leading x
    # if it duplicates the first hook capture.
    if len(activations) >= 2 and activations[1] is x:
        activations = activations[1:]
    elif len(activations) >= 2:
        activations = activations[1:]

    Ws = []
    for layer, a_in in zip(layers, activations):
        W = layer.weight.detach().cpu().abs().numpy()  # (out, in)
        a = a_in.detach().cpu().abs().numpy()           # (B, in)
        if aggregate == "mean":
            scale = a.mean(axis=0)
        elif aggregate == "max":
            scale = a.max(axis=0)
        else:
            raise ValueError(f"unknown aggregate: {aggregate}")
        Ws.append(W * scale[None, :])
    _set_edge_weights_from_layers(sys, Ws, prefix)


# ----------------------------------------------------------- Recurrent bridge

def rnncell_to_system(
    cell: nn.Module,
    in_prefix: str = "in",
    h_prefix: str = "h",
) -> System:
    """Build a System from an `nn.RNNCell` (or any cell exposing `weight_ih`
    and `weight_hh`). Hidden-to-hidden edges are recurrent (closed by `unroll`).

    Outputs are the hidden units (a downstream readout, if any, can be a
    separate `nn.Linear` and added as a second System layer manually).
    """
    if not (hasattr(cell, "weight_ih") and hasattr(cell, "weight_hh")):
        raise TypeError("expected a cell with `weight_ih` and `weight_hh` attrs")
    W_ih = cell.weight_ih.detach().cpu().abs().numpy()  # (H, I)
    W_hh = cell.weight_hh.detach().cpu().abs().numpy()  # (H, H)
    H, I = W_ih.shape

    sys = System()
    for k in range(I):
        sys.add_unit(f"{in_prefix}{k}", scale=0)
    for k in range(H):
        sys.add_unit(f"{h_prefix}{k}", scale=0)

    for j in range(H):
        for i in range(I):
            w = float(W_ih[j, i])
            if w > 0.0:
                sys.add_edge(f"{in_prefix}{i}", f"{h_prefix}{j}", weight=w)
    for j in range(H):
        for i in range(H):
            w = float(W_hh[j, i])
            if w > 0.0:
                sys.add_edge(f"{h_prefix}{i}", f"{h_prefix}{j}", weight=w, recurrent=True)

    sys.set_input(*[f"{in_prefix}{k}" for k in range(I)])
    sys.set_output(*[f"{h_prefix}{k}" for k in range(H)])
    return sys


# ---------------------------------------------------------- toy training task

def train_toy_mlp(
    hidden: list[int] = (8, 8),
    n_classes: int = 3,
    n_input: int = 2,
    n_samples: int = 600,
    epochs: int = 200,
    lr: float = 0.05,
    seed: int = 0,
):
    """Train a small MLP on a synthetic Gaussian-blob classification task and
    return `(model, X, y)`. Used by `demo_torch.py`.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    centers = rng.normal(size=(n_classes, n_input)) * 2.5
    X_np = np.zeros((n_samples, n_input), dtype=np.float32)
    y_np = np.zeros(n_samples, dtype=np.int64)
    per = n_samples // n_classes
    for c in range(n_classes):
        idx = slice(c * per, (c + 1) * per)
        X_np[idx] = rng.normal(loc=centers[c], scale=0.6, size=(per, n_input))
        y_np[idx] = c
    X = torch.from_numpy(X_np)
    y = torch.from_numpy(y_np)

    sizes = [n_input, *hidden, n_classes]
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    for _ in range(epochs):
        opt.zero_grad()
        loss = loss_fn(model(X), y)
        loss.backward()
        opt.step()
    return model, X, y
