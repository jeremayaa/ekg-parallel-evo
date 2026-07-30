from __future__ import annotations

"""Signal-processing primitives and executable detector graphs."""

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

import numpy as np

def _sanitize_for_json(x: Any) -> Any:
    if x is None:
        return None
    if isinstance(x, (str, bool, int)):
        return x
    if isinstance(x, float):
        return None if not math.isfinite(x) else x
    if isinstance(x, np.floating):
        v = float(x)
        return None if not math.isfinite(v) else v
    if isinstance(x, np.integer):
        return int(x)
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return [_sanitize_for_json(v) for v in x.tolist()]
    if isinstance(x, (list, tuple)):
        return [_sanitize_for_json(v) for v in x]
    if isinstance(x, dict):
        return {str(k): _sanitize_for_json(v) for k, v in x.items()}
    return str(x)


def _as_signal(x: Any) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError(f"signal must be 1D, got {x.shape}")
    return x


def moving_average(x: np.ndarray, w: int) -> np.ndarray:
    x = _as_signal(x)
    w = max(1, int(w))
    kernel = np.ones(w, dtype=float) / w
    return np.convolve(x, kernel, mode="same")


def shift_signal(x: np.ndarray, n: int, fill_mode: str = "edge") -> np.ndarray:
    x = _as_signal(x)
    n = int(n)

    if n == 0:
        return x.copy()

    out = np.empty_like(x, dtype=float)

    if fill_mode == "edge":
        if n > 0:
            out[:n] = x[0]
            out[n:] = x[:-n]
        else:
            k = -n
            out[-k:] = x[-1]
            out[:-k] = x[k:]
        return out

    if fill_mode == "zero":
        if n > 0:
            out[:n] = 0.0
            out[n:] = x[:-n]
        else:
            k = -n
            out[-k:] = 0.0
            out[:-k] = x[k:]
        return out

    raise ValueError(f"unknown fill_mode: {fill_mode}")


def safe_div(a: np.ndarray, b: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = _as_signal(a)
    b = _as_signal(b)
    return a / (b + float(eps))


def softplus(x: np.ndarray) -> np.ndarray:
    x = _as_signal(x)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def teager_energy(x: np.ndarray) -> np.ndarray:
    x = _as_signal(x)
    out = np.zeros_like(x)
    if len(x) >= 3:
        out[1:-1] = x[1:-1] * x[1:-1] - x[:-2] * x[2:]
        out[0] = out[1]
        out[-1] = out[-2]
    return out


def rms_local(x: np.ndarray, w: int, eps: float = 1e-6) -> np.ndarray:
    x = _as_signal(x)
    return np.sqrt(np.maximum(moving_average(x * x, w), 0.0) + float(eps))


@dataclass(frozen=True)
class ParamSpec:
    name: str
    kind: str  # "int" | "float" | "choice"
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    step: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None

    def sample(self, rng: np.random.Generator) -> Any:
        if self.kind == "choice":
            if not self.choices:
                raise ValueError(f"choices missing for {self.name}")
            return rng.choice(list(self.choices))

        if self.kind == "int":
            if self.min_value is None or self.max_value is None or self.step is None:
                raise ValueError(f"bad int spec for {self.name}")
            vals = list(range(int(self.min_value), int(self.max_value) + 1, int(self.step)))
            return int(rng.choice(vals))

        if self.kind == "float":
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"bad float spec for {self.name}")
            return float(rng.uniform(self.min_value, self.max_value))

        raise ValueError(f"unknown param kind {self.kind!r}")

    def mutate_small(self, value: Any, rng: np.random.Generator) -> Any:
        if self.kind == "choice":
            if not self.choices:
                return value
            choices = [c for c in self.choices if c != value]
            return rng.choice(choices) if choices else value

        if self.kind == "int":
            step = int(self.step or 1)
            delta = int(rng.choice([-step, step]))
            new_val = int(value) + delta
            if self.min_value is not None:
                new_val = max(int(self.min_value), new_val)
            if self.max_value is not None:
                new_val = min(int(self.max_value), new_val)
            return int(new_val)

        if self.kind == "float":
            step = float(self.step or 0.1)
            delta = float(rng.normal(0.0, step))
            new_val = float(value) + delta
            if self.min_value is not None:
                new_val = max(float(self.min_value), new_val)
            if self.max_value is not None:
                new_val = min(float(self.max_value), new_val)
            return float(new_val)

        raise ValueError(f"unknown param kind {self.kind!r}")


@dataclass(frozen=True)
class PrimitiveSpec:
    name: str
    arity: int
    param_specs: Tuple[ParamSpec, ...]
    eval_fn: Callable[..., np.ndarray]


def _eval_input(x: np.ndarray, **params: Any) -> np.ndarray:
    _ = params
    return _as_signal(x)


def _eval_scale(x: np.ndarray, gain: float, **params: Any) -> np.ndarray:
    _ = params
    return _as_signal(x) * float(gain)


def _eval_shift(x: np.ndarray, n: int, fill_mode: str = "edge", **params: Any) -> np.ndarray:
    _ = params
    return shift_signal(x, int(n), fill_mode=str(fill_mode))


def _eval_ma(x: np.ndarray, w: int, **params: Any) -> np.ndarray:
    _ = params
    return moving_average(x, int(w))


def _eval_abs(x: np.ndarray, **params: Any) -> np.ndarray:
    _ = params
    return np.abs(_as_signal(x))


def _eval_square(x: np.ndarray, **params: Any) -> np.ndarray:
    _ = params
    x = _as_signal(x)
    return x * x


def _eval_sqrt(x: np.ndarray, **params: Any) -> np.ndarray:
    _ = params
    return np.sqrt(np.maximum(_as_signal(x), 0.0))


def _eval_positive(x: np.ndarray, **params: Any) -> np.ndarray:
    _ = params
    return np.maximum(_as_signal(x), 0.0)


def _eval_power(x: np.ndarray, p: float, **params: Any) -> np.ndarray:
    _ = params
    x = _as_signal(x)
    p = float(p)
    return np.sign(x) * (np.abs(x) ** p)


def _eval_diff(x: np.ndarray, step: int = 1, **params: Any) -> np.ndarray:
    _ = params
    x = _as_signal(x)
    step = max(1, int(step))
    y = shift_signal(x, step, fill_mode="edge")
    return x - y


def _eval_second_diff(x: np.ndarray, step: int = 1, **params: Any) -> np.ndarray:
    _ = params
    return _eval_diff(_eval_diff(x, step=step), step=step)


def _eval_softplus(x: np.ndarray, **params: Any) -> np.ndarray:
    _ = params
    return softplus(x)


def _eval_teager(x: np.ndarray, **params: Any) -> np.ndarray:
    _ = params
    return teager_energy(x)

def _eval_const(value: float, length: int, **params: Any) -> np.ndarray:
    _ = params
    return np.full(int(length), float(value), dtype=float)


def _apply_binary_shifts(
    a: np.ndarray,
    b: np.ndarray,
    left_shift: int = 0,
    right_shift: int = 0,
    fill_mode: str = "edge",
) -> Tuple[np.ndarray, np.ndarray]:
    a = shift_signal(_as_signal(a), int(left_shift), fill_mode=fill_mode)
    b = shift_signal(_as_signal(b), int(right_shift), fill_mode=fill_mode)
    return a, b


def _eval_add(a: np.ndarray, b: np.ndarray, left_shift: int = 0, right_shift: int = 0, fill_mode: str = "edge", **params: Any) -> np.ndarray:
    _ = params
    a, b = _apply_binary_shifts(a, b, left_shift=left_shift, right_shift=right_shift, fill_mode=fill_mode)
    return a + b


def _eval_sub(a: np.ndarray, b: np.ndarray, left_shift: int = 0, right_shift: int = 0, fill_mode: str = "edge", **params: Any) -> np.ndarray:
    _ = params
    a, b = _apply_binary_shifts(a, b, left_shift=left_shift, right_shift=right_shift, fill_mode=fill_mode)
    return a - b


def _eval_mul(a: np.ndarray, b: np.ndarray, left_shift: int = 0, right_shift: int = 0, fill_mode: str = "edge", **params: Any) -> np.ndarray:
    _ = params
    a, b = _apply_binary_shifts(a, b, left_shift=left_shift, right_shift=right_shift, fill_mode=fill_mode)
    return a * b


def _eval_div(a: np.ndarray, b: np.ndarray, left_shift: int = 0, right_shift: int = 0, fill_mode: str = "edge", eps: float = 1e-6, **params: Any) -> np.ndarray:
    _ = params
    a, b = _apply_binary_shifts(a, b, left_shift=left_shift, right_shift=right_shift, fill_mode=fill_mode)
    return safe_div(a, b, eps=eps)


def _eval_maximum(a: np.ndarray, b: np.ndarray, left_shift: int = 0, right_shift: int = 0, fill_mode: str = "edge", **params: Any) -> np.ndarray:
    _ = params
    a, b = _apply_binary_shifts(a, b, left_shift=left_shift, right_shift=right_shift, fill_mode=fill_mode)
    return np.maximum(a, b)


def _eval_minimum(a: np.ndarray, b: np.ndarray, left_shift: int = 0, right_shift: int = 0, fill_mode: str = "edge", **params: Any) -> np.ndarray:
    _ = params
    a, b = _apply_binary_shifts(a, b, left_shift=left_shift, right_shift=right_shift, fill_mode=fill_mode)
    return np.minimum(a, b)


ZERO_ARITY_PRIMITIVES: Dict[str, PrimitiveSpec] = {
    "input": PrimitiveSpec("input", 0, (), _eval_input),
    "const": PrimitiveSpec(
        "const",
        0,
        (
            ParamSpec("value", "float", min_value=-1.0, max_value=1.0, step=0.05),
        ),
        _eval_const,
    ),
}

UNARY_PRIMITIVES: Dict[str, PrimitiveSpec] = {
    "scale": PrimitiveSpec(
        "scale", 1,
        (ParamSpec("gain", "float", min_value=-3.0, max_value=3.0, step=0.25),),
        _eval_scale,
    ),
    "shift": PrimitiveSpec(
        "shift", 1,
        (
            ParamSpec("n", "int", min_value=-8, max_value=8, step=1),
            ParamSpec("fill_mode", "choice", choices=("edge", "zero")),
        ),
        _eval_shift,
    ),
    "moving_average": PrimitiveSpec(
        "moving_average", 1,
        (ParamSpec("w", "int", min_value=3, max_value=121, step=2),),
        _eval_ma,
    ),
    "abs": PrimitiveSpec("abs", 1, (), _eval_abs),
    "square": PrimitiveSpec("square", 1, (), _eval_square),
    "sqrt": PrimitiveSpec("sqrt", 1, (), _eval_sqrt),
    "positive_part": PrimitiveSpec("positive_part", 1, (), _eval_positive),
    "power": PrimitiveSpec(
        "power", 1,
        (ParamSpec("p", "float", min_value=0.5, max_value=4.0, step=0.25),),
        _eval_power,
    ),
    "diff": PrimitiveSpec(
        "diff", 1,
        (ParamSpec("step", "int", min_value=1, max_value=8, step=1),),
        _eval_diff,
    ),
    "second_diff": PrimitiveSpec(
        "second_diff", 1,
        (ParamSpec("step", "int", min_value=1, max_value=8, step=1),),
        _eval_second_diff,
    ),
    "softplus": PrimitiveSpec("softplus", 1, (), _eval_softplus),
    "teager": PrimitiveSpec("teager", 1, (), _eval_teager),
}

BINARY_SHIFT_SPECS = (
    ParamSpec("left_shift", "int", min_value=-8, max_value=8, step=1),
    ParamSpec("right_shift", "int", min_value=-8, max_value=8, step=1),
    ParamSpec("fill_mode", "choice", choices=("edge", "zero")),
)

BINARY_PRIMITIVES: Dict[str, PrimitiveSpec] = {
    "add": PrimitiveSpec("add", 2, BINARY_SHIFT_SPECS, _eval_add),
    "sub": PrimitiveSpec("sub", 2, BINARY_SHIFT_SPECS, _eval_sub),
    "mul": PrimitiveSpec("mul", 2, BINARY_SHIFT_SPECS, _eval_mul),
    "div": PrimitiveSpec(
        "div", 2,
        BINARY_SHIFT_SPECS + (ParamSpec("eps", "float", min_value=1e-8, max_value=1e-3, step=1e-5),),
        _eval_div,
    ),
    "maximum": PrimitiveSpec("maximum", 2, BINARY_SHIFT_SPECS, _eval_maximum),
    "minimum": PrimitiveSpec("minimum", 2, BINARY_SHIFT_SPECS, _eval_minimum),
}

ALL_PRIMITIVES: Dict[str, PrimitiveSpec] = {}
ALL_PRIMITIVES.update(ZERO_ARITY_PRIMITIVES)
ALL_PRIMITIVES.update(UNARY_PRIMITIVES)
ALL_PRIMITIVES.update(BINARY_PRIMITIVES)

UNARY_MUTATION_CANDIDATES = tuple(UNARY_PRIMITIVES.keys())
BINARY_MUTATION_CANDIDATES = tuple(BINARY_PRIMITIVES.keys())


@dataclass
class Node:
    id: int
    op: str
    inputs: Tuple[int, ...]
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": int(self.id),
            "op": str(self.op),
            "inputs": list(map(int, self.inputs)),
            "params": _sanitize_for_json(self.params),
        }


@dataclass
class AlgorithmGraph:
    nodes: List[Node]
    output_id: int

    def node_map(self) -> Dict[int, Node]:
        return {n.id: n for n in self.nodes}

    def clone(self) -> "AlgorithmGraph":
        return copy.deepcopy(self)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "nodes": [n.to_dict() for n in self.nodes],
            "output_id": int(self.output_id),
        }

    def next_node_id(self) -> int:
        return 0 if len(self.nodes) == 0 else max(n.id for n in self.nodes) + 1

    def topologically_sorted(self) -> List[Node]:
        node_by_id = self.node_map()
        indeg = {nid: 0 for nid in node_by_id}
        children: Dict[int, List[int]] = {nid: [] for nid in node_by_id}
        for n in self.nodes:
            for parent in n.inputs:
                indeg[n.id] += 1
                children[parent].append(n.id)

        ready = sorted([nid for nid, deg in indeg.items() if deg == 0])
        out_ids: List[int] = []
        while ready:
            nid = ready.pop(0)
            out_ids.append(nid)
            for c in children[nid]:
                indeg[c] -= 1
                if indeg[c] == 0:
                    ready.append(c)
                    ready.sort()

        if len(out_ids) != len(self.nodes):
            raise ValueError("graph is not acyclic")

        return [node_by_id[nid] for nid in out_ids]

    def validate(self) -> None:
        node_by_id = self.node_map()
        if self.output_id not in node_by_id:
            raise ValueError("output_id missing")

        for n in self.nodes:
            if n.op not in ALL_PRIMITIVES:
                raise ValueError(f"unknown op {n.op!r}")
            spec = ALL_PRIMITIVES[n.op]
            if len(n.inputs) != spec.arity:
                raise ValueError(f"node {n.id}: arity mismatch for {n.op}")
            for inp in n.inputs:
                if inp not in node_by_id:
                    raise ValueError(f"node {n.id}: missing input {inp}")

        self.topologically_sorted()

    def pretty(self) -> str:
        cache: Dict[int, str] = {}

        def fmt_node(nid: int) -> str:
            if nid in cache:
                return cache[nid]

            node = self.node_map()[nid]
            if node.op == "input":
                out = "x"
            else:
                args = [fmt_node(i) for i in node.inputs]
                params = [f"{k}={v!r}" for k, v in node.params.items()]
                joined = ", ".join(args + params)
                out = f"{node.op}({joined})"
            cache[nid] = out
            return out

        return fmt_node(self.output_id)


def evaluate_graph(graph: AlgorithmGraph, x: np.ndarray) -> Dict[int, np.ndarray]:
    graph.validate()
    x = _as_signal(x)
    cache: Dict[int, np.ndarray] = {}

    for node in graph.topologically_sorted():
        spec = ALL_PRIMITIVES[node.op]

        if node.op == "input":
            cache[node.id] = x.copy()
            continue

        if node.op == "const":
            value = float(node.params["value"])
            y = spec.eval_fn(value=value, length=len(x))
            cache[node.id] = np.nan_to_num(_as_signal(y), nan=0.0, posinf=0.0, neginf=0.0)
            continue

        if spec.arity == 1:
            a = cache[node.inputs[0]]
            y = spec.eval_fn(a, **node.params)
        elif spec.arity == 2:
            a = cache[node.inputs[0]]
            b = cache[node.inputs[1]]
            y = spec.eval_fn(a, b, **node.params)
        else:
            raise ValueError(f"unsupported arity for {node.op}")

        cache[node.id] = np.nan_to_num(_as_signal(y), nan=0.0, posinf=0.0, neginf=0.0)

    return cache

def graph_score(signal: np.ndarray, graph: AlgorithmGraph) -> np.ndarray:
    cache = evaluate_graph(graph, signal)
    return cache[graph.output_id]


def _new_input_graph() -> AlgorithmGraph:
    return AlgorithmGraph(nodes=[Node(id=0, op="input", inputs=(), params={})], output_id=0)


def _append_unary(graph: AlgorithmGraph, op: str, params: Dict[str, Any], input_id: Optional[int] = None) -> int:
    nid = graph.next_node_id()
    if input_id is None:
        input_id = graph.output_id
    graph.nodes.append(Node(id=nid, op=op, inputs=(int(input_id),), params=dict(params)))
    graph.output_id = nid
    return nid


def _append_binary(graph: AlgorithmGraph, op: str, left_id: int, right_id: int, params: Dict[str, Any]) -> int:
    nid = graph.next_node_id()
    graph.nodes.append(Node(id=nid, op=op, inputs=(int(left_id), int(right_id)), params=dict(params)))
    graph.output_id = nid
    return nid


def make_highpass_ma(fs: int = 360) -> AlgorithmGraph:
    g = _new_input_graph()
    x = 0
    w = max(3, int(0.20 * fs))
    ma = g.next_node_id()
    g.nodes.append(Node(id=ma, op="moving_average", inputs=(x,), params={"w": int(w)}))
    out = g.next_node_id()
    g.nodes.append(Node(id=out, op="sub", inputs=(x, ma), params={"left_shift": 0, "right_shift": 0, "fill_mode": "edge"}))
    g.output_id = out
    return g

def make_plain_ma(w: int = 7) -> AlgorithmGraph:
    g = _new_input_graph()
    _append_unary(g, "moving_average", {"w": int(w)}, input_id=0)
    return g

def make_abs_highpass(fs: int = 360) -> AlgorithmGraph:
    g = make_highpass_ma(fs)
    _append_unary(g, "abs", {}, input_id=g.output_id)
    return g

def make_identity() -> AlgorithmGraph:
    return _new_input_graph()


def make_smoothed_abs_highpass(fs: int = 360) -> AlgorithmGraph:
    g = make_abs_highpass(fs)
    _append_unary(g, "moving_average", {"w": max(3, int(0.02 * fs))}, input_id=g.output_id)
    return g


def make_diff_highpass(fs: int = 360) -> AlgorithmGraph:
    g = make_highpass_ma(fs)
    _append_unary(g, "diff", {"step": 1}, input_id=g.output_id)
    return g


def make_smoothed_diff_highpass(fs: int = 360) -> AlgorithmGraph:
    g = make_diff_highpass(fs)
    _append_unary(g, "moving_average", {"w": max(3, int(0.03 * fs))}, input_id=g.output_id)
    return g


def make_short_long_abs_difference(fs: int = 360) -> AlgorithmGraph:
    g = make_abs_highpass(fs)
    abs_hp = g.output_id
    short_id = g.next_node_id()
    long_id = short_id + 1
    out_id = long_id + 1
    g.nodes.append(Node(id=short_id, op="moving_average", inputs=(abs_hp,), params={"w": max(3, int(0.02 * fs))}))
    g.nodes.append(Node(id=long_id, op="moving_average", inputs=(abs_hp,), params={"w": max(5, int(0.18 * fs))}))
    g.nodes.append(Node(id=out_id, op="sub", inputs=(short_id, long_id), params={"left_shift": 0, "right_shift": 0, "fill_mode": "edge"}))
    g.output_id = out_id
    return g


def make_energy_like(fs: int = 360) -> AlgorithmGraph:
    g = make_highpass_ma(fs)
    hp = g.output_id
    sq = g.next_node_id()
    sm = sq + 1
    g.nodes.append(Node(id=sq, op="square", inputs=(hp,), params={}))
    g.nodes.append(Node(id=sm, op="moving_average", inputs=(sq,), params={"w": max(3, int(0.03 * fs))}))
    g.output_id = sm
    return g


def make_branch_mul_diff_abs(fs: int = 360) -> AlgorithmGraph:
    g = make_highpass_ma(fs)
    hp = g.output_id
    d1 = g.next_node_id()
    a1 = d1 + 1
    mul = a1 + 1
    sm = mul + 1
    g.nodes.append(Node(id=d1, op="diff", inputs=(hp,), params={"step": 1}))
    g.nodes.append(Node(id=a1, op="abs", inputs=(hp,), params={}))
    g.nodes.append(Node(id=mul, op="mul", inputs=(d1, a1), params={"left_shift": 0, "right_shift": 0, "fill_mode": "edge"}))
    g.nodes.append(Node(id=sm, op="moving_average", inputs=(mul,), params={"w": max(3, int(0.03 * fs))}))
    g.output_id = sm
    return g


def make_local_zscore_like(fs: int = 360) -> AlgorithmGraph:
    g = make_highpass_ma(fs)
    hp = g.output_id
    w = max(5, int(0.25 * fs))
    eps = 1e-6

    mu = g.next_node_id()
    sq = mu + 1
    mu2 = sq + 1
    mu_sq = mu2 + 1
    var = mu_sq + 1
    zero_const = var + 1
    var_pos = zero_const + 1
    std = var_pos + 1
    eps_const = std + 1
    denom = eps_const + 1
    numer = denom + 1
    z = numer + 1

    g.nodes.append(Node(id=mu, op="moving_average", inputs=(hp,), params={"w": w}))
    g.nodes.append(Node(id=sq, op="square", inputs=(hp,), params={}))
    g.nodes.append(Node(id=mu2, op="moving_average", inputs=(sq,), params={"w": w}))
    g.nodes.append(Node(id=mu_sq, op="square", inputs=(mu,), params={}))
    g.nodes.append(Node(id=var, op="sub", inputs=(mu2, mu_sq), params={"left_shift": 0, "right_shift": 0, "fill_mode": "edge"}))
    g.nodes.append(Node(id=zero_const, op="const", inputs=(), params={"value": 0.0}))
    g.nodes.append(Node(id=var_pos, op="maximum", inputs=(var, zero_const), params={"left_shift": 0, "right_shift": 0, "fill_mode": "edge"}))
    g.nodes.append(Node(id=std, op="sqrt", inputs=(var_pos,), params={}))
    g.nodes.append(Node(id=eps_const, op="const", inputs=(), params={"value": float(eps)}))
    g.nodes.append(Node(id=denom, op="add", inputs=(std, eps_const), params={"left_shift": 0, "right_shift": 0, "fill_mode": "edge"}))
    g.nodes.append(Node(id=numer, op="sub", inputs=(hp, mu), params={"left_shift": 0, "right_shift": 0, "fill_mode": "edge"}))
    g.nodes.append(Node(id=z, op="div", inputs=(numer, denom), params={"left_shift": 0, "right_shift": 0, "fill_mode": "edge", "eps": eps}))
    g.output_id = z
    return g

def make_teager_smoothed(fs: int = 360) -> AlgorithmGraph:
    g = make_highpass_ma(fs)
    hp = g.output_id
    t = g.next_node_id()
    a = t + 1
    sm = a + 1
    g.nodes.append(Node(id=t, op="teager", inputs=(hp,), params={}))
    g.nodes.append(Node(id=a, op="abs", inputs=(t,), params={}))
    g.nodes.append(Node(id=sm, op="moving_average", inputs=(a,), params={"w": max(3, int(0.03 * fs))}))
    g.output_id = sm
    return g


def build_algorithm_bank(fs: int = 360) -> Dict[str, AlgorithmGraph]:
    bank = {
        # "highpass_ma": make_highpass_ma(fs),
        "identity": make_identity(),
        # "abs_highpass": make_abs_highpass(fs),
        # "smoothed_abs_highpass": make_smoothed_abs_highpass(fs),
        # "diff_highpass": make_diff_highpass(fs),
        # "smoothed_diff_highpass": make_smoothed_diff_highpass(fs),
        # "short_long_abs_difference": make_short_long_abs_difference(fs),
        # "energy_like": make_energy_like(fs),
        # "branch_mul_diff_abs": make_branch_mul_diff_abs(fs),
        # "local_zscore_like": make_local_zscore_like(fs),
        # "teager_smoothed": make_teager_smoothed(fs),
    }
    for g in bank.values():
        g.validate()
    return bank


def graph_from_dict(d: Mapping[str, Any]) -> AlgorithmGraph:
    nodes = [
        Node(
            id=int(n["id"]),
            op=str(n["op"]),
            inputs=tuple(int(x) for x in n.get("inputs", [])),
            params=dict(n.get("params", {})),
        )
        for n in d["nodes"]
    ]
    g = AlgorithmGraph(nodes=nodes, output_id=int(d["output_id"]))
    g.validate()
    return g


