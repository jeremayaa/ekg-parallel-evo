from __future__ import annotations

"""Graph mutation operators and evolutionary-search configuration."""

import copy
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np

from .graph import (
    ALL_PRIMITIVES,
    BINARY_MUTATION_CANDIDATES,
    UNARY_MUTATION_CANDIDATES,
    AlgorithmGraph,
    Node,
    ParamSpec,
    PrimitiveSpec,
    build_algorithm_bank,
    make_identity,
)


@dataclass
class EvolutionConfig:
    population_size: int = 20
    n_elite: int = 5
    children_per_parent: int = 20

    max_nodes: int = 20

    p_param_mutation: float = 0.45
    p_op_replace: float = 0.12
    p_unary_insert: float = 0.15
    p_unary_delete: float = 0.10
    p_branch_binary: float = 0.18

    random_seed_jitter: bool = True

    # Staged dataset curriculum.
    n_dataset_stages: int = 3
    max_generations_per_stage: int = 100
    stage_success_f1: float = 0.95
    dataset_seed_stride: int = 10_000

    # Dataset handling.
    resample_dataset_each_generation: bool = False
    success_consecutive_generations: int = 1
    force_identity_only_seed: bool = False

    # A child is accepted only when it differs structurally from its parent
    # and from every fingerprint supplied by the search loop.
    mutation_max_attempts: int = 100

    # Recording.
    save_top_k_children_per_generation: Optional[int] = 3
    verbose: bool = True


def graph_fingerprint(graph: AlgorithmGraph) -> str:
    """Return a stable structural fingerprint for a reachable graph.

    The fingerprint is independent of the original node IDs and node-list
    order. It preserves input order, shared subgraphs, parameters, and the
    number of reachable nodes.
    """
    graph.validate()
    node_by_id = graph.node_map()

    canonical_ids: Dict[int, int] = {}
    visit_order: List[int] = []

    def visit(node_id: int) -> int:
        if node_id in canonical_ids:
            return canonical_ids[node_id]

        canonical_id = len(canonical_ids)
        canonical_ids[node_id] = canonical_id
        visit_order.append(node_id)

        for input_id in node_by_id[node_id].inputs:
            visit(input_id)

        return canonical_id

    output_id = visit(graph.output_id)

    nodes = []
    for original_id in visit_order:
        node = node_by_id[original_id]
        nodes.append(
            {
                "id": canonical_ids[original_id],
                "op": node.op,
                "inputs": [canonical_ids[input_id] for input_id in node.inputs],
                "params": node.to_dict()["params"],
            }
        )

    canonical_graph = {
        "nodes": nodes,
        "output_id": output_id,
    }
    return json.dumps(canonical_graph, sort_keys=True, separators=(",", ":"))


def _random_params_for_primitive(
    spec: PrimitiveSpec,
    rng: np.random.Generator,
) -> Dict[str, Any]:
    return {parameter.name: parameter.sample(rng) for parameter in spec.param_specs}


def _mutate_params_small(
    params: Dict[str, Any],
    param_specs: Tuple[ParamSpec, ...],
    rng: np.random.Generator,
) -> Dict[str, Any]:
    if not param_specs:
        return dict(params)

    output = dict(params)
    chosen = rng.choice(list(param_specs))
    current = output.get(chosen.name, chosen.sample(rng))
    output[chosen.name] = chosen.mutate_small(current, rng)
    return output


def _reachable_node_ids(graph: AlgorithmGraph) -> List[int]:
    node_by_id = graph.node_map()
    seen: Set[int] = set()
    stack = [graph.output_id]

    while stack:
        node_id = stack.pop()
        if node_id in seen:
            continue
        seen.add(node_id)
        stack.extend(node_by_id[node_id].inputs)

    return sorted(seen)


def _used_by(graph: AlgorithmGraph) -> Dict[int, List[int]]:
    users: Dict[int, List[int]] = {node.id: [] for node in graph.nodes}
    for node in graph.nodes:
        for input_id in node.inputs:
            users[input_id].append(node.id)
    return users


def _prune_unreachable(graph: AlgorithmGraph) -> AlgorithmGraph:
    reachable = set(_reachable_node_ids(graph))
    nodes = [copy.deepcopy(node) for node in graph.nodes if node.id in reachable]
    result = AlgorithmGraph(nodes=nodes, output_id=graph.output_id)
    result.validate()
    return result


def mutate_parameter(
    graph: AlgorithmGraph,
    rng: np.random.Generator,
) -> Tuple[AlgorithmGraph, str]:
    child = graph.clone()
    candidates = [
        node
        for node in child.nodes
        if node.op != "input" and ALL_PRIMITIVES[node.op].param_specs
    ]
    if not candidates:
        return child, "param_noop"

    node = rng.choice(candidates)
    spec = ALL_PRIMITIVES[node.op]
    node.params = _mutate_params_small(node.params, spec.param_specs, rng)
    return child, "param_mutation"


def mutate_operation_replace(
    graph: AlgorithmGraph,
    rng: np.random.Generator,
) -> Tuple[AlgorithmGraph, str]:
    child = graph.clone()
    candidates = [node for node in child.nodes if len(node.inputs) in {1, 2}]
    if not candidates:
        return child, "op_replace_noop"

    node = rng.choice(candidates)

    if len(node.inputs) == 1:
        operations = [
            operation
            for operation in UNARY_MUTATION_CANDIDATES
            if operation != node.op
        ]
        if not operations:
            return child, "op_replace_noop"

        new_operation = str(rng.choice(operations))
        node.op = new_operation
        node.params = _random_params_for_primitive(
            ALL_PRIMITIVES[new_operation],
            rng,
        )
        return child, "op_replace_unary"

    operations = [
        operation
        for operation in BINARY_MUTATION_CANDIDATES
        if operation != node.op
    ]
    if not operations:
        return child, "op_replace_noop"

    new_operation = str(rng.choice(operations))
    node.op = new_operation
    node.params = _random_params_for_primitive(
        ALL_PRIMITIVES[new_operation],
        rng,
    )
    return child, "op_replace_binary"


def mutate_insert_unary(
    graph: AlgorithmGraph,
    rng: np.random.Generator,
) -> Tuple[AlgorithmGraph, str]:
    """Insert a unary operation on an existing edge or after the output."""
    child = graph.clone()

    # None means inserting after the current output. Otherwise the tuple is
    # (consumer_node_id, consumer_input_index).
    insertion_points: List[Optional[Tuple[int, int]]] = [None]
    for node in child.nodes:
        insertion_points.extend(
            (node.id, input_index)
            for input_index in range(len(node.inputs))
        )

    insertion_point = insertion_points[int(rng.integers(0, len(insertion_points)))]
    operation = str(rng.choice(UNARY_MUTATION_CANDIDATES))
    spec = ALL_PRIMITIVES[operation]
    new_id = child.next_node_id()

    if insertion_point is None:
        parent_id = child.output_id
        child.nodes.append(
            Node(
                id=new_id,
                op=operation,
                inputs=(parent_id,),
                params=_random_params_for_primitive(spec, rng),
            )
        )
        child.output_id = new_id
    else:
        consumer_id, input_index = insertion_point
        node_by_id = child.node_map()
        consumer = node_by_id[consumer_id]
        parent_id = consumer.inputs[input_index]

        child.nodes.append(
            Node(
                id=new_id,
                op=operation,
                inputs=(parent_id,),
                params=_random_params_for_primitive(spec, rng),
            )
        )

        new_inputs = list(consumer.inputs)
        new_inputs[input_index] = new_id
        consumer.inputs = tuple(new_inputs)

    child = _prune_unreachable(child)
    return child, "insert_unary"


def mutate_delete_unary(
    graph: AlgorithmGraph,
    rng: np.random.Generator,
) -> Tuple[AlgorithmGraph, str]:
    child = graph.clone()
    users = _used_by(child)
    candidates = [
        node
        for node in child.nodes
        if node.op != "input" and len(node.inputs) == 1
    ]
    if not candidates:
        return child, "delete_unary_noop"

    node = rng.choice(candidates)
    parent_id = node.inputs[0]
    node_by_id = child.node_map()

    for user_id in users[node.id]:
        user = node_by_id[user_id]
        user.inputs = tuple(
            parent_id if input_id == node.id else input_id
            for input_id in user.inputs
        )

    if child.output_id == node.id:
        child.output_id = parent_id

    child = _prune_unreachable(child)
    return child, "delete_unary"


def _ancestor_ids(graph: AlgorithmGraph, node_id: int) -> List[int]:
    node_by_id = graph.node_map()
    seen: Set[int] = set()
    stack = [node_id]

    while stack:
        current_id = stack.pop()
        if current_id in seen:
            continue
        seen.add(current_id)
        stack.extend(node_by_id[current_id].inputs)

    return sorted(seen)


def _make_random_const_node(
    node_id: int,
    rng: np.random.Generator,
) -> Node:
    spec = ALL_PRIMITIVES["const"]
    return Node(
        id=int(node_id),
        op="const",
        inputs=(),
        params=_random_params_for_primitive(spec, rng),
    )


def mutate_branch_binary(
    graph: AlgorithmGraph,
    rng: np.random.Generator,
) -> Tuple[AlgorithmGraph, str]:
    child = graph.clone()

    attach_candidates = [
        node
        for node in child.nodes
        if node.op != "input" and len(node.inputs) == 1
    ]
    if not attach_candidates:
        return child, "branch_binary_noop"

    attach_node = rng.choice(attach_candidates)
    attach_id = int(attach_node.id)
    attach_input = int(attach_node.inputs[0])

    # A branch may only start from an ancestor of the attachment node.
    # Excluding the attachment node prevents cycles.
    ancestor_ids = [
        node_id
        for node_id in _ancestor_ids(child, attach_id)
        if node_id != attach_id
    ]

    use_const_branch = rng.random() < 0.30

    if use_const_branch:
        branch_id = child.next_node_id()
        child.nodes.append(_make_random_const_node(branch_id, rng))
    else:
        if not ancestor_ids:
            return child, "branch_binary_noop"

        base_id = int(rng.choice(ancestor_ids))
        unary_operation = str(rng.choice(UNARY_MUTATION_CANDIDATES))
        unary_spec = ALL_PRIMITIVES[unary_operation]
        branch_id = child.next_node_id()
        child.nodes.append(
            Node(
                id=branch_id,
                op=unary_operation,
                inputs=(base_id,),
                params=_random_params_for_primitive(unary_spec, rng),
            )
        )

    binary_operation = str(rng.choice(BINARY_MUTATION_CANDIDATES))
    binary_spec = ALL_PRIMITIVES[binary_operation]

    node_by_id = child.node_map()
    node_by_id[attach_id].op = binary_operation
    node_by_id[attach_id].inputs = (attach_input, branch_id)
    node_by_id[attach_id].params = _random_params_for_primitive(
        binary_spec,
        rng,
    )

    child = _prune_unreachable(child)
    return child, "branch_binary"


def enforce_graph_limits(
    graph: AlgorithmGraph,
    evo_cfg: EvolutionConfig,
) -> AlgorithmGraph:
    """Validate a graph and reject it when it exceeds ``max_nodes``."""
    child = _prune_unreachable(graph)
    if len(child.nodes) > evo_cfg.max_nodes:
        raise ValueError(
            f"graph has {len(child.nodes)} nodes; max_nodes={evo_cfg.max_nodes}"
        )
    child.validate()
    return child


def _applicable_mutations(
    graph: AlgorithmGraph,
    evo_cfg: EvolutionConfig,
) -> List[Tuple[str, float]]:
    if len(graph.nodes) > evo_cfg.max_nodes:
        raise ValueError(
            f"parent graph has {len(graph.nodes)} nodes; "
            f"max_nodes={evo_cfg.max_nodes}"
        )

    mutations: List[Tuple[str, float]] = []

    has_parameters = any(
        node.op != "input" and ALL_PRIMITIVES[node.op].param_specs
        for node in graph.nodes
    )
    has_replaceable_operation = any(
        len(node.inputs) in {1, 2}
        for node in graph.nodes
    )
    has_unary_node = any(
        node.op != "input" and len(node.inputs) == 1
        for node in graph.nodes
    )
    has_node_space = len(graph.nodes) < evo_cfg.max_nodes

    if has_parameters and evo_cfg.p_param_mutation > 0:
        mutations.append(("parameter", evo_cfg.p_param_mutation))
    if has_replaceable_operation and evo_cfg.p_op_replace > 0:
        mutations.append(("replace", evo_cfg.p_op_replace))
    if has_node_space and evo_cfg.p_unary_insert > 0:
        mutations.append(("insert", evo_cfg.p_unary_insert))
    if has_unary_node and evo_cfg.p_unary_delete > 0:
        mutations.append(("delete", evo_cfg.p_unary_delete))
    if has_node_space and has_unary_node and evo_cfg.p_branch_binary > 0:
        mutations.append(("branch", evo_cfg.p_branch_binary))

    return mutations


def _mutate_graph_once(
    graph: AlgorithmGraph,
    rng: np.random.Generator,
    evo_cfg: EvolutionConfig,
) -> Tuple[AlgorithmGraph, str]:
    applicable = _applicable_mutations(graph, evo_cfg)
    if not applicable:
        raise RuntimeError(
            "No mutation is applicable. Increase max_nodes or enable an "
            "applicable mutation probability."
        )

    names = [name for name, _ in applicable]
    probabilities = np.asarray([weight for _, weight in applicable], dtype=float)
    probabilities /= probabilities.sum()
    mutation_name = names[int(rng.choice(len(names), p=probabilities))]

    mutation_functions = {
        "parameter": mutate_parameter,
        "replace": mutate_operation_replace,
        "insert": mutate_insert_unary,
        "delete": mutate_delete_unary,
        "branch": mutate_branch_binary,
    }

    child, mutation_type = mutation_functions[mutation_name](graph, rng)
    child = enforce_graph_limits(child, evo_cfg)
    return child, mutation_type


def mutate_graph(
    graph: AlgorithmGraph,
    rng: np.random.Generator,
    evo_cfg: EvolutionConfig,
    forbidden_fingerprints: Optional[Set[str]] = None,
) -> Tuple[AlgorithmGraph, str]:
    """Create a genuinely changed and previously unseen child graph.

    Accepted children:
    - differ structurally from the parent;
    - do not occur in ``forbidden_fingerprints``;
    - contain no more than ``max_nodes`` reachable nodes.

    If the configured mutation probabilities repeatedly create an unchanged or
    forbidden graph, the function retries. It raises ``RuntimeError`` instead
    of returning a no-op child.
    """
    parent_fingerprint = graph_fingerprint(graph)
    forbidden = set(forbidden_fingerprints or ())
    forbidden.add(parent_fingerprint)

    attempts = max(1, int(evo_cfg.mutation_max_attempts))

    for _ in range(attempts):
        child, mutation_type = _mutate_graph_once(graph, rng, evo_cfg)
        child_fingerprint = graph_fingerprint(child)

        if child_fingerprint in forbidden:
            continue

        return child, mutation_type

    raise RuntimeError(
        "Could not generate a unique changed child after "
        f"{attempts} mutation attempts"
    )


def _jitter_seed_graph(
    graph: AlgorithmGraph,
    rng: np.random.Generator,
    n_steps: int = 1,
) -> AlgorithmGraph:
    result = graph.clone()
    for _ in range(max(0, int(n_steps))):
        candidate, _ = mutate_parameter(result, rng)
        if graph_fingerprint(candidate) != graph_fingerprint(result):
            result = candidate
    result.validate()
    return result


def sample_seed_population(
    fs: int,
    population_size: int,
    rng: np.random.Generator,
    jitter: bool = True,
) -> List[AlgorithmGraph]:
    bank = build_algorithm_bank(fs=fs)
    names = list(bank.keys())
    if not names:
        raise ValueError("build_algorithm_bank() returned no seed graphs")

    chosen = rng.choice(
        names,
        size=int(population_size),
        replace=len(names) < population_size,
    )

    population: List[AlgorithmGraph] = []
    for name in chosen:
        graph = bank[str(name)].clone()
        if jitter:
            graph = _jitter_seed_graph(
                graph,
                rng,
                n_steps=int(rng.integers(0, 3)),
            )
        population.append(graph)

    return population


def sample_identity_population(population_size: int) -> List[AlgorithmGraph]:
    return [make_identity() for _ in range(int(population_size))]


def sample_population_from_graph(
    graph: AlgorithmGraph,
    population_size: int,
    rng: np.random.Generator,
    jitter: bool = True,
) -> List[AlgorithmGraph]:
    population: List[AlgorithmGraph] = []
    for _ in range(int(population_size)):
        candidate = graph.clone()
        if jitter:
            candidate = _jitter_seed_graph(
                candidate,
                rng,
                n_steps=int(rng.integers(0, 3)),
            )
        population.append(candidate)
    return population
