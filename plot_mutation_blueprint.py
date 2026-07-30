from pathlib import Path
from typing import Any
import json

from detector_engine.graph import (
    ALL_PRIMITIVES,
    AlgorithmGraph,
    Node,
)
from graph_visualization import TrialGraphVisualizer


OUTPUT_ROOT = Path("mutation_blueprint")


# ---------------------------------------------------------------------
# Exact mutation blueprint
#
# Node IDs refer to the graph state at that particular step.

BLUEPRINT = [
    {
        "name": "parameter_mutation",
        "type": "set_parameters",
        "node_id": 1,
        "parameters": {
            "w": 80,
        },
    },
    {
        "name": "operation_replace_diff_with_abs",
        "type": "replace_operation",
        "node_id": 3,
        "primitive": "abs",
        "parameters": {},
    },
    {
        "name": "insert_square_before_abs",
        "type": "insert_unary",
        "target_node_id": 3,
        "input_index": 0,
        "primitive": "square",
        "parameters": {},
    },
    {
        "name": "add_constant_to_output",
        "type": "insert_constant_binary",
        # After inserting square, the output abs node has ID 4.
        "target_node_id": 4,
        "binary_primitive": "add",
        "binary_parameters": {
            "left_shift": 0,
            "right_shift": 0,
            "fill_mode": "edge",
        },
        "constant_value": 0.5,
        "constant_input_index": 1,
    },
    {
        "name": "branch_mul_with_moving_average",
        "type": "branch_binary",
        # The abs node remains ID 4 after constant insertion.
        # It is replaced by mul inside the outer add operation.
        "attach_node_id": 4,
        "binary_primitive": "mul",
        "binary_parameters": {
            "left_shift": 0,
            "right_shift": 0,
            "fill_mode": "edge",
        },
        "branch_source_id": 0,
        "branch_primitive": "moving_average",
        "branch_parameters": {
            "w": 18,
        },
    },
]


class MutationGraphVisualizer(TrialGraphVisualizer):
    def _title(self) -> str:
        return str(
            self.record.get(
                "mutation_type",
                "unknown_mutation",
            )
        )


# ---------------------------------------------------------------------
# Exact initial graph
#
# Equivalent to:
#
# diff(
#     sub(
#         x,
#         moving_average(x, w=72),
#         left_shift=0,
#         right_shift=0,
#         fill_mode="edge",
#     ),
#     step=1,
# )
# ---------------------------------------------------------------------

def make_initial_graph() -> AlgorithmGraph:
    graph = AlgorithmGraph(
        nodes=[
            Node(
                id=0,
                op="input",
                inputs=(),
                params={},
            ),
            Node(
                id=1,
                op="moving_average",
                inputs=(0,),
                params={
                    "w": 72,
                },
            ),
            Node(
                id=2,
                op="sub",
                inputs=(0, 1),
                params={
                    "left_shift": 0,
                    "right_shift": 0,
                    "fill_mode": "edge",
                },
            ),
            Node(
                id=3,
                op="diff",
                inputs=(2,),
                params={
                    "step": 1,
                },
            ),
        ],
        output_id=3,
    )

    graph.validate()
    return graph


# ---------------------------------------------------------------------
# Graph utilities
# ---------------------------------------------------------------------

def get_node(
    graph: AlgorithmGraph,
    node_id: int,
) -> Node:
    node = graph.node_map().get(node_id)

    if node is None:
        available_ids = sorted(
            graph.node_map()
        )

        raise ValueError(
            f"Node {node_id} does not exist. "
            f"Available node IDs: {available_ids}"
        )

    return node


def check_primitive(
    primitive: str,
    expected_arity: int | None = None,
):
    if primitive not in ALL_PRIMITIVES:
        available = sorted(ALL_PRIMITIVES)

        raise ValueError(
            f"Unknown primitive {primitive!r}. "
            f"Available primitives: {available}"
        )

    specification = ALL_PRIMITIVES[primitive]

    if (
        expected_arity is not None
        and specification.arity != expected_arity
    ):
        raise ValueError(
            f"Primitive {primitive!r} has arity "
            f"{specification.arity}, but arity "
            f"{expected_arity} is required."
        )

    return specification


def reachable_node_ids(
    graph: AlgorithmGraph,
) -> set[int]:
    node_map = graph.node_map()

    reachable: set[int] = set()
    stack = [graph.output_id]

    while stack:
        node_id = stack.pop()

        if node_id in reachable:
            continue

        reachable.add(node_id)

        node = node_map[node_id]
        stack.extend(node.inputs)

    return reachable


def prune_unreachable(
    graph: AlgorithmGraph,
) -> AlgorithmGraph:
    child = graph.clone()
    reachable = reachable_node_ids(child)

    child.nodes = [
        node
        for node in child.nodes
        if node.id in reachable
    ]

    child.validate()
    return child


def ancestor_node_ids(
    graph: AlgorithmGraph,
    node_id: int,
) -> set[int]:
    node_map = graph.node_map()

    ancestors: set[int] = set()
    stack = [node_id]

    while stack:
        current_id = stack.pop()

        if current_id in ancestors:
            continue

        ancestors.add(current_id)
        stack.extend(
            node_map[current_id].inputs
        )

    return ancestors


# ---------------------------------------------------------------------
# Deterministic mutation operations
# ---------------------------------------------------------------------

def set_parameters(
    graph: AlgorithmGraph,
    node_id: int,
    parameters: dict[str, Any],
) -> AlgorithmGraph:
    child = graph.clone()
    node = get_node(child, node_id)

    if node.op == "input":
        raise ValueError(
            "The input node has no mutable parameters."
        )

    node.params.update(parameters)

    child.validate()
    return child


def replace_operation(
    graph: AlgorithmGraph,
    node_id: int,
    primitive: str,
    parameters: dict[str, Any],
) -> AlgorithmGraph:
    child = graph.clone()
    node = get_node(child, node_id)

    if node.op == "input":
        raise ValueError(
            "The input operation cannot be replaced."
        )

    specification = check_primitive(primitive)

    if len(node.inputs) != specification.arity:
        raise ValueError(
            f"Cannot replace node {node_id} with "
            f"{primitive!r}. Node {node_id} has "
            f"{len(node.inputs)} inputs, while "
            f"{primitive!r} requires "
            f"{specification.arity}."
        )

    node.op = primitive
    node.params = dict(parameters)

    child.validate()
    return child


def insert_unary(
    graph: AlgorithmGraph,
    target_node_id: int,
    primitive: str,
    parameters: dict[str, Any],
    input_index: int = 0,
) -> AlgorithmGraph:
    child = graph.clone()

    check_primitive(
        primitive,
        expected_arity=1,
    )

    target = get_node(
        child,
        target_node_id,
    )

    if not target.inputs:
        raise ValueError(
            f"Node {target_node_id} has no inputs. "
            "A unary node cannot be inserted before it."
        )

    if not 0 <= input_index < len(target.inputs):
        raise ValueError(
            f"input_index={input_index} is invalid for "
            f"node {target_node_id}, which has "
            f"{len(target.inputs)} inputs."
        )

    parent_id = target.inputs[input_index]
    new_node_id = child.next_node_id()

    child.nodes.append(
        Node(
            id=new_node_id,
            op=primitive,
            inputs=(parent_id,),
            params=dict(parameters),
        )
    )

    new_inputs = list(target.inputs)
    new_inputs[input_index] = new_node_id
    target.inputs = tuple(new_inputs)

    child.validate()
    return child


def delete_unary(
    graph: AlgorithmGraph,
    node_id: int,
) -> AlgorithmGraph:
    child = graph.clone()
    node = get_node(child, node_id)

    if node.op == "input":
        raise ValueError(
            "The input node cannot be deleted."
        )

    if len(node.inputs) != 1:
        raise ValueError(
            f"Node {node_id} is not unary. "
            f"It has {len(node.inputs)} inputs."
        )

    parent_id = node.inputs[0]

    for other_node in child.nodes:
        other_node.inputs = tuple(
            parent_id
            if input_id == node_id
            else input_id
            for input_id in other_node.inputs
        )

    if child.output_id == node_id:
        child.output_id = parent_id

    child.nodes = [
        existing_node
        for existing_node in child.nodes
        if existing_node.id != node_id
    ]

    child = prune_unreachable(child)
    child.validate()

    return child


def create_binary_branch(
    graph: AlgorithmGraph,
    attach_node_id: int,
    binary_primitive: str,
    binary_parameters: dict[str, Any],
    branch_source_id: int,
    branch_primitive: str | None = None,
    branch_parameters: dict[str, Any] | None = None,
) -> AlgorithmGraph:
    child = graph.clone()

    check_primitive(
        binary_primitive,
        expected_arity=2,
    )

    attach_node = get_node(
        child,
        attach_node_id,
    )

    get_node(
        child,
        branch_source_id,
    )

    if len(attach_node.inputs) != 1:
        raise ValueError(
            f"Attachment node {attach_node_id} must "
            "initially be unary."
        )

    ancestors = ancestor_node_ids(
        child,
        attach_node_id,
    )

    if branch_source_id not in ancestors:
        raise ValueError(
            f"Node {branch_source_id} is not an ancestor "
            f"of node {attach_node_id}. Using it could "
            "create a cycle."
        )

    original_input_id = attach_node.inputs[0]
    second_input_id = branch_source_id

    if branch_primitive is not None:
        check_primitive(
            branch_primitive,
            expected_arity=1,
        )

        second_input_id = child.next_node_id()

        child.nodes.append(
            Node(
                id=second_input_id,
                op=branch_primitive,
                inputs=(branch_source_id,),
                params=dict(
                    branch_parameters or {}
                ),
            )
        )

    attach_node.op = binary_primitive
    attach_node.inputs = (
        original_input_id,
        second_input_id,
    )
    attach_node.params = dict(
        binary_parameters
    )

    child = prune_unreachable(child)
    child.validate()

    return child


def insert_constant_binary(
    graph: AlgorithmGraph,
    target_node_id: int,
    binary_primitive: str,
    binary_parameters: dict[str, Any],
    constant_value: float,
    constant_input_index: int = 1,
) -> AlgorithmGraph:
    """Insert a binary operation combining a node with a constant signal.

    For example, with ``binary_primitive="add"`` this changes::

        target

    into::

        add(target, const(value))

    All existing consumers of ``target`` are rewired to the new binary node.
    If ``target`` is the graph output, the new binary node becomes the output.
    """
    child = graph.clone()

    check_primitive(
        binary_primitive,
        expected_arity=2,
    )
    check_primitive(
        "const",
        expected_arity=0,
    )

    get_node(
        child,
        target_node_id,
    )

    if constant_input_index not in (0, 1):
        raise ValueError(
            "constant_input_index must be 0 or 1."
        )

    constant_node_id = child.next_node_id()
    binary_node_id = constant_node_id + 1

    # Rewire only the nodes that existed before this mutation.
    for node in child.nodes:
        node.inputs = tuple(
            binary_node_id
            if input_id == target_node_id
            else input_id
            for input_id in node.inputs
        )

    if constant_input_index == 0:
        binary_inputs = (
            constant_node_id,
            target_node_id,
        )
    else:
        binary_inputs = (
            target_node_id,
            constant_node_id,
        )

    child.nodes.append(
        Node(
            id=constant_node_id,
            op="const",
            inputs=(),
            params={
                "value": float(constant_value),
            },
        )
    )

    child.nodes.append(
        Node(
            id=binary_node_id,
            op=binary_primitive,
            inputs=binary_inputs,
            params=dict(binary_parameters),
        )
    )

    if child.output_id == target_node_id:
        child.output_id = binary_node_id

    child.validate()
    return child


# ---------------------------------------------------------------------
# Blueprint dispatcher
# ---------------------------------------------------------------------

def renumber_in_execution_order(
    graph: AlgorithmGraph,
) -> AlgorithmGraph:
    """
    Renumber nodes according to topological execution order.

    After renumbering:
    - the first executed node has ID 0;
    - every node has a higher ID than its inputs;
    - the output node has the highest ID.
    """
    graph.validate()

    sorted_nodes = graph.topologically_sorted()

    old_to_new_id = {
        node.id: new_id
        for new_id, node in enumerate(sorted_nodes)
    }

    new_nodes = []

    for new_id, old_node in enumerate(sorted_nodes):
        new_nodes.append(
            Node(
                id=new_id,
                op=old_node.op,
                inputs=tuple(
                    old_to_new_id[input_id]
                    for input_id in old_node.inputs
                ),
                params=dict(old_node.params),
            )
        )

    normalized_graph = AlgorithmGraph(
        nodes=new_nodes,
        output_id=old_to_new_id[
            graph.output_id
        ],
    )

    normalized_graph.validate()
    return normalized_graph

def apply_blueprint_step(
    graph: AlgorithmGraph,
    definition: dict[str, Any],
) -> AlgorithmGraph:
    mutation_type = definition["type"]

    if mutation_type == "set_parameters":
        result = set_parameters(
            graph=graph,
            node_id=definition["node_id"],
            parameters=definition["parameters"],
        )

    elif mutation_type == "replace_operation":
        result = replace_operation(
            graph=graph,
            node_id=definition["node_id"],
            primitive=definition["primitive"],
            parameters=definition["parameters"],
        )

    elif mutation_type == "insert_unary":
        result = insert_unary(
            graph=graph,
            target_node_id=definition[
                "target_node_id"
            ],
            primitive=definition["primitive"],
            parameters=definition["parameters"],
            input_index=definition.get(
                "input_index",
                0,
            ),
        )

    elif mutation_type == "delete_unary":
        result = delete_unary(
            graph=graph,
            node_id=definition["node_id"],
        )

    elif mutation_type == "branch_binary":
        result = create_binary_branch(
            graph=graph,
            attach_node_id=definition[
                "attach_node_id"
            ],
            binary_primitive=definition[
                "binary_primitive"
            ],
            binary_parameters=definition[
                "binary_parameters"
            ],
            branch_source_id=definition[
                "branch_source_id"
            ],
            branch_primitive=definition.get(
                "branch_primitive"
            ),
            branch_parameters=definition.get(
                "branch_parameters"
            ),
        )

    elif mutation_type == "insert_constant_binary":
        result = insert_constant_binary(
            graph=graph,
            target_node_id=definition[
                "target_node_id"
            ],
            binary_primitive=definition[
                "binary_primitive"
            ],
            binary_parameters=definition[
                "binary_parameters"
            ],
            constant_value=definition[
                "constant_value"
            ],
            constant_input_index=definition.get(
                "constant_input_index",
                1,
            ),
        )

    else:
        raise ValueError(
            f"Unknown mutation type: "
            f"{mutation_type!r}"
        )

    return renumber_in_execution_order(
        result
    )
# ---------------------------------------------------------------------
# Saving and visualization
# ---------------------------------------------------------------------

def get_execution_order(
    graph: AlgorithmGraph,
) -> list[int]:
    return [
        node.id
        for node in graph.topologically_sorted()
    ]


def save_graph(
    graph: AlgorithmGraph,
    step: int,
    mutation_name: str,
    mutation_definition: dict[str, Any] | None,
):
    step_dir = (
        OUTPUT_ROOT
        / f"step_{step:02d}_{mutation_name}"
    )

    step_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    execution_order = get_execution_order(
        graph
    )

    record = {
        "eval_id": step,
        "dataset_stage": 0,
        "generation_in_stage": step,
        "global_generation": step,
        "dataset_seed": 0,
        "parent_id": (
            None if step == 0 else step - 1
        ),
        "mutation_type": mutation_name,
        "mutation_definition": (
            mutation_definition
        ),
        "objective": 0.0,
        "output_id": graph.output_id,
        "execution_order": execution_order,
        "graph": graph.to_dict(),
        "graph_pretty": graph.pretty(),
        "overall_metrics": {},
    }

    with (
        step_dir / "best_graph.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            record,
            file,
            indent=2,
        )

    visualizer = MutationGraphVisualizer(
        step_dir
    )

    visualizer.plot_best_graph(
        output_path=step_dir / "graph.png",
        show=False,
        output_format="png",
        show_params=True,
        show_input_order=True,
        rank_direction="LR",
    )


def print_graph_summary(
    graph: AlgorithmGraph,
    step: int,
    mutation_name: str,
):
    print()
    print(
        f"Step {step}: {mutation_name}"
    )
    print(
        f"Output node: {graph.output_id}"
    )
    print(
        "Execution order:",
        get_execution_order(graph),
    )
    print(
        "Graph:",
        graph.pretty(),
    )


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main():
    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    graph = make_initial_graph()

    saved_blueprint = []

    save_graph(
        graph=graph,
        step=0,
        mutation_name="initial_graph",
        mutation_definition=None,
    )

    print_graph_summary(
        graph=graph,
        step=0,
        mutation_name="initial_graph",
    )

    saved_blueprint.append(
        {
            "step": 0,
            "mutation": "initial_graph",
            "definition": None,
            "output_id": graph.output_id,
            "execution_order": get_execution_order(
                graph
            ),
            "graph_pretty": graph.pretty(),
        }
    )

    for step_number, definition in enumerate(
        BLUEPRINT,
        start=1,
    ):
        mutation_name = definition["name"]

        graph = apply_blueprint_step(
            graph=graph,
            definition=definition,
        )

        save_graph(
            graph=graph,
            step=step_number,
            mutation_name=mutation_name,
            mutation_definition=definition,
        )

        print_graph_summary(
            graph=graph,
            step=step_number,
            mutation_name=mutation_name,
        )

        saved_blueprint.append(
            {
                "step": step_number,
                "mutation": mutation_name,
                "definition": definition,
                "output_id": graph.output_id,
                "execution_order": (
                    get_execution_order(graph)
                ),
                "graph_pretty": graph.pretty(),
            }
        )

    with (
        OUTPUT_ROOT / "blueprint.json"
    ).open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            saved_blueprint,
            file,
            indent=2,
        )


if __name__ == "__main__":
    main()