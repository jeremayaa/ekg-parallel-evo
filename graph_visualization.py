from __future__ import annotations

"""Graphviz-based visualization of detector graphs stored in trial artifacts."""

import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from graphviz import Digraph

from detector_engine import AlgorithmGraph, graph_from_dict


class TrialGraphVisualizer:
    """Load and visualize the best graph from one detector trial.

    Parameters
    ----------
    trial_path:
        A trial directory such as ``artifacts4/trial_005`` or a direct path
        to its ``best_graph.json`` file.
    """

    def __init__(self, trial_path: str | Path) -> None:
        self.trial_path, self.best_graph_path = self._resolve_paths(trial_path)
        self.record, self.graph = self._load_best_graph(self.best_graph_path)

    @classmethod
    def from_artifacts_root(
        cls,
        artifacts_root: str | Path,
        trial_name: str,
    ) -> "TrialGraphVisualizer":
        """Load a named trial from an artifact root directory."""
        return cls(Path(artifacts_root) / trial_name)

    @staticmethod
    def _resolve_paths(trial_path: str | Path) -> Tuple[Path, Path]:
        path = Path(trial_path).expanduser().resolve()

        if path.is_dir():
            trial_dir = path
            best_graph_path = path / "best_graph.json"
        elif path.is_file():
            trial_dir = path.parent
            best_graph_path = path
        else:
            raise FileNotFoundError(f"Trial path does not exist: {path}")

        if not best_graph_path.is_file():
            raise FileNotFoundError(
                f"Could not find best_graph.json in: {trial_dir}"
            )

        return trial_dir, best_graph_path

    @staticmethod
    def _load_best_graph(
        best_graph_path: Path,
    ) -> Tuple[Dict[str, Any], AlgorithmGraph]:
        with best_graph_path.open("r", encoding="utf-8") as file:
            data = json.load(file)

        if not isinstance(data, dict):
            raise ValueError("best_graph.json must contain a JSON object")

        if "graph" in data:
            graph_data = data["graph"]
            record = dict(data)
        elif "nodes" in data and "output_id" in data:
            graph_data = data
            record = {"graph": data}
        else:
            raise ValueError(
                "best_graph.json must contain a 'graph' field or directly "
                "contain 'nodes' and 'output_id'"
            )

        graph = graph_from_dict(graph_data)
        return record, graph

    @property
    def trial_name(self) -> str:
        return self.trial_path.name

    @property
    def metrics(self) -> Mapping[str, Any]:
        metrics = self.record.get("overall_metrics", {})
        return metrics if isinstance(metrics, Mapping) else {}

    def summary(self) -> Dict[str, Any]:
        """Return basic trial, graph, and metric information."""
        return {
            "trial": self.trial_name,
            "best_graph_path": str(self.best_graph_path),
            "eval_id": self.record.get("eval_id"),
            "objective": self.record.get("objective"),
            "n_nodes": len(self.graph.nodes),
            "output_id": self.graph.output_id,
            "graph_pretty": self.record.get("graph_pretty", self.graph.pretty()),
            "metrics": dict(self.metrics),
        }

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    @classmethod
    def _node_label(cls, node: Any, show_params: bool) -> str:
        lines = [f"{node.id}: {node.op}"]
        if show_params:
            lines.extend(
                f"{name}={cls._format_value(value)}"
                for name, value in node.params.items()
            )
        return "\n".join(lines)

    def _title(self) -> str:
        lines = [f"Best graph — {self.trial_name}"]
        metric_parts = []

        for key in ("f1", "precision", "recall", "mae"):
            value = self.metrics.get(key)
            if isinstance(value, (int, float)):
                digits = 2 if key == "mae" else 4
                metric_parts.append(f"{key}={value:.{digits}f}")

        objective = self.record.get("objective")
        if isinstance(objective, (int, float)):
            metric_parts.append(f"objective={objective:.4f}")

        if metric_parts:
            lines.append(" | ".join(metric_parts))

        return "\n".join(lines)

    def build_best_graph(
        self,
        *,
        show_params: bool = True,
        show_input_order: bool = True,
        rank_direction: str = "LR",
    ) -> Digraph:
        """Build and return a Graphviz ``Digraph`` for the best graph."""
        self.graph.validate()

        dot = Digraph(
            name=f"best_graph_{self.trial_name}",
            comment=f"Best detector graph from {self.trial_name}",
        )
        dot.attr(
            rankdir=rank_direction,
            label=self._title(),
            labelloc="t",
            fontsize="15",
            fontname="Helvetica",
            pad="0.25",
            nodesep="0.45",
            ranksep="0.70",
            bgcolor="transparent",
        )
        dot.attr(
            "node",
            shape="box",
            style="rounded,filled",
            fillcolor="white",
            color="#404040",
            fontname="Helvetica",
            fontsize="10",
            margin="0.14,0.09",
        )
        dot.attr(
            "edge",
            color="#606060",
            fontname="Helvetica",
            fontsize="9",
            arrowsize="0.75",
        )

        for node in self.graph.topologically_sorted():
            attributes: Dict[str, str] = {}

            if node.op == "input":
                attributes.update(shape="ellipse", fillcolor="#eef5ff")
            elif node.op == "const":
                attributes.update(shape="ellipse", fillcolor="#f5f5f5")

            if node.id == self.graph.output_id:
                attributes.update(
                    peripheries="2",
                    penwidth="2.0",
                    fillcolor="#fff4d6",
                )

            dot.node(
                str(node.id),
                label=self._node_label(node, show_params),
                **attributes,
            )

        for child in self.graph.topologically_sorted():
            for input_index, parent_id in enumerate(child.inputs):
                edge_attributes: Dict[str, str] = {}
                if show_input_order and len(child.inputs) > 1:
                    edge_attributes["label"] = str(input_index)

                dot.edge(
                    str(parent_id),
                    str(child.id),
                    **edge_attributes,
                )

        return dot

    def plot_best_graph(
        self,
        output_path: Optional[str | Path] = None,
        *,
        show: bool = True,
        output_format: Optional[str] = None,
        show_params: bool = True,
        show_input_order: bool = True,
        rank_direction: str = "LR",
    ) -> Digraph:
        """Display the best graph and optionally save it as PNG, SVG, or PDF.

        The output format is inferred from ``output_path`` when it has a file
        extension. Otherwise ``output_format`` is used, defaulting to ``png``.
        The file is written to the exact requested path.
        """
        dot = self.build_best_graph(
            show_params=show_params,
            show_input_order=show_input_order,
            rank_direction=rank_direction,
        )

        if output_path is not None:
            output = Path(output_path).expanduser().resolve()
            inferred_format = output.suffix.lstrip(".").lower()
            image_format = (output_format or inferred_format or "png").lower()

            if not output.suffix:
                output = output.with_suffix(f".{image_format}")

            output.parent.mkdir(parents=True, exist_ok=True)
            output.write_bytes(dot.pipe(format=image_format))

            if not output.is_file() or output.stat().st_size == 0:
                raise RuntimeError(f"Graphviz did not produce a valid file: {output}")

        if show:
            try:
                from IPython.display import display

                display(dot)
            except ImportError:
                pass

        return dot
