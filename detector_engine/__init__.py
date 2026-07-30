"""Evolutionary signal-detector search engine."""

from .evaluation import (
    ObjectiveConfig,
    TrialMetrics,
    evaluate_graph_on_segments,
    evaluate_graph_on_single_row,
    fast_find_peaks,
    greedy_match_metrics,
)
from .graph import (
    AlgorithmGraph,
    Node,
    build_algorithm_bank,
    evaluate_graph,
    graph_from_dict,
    graph_score,
)
from .mutation import EvolutionConfig
from .sampling import SegmentDataset, SegmentSpec, SignalSampler, sample_signal
from .search import EvolutionResult, load_best_graph, run_evolutionary_search

__all__ = [
    "AlgorithmGraph",
    "EvolutionConfig",
    "EvolutionResult",
    "Node",
    "ObjectiveConfig",
    "SegmentDataset",
    "SegmentSpec",
    "SignalSampler",
    "TrialMetrics",
    "build_algorithm_bank",
    "evaluate_graph",
    "evaluate_graph_on_segments",
    "evaluate_graph_on_single_row",
    "fast_find_peaks",
    "graph_from_dict",
    "graph_score",
    "greedy_match_metrics",
    "load_best_graph",
    "run_evolutionary_search",
    "sample_signal",
]
