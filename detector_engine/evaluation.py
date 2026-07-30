from __future__ import annotations

"""Peak detection, annotation matching, and detector objectives."""

from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .graph import AlgorithmGraph, _as_signal, _sanitize_for_json, evaluate_graph, graph_score
from .sampling import SegmentDataset

def fast_find_peaks(
    score: np.ndarray,
    min_distance: int = 50,
    threshold: Optional[float] = None,
) -> np.ndarray:
    score = _as_signal(score)
    n = len(score)
    if n < 3:
        return np.array([], dtype=int)

    if threshold is None:
        threshold = float(np.median(score))

    local_max = (
        (score[1:-1] > score[:-2]) &
        (score[1:-1] >= score[2:]) &
        (score[1:-1] >= threshold)
    )
    peaks = np.where(local_max)[0] + 1

    if len(peaks) <= 1 or min_distance <= 1:
        return peaks.astype(int)

    order = np.argsort(score[peaks])[::-1]
    kept: List[int] = []
    for idx in order:
        p = int(peaks[idx])
        if all(abs(p - q) >= min_distance for q in kept):
            kept.append(p)

    kept = sorted(kept)
    return np.asarray(kept, dtype=int)


def _get_match_window_for_label(
    label: str,
    tol: int,
    match_windows_by_label: Optional[Mapping[str, Tuple[int, int]]] = None,
    default_label: Optional[str] = None,
) -> Tuple[int, int]:
    label = str(label)

    if match_windows_by_label is None:
        return (-int(tol), int(tol))

    if label in match_windows_by_label:
        lo, hi = match_windows_by_label[label]
        return int(lo), int(hi)

    if default_label is not None and default_label in match_windows_by_label:
        lo, hi = match_windows_by_label[default_label]
        return int(lo), int(hi)

    return (-int(tol), int(tol))

def greedy_match_metrics(
    peaks: np.ndarray,
    timestamps: np.ndarray,
    labels: np.ndarray,
    tol: int = 60,
    target_labels: Optional[Sequence[str]] = None,
    ignore_non_target_labels: bool = False,
    penalize_non_target_detections: bool = False,
    non_target_penalty_labels: Optional[Sequence[str]] = None,
    non_target_penalty_tol: Optional[int] = None,
    match_windows_by_label: Optional[Mapping[str, Tuple[int, int]]] = None,
    default_match_window_label: Optional[str] = None,
) -> Dict[str, Any]:
    peaks = np.asarray(peaks, dtype=int)
    timestamps = np.asarray(timestamps, dtype=int)
    labels = np.asarray(labels, dtype=object)

    penalty_tol = int(non_target_penalty_tol if non_target_penalty_tol is not None else tol)


    if (
        not ignore_non_target_labels
        and not penalize_non_target_detections
        and match_windows_by_label is None
    ):
        if len(timestamps) == 0 and len(peaks) == 0:
            return {
                "missed": 0.0,
                "extra": 0.0,
                "matched": 0.0,
                "signed_offsets": [],
                "matches": [],
                "non_target_hits": 0.0,
                "non_target_matches": [],
            }

        if len(timestamps) == 0:
            return {
                "missed": 0.0,
                "extra": float(len(peaks)),
                "matched": 0.0,
                "signed_offsets": [],
                "matches": [],
                "non_target_hits": 0.0,
                "non_target_matches": [],
            }

        if len(peaks) == 0:
            return {
                "missed": float(len(timestamps)),
                "extra": 0.0,
                "matched": 0.0,
                "signed_offsets": [],
                "matches": [],
                "non_target_hits": 0.0,
                "non_target_matches": [],
            }

        used_p = set()
        signed_offsets: List[float] = []
        matches: List[Dict[str, Any]] = []

        matched = 0
        for i, t in enumerate(timestamps):
            best_j = None
            best_d = None
            best_signed = None

            for j, p in enumerate(peaks):
                if j in used_p:
                    continue
                signed = int(p) - int(t)
                d = abs(signed)
                if best_d is None or d < best_d:
                    best_d = d
                    best_j = j
                    best_signed = signed

            if best_j is not None and best_d is not None and best_d <= tol:
                used_p.add(best_j)
                matched += 1
                signed_offsets.append(float(best_signed))
                matches.append(
                    {
                        "label": str(labels[i]),
                        "timestamp": int(t),
                        "peak": int(peaks[best_j]),
                        "signed_offset": float(best_signed),
                        "abs_dist": float(best_d),
                    }
                )

        return {
            "missed": float(len(timestamps) - matched),
            "extra": float(len(peaks) - matched),
            "matched": float(matched),
            "signed_offsets": signed_offsets,
            "matches": matches,
            "non_target_hits": 0.0,
            "non_target_matches": [],
        }

    # -------------------------------------------------------------------------
    # Label-aware mode
    # -------------------------------------------------------------------------
    if target_labels is None:
        target_mask = np.ones(len(labels), dtype=bool)
    else:
        target_set = set(map(str, target_labels))
        target_mask = np.array([str(lbl) in target_set for lbl in labels], dtype=bool)

    target_ts = timestamps[target_mask]
    target_lbls = labels[target_mask]

    other_ts = timestamps[~target_mask]
    other_lbls = labels[~target_mask]

    if non_target_penalty_labels is not None:
        allowed_other = set(map(str, non_target_penalty_labels))
        other_mask = np.array([str(lbl) in allowed_other for lbl in other_lbls], dtype=bool)
        other_ts = other_ts[other_mask]
        other_lbls = other_lbls[other_mask]

    if len(target_ts) == 0 and len(peaks) == 0:
        return {
            "missed": 0.0,
            "extra": 0.0,
            "matched": 0.0,
            "signed_offsets": [],
            "matches": [],
            "non_target_hits": 0.0,
            "non_target_matches": [],
        }

    if len(peaks) == 0:
        return {
            "missed": float(len(target_ts)),
            "extra": 0.0,
            "matched": 0.0,
            "signed_offsets": [],
            "matches": [],
            "non_target_hits": 0.0,
            "non_target_matches": [],
        }

    used_p = set()
    signed_offsets: List[float] = []
    matches: List[Dict[str, Any]] = []

    matched = 0
    for i, t in enumerate(target_ts):
        label_i = str(target_lbls[i])
        lo, hi = _get_match_window_for_label(
            label=label_i,
            tol=tol,
            match_windows_by_label=match_windows_by_label,
            default_label=default_match_window_label,
        )

        best_j = None
        best_d = None
        best_signed = None

        for j, p in enumerate(peaks):
            if j in used_p:
                continue
            signed = int(p) - int(t)
            if not (lo <= signed <= hi):
                continue
            d = abs(signed)
            if best_d is None or d < best_d:
                best_d = d
                best_j = j
                best_signed = signed

        if best_j is not None:
            used_p.add(best_j)
            matched += 1
            signed_offsets.append(float(best_signed))
            matches.append(
                {
                    "label": label_i,
                    "timestamp": int(t),
                    "peak": int(peaks[best_j]),
                    "signed_offset": float(best_signed),
                    "abs_dist": float(best_d),
                }
            )

    non_target_hits = 0.0
    non_target_matches: List[Dict[str, Any]] = []

    if penalize_non_target_detections and len(other_ts) > 0:
        unmatched_peak_indices = [j for j in range(len(peaks)) if j not in used_p]

        for j in unmatched_peak_indices:
            p = int(peaks[j])
            dists = np.abs(other_ts - p)
            if len(dists) == 0:
                continue
            k = int(np.argmin(dists))
            if dists[k] <= penalty_tol:
                non_target_hits += 1.0
                non_target_matches.append(
                    {
                        "label": str(other_lbls[k]),
                        "timestamp": int(other_ts[k]),
                        "peak": p,
                        "abs_dist": float(dists[k]),
                    }
                )

    return {
        "missed": float(len(target_ts) - matched),
        "extra": float(len(peaks) - matched),
        "matched": float(matched),
        "signed_offsets": signed_offsets,
        "matches": matches,
        "non_target_hits": float(non_target_hits),
        "non_target_matches": non_target_matches,
    }


@dataclass
class ObjectiveConfig:
    tol: int = 60
    threshold_quantile: float = 0.97
    min_distance: int = 50

    f1_weight: float = 1.0
    mae_weight: float = 0.01
    complexity_weight: float = 0.001

    target_labels: Optional[Tuple[str, ...]] = None
    ignore_non_target_labels: bool = False

    penalize_non_target_detections: bool = False
    non_target_penalty_weight: float = 1.0
    non_target_penalty_labels: Optional[Tuple[str, ...]] = None
    non_target_penalty_tol: Optional[int] = None

    match_windows_by_label: Optional[Dict[str, Tuple[int, int]]] = None
    default_match_window_label: Optional[str] = None
    experiment_name: Optional[str] = None
    

@dataclass
class TrialMetrics:
    precision: float
    recall: float
    f1: float
    mae: float
    objective: float
    tp: float
    fp: float
    fn: float
    n_segments: int
    n_nodes: int

    def to_dict(self) -> Dict[str, Any]:
        return _sanitize_for_json(asdict(self))


def _estimate_graph_size(graph: AlgorithmGraph) -> int:
    return len(graph.nodes)

def evaluate_graph_on_segments(
    graph: AlgorithmGraph,
    dataset: SegmentDataset,
    cfg: ObjectiveConfig,
) -> Dict[str, Any]:
    tp_total = 0.0
    fp_total = 0.0
    fn_total = 0.0
    non_target_hits_total = 0.0
    abs_errors: List[float] = []

    for signal, ts, labs in zip(dataset.signals, dataset.timestamps, dataset.labels):
        score = graph_score(signal, graph)
        threshold = float(np.quantile(score, cfg.threshold_quantile))
        peaks = fast_find_peaks(
            score,
            min_distance=cfg.min_distance,
            threshold=threshold,
        )

        m = greedy_match_metrics(
            peaks=peaks,
            timestamps=ts,
            labels=labs,
            tol=cfg.tol,
            target_labels=cfg.target_labels,
            ignore_non_target_labels=cfg.ignore_non_target_labels,
            penalize_non_target_detections=cfg.penalize_non_target_detections,
            non_target_penalty_labels=cfg.non_target_penalty_labels,
            non_target_penalty_tol=cfg.non_target_penalty_tol,
            match_windows_by_label=cfg.match_windows_by_label,
            default_match_window_label=cfg.default_match_window_label,
        )

        tp_total += float(m["matched"])
        fp_total += float(m["extra"])
        fn_total += float(m["missed"])
        non_target_hits_total += float(m["non_target_hits"])

        if len(m["matches"]) > 0:
            abs_errors.extend(float(mm["abs_dist"]) for mm in m["matches"])

    recall = tp_total / (tp_total + fn_total + 1e-8)
    precision = tp_total / (tp_total + fp_total + 1e-8)
    f1 = 2.0 * precision * recall / (precision + recall + 1e-8)

    mae = float(np.mean(abs_errors)) if len(abs_errors) > 0 else float(cfg.tol)
    n_nodes = _estimate_graph_size(graph)

    objective = (
        -cfg.f1_weight * f1
        + cfg.mae_weight * mae
        + cfg.complexity_weight * float(n_nodes)
    )

    if cfg.penalize_non_target_detections:
        objective += cfg.non_target_penalty_weight * non_target_hits_total

    metrics = TrialMetrics(
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        mae=float(mae),
        objective=float(objective),
        tp=float(tp_total),
        fp=float(fp_total),
        fn=float(fn_total),
        n_segments=int(len(dataset.segments)),
        n_nodes=int(n_nodes),
    )

    return {
        "metrics": metrics,
        "non_target_hits": float(non_target_hits_total),
    }


def evaluate_graph_on_single_row(
    df: pd.DataFrame,
    row_idx: int,
    graph: AlgorithmGraph,
    threshold_quantile: float = 0.97,
    min_distance: int = 50,
    tol: int = 60,
    max_len: Optional[int] = None,
) -> Dict[str, Any]:
    signal = np.asarray(df.iloc[row_idx]["signal"], dtype=float)
    timestamps = np.asarray(df.iloc[row_idx]["timestamps"], dtype=int)
    annotations = np.asarray(df.iloc[row_idx]["annotations"], dtype=object)

    if max_len is not None:
        signal = signal[:max_len]
        mask = timestamps < max_len
        timestamps = timestamps[mask]
        annotations = annotations[mask]

    cache = evaluate_graph(graph, signal)
    score = cache[graph.output_id]
    threshold = float(np.quantile(score, threshold_quantile))
    peaks = fast_find_peaks(score, min_distance=min_distance, threshold=threshold)
    match_dict = greedy_match_metrics(peaks, timestamps, annotations, tol=tol)

    return {
        "signal": signal,
        "timestamps": timestamps,
        "annotations": annotations,
        "score": score,
        "cache": cache,
        "threshold": threshold,
        "peaks": peaks,
        "match_dict": match_dict,
    }
