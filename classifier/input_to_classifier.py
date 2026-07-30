from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, TypeAlias

import numpy as np
import pandas as pd

from detector_engine import AlgorithmGraph, fast_find_peaks, graph_score


DetectorFunction: TypeAlias = Callable[[np.ndarray], Sequence[int] | np.ndarray]
Detector: TypeAlias = DetectorFunction | AlgorithmGraph


class ClassifierInputBuilder:
    """
    Build the decision-tree input dataframe from ECG peak detectors.

    Each output row corresponds to one detection made by the reference
    detector. Every feature contains the signed offset, in samples, from that
    reference detection to the nearest detection made by another detector.

    A detector can be either:
    - a regular callable from ``algorithms.py`` returning peak indices, or
    - an ``AlgorithmGraph`` evaluated with ``detector_engine.graph_score``.
    """

    _SCHEMA_CANDIDATES = (
        ("signal", "timestamps", "annotations"),
        ("ECG", "True_peak", "Peak_type"),
    )
    _RESERVED_COLUMNS = {"record", "reference_peak", "annotation"}

    def __init__(
        self,
        *,
        window: int = 50,
        graph_threshold_quantile: float = 0.97,
        graph_min_distance: int = 80,
        missing_offset: int | None = None,
    ) -> None:
        if window <= 0:
            raise ValueError("window must be positive")
        if not 0.0 <= graph_threshold_quantile <= 1.0:
            raise ValueError("graph_threshold_quantile must be between 0 and 1")
        if graph_min_distance < 1:
            raise ValueError("graph_min_distance must be at least 1")

        self.window = int(window)
        self.graph_threshold_quantile = float(graph_threshold_quantile)
        self.graph_min_distance = int(graph_min_distance)
        self.missing_offset = (
            int(missing_offset)
            if missing_offset is not None
            else -2 * self.window
        )

    def build(
        self,
        dataset: pd.DataFrame,
        detectors: Mapping[str, Detector],
        *,
        reference_detector: str,
        graph_settings: Mapping[str, Mapping[str, float | int]] | None = None,
        signal_column: str | None = None,
        true_peak_column: str | None = None,
        annotation_column: str | None = None,
        record_column: str | None = None,
    ) -> pd.DataFrame:
        """
        Compute classifier features for all records in ``dataset``.

        Parameters
        ----------
        dataset:
            Dataframe containing ECG signals, true peak positions, and labels.
            The current engine schema
            ``signal / timestamps / annotations`` and the legacy schema
            ``ECG / True_peak / Peak_type`` are detected automatically.
        detectors:
            Mapping from feature name to a callable detector or AlgorithmGraph.
        reference_detector:
            Name of the detector whose peaks define the dataframe rows.
            The reference detector is not included as a feature column.
        graph_settings:
            Optional per-graph overrides, for example::

                {
                    "evolved_1": {
                        "threshold_quantile": 0.98,
                        "min_distance": 90,
                    }
                }
        record_column:
            Optional source column used as the record identifier. When omitted,
            the builder uses the ``record`` column when present, otherwise the
            dataframe index.
        """
        if not isinstance(dataset, pd.DataFrame):
            raise TypeError("dataset must be a pandas DataFrame")
        if not detectors:
            raise ValueError("detectors cannot be empty")
        if reference_detector not in detectors:
            raise KeyError(
                f"reference_detector {reference_detector!r} is not present "
                f"in detectors: {list(detectors)}"
            )

        invalid_names = set(detectors) & self._RESERVED_COLUMNS
        if invalid_names:
            raise ValueError(
                "Detector names cannot use reserved output columns: "
                f"{sorted(invalid_names)}"
            )

        signal_column, true_peak_column, annotation_column = self._resolve_schema(
            dataset=dataset,
            signal_column=signal_column,
            true_peak_column=true_peak_column,
            annotation_column=annotation_column,
        )

        if record_column is not None and record_column not in dataset.columns:
            raise KeyError(f"record column {record_column!r} does not exist")

        graph_settings = graph_settings or {}
        feature_names = [
            name for name in detectors
            if name != reference_detector
        ]
        rows: list[dict[str, Any]] = []

        for row_position, (row_index, source_row) in enumerate(dataset.iterrows()):
            signal = self._as_signal(
                source_row[signal_column],
                context=f"row {row_index!r}",
            )
            true_peaks = self._as_peak_array(
                source_row[true_peak_column],
                context=f"true peaks in row {row_index!r}",
            )
            annotations = np.asarray(
                source_row[annotation_column],
                dtype=object,
            )

            if len(true_peaks) != len(annotations):
                raise ValueError(
                    f"Row {row_index!r}: true peak count ({len(true_peaks)}) "
                    f"does not match annotation count ({len(annotations)})"
                )

            record_id = self._record_id(
                source_row=source_row,
                row_index=row_index,
                record_column=record_column,
            )

            detected_peaks: dict[str, np.ndarray] = {}
            for detector_name, detector in detectors.items():
                try:
                    detected_peaks[detector_name] = self._run_detector(
                        name=detector_name,
                        detector=detector,
                        signal=signal,
                        graph_settings=graph_settings,
                    )
                except Exception as exc:
                    raise RuntimeError(
                        f"Detector {detector_name!r} failed for record "
                        f"{record_id!r} at dataframe row {row_position}"
                    ) from exc

            reference_peaks = detected_peaks[reference_detector]

            for reference_peak in reference_peaks:
                output_row: dict[str, Any] = {
                    "record": record_id,
                    "reference_peak": int(reference_peak),
                    "annotation": self._annotation_for_detection(
                        detection=int(reference_peak),
                        true_peaks=true_peaks,
                        annotations=annotations,
                    ),
                }

                for feature_name in feature_names:
                    output_row[feature_name] = self._nearest_offset(
                        reference_peak=int(reference_peak),
                        other_peaks=detected_peaks[feature_name],
                    )

                rows.append(output_row)

        columns = [
            "record",
            "reference_peak",
            *feature_names,
            "annotation",
        ]
        return pd.DataFrame(rows, columns=columns)

    def build_and_save(
        self,
        dataset: pd.DataFrame,
        detectors: Mapping[str, Detector],
        output_path: str | Path,
        *,
        reference_detector: str,
        **build_kwargs: Any,
    ) -> pd.DataFrame:
        """Build the dataframe and save it as pickle or CSV."""
        result = self.build(
            dataset=dataset,
            detectors=detectors,
            reference_detector=reference_detector,
            **build_kwargs,
        )

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        suffix = output_path.suffix.lower()
        if suffix in {".pkl", ".pickle"}:
            result.to_pickle(output_path)
        elif suffix == ".csv":
            result.to_csv(output_path, index=False)
        else:
            raise ValueError(
                "output_path must end with .pkl, .pickle, or .csv"
            )

        return result

    def _run_detector(
        self,
        *,
        name: str,
        detector: Detector,
        signal: np.ndarray,
        graph_settings: Mapping[str, Mapping[str, float | int]],
    ) -> np.ndarray:
        if isinstance(detector, AlgorithmGraph):
            settings = graph_settings.get(name, {})
            threshold_quantile = float(
                settings.get(
                    "threshold_quantile",
                    self.graph_threshold_quantile,
                )
            )
            min_distance = int(
                settings.get(
                    "min_distance",
                    self.graph_min_distance,
                )
            )

            if not 0.0 <= threshold_quantile <= 1.0:
                raise ValueError(
                    f"{name}: threshold_quantile must be between 0 and 1"
                )
            if min_distance < 1:
                raise ValueError(
                    f"{name}: min_distance must be at least 1"
                )

            score = graph_score(signal, detector)
            threshold = float(np.quantile(score, threshold_quantile))
            peaks = fast_find_peaks(
                score,
                min_distance=min_distance,
                threshold=threshold,
            )
            return self._as_peak_array(peaks, context=name)

        if not callable(detector):
            raise TypeError(
                f"{name}: expected a callable or AlgorithmGraph, "
                f"got {type(detector).__name__}"
            )

        result = detector(signal)

        # Some detector helpers return ``(peaks, score, threshold)`` or
        # ``(peaks, operation_counts)``. In both cases, the first item is the
        # peak sequence required by the classifier.
        if isinstance(result, tuple):
            if not result:
                return np.array([], dtype=int)
            result = result[0]

        return self._as_peak_array(result, context=name)

    def _annotation_for_detection(
        self,
        *,
        detection: int,
        true_peaks: np.ndarray,
        annotations: np.ndarray,
    ) -> str:
        mask = (
            (true_peaks >= detection - self.window)
            & (true_peaks <= detection + self.window)
        )
        matching_indices = np.flatnonzero(mask)

        if len(matching_indices) == 1:
            return str(annotations[matching_indices[0]])
        if len(matching_indices) > 1:
            return "AMB"
        return "FP"

    def _nearest_offset(
        self,
        *,
        reference_peak: int,
        other_peaks: np.ndarray,
    ) -> int:
        if len(other_peaks) == 0:
            return self.missing_offset

        offsets = other_peaks - int(reference_peak)
        valid = np.abs(offsets) < self.window

        if not np.any(valid):
            return self.missing_offset

        valid_offsets = offsets[valid]
        nearest_index = int(np.argmin(np.abs(valid_offsets)))
        return int(valid_offsets[nearest_index])

    @classmethod
    def _resolve_schema(
        cls,
        *,
        dataset: pd.DataFrame,
        signal_column: str | None,
        true_peak_column: str | None,
        annotation_column: str | None,
    ) -> tuple[str, str, str]:
        explicit = (signal_column, true_peak_column, annotation_column)
        if any(value is not None for value in explicit):
            if not all(value is not None for value in explicit):
                raise ValueError(
                    "signal_column, true_peak_column, and annotation_column "
                    "must be provided together"
                )
            selected = (
                str(signal_column),
                str(true_peak_column),
                str(annotation_column),
            )
            missing = [name for name in selected if name not in dataset.columns]
            if missing:
                raise KeyError(f"Dataset columns do not exist: {missing}")
            return selected

        for candidate in cls._SCHEMA_CANDIDATES:
            if all(column in dataset.columns for column in candidate):
                return candidate

        raise ValueError(
            "Could not detect the dataset schema. Expected either "
            "['signal', 'timestamps', 'annotations'] or "
            "['ECG', 'True_peak', 'Peak_type']. You can also pass explicit "
            "column names."
        )

    @staticmethod
    def _record_id(
        *,
        source_row: pd.Series,
        row_index: Any,
        record_column: str | None,
    ) -> Any:
        if record_column is not None:
            return source_row[record_column]
        if "record" in source_row.index:
            return source_row["record"]
        return row_index

    @staticmethod
    def _as_signal(value: Any, *, context: str) -> np.ndarray:
        signal = np.asarray(value, dtype=float)
        if signal.ndim != 1:
            raise ValueError(
                f"{context}: signal must be one-dimensional, got "
                f"shape {signal.shape}"
            )
        if len(signal) == 0:
            raise ValueError(f"{context}: signal cannot be empty")
        return signal

    @staticmethod
    def _as_peak_array(value: Any, *, context: str) -> np.ndarray:
        peaks = np.asarray(value)
        if peaks.ndim == 0:
            peaks = peaks.reshape(1)
        if peaks.ndim != 1:
            raise ValueError(
                f"{context}: peak positions must be one-dimensional, "
                f"got shape {peaks.shape}"
            )
        if len(peaks) == 0:
            return np.array([], dtype=int)

        try:
            peaks_float = peaks.astype(float)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"{context}: peak positions must be numeric"
            ) from exc

        if not np.all(np.isfinite(peaks_float)):
            raise ValueError(
                f"{context}: peak positions contain NaN or infinity"
            )

        return np.unique(peaks_float.astype(int))


def build_classifier_input(
    dataset: pd.DataFrame,
    detectors: Mapping[str, Detector],
    *,
    reference_detector: str,
    window: int = 50,
    graph_threshold_quantile: float = 0.97,
    graph_min_distance: int = 80,
    graph_settings: Mapping[str, Mapping[str, float | int]] | None = None,
    output_path: str | Path | None = None,
    **column_kwargs: Any,
) -> pd.DataFrame:
    """
    Functional wrapper around :class:`ClassifierInputBuilder`.

    Example
    -------
    ``detectors`` may mix functions and graphs::

        detectors = {
            "reference": algorithms.alg4_polish_20210222,
            "pan_tompkins": algorithms.alg5_pan_tompkins,
            "evolved": evolved_graph,
        }
    """
    builder = ClassifierInputBuilder(
        window=window,
        graph_threshold_quantile=graph_threshold_quantile,
        graph_min_distance=graph_min_distance,
    )

    if output_path is None:
        return builder.build(
            dataset=dataset,
            detectors=detectors,
            reference_detector=reference_detector,
            graph_settings=graph_settings,
            **column_kwargs,
        )

    return builder.build_and_save(
        dataset=dataset,
        detectors=detectors,
        output_path=output_path,
        reference_detector=reference_detector,
        graph_settings=graph_settings,
        **column_kwargs,
    )
