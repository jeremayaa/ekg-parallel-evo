from __future__ import annotations

"""Sampling of fixed-length signal windows around annotations."""

from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
import pandas as pd

class SignalSampler:
    def __init__(
        self,
        dataset,
        target_class,
        sampling_method="random",
        window=(2000, 2000),
        batch_size=1,
        seed=None,
        pad_value=0.0,
        drop_edge=False,
    ):
        """
        Sample signal windows centered around occurrences of `target_class`.

        Parameters
        ----------
        dataset : pd.DataFrame
            Must contain columns: ['record', 'signal', 'timestamps', 'annotations']
        target_class : str
            Annotation label to center windows on, e.g. "N", "A", "V"
        sampling_method : {"random", "ordered"}
            random  -> sample matching events randomly forever
            ordered -> iterate through matching events in dataset order
        window : tuple[int, int]
            (n_left, n_right), number of samples before and after the center
        batch_size : int
            Number of windows per batch
        seed : int or None
            RNG seed for random sampling
        pad_value : float
            Value used to pad signal when window exceeds signal boundaries
        drop_edge : bool
            If True, skip windows that would exceed signal boundaries.
            If False, pad them to fixed length.
        """
        self.dataset = dataset.reset_index(drop=True)
        self.target_class = target_class
        self.sampling_method = sampling_method
        self.left, self.right = window
        self.batch_size = batch_size
        self.pad_value = pad_value
        self.drop_edge = drop_edge
        self.rng = np.random.default_rng(seed)

        if sampling_method not in {"random", "ordered"}:
            raise ValueError("sampling_method must be 'random' or 'ordered'")

        self.window_len = self.left + self.right + 1

        # Build candidate list: one entry per matching annotation
        self.candidates = self._build_candidates()

        if len(self.candidates) == 0:
            raise ValueError(f"No annotations with class {target_class!r} found.")

        self._ordered_pos = 0

    def _build_candidates(self):
        candidates = []

        for row_idx, row in self.dataset.iterrows():
            anns = row["annotations"]
            ts = row["timestamps"]
            sig = row["signal"]

            # Basic checks
            if len(anns) != len(ts):
                raise ValueError(
                    f"Row {row_idx}: len(annotations) != len(timestamps)"
                )

            signal_len = len(sig)

            for ann_idx, ann in enumerate(anns):
                if ann != self.target_class:
                    continue

                center = ts[ann_idx]
                start = center - self.left
                end = center + self.right

                if self.drop_edge and (start < 0 or end >= signal_len):
                    continue

                candidates.append((row_idx, ann_idx))

        return candidates

    def __len__(self):
        return len(self.candidates)

    def reset(self):
        self._ordered_pos = 0

    def _extract_window(self, row_idx, ann_idx):
        row = self.dataset.iloc[row_idx]

        signal = np.asarray(row["signal"])
        timestamps = np.asarray(row["timestamps"])
        annotations = list(row["annotations"])
        record = row["record"]

        center = int(timestamps[ann_idx])

        start = center - self.left
        end = center + self.right

        # Fixed relative timestamp axis for the extracted signal window
        rel_timestamps = np.arange(-self.left, self.right + 1, dtype=int)

        if self.drop_edge:
            # Safe because edge-invalid candidates were filtered earlier
            signal_window = signal[start:end + 1]
        else:
            # Pad signal to fixed size if needed
            signal_window = np.full(self.window_len, self.pad_value, dtype=float)

            src_start = max(start, 0)
            src_end = min(end, len(signal) - 1)

            dst_start = src_start - start
            dst_end = dst_start + (src_end - src_start + 1)

            signal_window[dst_start:dst_end] = signal[src_start:src_end + 1]

        # Collect annotations falling inside the window
        in_window = [
            {
                "rel_pos": int(timestamps[i] - center),
                "abs_pos": int(timestamps[i]),
                "label": annotations[i],
            }
            for i in range(len(annotations))
            if start <= timestamps[i] <= end
        ]

        return {
            "signal": signal_window,
            "timestamps": rel_timestamps,
            "annotations": in_window,
            "record": record,
            "row_idx": row_idx,
            "ann_idx": ann_idx,
            "center_abs": center,
            "center_label": annotations[ann_idx],
        }

    def _next_candidate_indices(self):
        if self.sampling_method == "random":
            idxs = self.rng.integers(0, len(self.candidates), size=self.batch_size)
            return [self.candidates[i] for i in idxs]

        # ordered
        if self._ordered_pos >= len(self.candidates):
            raise StopIteration

        end_pos = min(self._ordered_pos + self.batch_size, len(self.candidates))
        batch = self.candidates[self._ordered_pos:end_pos]
        self._ordered_pos = end_pos
        return batch

    def __iter__(self):
        return self

    def __next__(self):
        batch_candidates = self._next_candidate_indices()
        batch_items = [self._extract_window(row_idx, ann_idx) for row_idx, ann_idx in batch_candidates]

        signals = np.stack([item["signal"] for item in batch_items], axis=0)
        timestamps = np.stack([item["timestamps"] for item in batch_items], axis=0)

        annotations = [item["annotations"] for item in batch_items]
        records = np.array([item["record"] for item in batch_items])
        centers = np.array([item["center_abs"] for item in batch_items])
        labels = np.array([item["center_label"] for item in batch_items])

        row_indices = np.array([item["row_idx"] for item in batch_items])
        ann_indices = np.array([item["ann_idx"] for item in batch_items])

        return {
            "signals": signals,
            "timestamps": timestamps,
            "annotations": annotations,
            "records": records,
            "centers": centers,
            "labels": labels,
            "row_indices": row_indices,
            "ann_indices": ann_indices,
        }


def sample_signal(
    dataset,
    target_class,
    sampling_method="random",
    window=(2000, 2000),
    batch_size=1,
    seed=None,
    pad_value=0.0,
    drop_edge=False,
):
    return SignalSampler(
        dataset=dataset,
        target_class=target_class,
        sampling_method=sampling_method,
        window=window,
        batch_size=batch_size,
        seed=seed,
        pad_value=pad_value,
        drop_edge=drop_edge,
    )


@dataclass(frozen=True)
class SegmentSpec:
    row_id: int
    start: int
    end: int
    center_label: str


@dataclass
class SegmentDataset:
    segments: List[SegmentSpec]
    signals: List[np.ndarray]
    timestamps: List[np.ndarray]
    labels: List[np.ndarray]


def sample_segments_with_sampler(
    df: pd.DataFrame,
    target_classes: Sequence[str] = ("N",),
    n_segments_per_class: int = 30,
    segment_len: int = 10_000,
    seed: int = 0,
) -> SegmentDataset:
    left = segment_len // 2
    right = segment_len - left - 1

    segments: List[SegmentSpec] = []
    signals: List[np.ndarray] = []
    timestamps_out: List[np.ndarray] = []
    labels_out: List[np.ndarray] = []

    for class_idx, target_class in enumerate(target_classes):
        sampler = sample_signal(
            dataset=df,
            target_class=target_class,
            sampling_method="random",
            window=(left, right),
            batch_size=1,
            seed=seed + class_idx,
        )

        for _ in range(int(n_segments_per_class)):
            batch = next(sampler)

            signal = np.asarray(batch["signals"][0], dtype=float)
            ann_list = batch["annotations"][0]
            center_abs = int(batch["centers"][0])
            row_id = int(batch["row_indices"][0])

            start = center_abs - left
            end = center_abs + right + 1

            local_ts = np.array([a["rel_pos"] + left for a in ann_list], dtype=int)
            local_ann = np.array([a["label"] for a in ann_list], dtype=object)

            segments.append(
                SegmentSpec(
                    row_id=row_id,
                    start=start,
                    end=end,
                    center_label=str(target_class),
                )
            )
            signals.append(signal)
            timestamps_out.append(local_ts)
            labels_out.append(local_ann)

    return SegmentDataset(
        segments=segments,
        signals=signals,
        timestamps=timestamps_out,
        labels=labels_out,
    )


