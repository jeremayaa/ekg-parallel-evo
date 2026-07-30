from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Sequence

import pandas as pd

from detector_engine import EvolutionConfig, ObjectiveConfig, run_evolutionary_search

TARGET_CLASSES = ("N", "V", "R", "L", "A", "/")
MATCH_WINDOWS = {
    "N": (0, 50),
    "V": (0, 50),
    "R": (-50, 0),
    "L": (-50, 0),
    "A": (0, 50),
    "/": (-50, 0),
}

def build_objective_config() -> ObjectiveConfig:
    return ObjectiveConfig(
        tol=60,
        threshold_quantile=0.97,
        min_distance=80,
        f1_weight=1.0,
        mae_weight=0.0,
        complexity_weight=0.0,
        target_labels=TARGET_CLASSES,
        ignore_non_target_labels=False,
        penalize_non_target_detections=False,
        match_windows_by_label=MATCH_WINDOWS,
    )


def build_evolution_config() -> EvolutionConfig:
    return EvolutionConfig(
        population_size=10,
        n_elite=5,
        children_per_parent=20,
        max_nodes=25,
        n_dataset_stages=1,
        max_generations_per_stage=50,
        stage_success_f1=0.97,
        dataset_seed_stride=10_000,
        resample_dataset_each_generation=True,
        success_consecutive_generations=5,
        force_identity_only_seed=True,
        random_seed_jitter=False,
        save_top_k_children_per_generation=3,
        verbose=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train detector graphs with evolutionary search."
    )
    parser.add_argument("--train", type=Path, default=Path("dataset/train.pkl"))
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts4"))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(range(15)))
    parser.add_argument("--fs", type=int, default=360)
    parser.add_argument("--segments-per-class", type=int, default=25)
    parser.add_argument("--segment-length", type=int, default=20_000)
    return parser.parse_args()


def run_trials(
    train_df: pd.DataFrame,
    artifacts_root: Path,
    seeds: Sequence[int],
    fs: int,
    n_segments_per_class: int,
    segment_len: int,
    training_dataset: Path,
) -> None:
    artifacts_root.mkdir(parents=True, exist_ok=True)
    objective_cfg = build_objective_config()
    evolution_cfg = build_evolution_config()
    summary_rows = []

    for seed in seeds:
        run_name = f"trial_{seed:03d}"
        out_dir = artifacts_root / run_name
        out_dir.mkdir(parents=True, exist_ok=True)

        print("=" * 100)
        print(f"RUNNING {run_name}")
        print("=" * 100)

        result, _, _ = run_evolutionary_search(
            df=train_df,
            fs=fs,
            target_classes=TARGET_CLASSES,
            n_segments_per_class=n_segments_per_class,
            segment_len=segment_len,
            seed=seed,
            out_dir=out_dir,
            objective_cfg=objective_cfg,
            evolution_cfg=evolution_cfg,
        )

        manifest = {
            "mode": "base_search",
            "run_name": run_name,
            "seed": seed,
            "training_dataset": str(training_dataset),
            "target_classes": list(TARGET_CLASSES),
            "n_segments_per_class": n_segments_per_class,
            "segment_len": segment_len,
            "objective_config": asdict(objective_cfg),
            "evolution_config": asdict(evolution_cfg),
            "best_result": result.best_record,
        }
        with open(out_dir / "run_manifest.json", "w", encoding="utf-8") as file:
            json.dump(manifest, file, indent=2)

        metrics = result.best_record["overall_metrics"]
        summary_rows.append(
            {
                "run_name": run_name,
                "seed": seed,
                "artifact_dir": str(out_dir),
                "best_f1": metrics["f1"],
                "best_precision": metrics["precision"],
                "best_recall": metrics["recall"],
                "best_objective": result.best_objective,
                "graph_pretty": result.best_record["graph_pretty"],
            }
        )

        summary = pd.DataFrame(summary_rows).sort_values(
            ["best_f1", "best_objective"],
            ascending=[False, True],
        )
        summary.to_csv(artifacts_root / "batch_summary.csv", index=False)


def main() -> None:
    args = parse_args()
    train_df = pd.read_pickle(args.train)
    run_trials(
        train_df=train_df,
        artifacts_root=args.artifacts,
        seeds=args.seeds,
        fs=args.fs,
        n_segments_per_class=args.segments_per_class,
        segment_len=args.segment_length,
        training_dataset=args.train,
    )
    print(f"Finished {len(args.seeds)} trial(s).")
    print("Saved:", args.artifacts / "batch_summary.csv")


if __name__ == "__main__":
    main()
