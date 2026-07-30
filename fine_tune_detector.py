from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from detector_engine import (
    EvolutionConfig,
    ObjectiveConfig,
    load_best_graph,
    run_evolutionary_search,
)

TARGET_CLASSES = ("N", "V", "R", "L", "A", "/")
MATCH_WINDOWS = {
    "N": (-30, 0),
    "V": (0, 30),
    "R": (0, 30),
    "L": (0, 30),
    "A": (-30, 0),
    "/": (0, 30),
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
        population_size=20,
        n_elite=2,
        children_per_parent=20,
        max_nodes=20,
        n_dataset_stages=1,
        max_generations_per_stage=50,
        stage_success_f1=0.95,
        dataset_seed_stride=10_000,
        resample_dataset_each_generation=True,
        success_consecutive_generations=5,
        force_identity_only_seed=False,
        random_seed_jitter=False,
        save_top_k_children_per_generation=3,
        verbose=True,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune a detector graph saved by a previous search."
    )
    parser.add_argument("--validation", type=Path, default=Path("dataset/val.pkl"))
    parser.add_argument(
        "--pretrained",
        type=Path,
        default=Path("artifacts4/trial_005"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "artifacts4/finetune_trial_005_on_val_to_delay_each_class_but_NA"
        ),
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--fs", type=int, default=360)
    parser.add_argument("--segments-per-class", type=int, default=10)
    parser.add_argument("--segment-length", type=int, default=20_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    validation_df = pd.read_pickle(args.validation)
    args.output.mkdir(parents=True, exist_ok=True)

    pretrained_record, pretrained_graph = load_best_graph(args.pretrained)
    objective_cfg = build_objective_config()
    evolution_cfg = build_evolution_config()

    result, _, _ = run_evolutionary_search(
        df=validation_df,
        fs=args.fs,
        target_classes=TARGET_CLASSES,
        n_segments_per_class=args.segments_per_class,
        segment_len=args.segment_length,
        seed=args.seed,
        out_dir=args.output,
        objective_cfg=objective_cfg,
        evolution_cfg=evolution_cfg,
        initial_graph=pretrained_graph,
    )

    manifest = {
        "mode": "fine_tune_from_pretrained_graph",
        "pretrained_source_dir": str(args.pretrained),
        "pretrained_eval_id": pretrained_record.get("eval_id"),
        "pretrained_graph_pretty": pretrained_graph.pretty(),
        "training_dataset": str(args.validation),
        "target_classes": list(TARGET_CLASSES),
        "n_segments_per_class": args.segments_per_class,
        "segment_len": args.segment_length,
        "objective_config": asdict(objective_cfg),
        "evolution_config": asdict(evolution_cfg),
        "best_result": result.best_record,
    }
    with open(args.output / "finetune_manifest.json", "w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    print("Finished fine-tuning.")
    print("Pretrained graph:")
    print(pretrained_graph.pretty())
    print("Best fine-tuned graph:")
    print(result.best_graph.pretty())
    print("Saved artifacts to:", args.output)


if __name__ == "__main__":
    main()
