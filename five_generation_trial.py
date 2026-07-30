from pathlib import Path
import json

import pandas as pd

from detector_engine import (
    EvolutionConfig,
    ObjectiveConfig,
    run_evolutionary_search,
)
from graph_visualization import TrialGraphVisualizer


TRAIN_PATH = Path("dataset/train.pkl")
OUTPUT_ROOT = Path("five_generation_demo")

TARGET_CLASSES = ("N", "V", "R", "L", "A", "/")

EARLY_WINDOW = (-50, 0)
LATE_WINDOW = (0, 50)

MATCH_WINDOWS = {
    "N": EARLY_WINDOW,
    "V": LATE_WINDOW,
    "R": LATE_WINDOW,
    "L": LATE_WINDOW,
    "A": EARLY_WINDOW,
    "/": LATE_WINDOW,
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


def build_evolution_config(
    number_of_generations: int,
) -> EvolutionConfig:
    return EvolutionConfig(
        population_size=10,
        n_elite=5,
        children_per_parent=20,
        max_nodes=25,
        n_dataset_stages=1,
        max_generations_per_stage=number_of_generations,

        # Prevent stopping before the requested generation.
        stage_success_f1=2.0,
        success_consecutive_generations=10,

        dataset_seed_stride=10_000,
        resample_dataset_each_generation=True,
        force_identity_only_seed=True,
        random_seed_jitter=False,
        save_top_k_children_per_generation=5,
        verbose=True,
    )


def main() -> None:
    if not TRAIN_PATH.exists():
        raise FileNotFoundError(
            f"Training dataset not found: {TRAIN_PATH.resolve()}"
        )

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_df = pd.read_pickle(TRAIN_PATH)

    previous_record = None

    for generation_count in range(1, 6):
        print("=" * 80)
        print(
            f"RUNNING TRIAL WITH "
            f"{generation_count} GENERATION(S)"
        )
        print("=" * 80)

        run_dir = (
            OUTPUT_ROOT
            / f"generation_{generation_count}"
        )

        run_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        result, _, _ = run_evolutionary_search(
            df=train_df,
            fs=360,
            target_classes=TARGET_CLASSES,
            n_segments_per_class=10,
            segment_len=20_000,
            seed=0,
            out_dir=run_dir,
            objective_cfg=build_objective_config(),
            evolution_cfg=build_evolution_config(
                generation_count
            ),
        )

        graph_output_path = (
            run_dir
            / "best_graph.png"
        )

        visualizer = TrialGraphVisualizer(
            run_dir
        )

        visualizer.plot_best_graph(
            output_path=graph_output_path,
            show=False,
            output_format="png",
            show_params=True,
            show_input_order=True,
            rank_direction="LR",
        )

        best_graph_path = (
            run_dir
            / "best_graph.json"
        )

        with best_graph_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            current_record = json.load(file)

        if previous_record is not None:
            change_record = {
                "from_generation": (
                    generation_count - 1
                ),
                "to_generation": generation_count,
                "previous_graph": previous_record.get(
                    "graph_pretty"
                ),
                "current_graph": current_record.get(
                    "graph_pretty"
                ),
                "current_mutation_type": (
                    current_record.get(
                        "mutation_type"
                    )
                ),
                "current_parent_id": (
                    current_record.get(
                        "parent_id"
                    )
                ),
            }

            change_path = (
                run_dir
                / "change_from_previous.json"
            )

            with change_path.open(
                "w",
                encoding="utf-8",
            ) as file:
                json.dump(
                    change_record,
                    file,
                    indent=2,
                )

        previous_record = current_record

        print(
            f"Saved trial: {run_dir.resolve()}"
        )
        print(
            f"Saved graph: "
            f"{graph_output_path.resolve()}"
        )
        print(
            f"Best F1: "
            f"{result.best_record['overall_metrics']['f1']:.4f}"
        )

    print("\nFinished.")
    print(
        f"Results: {OUTPUT_ROOT.resolve()}"
    )


if __name__ == "__main__":
    main()