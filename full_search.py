from __future__ import annotations

import argparse
import json
import multiprocessing as mp
from concurrent.futures import Future, ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from detector_engine import (
    EvolutionConfig,
    ObjectiveConfig,
    run_evolutionary_search,
)
from graph_visualization import TrialGraphVisualizer


TARGET_CLASSES = ("N", "V", "R", "L", "A", "/")

EARLY_WINDOW = (-50, 0)
LATE_WINDOW = (0, 50)


# =============================================================================
# Matching-window configurations
# =============================================================================

EARLY_NA = {
    "N": EARLY_WINDOW,
    "V": LATE_WINDOW,
    "R": LATE_WINDOW,
    "L": LATE_WINDOW,
    "A": EARLY_WINDOW,
    "/": LATE_WINDOW,
}

EARLY_V = {
    "N": LATE_WINDOW,
    "V": EARLY_WINDOW,
    "R": LATE_WINDOW,
    "L": LATE_WINDOW,
    "A": LATE_WINDOW,
    "/": LATE_WINDOW,
}

EARLY_OTHERS = {
    "N": LATE_WINDOW,
    "V": LATE_WINDOW,
    "R": EARLY_WINDOW,
    "L": EARLY_WINDOW,
    "A": LATE_WINDOW,
    "/": EARLY_WINDOW,
}

LATE_NA = {
    "N": LATE_WINDOW,
    "V": EARLY_WINDOW,
    "R": EARLY_WINDOW,
    "L": EARLY_WINDOW,
    "A": LATE_WINDOW,
    "/": EARLY_WINDOW,
}

LATE_V = {
    "N": EARLY_WINDOW,
    "V": LATE_WINDOW,
    "R": EARLY_WINDOW,
    "L": EARLY_WINDOW,
    "A": EARLY_WINDOW,
    "/": EARLY_WINDOW,
}

LATE_OTHERS = {
    "N": EARLY_WINDOW,
    "V": EARLY_WINDOW,
    "R": LATE_WINDOW,
    "L": LATE_WINDOW,
    "A": EARLY_WINDOW,
    "/": LATE_WINDOW,
}


WINDOW_CONFIGURATIONS = {
    "EARLY_NA": EARLY_NA,
    "EARLY_V": EARLY_V,
    "EARLY_OTHERS": EARLY_OTHERS,
    "LATE_NA": LATE_NA,
    "LATE_V": LATE_V,
    "LATE_OTHERS": LATE_OTHERS,
}


# Loaded once in every worker process.
_WORKER_TRAIN_DF: pd.DataFrame | None = None


def initialize_worker(train_path: Path) -> None:
    """
    Load the training dataset once per worker process.

    This avoids sending the complete DataFrame through multiprocessing
    for every individual trial.
    """
    global _WORKER_TRAIN_DF
    _WORKER_TRAIN_DF = pd.read_pickle(train_path)


def build_objective_config(
    match_windows: Mapping[str, tuple[int, int]],
) -> ObjectiveConfig:
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
        match_windows_by_label=dict(match_windows),
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


def render_best_graph(trial_dir: Path) -> tuple[str, str]:
    """
    Render the best saved graph.

    Returns
    -------
    graph_path
        Path to the created PNG file.
    graph_error
        Empty when rendering succeeded; otherwise, the error message.
    """
    output_path = trial_dir / "best_graph.png"

    try:
        visualizer = TrialGraphVisualizer(trial_dir)

        visualizer.plot_best_graph(
            output_path=output_path,
            show=False,
            output_format="png",
            show_params=True,
            show_input_order=True,
            rank_direction="LR",
        )

        return str(output_path), ""

    except Exception as error:
        return str(output_path), repr(error)


def run_single_trial(
    configuration_name: str,
    match_windows: Mapping[str, tuple[int, int]],
    seed: int,
    train_path: Path,
    artifacts_root: Path,
    fs: int,
    n_segments_per_class: int,
    segment_len: int,
) -> dict:
    """
    Run one configuration/seed combination.

    This function is executed inside a worker process.
    """
    if _WORKER_TRAIN_DF is None:
        raise RuntimeError(
            "Training DataFrame was not initialized in the worker."
        )

    configuration_dir = artifacts_root / configuration_name
    run_name = f"trial_{seed:03d}"
    trial_dir = configuration_dir / run_name

    trial_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 100, flush=True)
    print(
        f"RUNNING {configuration_name}/{run_name}",
        flush=True,
    )
    print(f"MATCH WINDOWS: {dict(match_windows)}", flush=True)
    print("=" * 100, flush=True)

    objective_cfg = build_objective_config(match_windows)
    evolution_cfg = build_evolution_config()

    result, _, _ = run_evolutionary_search(
        df=_WORKER_TRAIN_DF,
        fs=fs,
        target_classes=TARGET_CLASSES,
        n_segments_per_class=n_segments_per_class,
        segment_len=segment_len,
        seed=seed,
        out_dir=trial_dir,
        objective_cfg=objective_cfg,
        evolution_cfg=evolution_cfg,
    )

    manifest = {
        "mode": "full_search",
        "configuration": configuration_name,
        "run_name": run_name,
        "seed": seed,
        "training_dataset": str(train_path),
        "target_classes": list(TARGET_CLASSES),
        "match_windows": dict(match_windows),
        "n_segments_per_class": n_segments_per_class,
        "segment_len": segment_len,
        "objective_config": asdict(objective_cfg),
        "evolution_config": asdict(evolution_cfg),
        "best_result": result.best_record,
    }

    manifest_path = trial_dir / "run_manifest.json"

    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2)

    graph_path, graph_error = render_best_graph(trial_dir)

    metrics = result.best_record["overall_metrics"]

    return {
        "status": "completed",
        "configuration": configuration_name,
        "run_name": run_name,
        "seed": seed,
        "artifact_dir": str(trial_dir),
        "manifest_path": str(manifest_path),
        "graph_path": graph_path,
        "graph_error": graph_error,
        "best_f1": metrics["f1"],
        "best_precision": metrics["precision"],
        "best_recall": metrics["recall"],
        "best_objective": result.best_objective,
        "graph_pretty": result.best_record["graph_pretty"],
    }


def save_summaries(
    summary_rows: list[dict],
    artifacts_root: Path,
) -> None:
    if not summary_rows:
        return

    summary = pd.DataFrame(summary_rows)

    existing_sort_columns = [
        column
        for column in ["configuration", "seed"]
        if column in summary.columns
    ]

    if existing_sort_columns:
        summary = summary.sort_values(existing_sort_columns)

    # Summary containing all 18 searches.
    summary.to_csv(
        artifacts_root / "batch_summary.csv",
        index=False,
    )

    # Separate summary inside each configuration folder.
    if "configuration" in summary.columns:
        for configuration_name, configuration_summary in summary.groupby(
            "configuration",
            dropna=False,
        ):
            configuration_dir = artifacts_root / str(configuration_name)
            configuration_dir.mkdir(parents=True, exist_ok=True)

            configuration_summary.to_csv(
                configuration_dir / "batch_summary.csv",
                index=False,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run all six detector matching-window configurations "
            "for multiple seeds."
        )
    )

    parser.add_argument(
        "--train",
        type=Path,
        default=Path("dataset/train.pkl"),
    )
    parser.add_argument(
        "--artifacts",
        type=Path,
        default=Path("Full_search_artifacts"),
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[0, 1, 2],
    )
    parser.add_argument(
        "--fs",
        type=int,
        default=360,
    )
    parser.add_argument(
        "--segments-per-class",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--segment-length",
        type=int,
        default=20_000,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=3,
        help=(
            "Number of searches running simultaneously. "
            "Use 1 to disable parallel execution."
        ),
    )

    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.workers < 1:
        raise ValueError("--workers must be at least 1.")

    if not args.train.exists():
        raise FileNotFoundError(
            f"Training dataset does not exist: {args.train}"
        )

    args.artifacts.mkdir(parents=True, exist_ok=True)

    # Create all six top-level configuration folders immediately.
    for configuration_name in WINDOW_CONFIGURATIONS:
        configuration_dir = args.artifacts / configuration_name
        configuration_dir.mkdir(parents=True, exist_ok=True)

    jobs = [
        (configuration_name, match_windows, seed)
        for configuration_name, match_windows
        in WINDOW_CONFIGURATIONS.items()
        for seed in args.seeds
    ]

    print(f"Configurations: {len(WINDOW_CONFIGURATIONS)}")
    print(f"Seeds: {args.seeds}")
    print(f"Total searches: {len(jobs)}")
    print(f"Parallel workers: {args.workers}")
    print(f"Artifacts: {args.artifacts.resolve()}")

    summary_rows: list[dict] = []

    if args.workers == 1:
        # Sequential execution, useful for debugging and readable logs.
        initialize_worker(args.train)

        for configuration_name, match_windows, seed in jobs:
            try:
                row = run_single_trial(
                    configuration_name=configuration_name,
                    match_windows=match_windows,
                    seed=seed,
                    train_path=args.train,
                    artifacts_root=args.artifacts,
                    fs=args.fs,
                    n_segments_per_class=args.segments_per_class,
                    segment_len=args.segment_length,
                )

            except Exception as error:
                row = {
                    "status": "failed",
                    "configuration": configuration_name,
                    "run_name": f"trial_{seed:03d}",
                    "seed": seed,
                    "error": repr(error),
                }

            summary_rows.append(row)
            save_summaries(summary_rows, args.artifacts)

    else:
        # "spawn" is the safest multiprocessing mode on macOS.
        process_context = mp.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=args.workers,
            mp_context=process_context,
            initializer=initialize_worker,
            initargs=(args.train,),
        ) as executor:
            future_to_job: dict[
                Future,
                tuple[str, int],
            ] = {}

            for configuration_name, match_windows, seed in jobs:
                future = executor.submit(
                    run_single_trial,
                    configuration_name,
                    match_windows,
                    seed,
                    args.train,
                    args.artifacts,
                    args.fs,
                    args.segments_per_class,
                    args.segment_length,
                )

                future_to_job[future] = (
                    configuration_name,
                    seed,
                )

            for future in as_completed(future_to_job):
                configuration_name, seed = future_to_job[future]

                try:
                    row = future.result()

                    print(
                        f"COMPLETED: "
                        f"{configuration_name}/trial_{seed:03d} "
                        f"F1={row['best_f1']:.4f}",
                        flush=True,
                    )

                    if row["graph_error"]:
                        print(
                            f"Graph rendering failed for "
                            f"{configuration_name}/trial_{seed:03d}: "
                            f"{row['graph_error']}",
                            flush=True,
                        )

                except Exception as error:
                    row = {
                        "status": "failed",
                        "configuration": configuration_name,
                        "run_name": f"trial_{seed:03d}",
                        "seed": seed,
                        "error": repr(error),
                    }

                    print(
                        f"FAILED: "
                        f"{configuration_name}/trial_{seed:03d}: "
                        f"{error!r}",
                        flush=True,
                    )

                summary_rows.append(row)

                # Save progress after every finished trial.
                save_summaries(
                    summary_rows,
                    args.artifacts,
                )

    completed = sum(
        row.get("status") == "completed"
        for row in summary_rows
    )
    failed = len(summary_rows) - completed

    print("=" * 100)
    print(f"Finished searches: {completed}")
    print(f"Failed searches: {failed}")
    print(
        "Combined summary:",
        args.artifacts / "batch_summary.csv",
    )


if __name__ == "__main__":
    main()