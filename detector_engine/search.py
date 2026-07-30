from __future__ import annotations

"""Evolutionary search loop and artifact persistence."""

import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

import numpy as np
import pandas as pd

from .evaluation import ObjectiveConfig, TrialMetrics, evaluate_graph_on_segments
from .graph import AlgorithmGraph, _sanitize_for_json, graph_from_dict
from .mutation import (
    EvolutionConfig,
    graph_fingerprint,
    mutate_graph,
    sample_identity_population,
    sample_population_from_graph,
    sample_seed_population,
)
from .sampling import SegmentDataset, sample_segments_with_sampler


class EvolutionRecorder:
    def __init__(self) -> None:
        self.records: List[Dict[str, Any]] = []

    def add(
        self,
        eval_id: int,
        dataset_stage: int,
        generation_in_stage: int,
        global_generation: int,
        dataset_seed: int,
        parent_id: Optional[int],
        mutation_type: str,
        graph: AlgorithmGraph,
        metrics: TrialMetrics,
    ) -> None:
        record = {
            "eval_id": int(eval_id),
            "dataset_stage": int(dataset_stage),
            "generation_in_stage": int(generation_in_stage),
            "global_generation": int(global_generation),
            "dataset_seed": int(dataset_seed),
            "parent_id": None if parent_id is None else int(parent_id),
            "mutation_type": str(mutation_type),
            "objective": float(metrics.objective),
            "graph": graph.to_dict(),
            "graph_pretty": graph.pretty(),
            "overall_metrics": metrics.to_dict(),
        }
        self.records.append(record)

    def to_frame(self) -> pd.DataFrame:
        rows: List[Dict[str, Any]] = []
        for record in self.records:
            row = {
                key: value
                for key, value in record.items()
                if key not in {"graph", "overall_metrics"}
            }
            row.update(
                {
                    f"overall__{key}": value
                    for key, value in record["overall_metrics"].items()
                }
            )
            rows.append(row)

        if not rows:
            return pd.DataFrame()

        # Different dataset seeds are not directly comparable, so preserve
        # chronological generation order and rank only within that context.
        return (
            pd.DataFrame(rows)
            .sort_values(
                ["global_generation", "dataset_seed", "objective", "eval_id"],
                ascending=[True, True, True, True],
            )
            .reset_index(drop=True)
        )

    def save(self, out_dir: str | Path) -> None:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)

        frame = self.to_frame()
        frame.to_csv(output / "evaluation_summary.csv", index=False)
        frame.to_pickle(output / "evaluation_summary.pkl")

        with open(output / "evaluation_records.json", "w", encoding="utf-8") as file:
            json.dump(_sanitize_for_json(self.records), file, indent=2)


@dataclass
class EvolutionResult:
    best_graph: AlgorithmGraph
    best_objective: float
    best_record: Dict[str, Any]
    recorder: EvolutionRecorder
    dataset: SegmentDataset


def _build_dataset_for_generation(
    df: pd.DataFrame,
    target_classes: Sequence[str],
    n_segments_per_class: int,
    segment_len: int,
    seed: int,
) -> SegmentDataset:
    return sample_segments_with_sampler(
        df=df,
        target_classes=target_classes,
        n_segments_per_class=n_segments_per_class,
        segment_len=segment_len,
        seed=seed,
    )


def _print_generation_summary(
    stage_idx: int,
    gen_in_stage: int,
    global_generation: int,
    dataset_seed: int,
    best_metrics: TrialMetrics,
    population_size: int,
    n_children: int,
    cache_size: int,
) -> None:
    print(
        f"[stage {stage_idx}] "
        f"[gen_in_stage {gen_in_stage}] "
        f"[global_gen {global_generation}] "
        f"[dataset_seed {dataset_seed}] "
        f"f1={best_metrics.f1:.4f} "
        f"recall={best_metrics.recall:.4f} "
        f"precision={best_metrics.precision:.4f} "
        f"mae={best_metrics.mae:.2f} "
        f"objective={best_metrics.objective:.4f} "
        f"population={population_size} "
        f"children={n_children} "
        f"cache={cache_size}"
    )


def _record_to_json(record: Dict[str, Any]) -> Dict[str, Any]:
    graph: AlgorithmGraph = record["graph"]
    metrics: TrialMetrics = record["metrics"]

    return {
        "eval_id": int(record["eval_id"]),
        "dataset_stage": int(record["dataset_stage"]),
        "generation_in_stage": int(record["generation_in_stage"]),
        "global_generation": int(record["global_generation"]),
        "dataset_seed": int(record["dataset_seed"]),
        "parent_id": record["parent_id"],
        "mutation_type": str(record["mutation_type"]),
        "objective": float(metrics.objective),
        "graph": graph.to_dict(),
        "graph_pretty": graph.pretty(),
        "overall_metrics": metrics.to_dict(),
    }


def _save_progress_snapshot(
    out_dir: Path,
    recorder: EvolutionRecorder,
    best_record: Optional[Dict[str, Any]],
    evo_cfg: EvolutionConfig,
    obj_cfg: ObjectiveConfig,
    last_dataset: Optional[SegmentDataset],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    recorder.save(out_dir)

    if best_record is not None:
        with open(out_dir / "best_graph.json", "w", encoding="utf-8") as file:
            json.dump(_sanitize_for_json(_record_to_json(best_record)), file, indent=2)

    if last_dataset is not None:
        with open(out_dir / "sampled_segments.json", "w", encoding="utf-8") as file:
            json.dump(
                _sanitize_for_json([asdict(segment) for segment in last_dataset.segments]),
                file,
                indent=2,
            )

    with open(out_dir / "evolution_config.json", "w", encoding="utf-8") as file:
        json.dump(_sanitize_for_json(asdict(evo_cfg)), file, indent=2)

    with open(out_dir / "objective_config.json", "w", encoding="utf-8") as file:
        json.dump(_sanitize_for_json(asdict(obj_cfg)), file, indent=2)


def _select_unique_records(
    records: Sequence[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """Return the best records while keeping one record per graph."""
    selected: List[Dict[str, Any]] = []
    fingerprints: Set[str] = set()

    for record in sorted(records, key=lambda item: item["objective"]):
        fingerprint = graph_fingerprint(record["graph"])
        if fingerprint in fingerprints:
            continue

        selected.append(record)
        fingerprints.add(fingerprint)

        if len(selected) >= limit:
            break

    return selected


def run_evolutionary_search(
    df: pd.DataFrame,
    fs: int = 360,
    target_classes: Sequence[str] = ("N",),
    n_segments_per_class: int = 30,
    segment_len: int = 10_000,
    seed: int = 0,
    out_dir: Optional[str | Path] = None,
    objective_cfg: Optional[ObjectiveConfig] = None,
    evolution_cfg: Optional[EvolutionConfig] = None,
    initial_graph: Optional[AlgorithmGraph] = None,
) -> Tuple[EvolutionResult, EvolutionRecorder, SegmentDataset]:
    rng = np.random.default_rng(seed)
    random.seed(seed)

    obj_cfg = objective_cfg or ObjectiveConfig()
    evo_cfg = evolution_cfg or EvolutionConfig()
    recorder = EvolutionRecorder()

    if evo_cfg.population_size < 1:
        raise ValueError("population_size must be at least 1")
    if evo_cfg.n_elite < 1:
        raise ValueError("n_elite must be at least 1")
    if evo_cfg.children_per_parent < 1:
        raise ValueError("children_per_parent must be at least 1")
    if evo_cfg.n_dataset_stages < 1:
        raise ValueError("n_dataset_stages must be at least 1")
    if evo_cfg.max_generations_per_stage < 0:
        raise ValueError("max_generations_per_stage cannot be negative")

    # A result is reusable only for the same graph on the same sampled dataset.
    evaluation_cache: Dict[Tuple[int, str], TrialMetrics] = {}

    def evaluate_cached(
        graph: AlgorithmGraph,
        dataset: SegmentDataset,
        dataset_seed: int,
    ) -> TrialMetrics:
        cache_key = (int(dataset_seed), graph_fingerprint(graph))
        metrics = evaluation_cache.get(cache_key)

        if metrics is None:
            result = evaluate_graph_on_segments(graph, dataset, obj_cfg)
            metrics = result["metrics"]
            evaluation_cache[cache_key] = metrics

        return metrics

    def evaluate_population(
        population: Sequence[Dict[str, Any]],
        dataset: SegmentDataset,
        dataset_seed: int,
        stage_idx: int,
        generation_in_stage: int,
        global_generation: int,
    ) -> List[Dict[str, Any]]:
        """Reevaluate every survivor on the current sampled dataset."""
        reevaluated: List[Dict[str, Any]] = []

        for record in population:
            metrics = evaluate_cached(
                graph=record["graph"],
                dataset=dataset,
                dataset_seed=dataset_seed,
            )

            updated = dict(record)
            updated.update(
                {
                    "dataset_stage": int(stage_idx),
                    "generation_in_stage": int(generation_in_stage),
                    "global_generation": int(global_generation),
                    "dataset_seed": int(dataset_seed),
                    "metrics": metrics,
                    "objective": float(metrics.objective),
                }
            )
            reevaluated.append(updated)

        return reevaluated

    def make_stage_dataset(stage_idx: int) -> Tuple[int, SegmentDataset]:
        dataset_seed = seed + stage_idx * evo_cfg.dataset_seed_stride
        dataset = _build_dataset_for_generation(
            df=df,
            target_classes=target_classes,
            n_segments_per_class=n_segments_per_class,
            segment_len=segment_len,
            seed=dataset_seed,
        )
        return dataset_seed, dataset

    if initial_graph is not None:
        initial_graph.validate()
        population_graphs = sample_population_from_graph(
            graph=initial_graph,
            population_size=evo_cfg.population_size,
            rng=rng,
            jitter=evo_cfg.random_seed_jitter,
        )
    elif evo_cfg.force_identity_only_seed:
        population_graphs = sample_identity_population(
            population_size=evo_cfg.population_size,
        )
    else:
        population_graphs = sample_seed_population(
            fs=fs,
            population_size=evo_cfg.population_size,
            rng=rng,
            jitter=evo_cfg.random_seed_jitter,
        )

    eval_counter = 0
    global_generation = 0
    current_population: List[Dict[str, Any]] = []
    current_best: Optional[Dict[str, Any]] = None
    last_dataset: Optional[SegmentDataset] = None

    # Prevent regenerating any graph that has already occurred in this run.
    seen_graph_fingerprints: Set[str] = {
        graph_fingerprint(graph)
        for graph in population_graphs
    }

    stop_search = False

    for stage_idx in range(evo_cfg.n_dataset_stages):
        success_streak = 0
        dataset_seed, dataset = make_stage_dataset(stage_idx)
        last_dataset = dataset

        if stage_idx == 0:
            seeded_population: List[Dict[str, Any]] = []

            for graph in population_graphs:
                metrics = evaluate_cached(graph, dataset, dataset_seed)
                record = {
                    "eval_id": eval_counter,
                    "dataset_stage": stage_idx,
                    "generation_in_stage": 0,
                    "global_generation": global_generation,
                    "dataset_seed": dataset_seed,
                    "parent_id": None,
                    "mutation_type": "seed",
                    "graph": graph,
                    "metrics": metrics,
                    "objective": float(metrics.objective),
                }
                seeded_population.append(record)

                recorder.add(
                    eval_id=eval_counter,
                    dataset_stage=stage_idx,
                    generation_in_stage=0,
                    global_generation=global_generation,
                    dataset_seed=dataset_seed,
                    parent_id=None,
                    mutation_type="seed",
                    graph=graph,
                    metrics=metrics,
                )
                eval_counter += 1

            current_population = _select_unique_records(
                seeded_population,
                limit=evo_cfg.population_size,
            )
        else:
            current_population = evaluate_population(
                population=current_population,
                dataset=dataset,
                dataset_seed=dataset_seed,
                stage_idx=stage_idx,
                generation_in_stage=0,
                global_generation=global_generation,
            )
            current_population = _select_unique_records(
                current_population,
                limit=evo_cfg.population_size,
            )

        if not current_population:
            raise RuntimeError("the population is empty")

        current_best = min(current_population, key=lambda item: item["objective"])

        if evo_cfg.verbose:
            _print_generation_summary(
                stage_idx=stage_idx,
                gen_in_stage=0,
                global_generation=global_generation,
                dataset_seed=dataset_seed,
                best_metrics=current_best["metrics"],
                population_size=len(current_population),
                n_children=0,
                cache_size=len(evaluation_cache),
            )

        if out_dir is not None:
            _save_progress_snapshot(
                out_dir=Path(out_dir),
                recorder=recorder,
                best_record=current_best,
                evo_cfg=evo_cfg,
                obj_cfg=obj_cfg,
                last_dataset=last_dataset,
            )

        if current_best["metrics"].f1 >= evo_cfg.stage_success_f1:
            success_streak = 1

        if success_streak >= evo_cfg.success_consecutive_generations:
            if evo_cfg.verbose:
                print(
                    f"Search success at stage {stage_idx} before mutation "
                    f"with f1={current_best['metrics'].f1:.4f}"
                )
            if stage_idx == evo_cfg.n_dataset_stages - 1:
                stop_search = True
                break
            continue

        for generation_in_stage in range(
            1,
            evo_cfg.max_generations_per_stage + 1,
        ):
            global_generation += 1

            if evo_cfg.resample_dataset_each_generation:
                generation_dataset_seed = (
                    seed
                    + stage_idx * evo_cfg.dataset_seed_stride
                    + generation_in_stage
                )
                dataset = _build_dataset_for_generation(
                    df=df,
                    target_classes=target_classes,
                    n_segments_per_class=n_segments_per_class,
                    segment_len=segment_len,
                    seed=generation_dataset_seed,
                )
                last_dataset = dataset

                # Critical fix: all survivors are evaluated on the new dataset
                # before parent selection and before comparison with children.
                current_population = evaluate_population(
                    population=current_population,
                    dataset=dataset,
                    dataset_seed=generation_dataset_seed,
                    stage_idx=stage_idx,
                    generation_in_stage=generation_in_stage,
                    global_generation=global_generation,
                )
            else:
                generation_dataset_seed = dataset_seed

            current_population = _select_unique_records(
                current_population,
                limit=evo_cfg.population_size,
            )
            parents = _select_unique_records(
                current_population,
                limit=min(evo_cfg.n_elite, len(current_population)),
            )

            children: List[Dict[str, Any]] = []

            for parent in parents:
                for _ in range(evo_cfg.children_per_parent):
                    try:
                        child_graph, mutation_type = mutate_graph(
                            graph=parent["graph"],
                            rng=rng,
                            evo_cfg=evo_cfg,
                            forbidden_fingerprints=seen_graph_fingerprints,
                        )
                    except RuntimeError as error:
                        if evo_cfg.verbose:
                            print(
                                "Could not generate another unique child for "
                                f"parent {parent['eval_id']}: {error}"
                            )
                        break

                    child_fingerprint = graph_fingerprint(child_graph)
                    seen_graph_fingerprints.add(child_fingerprint)

                    metrics = evaluate_cached(
                        graph=child_graph,
                        dataset=dataset,
                        dataset_seed=generation_dataset_seed,
                    )

                    child_record = {
                        "eval_id": eval_counter,
                        "dataset_stage": stage_idx,
                        "generation_in_stage": generation_in_stage,
                        "global_generation": global_generation,
                        "dataset_seed": generation_dataset_seed,
                        "parent_id": parent["eval_id"],
                        "mutation_type": mutation_type,
                        "graph": child_graph,
                        "metrics": metrics,
                        "objective": float(metrics.objective),
                    }
                    children.append(child_record)
                    eval_counter += 1

            if evo_cfg.save_top_k_children_per_generation is None:
                children_to_record = children
            else:
                top_k = max(0, int(evo_cfg.save_top_k_children_per_generation))
                children_to_record = sorted(
                    children,
                    key=lambda item: item["objective"],
                )[:top_k]

            for record in children_to_record:
                recorder.add(
                    eval_id=record["eval_id"],
                    dataset_stage=record["dataset_stage"],
                    generation_in_stage=record["generation_in_stage"],
                    global_generation=record["global_generation"],
                    dataset_seed=record["dataset_seed"],
                    parent_id=record["parent_id"],
                    mutation_type=record["mutation_type"],
                    graph=record["graph"],
                    metrics=record["metrics"],
                )

            current_population = _select_unique_records(
                current_population + children,
                limit=evo_cfg.population_size,
            )

            if not current_population:
                raise RuntimeError("the population became empty")

            # This is the best graph on the current dataset. Historical scores
            # from other dataset seeds are deliberately not compared with it.
            current_best = current_population[0]

            if evo_cfg.verbose:
                _print_generation_summary(
                    stage_idx=stage_idx,
                    gen_in_stage=generation_in_stage,
                    global_generation=global_generation,
                    dataset_seed=generation_dataset_seed,
                    best_metrics=current_best["metrics"],
                    population_size=len(current_population),
                    n_children=len(children),
                    cache_size=len(evaluation_cache),
                )

            if out_dir is not None:
                _save_progress_snapshot(
                    out_dir=Path(out_dir),
                    recorder=recorder,
                    best_record=current_best,
                    evo_cfg=evo_cfg,
                    obj_cfg=obj_cfg,
                    last_dataset=last_dataset,
                )

            if current_best["metrics"].f1 >= evo_cfg.stage_success_f1:
                success_streak += 1
            else:
                success_streak = 0

            if success_streak >= evo_cfg.success_consecutive_generations:
                if evo_cfg.verbose:
                    print(
                        f"Search success at stage {stage_idx}, "
                        f"generation {generation_in_stage} after "
                        f"{success_streak} consecutive successful datasets "
                        f"with f1={current_best['metrics'].f1:.4f}"
                    )
                break
        else:
            if evo_cfg.verbose and current_best is not None:
                print(
                    f"Stage {stage_idx} timeout after "
                    f"{evo_cfg.max_generations_per_stage} generations "
                    f"with best current-dataset f1="
                    f"{current_best['metrics'].f1:.4f}"
                )

        if stage_idx == evo_cfg.n_dataset_stages - 1:
            stop_search = True

        if stop_search:
            break

    if current_best is None or last_dataset is None:
        raise RuntimeError("no evaluations were performed")

    best_graph: AlgorithmGraph = current_best["graph"]
    best_metrics: TrialMetrics = current_best["metrics"]
    best_record = _record_to_json(current_best)

    result = EvolutionResult(
        best_graph=best_graph,
        best_objective=float(best_metrics.objective),
        best_record=best_record,
        recorder=recorder,
        dataset=last_dataset,
    )

    if out_dir is not None:
        output = Path(out_dir)
        output.mkdir(parents=True, exist_ok=True)
        recorder.save(output)

        with open(output / "best_graph.json", "w", encoding="utf-8") as file:
            json.dump(_sanitize_for_json(best_record), file, indent=2)

        with open(output / "sampled_segments.json", "w", encoding="utf-8") as file:
            json.dump(
                _sanitize_for_json([asdict(segment) for segment in last_dataset.segments]),
                file,
                indent=2,
            )

        with open(output / "evolution_config.json", "w", encoding="utf-8") as file:
            json.dump(_sanitize_for_json(asdict(evo_cfg)), file, indent=2)

        with open(output / "objective_config.json", "w", encoding="utf-8") as file:
            json.dump(_sanitize_for_json(asdict(obj_cfg)), file, indent=2)

    return result, recorder, last_dataset


def load_best_graph(
    artifacts_path: str | Path,
) -> Tuple[Dict[str, Any], AlgorithmGraph]:
    artifacts_path = Path(artifacts_path)
    with open(artifacts_path / "best_graph.json", "r", encoding="utf-8") as file:
        record = json.load(file)

    graph = graph_from_dict(record["graph"])
    return record, graph
