# Detector evolution engine

This project evolves signal-processing graphs that produce a score signal, detects peaks in that score, and evaluates the detections against annotated events.

## Structure

```text
detector_engine/
├── graph.py       # primitives, graph model, execution and serialization
├── sampling.py    # signal-window sampling and sampled datasets
├── evaluation.py  # peak detection, matching and objective calculation
├── mutation.py    # graph mutations and evolution configuration
├── search.py      # evolutionary loop, recording and artifact loading
└── __init__.py    # public API

run_base_detector_search.py  # train detectors from identity graphs
fine_tune_detector.py        # continue search from a saved graph
requirements.txt
documentation.md
```

## Dataset format

Training and validation files are pandas pickles containing one row per signal record.

Required columns:

| Column | Content |
|---|---|
| `record` | record identifier |
| `signal` | one-dimensional numeric signal |
| `timestamps` | integer sample positions of annotations |
| `annotations` | labels corresponding to `timestamps` |

For every row, `timestamps` and `annotations` must have equal lengths.

## Installation

```bash
python -m pip install -r requirements.txt
```

Run commands from the project root so that `detector_engine` is importable.

## Base detector search

Default command:

```bash
python run_base_detector_search.py
```

Equivalent explicit command:

```bash
python run_base_detector_search.py \
  --train dataset/train.pkl \
  --artifacts artifacts4 \
  --seeds 0 1 2 3 4 5 6 7 8 9 10 11 12 13 14
```

Important options:

```text
--fs                    sampling frequency
--segments-per-class    sampled windows per annotation class
--segment-length        number of samples in each window
--seeds                 independent trial seeds
```

Detector and evolution settings are defined in `build_objective_config()` and `build_evolution_config()` inside the script.

## Fine-tuning

Fine-tuning loads `best_graph.json` from a previous artifact directory and uses that graph as the initial population.

```bash
python fine_tune_detector.py \
  --validation dataset/val.pkl \
  --pretrained artifacts4/trial_005 \
  --output artifacts4/finetune_trial_005
```

Class-specific accepted detection delays are defined by `MATCH_WINDOWS` in `fine_tune_detector.py`.

## Main output files

Each trial directory contains:

| File | Content |
|---|---|
| `best_graph.json` | best graph and its metrics |
| `evaluation_summary.csv` | recorded evaluations in tabular form |
| `evaluation_summary.pkl` | pandas version of the summary |
| `evaluation_records.json` | recorded graph definitions and metrics |
| `sampled_segments.json` | metadata of the last sampled dataset |
| `objective_config.json` | objective settings |
| `evolution_config.json` | evolution settings |
| `run_manifest.json` or `finetune_manifest.json` | complete run description |

The base-search script also writes `batch_summary.csv` in the artifact root.

## Python API

```python
from detector_engine import (
    EvolutionConfig,
    ObjectiveConfig,
    load_best_graph,
    run_evolutionary_search,
)
```

`best_graph.json` keeps the original graph format, so previously saved detector graphs can be loaded by the refactored engine.

## Graph visualization

Plot the best graph from a saved trial:

```bash
python plot_trial_graph.py artifacts4/trial_005
```

The image is saved as `artifacts4/trial_005/best_graph.png`.
For a non-interactive environment, add `--no-show`.

Python usage:

```python
from graph_visualization import TrialGraphVisualizer

visualizer = TrialGraphVisualizer("artifacts4/trial_005")
visualizer.plot_best_graph(output_path="best_graph.png")
```

Binary-operation edges are labeled `0` and `1` to preserve operand order.
