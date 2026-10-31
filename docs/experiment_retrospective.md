# Reviewing completed experiments

[README](../README.md#review-completed-experiments) → This guide → [Runnable example](../examples/experiment_retrospective.py)
See also the [lifecycle contract](experiment_lifecycle.md) and [saving and reloading](persistence.md).

## Plot your experiments

Call this on the manager after all Experiments have finished:

```python
lab.render_retrospective("review", time_unit="min")
```

Open `review/review.html` in a browser. Selection works with this HTML file alone.
GitHub displays HTML source; download the file and open it locally.

Use a new output directory. Existing outputs are not overwritten.
If any Experiment is active or on hold, wait until it finishes.

## Try the example

Run from the repository root:

```bash
MPLBACKEND=Agg poetry run python -m scripts.run_experiment_retrospective_example \
  --output /tmp/gems-review-example
```

Open `/tmp/gems-review-example/figures/review.html`.
The [example](../examples/experiment_retrospective.py) runs virtual experiments
that split two parents into children and grandchildren sharing two machines.
It finishes all experiments, saves JSON and pickle snapshots, reloads them,
and generates the plots. Inputs and figures are generated when it runs.

Add `--backend polars` to try Polars. Choose another output directory when
running the example again.

## Plot a saved run

Read the final snapshot without rerunning the experiments:

```bash
MPLBACKEND=Agg poetry run python -m scripts.render_experiment_retrospective \
  --snapshot /path/to/snapshot --output /path/to/new-review
```

`--snapshot` accepts a directory written by `save_snapshot` or a saved step
such as `step_00000000`. Use `--source json` or `--source pickle` to load just
one format. The default `auto` checks consistency between both formats.

The code defining the saved State classes must also be importable.
The example defines its States in `examples.experiment_retrospective`,
so this command can load them in a separate process.

## Read the plots and use selection

| Output | Interpretation |
| --- | --- |
| `overview-01.png` / `.svg` | Overall timeline. Pale bands run from registration to completion; dark bars show Tasks' final reservations. ◆ marks a split; ■ marks completion. Lines connect parents and children. |
| `family-001-01.png` / `.svg` | Lineage and timeline for each family. Each row represents the same Experiment in both panels. Time ranges match the overview. |
| `review.html` | Linked selection in lineage and machine views. Select an experiment name, box, or Task bar. Choose **Ancestors + descendants**, **Family**, or **Selected only**; use **All** to reset. |
| `review-data.json` / `manifest.json` | Rendering records and hashes of the inputs, outputs, and renderer. |

In the HTML, move between shapes with Tab and select with Enter or Space.
The selected scope is highlighted in both views; other experiments remain visible
but dimmed. Hover over a Task for details. Shapes also have screen-reader labels.

Overview and family plots paginate at 24 Experiments per page.
Ancestors spanning pages reappear for context. The HTML displays all Experiments.
Experiments with duplicate names receive display numbers.

## Fonts and time units

For Japanese experiment names, supply a font containing Japanese glyphs:

```python
lab.render_retrospective(
    "review",
    font="/path/to/JapaneseFont.ttf",
    time_unit="min",
)
```

For the CLI, add `--font /path/to/JapaneseFont.ttf --time-unit min`.
Without a supplied font, Matplotlib uses its default font.
SVG exports convert text to paths.

Choose the time unit to match your protocol's logical time.
The default is `min`; this labels the values without converting them.

## What the records describe

Bars show **final reservations at completion**, rather than measured physical
start and end times. The manager's logical time when a result arrives is not
treated as a measured completion time. Reasons for gaps between Tasks are not inferred.

`retrospective_trace` stores the registered name, parent UUID, logical registration
time, and completed Tasks' final reservations and machines.
Plots can be regenerated after Experiments and TaskGroups leave the active lists.
Inherited parent history is not displayed again as work performed by a child.

Missing times or Tasks in legacy saves are not invented. Partial plots are
generated with warnings and `complete: false` to identify gaps.
The full history of schedule changes, reasons for waiting, and physical execution
records are outside the scope. Record size grows with the number of completed Tasks.

## Development and verification

Regression tests are in `tests/test_retrospective.py`.
The [browser verification script](../scripts/verify_experiment_retrospective_browser.py)
checks selection behavior. Install Playwright and Chromium in an optional
verification environment; they are not product dependencies.

Historical plots, saved inputs, and detailed verification records are available
at the [pinned archive commit](https://github.com/bioinfo-tsukuba/GEMS-python/blob/cd78405f3c66061c3b71b72623a83540aaca83fb/docs/design/experiment_retrospective/README.md).
The archived [capacity evaluation after splitting](https://github.com/bioinfo-tsukuba/GEMS-python/blob/8ca9080bf62a4e6c2b70fe73a0f830c5f627b183/docs/benchmarks/experiment_lifecycle_capacity_2026-10-30/README.md)
measured an earlier implementation, before the visualization trace was added;
it does not measure the current trace-enabled implementation.
