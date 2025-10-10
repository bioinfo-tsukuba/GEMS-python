# GEMS Python

GEMS Python is a workflow toolkit for building, simulating, and scheduling
laboratory or shop-floor experiments. It provides reusable state-machine
primitives, task scheduling utilities, and a plugin-driven command-line
interface that help you orchestrate interval-based processes on both
single-machine and multi-machine setups.

## Core Packages

- `gems_python.one_machine_problem_interval_task` — finite-state orchestration
  for a single resource, including simulation helpers and linear penalty models.
- `gems_python.multi_machine_problem_interval_task` — extends the same model to
  multiple machines, adding resource allocation utilities and a plugin manager
  for interactive control.

Detailed API documentation is generated with Sphinx under `docs/`.

## Installation

Use Poetry (recommended):

```bash
poetry install
poetry shell
```

Or install with pip:

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Quick Start Workflow

1. **Simulate a one-machine experiment**
   - Review `tests/test_simulate_one.py` for a minimal example that defines
     three custom states (`InitState`, `MeasureState`, `FinishState`).
   - Run the simulation directly:
     ```bash
     python tests/test_simulate_one.py
     ```
     This builds an `Experiments` collection and executes
     `Experiments.simulate(max_steps=3, save_each_step=True)`, emitting task
     history snapshots to the working directory.

2. **Launch the multi-machine plugin manager**
   - Prepare an experiment directory (for example `examples/multi_machine_demo/`)
     that contains your saved experiment steps and optional `experimental_setting/`
     plugins.
   - Start the interactive loop:
     ```bash
     python main.py
     ```
   - Use the generated `mode/mode.txt` file to switch between automation modes
     (for example `loop`, `add_experiment`, or `delete_experiment`). The
     `PluginManager` automatically scans `experimental_setting/*.py` for new
     experiment generators.

3. **Refine experiment logic**
   - Extend `State` subclasses inside your project or plugin modules. Implement
     `task_generator`, `transition_function`, and optionally override
     `dummy_output` to support offline simulation.
   - Combine `TaskGroup` penalties (linear, cyclical rest, etc.) to fine-tune
     scheduling behaviour.

## Build the Documentation

The repository ships with a Sphinx project that publishes detailed reference
pages for both interval-task packages.

```bash
sphinx-build -b html docs/source docs/build/html
open docs/build/html/index.html  # Windows: start, Linux: xdg-open
```

The `quickstart` section in the generated site mirrors the steps above and links
to API reference material powered by `autodoc` and `napoleon`.

## Detailed Example

The walkthrough below demonstrates how to blend a one-machine simulation with a
multi-machine session.

1. **Create a sandbox experiment**
   - Copy `tests/test_simulate_one.py` to `examples/demo_states.py`.
   - Adjust the state logic and penalties to match your workflow, then run:
     ```bash
     python examples/demo_states.py
     ```
     This seeds a directory such as `experiments_dir_demo_one_machine/` with
     simulation artefacts.

2. **Turn the states into a plugin**
   - Inside `experimental_setting/`, create `demo_plugin.py`:
     ```python
     from examples.demo_states import build_experiment
     
     def demo_plugin():
         return build_experiment()
     ```
   - Restart `main.py` so the `PluginManager` auto-loads the plugin.

3. **Drive the multi-machine workflow**
   - Set `mode/mode.txt` to `add_experiment`, supplying `demo_plugin.demo_plugin`
     via `mode/mode_add_experiment.txt` to register the experiment.
   - Switch the mode to `loop` to let the scheduler advance using the saved
     artefacts.
   - Inspect the regenerated schedule CSV, Gantt chart, and experiment snapshots
     under the step directory reported in the console.

## Defining States and Experiments

Both interval-task packages share the same building blocks: custom `State`
subclasses, `TaskGroup` definitions, and a reusable factory that produces an
`Experiment`.

1. **Implement State subclasses**
   - Inherit from `State` and implement:
     - `task_generator(self, df: pl.DataFrame) -> TaskGroup` — compose a task
       group with penalties, tasks, and scheduling hints.
     - `transition_function(self, df: pl.DataFrame) -> str` — determine the next
       state's name based on the shared variable history.
     - Optionally override `dummy_output` to emit simulated results during
       offline testing.
   - Populate each `TaskGroup` with `Task` instances that declare
     `processing_time`, `interval`, and `experiment_operation`. Select a penalty
     class (for example `LinearPenalty`) that matches your scheduling goals.

2. **Construct the experiment factory**
   - Return an `Experiment` with:
     - `experiment_name`: descriptive label that appears in persistence outputs.
     - `states`: ordered list of the state instances you created.
     - `current_state_name`: initial state's class name.
     - `shared_variable_history`: usually `pl.DataFrame()` when starting fresh.
   - Wrap the logic in helpers such as `build_experiment()` and
     `build_experiments()` so tests, scripts, and plugins can import them.

3. **Bundle into Experiments**
   - Instantiate `Experiments(experiments=[build_experiment()], parent_dir_path=...)`
     to manage persistence and multi-experiment scheduling.
   - Use `simulate(..., save_each_step=True)` during local testing, then call
     `proceed_to_next_step()` to materialise the directory structure expected by
     the plugin manager.

Refer to `tests/test_simulate_one.py` for a comprehensive template that you can
adapt to new workflows.

## Project Layout

- `gems_python/one_machine_problem_interval_task/` — single-machine transitions,
  tasks, and penalties.
- `gems_python/multi_machine_problem_interval_task/` — multi-machine extensions,
  including machine abstractions and CLI utilities.
- `docs/` — Sphinx sources (`index.rst`, `quickstart.rst`, and API stubs).
- `tests/` — sample simulations and regression tests.
- `main.py` — entry point for the multi-machine plugin manager.

## Next Steps

- Adapt the example states in `tests/test_simulate_one.py` to match your
  workflow, then iterate with `Experiments.simulate`.
- Customize `experimental_setting/` plugins to enrol new experiments in the
  multi-machine manager.
- Expand docstrings and rebuild the Sphinx site to keep the published API up to
  date.
