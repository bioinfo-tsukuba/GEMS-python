# Overview

`tests/test_simulate_one.py` verifies the single-machine simulation features in `gems_python/one_machine_problem_interval_task`. It covers state transitions, task generation, dummy outputs for automated runs, and the behaviour of the scheduler.

# Prerequisites

- Run `poetry install` at the project root to install dependencies including `polars`.
- Execute the script through `poetry run` so that the expected Python interpreter is used.
- `build_experiments()` sets `parent_dir_path="experiments_dir_demo_one_machine"`. Constructing the manager creates this directory. The executable example calls `simulate(max_steps=3, save_each_step=True)` and saves each step; the API default for `simulate()` is `save_each_step=False`.

# How to Run

1. Execute the commands below at the project root:
   ```bash
   poetry install
   poetry run python -m tests.test_simulate_one
   ```
2. OR-Tools is the default scheduler. The script prints manager metadata and scheduling diagnostics. `simulate()` returns per-step dictionaries containing fields such as `status`, `from_state`, and `to_state`; this example does not print those dictionaries.
3. The example already uses `save_each_step=True`. Directories such as `experiments_dir_demo_one_machine/step_00000000` contain CSV schedules and experiment snapshots. Set the flag to `False` to suppress per-step saves.

# Customising States and Tasks

- `InitState`, `MeasureState`, and `FinishState` inherit from `State` and provide `task_generator`, `transition_function`, and `dummy_output`.
- Each `TaskGroup` models work on a single machine; adjust `penalty_type` and `optimal_start_time` to reflect scheduling constraints.
- Ensure `dummy_output` returns the columns expected by `transition_function`, such as `measurement` in `MeasureState`; missing data prevents transitions.

# Logs and Artifacts

- Inspect the dictionaries returned by `simulate()` to confirm transitions from `InitState` to `MeasureState` to `FinishState`. After reaching `FinishState`, the state remains there for subsequent steps.
- When `save_each_step=True`, inspect `schedule.csv` and `experiments.json` under the generated directory to review scheduling results and shared variables.
- Remove unnecessary directories manually, or keep `save_each_step=False` to suppress per-step artifacts. The manager still creates its parent directory.

Running the example with `-m` records its State classes under `__main__`.
A fresh process cannot reload that snapshot unless those classes are available
in its `__main__` namespace. For reusable snapshots, import `build_experiments`
from `tests.test_simulate_one` and call `simulate()` from Python, so the saved
classes have importable module names. See the
[persistence constraints](../docs/persistence.md#constraints-and-deliberate-decisions).

# Common Adjustments

- Shorten the run by lowering `simulate(max_steps=...)`.
- If transitions halt, confirm that `dummy_output` returns a `pl.DataFrame` containing all required columns.
- To experiment with penalties, modify the coefficient in `LinearPenalty` or adjust task intervals to change scheduling pressure.
