# Overview
`tests/test_simulate_multi.py` demonstrates how to use the simulation utilities in `gems_python/multi_machine_problem_interval_task` for multi-machine workflows. The script showcases state transitions, task generation, dummy outputs for automated progression, and scheduling validation.

# Prerequisites
- Run `poetry install` at the project root to prepare dependencies such as `polars`.
- Always execute the script via `poetry run` so that the expected Python environment is used.
- `build_experiments()` sets `parent_dir_path="experiments_dir_demo"`. Constructing the manager creates this directory. The executable example calls `simulate(max_steps=10, save_each_step=True)`, saving each step and printing its returned summary. The API default for `simulate()` is `save_each_step=False`.

# How to Run
1. Execute the following commands at the project root:
   ```bash
   poetry install
   poetry run python -m tests.test_simulate_multi
   ```
2. OR-Tools is the default scheduler. The console prints per-step transition results (fields such as `status`, `from_state`, `to_state`).
3. When `save_each_step` remains `True`, directories like `experiments_dir_demo/step_00000000` are created automatically with CSV schedules, JSON experiment data, and generated graphs. Adjust the output path by editing `parent_dir_path`.

# Customising States and Tasks
- Define `InitState`, `MeasureState`, and `FinishState`. Each state inherits from `State` and implements `task_generator`, `transition_function`, and `dummy_output`.
- `task_generator` must set `TaskGroup` fields consistent with the registered machines in `MachineList`. The sample registers two machines (`machine_type=0` and `machine_type=1`).
- Ensure that `dummy_output` mimics actual experiment results; missing columns or unexpected values will block correct state transitions.

# Logs and Artifacts
- Console lines like `[step N] {...}` report each transition; verify progression and terminal behaviour.
- Saved directories contain `schedule.csv`, `experiments.json`, and graphs under `experiments/`. Review them to inspect scheduling quality and state graphs.
- Delete unused directories manually, or run with `save_each_step=False` to suppress per-step saves. The manager still creates its parent directory.

# Common Adjustments
- Reduce simulation length by lowering `simulate(max_steps=...)`.
- If transitions stall, confirm that `dummy_output` returns a `pl.DataFrame` with all required columns.
- Setting `Task.optimal_machine_type` to a machine not registered in `MachineList` causes scheduling errors; align machine registration with task definitions.
