# GEMS

![GEMS logo](assets/logo.png)

> **推奨 / Recommended scheduler: Google OR-Tools (`"o"`, the default).**
> In 168 feasible comparisons across 64 synthetic scheduling patterns,
> OR-Tools had a lower (better) timing-penalty score in 110, tied in 58, and
> had a higher score in 0. See the
> [evaluation conditions and limitations](docs/benchmarks/scheduler_pattern_study_2026-10-28/summary.md).

## Overview
GEMS is a Python toolkit for orchestrating, simulating, and supervising multi-machine laboratory experiments such as cell culture workflows. The project offers two primary entry points:

- A simulation layer that lets you design experiment states, define transitions, and validate dummy outputs before running real experiments.
- An interactive command-line workflow (`gems_python/multi_machine_problem_interval_task/interactive_ui.py`) that manages experiments and machine configurations in real time through simple mode files.

This document describes how to set up the environment, run tests, simulate experiments, and work with the interactive UI.

## Prerequisites
- Python 3.14.4 (the repository's `.python-version`)
- [Poetry](https://python-poetry.org/) for dependency management

## Installation
Install the project dependencies into the Poetry-managed virtual environment:

```bash
poetry install
```

The local package version is **0.1.4**. Poetry uses `pyproject.toml` and
`poetry.lock` as the dependency source of truth.

## Scheduling

For normal use, leave `scheduling_method` unset to use the recommended
OR-Tools scheduler. Select `"s"` explicitly when comparing with the legacy
simulated annealer; in the synthetic evaluation, 24 of its 192 runs had
overlapping tasks assigned to the same physical machine ID.

The three task variants use Google OR-Tools CP-SAT when `scheduling_method` is
omitted or set to `"o"`. Use `"s"` to select simulated annealing explicitly.
The interval-task variants also retain `"f"` for sequential scheduling.
OR-Tools forbids overlapping use of the same machine, minimizes the existing
timing penalties, and searches for up to five seconds per scheduling call. A
feasible schedule may be returned before optimality is proven; the solver
status, objective, and best bound are recorded through Python logging at
`INFO` level. Missing resources or a failure to find a feasible schedule
raise an error.

Cyclical rest is a **group-start penalty**: if the start falls in an inclusive
rest range of its cycle, the configured maximum penalty applies. It does not
forbid a task from running through a rest range. Scheduling times and cycle
parameters use integer minutes in the same absolute time scale.


## Simulating Experiments with Dummy Outputs
Use `Experiments.simulate` or `Experiments.simulate_one` to validate state transitions with mock data before running physical experiments.

1. **Implement `dummy_output` for every state**  
   Each class that inherits from `State` must implement `dummy_output` in addition to its existing `task_generator` and `transition_function`. Return a `polars.DataFrame` whose columns include every field referenced by the transition logic (for example, `confluence`, `measurement`). During simulation, this DataFrame stands in for the experimental measurements.

2. **Assemble `Experiment` and `Experiments` objects**  
   Build an `Experiment` for each protocol, register the required `MachineList`, and hand the experiments to an `Experiments` instance. Set `parent_dir_path` to a writable directory if you plan to persist intermediate results.

3. **Run the simulation**  
   Use `simulate_one` to advance a single step when you only need a quick check, or `simulate(max_steps=<steps>)` for multi-step validation. The return value is a list of dictionaries that capture the state and task metadata for each step. Pass `save_each_step=True` when you need to confirm that the expected step directories are created.

Sample scripts are available under `tests/`:

```bash
# Minimal three-state dummy experiment
poetry run python tests/test_simulate.py

# HEK cell growth from Passage to Observation
poetry run python tests/HEK_two_state_growth_sim.py

# Run every unit test in the suite
poetry run python -m unittest discover -s tests -p "test_*.py"
```

Inspect the logged transitions in the console output and adjust `dummy_output` values or threshold parameters until the simulated behavior matches expectations.

Reusable, deterministic examples for cell passage/observation and synthetic
reaction-condition search live in `tests/simulators/`. They inject observations
through the optional `observation_provider` argument to `simulate_one` and
`simulate`; the provider receives `(state, history, task_group, task,
event_time)` and returns a `polars.DataFrame`. `event_time` is the scheduled
completion time in integer minutes, and the scheduler advances its reference
time to that value. When tasks run concurrently, the provider receives
results in completion order. Calls without a provider continue to use
`dummy_output`.
See [the test plan](docs/test_plan.md) for commands, assumptions, and the
research used to select these test scenarios.

## Interactive UI Workflow
The interactive UI loads experiments and machine definitions from files under your working directory. Follow the steps below to manage experiments with `interactive_ui.py`.

### 1. Prepare the Project Layout
Ensure the following directory structure exists:

```
sample_experiment/
├── experimental_setting/
│   └── sample_setting.py
└── mode/
    └── mode.txt
```

### 2. Implement Experiment Modules
Define at least one experiment state class in `sample_setting.py` (or another module of your choice). A minimal structure looks like this:

```python
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, Experiments, State

class StandardState1(State):
    def task_generator(self, df):
        # Task generation logic goes here
        ...

    def transition_function(self, df):
        return "StandardState1"

def gen_sample_experiment(experiment_name="sample_experiment"):
    return Experiment(
        experiment_name=experiment_name,
        states=[StandardState1()],
        current_state_name="StandardState1",
        shared_variable_history=pl.DataFrame({
            "time": [0],
            "temperature": [0],
            "pressure": [0],
        }),
    )
```

### 3. Configure the Active Mode
`mode/mode.txt` determines which operation the UI performs. For example:

```
add_experiments
```

### 4. Add Experiments
Create `mode/mode_add_experiments.txt` with one callable per line:

```
sample_setting.gen_sample_experiment
```

When you run the UI, it loads the experiments and deletes the file after a successful import.

### 5. Run the Interactive UI
Launch the UI from the project root:

```bash
python gems_python/multi_machine_problem_interval_task/interactive_ui.py
```

### 6. Available Modes
Update `mode/mode.txt` with any of the following values to control the system:

- `module_load`: Reload all plugins.
- `add_experiment`: Add one experiment.
- `delete_experiment`: Remove an experiment by UUID.
- `show_experiments`: List loaded experiments.
- `add_machines`: Register machines.
- `delete_machines`: Remove machines by ID.
- `proceed`: Advance to the next experiment step.
- `stop`: Pause execution.
- `exit`: Terminate the program.

### 7. Manage Machines
- **Add machines**: create `mode/mode_add_machines.txt` with comma-separated ID and name pairs:
  ```
  0,RobotA
  1,Human
  ```
- **Delete machines**: create `mode/mode_delete_machines.txt` and list one machine ID per line:
  ```
  0
  1
  ```

### 8. Discover Modes
Set `mode/mode.txt` to `help` to print the list of available modes:

```
help
```

### 9. Advance or Stop Execution
- Write `proceed` to `mode/mode.txt` to move to the next state.
- Write `stop` to pause the workflow.
- Write `exit` or `eof` to terminate the UI.
