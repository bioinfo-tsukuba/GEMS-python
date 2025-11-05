# GEMS

## Overview
GEMS is a Python toolkit for orchestrating, simulating, and supervising multi-machine laboratory experiments such as cell culture workflows. The project offers two primary entry points:

- A simulation layer that lets you design experiment states, define transitions, and validate dummy outputs before running real experiments.
- An interactive command-line workflow (`gems_python/multi_machine_problem_interval_task/interactive_ui.py`) that manages experiments and machine configurations in real time through simple mode files.

This document describes how to set up the environment, run tests, simulate experiments, and work with the interactive UI.

## Prerequisites
- Python 3.10 or later
- [Poetry](https://python-poetry.org/) for dependency management

## Installation
Install the project dependencies into the Poetry-managed virtual environment:

```bash
poetry install
```


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
