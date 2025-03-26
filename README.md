# What is GEMS?

# How to Use

This guide outlines the necessary steps to define and run a cell culture experiment simulation using Rust. It focuses on setting up the experiment without providing direct code examples.

## Prerequisites

- poetry
- python >= 3.10

## setup

```shell
poetry install
```

## Test

```shell
python -m unittest
```

## Steps to Define an Experiment

To define an experiment using the `interactive_ui.py` script with the updated directory and settings, follow these steps:

### 1. **Set Up the Project Directory**
- Ensure that your working directory has the following structure:
  ```
  sample_experiment/
  ├── experimental_setting/
  │   └── sample_setting.py
  └── mode/
      └── mode.txt
  ```

### 2. **Create Experiment Modules**
- The `ot2_setting.py` file should define at least one class representing an experiment and its state.
- Example structure in `ot2_setting.py`:
  ```python
  from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, Experiments, State

  class StandardState1(State):
      def task_generator(self, df):
          # Task generation logic here
          pass

      def transition_function(self, df):
          return "StandardState1"

  def gen_sample_experiment(experiment_name="sample_experiment"):
      return Experiment(
          experiment_name=experiment_name,
          states=[StandardState1()],
          current_state_name="StandardState1",
          shared_variable_history=
            pl.DataFrame({
                "time": [0],
                "temperature": [0],
                "pressure": [0]
            })
      )
  ```

### 3. **Configure Mode Settings**
- The `mode/mode.txt` file determines the current operational mode.
- Example `mode.txt`:
  ```
  add_experiments
  ```

### 4. **Add Experiments**
- Create a file named `mode_add_experiments.txt` in the `mode/` directory:
  ```
  sample_setting.gen_sample_experiment
  ```
- The script will automatically read this file and add the experiment. The file is deleted after successful loading.

### 5. **Run the Script**
- From your project root directory, run the script:
  ```bash
  python gems_python/multi_machine_problem_interval_task/interactive_ui.py
  ```

### 6. **Manage Modes**
- Available modes:
  - `module_load`: Reload all plugins.
  - `add_experiment`: Add a single experiment.
  - `delete_experiment`: Delete an experiment by UUID.
  - `show_experiments`: List all loaded experiments.
  - `add_machines`: Add machine configurations.
  - `delete_machines`: Delete specified machines.
  - `proceed`: Move to the next experiment step.
  - `stop`: Pause execution.
  - `exit`: Stop the program.

### 7. **Adding Machines**
- Create `mode_add_machines.txt` in the `mode/` directory:
  ```
  0,OT-2
  1,Human
  ```

### 8. **Deleting Machines**
- Specify machine IDs in `mode_delete_machines.txt`:
  ```
  0
  1
  ```

### 9. **Checking Available Modes**
- Set `mode.txt` to `help` to display all available modes:
  ```
  help
  ```

### 10. **Proceeding Through Experiments**
- Update `mode.txt` to `proceed` to move to the next state:
  ```
  proceed
  ```

### 11. **Stopping and Exiting**
- To pause execution, write `stop` in `mode.txt`.
- To terminate, write `exit` or `eof` in `mode.txt`.

---

By following these steps with the given directory and experiment settings, you can successfully define, manage, and execute OT-2 experiments using the provided interactive UI.
