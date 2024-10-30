This readme file is automatically generated. 
Note that the content of this file will be changed in the future.

# Plugin Manager

This project provides a **Plugin Manager** for dynamically loading and reloading Python plugins, integrating with experiments defined in the `gems_python` library. The manager also offers a command-line interface (CLI) to interact with the experiments.

## Features

- **Plugin auto-loader**: Monitors a directory for `.py` files and loads or reloads plugins on file changes.
- **Experiment management**: Add, delete, and display experiments via CLI.
- **Command-line interface**: Built using `cmd2`, providing interactive control over experiments.
- **Auto-load timer**: Automatically loads experiments if no commands are received within a specified time.

## Installation

1. Ensure you have `Python 3.x` installed.
2. Install required dependencies by running:

   ```bash
   pip install cmd2 watchdog
   ```

3. Clone or download the repository and place it in your working directory.

## Usage

1. Start the Plugin Manager by running:

   ```bash
   python gems_python/one_machine_problem_interval_task/interactive_ui.py"
   ```

2. On startup, the manager will ask if you want to reload experiments.

3. Use the available commands within the CLI:

   - `add <module.class>`: Add an experiment.
   - `delete <experiment_uuid>`: Delete an experiment by UUID.
   - `show`: Display all loaded experiments.
   - `cmdlist`: Show all available commands.
   - `stop`: Disable auto-load.
   - `reloop`: Enable auto-load.
   - `reload <step>`: Reload experiments to a specific step.

## Directory Structure

- **`experimental_setting/`**: Directory where plugins are stored.
- **`gems_python/`**: Contains the experiment management logic.

## How it Works

- **Plugin Auto-Loader**: Uses `watchdog` to monitor for `.py` file changes.
- **Experiment Management**: Experiment objects are dynamically added to the system using their module and class names.

## Contributing

Feel free to submit issues or pull requests to improve the Plugin Manager.

## License

This project is licensed under the MIT License.

