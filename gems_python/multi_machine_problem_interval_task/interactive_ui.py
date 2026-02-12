from datetime import datetime
import time
import os
from pathlib import Path
import sys
from importlib import import_module, reload
import inspect

from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments

class PluginManager:
    def __init__(self, experiments: Experiments, module_path: Path = "experimental_setting/", mode_path: Path = "mode"):
        self.plugins = {}
        self.experiments = experiments
        self.module_path = self.experiments.parent_dir_path / module_path
        # If the directory does not exist, create it.
        if not os.path.exists(self.module_path):
            os.makedirs(self.module_path, exist_ok=True)
            print(f"Module directory {self.module_path} created.")
        self.mode_path = self.experiments.parent_dir_path / mode_path
        # If the directory does not exist, create it.
        if not os.path.exists(self.mode_path):
            os.makedirs(self.mode_path, exist_ok=True)
            print(f"Mode directory {self.mode_path} created.")
        self.plugin_timestamps = {}  # ファイルの最終更新時刻を保持
        sys.path.append(str(self.module_path))
        self.mode = "stop"

        # Automatically load all plugins when the PluginManager is created
        self.load_all_plugins()

    def load_plugin(self, file_path):
        """Load or reload a plugin from a specific file path."""
        module_name = file_path.stem
        if module_name not in self.plugins:
            print(f'{module_name} loading.')
            try:
                self.plugins[module_name] = import_module(module_name)
                print(f'{module_name} loaded.')
            except Exception as e:
                print(f"Error loading module {module_name}: {e}")
        else:
            print(f'{module_name} reloading.')
            try:
                self.plugins[module_name] = reload(self.plugins[module_name])
                print(f'{module_name} reloaded.')
            except Exception as e:
                print(f"Error reloading module {module_name}: {e}")

    def load_all_plugins(self):
        """Scan the target directory and load all Python plugins."""
        for file_path in Path(self.module_path).glob('*.py'):
            last_modified = os.path.getmtime(file_path)
            if file_path not in self.plugin_timestamps or self.plugin_timestamps[file_path] < last_modified:
                # 新しいか更新されたプラグインのみをロード
                self.load_plugin(file_path)
                self.plugin_timestamps[file_path] = last_modified

    def get_mode(self):
        """Read mode.txt and return the current mode."""
        mode_file = self.mode_path / "mode.txt"
        try:
            with open(mode_file, "r") as file:
                mode = file.read().strip().lower()
                return mode
        except FileNotFoundError:
            # mode.txtを作成する
            with open(mode_file, "w") as file:
                file.write(self.mode)
            print(f"Mode file {mode_file} not found. Created mode file with mode {self.mode}.")
            return self.mode

    def run(self, interval=5):
        """Main loop that executes processing based on the mode every N seconds."""
        print("PluginManager started. Waiting for mode changes...")
        while True:
            mode = self.get_mode()
            print(f"{datetime.now().astimezone()} - Current mode: {mode}")

            if mode == "help":
                self.display_help()
            else:
                # モードに対応するメソッドを動的に取得
                mode_method_name = f"mode_{mode}"
                mode_method = getattr(self, mode_method_name, None)

                if callable(mode_method):
                    print(f"Running mode '{mode}'...")
                    mode_method()
                    self.mode = mode  # 有効なモードの場合のみ現在のモードを更新
                else:
                    print(f"Unknown mode: {mode}")

            # インターバルの間隔を待機
            print(f"Checking mode every {interval} seconds...")
            time.sleep(interval)

    def display_help(self):
        """Display all available modes and their descriptions."""
        print("Available modes and descriptions:")
        # クラス内のすべてのメソッドを調べ、'mode_'で始まるものを探す
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("mode_"):
                # メソッド名から 'mode_' を取り除いてモード名を取得
                mode_name = name[5:]
                # メソッドのdocstringから説明を取得
                description = inspect.getdoc(method) or "No description"
                print(f" - {mode_name}: {description}")

    def proceed_to_next_step(self):
        """Proceed to the next step."""
        print("Proceeding to next step...")
        self.experiments.proceed_to_next_step()

    # モードごとの処理を以下に定義します

    def mode_loop(self):
        """
        Run auto-load.
        """
        print("Running auto_load...")
        self.experiments.auto_load()

    def mode_module_load(self):
        """
        Load or reload all plugins.
        """
        print("Loading all plugins...")
        self.load_all_plugins()

    # TODO: reschedule+proceed_nextstepを一回行うようにする # 現状は毎回リスケジュールさせている
    def mode_add_experiment(self):
        """
        Read a command from 'mode_add_experiment.txt' and add an experiment.
        The command must be in 'module.class' format.
        'mode_add_experiment.txt' is deleted automatically after reading.
        Example: 'my_module.MyExperimentClass'
        """
        command_file = self.mode_path / "mode_add_experiment.txt"
        try:
            with open(command_file, "r") as file:
                experiment_generator_function = file.read().strip()
                print(f"Add experiment command: {experiment_generator_function}")
            # ファイルを読み取ったら削除
            os.remove(command_file)
        except FileNotFoundError:
            print(f"Add experiment command file {command_file} not found.")
            return

        parts = experiment_generator_function.split('.')
        if len(parts) == 2:
            module_name, experiment_generator_function = parts
            if module_name in self.plugins:
                module = self.plugins[module_name]
                cls = getattr(module, experiment_generator_function, None)
                if cls:
                    try:
                        experiment_instance = cls()
                        self.experiments.add_experiment(experiment_instance)
                        print(f"Class {experiment_generator_function} from module {module_name} added as an experiment.")
                    except Exception as e:
                        print(f"Error instantiating class {experiment_generator_function}: {e}")
                else:
                    print(f"Class {experiment_generator_function} not found in module {module_name}.")
            else:
                print(f"Module {module_name} is not loaded. Please load the module before adding experiments.")
        else:
            print("Invalid command format in mode_add_experiment.txt. Use 'module.class'.")

    def mode_delete_experiment(self):
        """
        Method for 'delete_experiment' mode.
        Read a command from 'mode_delete_experiment.txt' and delete the target experiment.
        The command must be an experiment UUID.
        'mode_delete_experiment.txt' is deleted automatically after reading.
        """
        # 引数が指定されていない場合はファイルから読み取る
        command_file = self.mode_path / "mode_delete_experiment.txt"
        try:
            with open(command_file, "r") as file:
                experiment_uuid = file.read().strip()
                print(f"Delete experiment UUID: {experiment_uuid}")
            # ファイルを読み取ったら削除
            os.remove(command_file)
            if experiment_uuid:
                try:
                    self.experiments.delete_experiment_with_experiment_uuid(experiment_uuid)
                    print(f"Experiment with UUID {experiment_uuid} has been deleted.")
                except Exception as e:
                    print(f"Error deleting experiment with UUID {experiment_uuid}: {e}")
            else:
                print("No UUID provided for deletion.")

        except FileNotFoundError:
            print(f"Delete experiment command file {command_file} not found.")
            return
    # TODO: reschedule+proceed_nextstepを一回行うようにする # 現状は毎回リスケジュールさせている
    def mode_add_experiments(self):
        """
        Read multiple commands from 'mode_add_experiments.txt' and add experiments.
        Each command must be in 'module.class' format.
        The file is deleted automatically after reading.
        Example:
        ```mode_add_experiments.txt
            my_module.ExperimentClass1
            other_module.ExperimentClass2
        ```
        """
        command_file = self.mode_path / "mode_add_experiments.txt"
        try:
            with open(command_file, "r") as file:
                lines = file.readlines()
                # 空行やコメント行を除外
                commands = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
                if not commands:
                    print(f"Add experiments command file {command_file} is empty.")
                    return
                print(f"Experiment commands to add: {commands}")
            # ファイルを読み取ったら削除
            os.remove(command_file)
        except FileNotFoundError:
            print(f"Add experiments command file {command_file} not found.")
            # templateファイルを作成する
            command_template_file = Path(str(command_file).replace(".txt", "_template.txt"))
            command_template = f"# Write commands in 'module.class' format to {command_file}.\n"
            command_template += "# Example:\n"
            command_template += "# my_module.ExperimentClass1\n"
            command_template += "# other_module.ExperimentClass2\n"
            with open(command_template_file, "w") as file:
                file.write(command_template)

            print(f"Created template file {command_template_file}.")
            print(command_template)
            return
        except Exception as e:
            print(f"Error reading command file: {e}")
            return

        for command in commands:
            parts = command.split('.')
            if len(parts) == 2:
                module_name, class_name = parts
                if module_name in self.plugins:
                    module = self.plugins[module_name]
                    cls = getattr(module, class_name, None)
                    if cls:
                        try:
                            experiment_instance = cls()
                            self.experiments.add_experiment(experiment_instance)
                            print(f"Added class {class_name} from module {module_name} as an experiment.")
                        except Exception as e:
                            print(f"Error instantiating class {class_name}: {e}")
                    else:
                        print(f"Class {class_name} not found in module {module_name}.")
                else:
                    print(f"Module {module_name} is not loaded. Load the module before adding experiments.")
            else:
                print(f"Invalid command format: '{command}'. Use 'module.class'.")

    # TODO: reschedule+proceed_nextstepを一回行うようにする # 現状は毎回リスケジュールさせている
    def mode_delete_experiments(self):
        """
        Method for 'delete_experiments' mode.
        Read multiple UUIDs from 'mode_delete_experiments.txt' and delete target experiments.
        Each UUID must be written on a new line.
        The file is deleted automatically after reading.
        Example:
            uuid1
            uuid2
            uuid3
        """
        command_file = self.mode_path / "mode_delete_experiments.txt"
        try:
            with open(command_file, "r") as file:
                lines = file.readlines()
                # 空行やコメント行を除外
                uuids = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
                if not uuids:
                    print(f"Delete experiments command file {command_file} is empty.")
                    return
                print(f"Experiment UUIDs to delete: {uuids}")
            # ファイルを読み取ったら削除
            os.remove(command_file)
        except FileNotFoundError:
            print(f"Delete experiments command file {command_file} not found.")
            # templateファイルを作成する
            command_template_file = Path(str(command_file).replace(".txt", "_template.txt"))
            command_template = f"# Write UUIDs to {command_file}.\n"
            command_template += "# Example:\n"
            command_template += "# uuid1\n"
            command_template += "# uuid2\n"
            with open(command_template_file, "w") as file:
                file.write(command_template)

            print(f"Created template file {command_template_file}.")
            print(command_template)
            return
        except Exception as e:
            print(f"Error reading command file: {e}")
            return

        for uuid in uuids:
            try:
                self.experiments.delete_experiment_with_experiment_uuid(uuid)
                print(f"Deleted experiment with UUID {uuid}.")
            except Exception as e:
                print(f"Error deleting experiment with UUID {uuid}: {e}")

    def mode_show_experiments(self):
        """
        Show the list of experiments.
        """
        # 実験クラスの表示メソッド
        if hasattr(self.experiments, 'list'):
            self.experiments.list()
        else:
            print("No experiments to show.")

    def mode_add_machines(self):
        """
        Read commands from 'mode_add_machines.txt' and add machines.
        Commands must be in 'machine_type[,description]' format.
        ```mode_add_machines.txt
        0,Pippeting machine 1
        0,Pippeting machine 2
        1,Heating machine 1
        ```
        """
        command_file = self.mode_path / "mode_add_machines.txt"
        try:
            with open(command_file, "r") as file:
                lines = file.readlines()
                # 空行やコメント行を除外
                commands = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
                if not commands:
                    print(f"Add machines command file {command_file} is empty.")
                    return
                print(f"Machine commands to add: {commands}")
            # ファイルを読み取ったら削除
            os.remove(command_file)

            for command in commands:
                parts = command.split(',')
                if len(parts) == 2:
                    machine_type, description = parts
                    try:
                        machine_type = int(machine_type)
                        self.experiments.add_machine(machine_type = machine_type, description = description)
                    except Exception as e:
                        print(f"Error adding machine {machine_type}: {e}")
                elif len(parts) == 1:
                    machine_type = parts[0]
                    try:
                        machine_type = int(machine_type)
                        self.experiments.add_machine(machine_type = machine_type)
                    except Exception as e:
                        print(f"Error adding machine {machine_type}: {e}")
                else:
                    print(f"Invalid command format: '{command}'. Use 'machine_type,description'.")

            

        except FileNotFoundError:
            print(f"Add machines command file {command_file} not found.")
            # templateファイルを作成する
            command_template_file = Path(str(command_file).replace(".txt", "_template.txt"))
            command_template = f"# Write commands in 'machine_type,description' format to {command_file}.\n"
            command_template += "# Example:\n"
            command_template += "# 0,Pippeting machine 1\n"
            command_template += "# 0,Pippeting machine 2\n"
            command_template += "# 1,Heating machine 1\n"
            with open(command_template_file, "w") as file:
                file.write(command_template)

            print(f"Created template file {command_template_file}.")
            print(command_template)
            return
        except Exception as e:
            print(f"Error reading command file: {e}")
            return
        
    def mode_delete_machines(self):
        """
        Read commands from 'mode_delete_machines.txt' and delete target machines.
        Commands must be in 'machine_id' format.
        ```mode_delete_machines.txt
        0
        1
        2
        ```
        """
        command_file = self.mode_path / "mode_delete_machines.txt"
        try:
            with open(command_file, "r") as file:
                lines = file.readlines()
                # 空行やコメント行を除外
                machine_ids = [line.strip() for line in lines if line.strip() and not line.strip().startswith('#')]
                if not machine_ids:
                    print(f"Delete machines command file {command_file} is empty.")
                    return
                print(f"Machine IDs to delete: {machine_ids}")
            # ファイルを読み取ったら削除
            os.remove(command_file)

            for machine_id in machine_ids:
                try:
                    machine_id = int(machine_id)
                    self.experiments.delete_machine(machine_id)
                    print(f"Machine {machine_id} deleted.")
                except Exception as e:
                    print(f"Error deleting machine {machine_id}: {e}")

        except FileNotFoundError:
            print(f"Delete machines command file {command_file} not found.")
            # templateファイルを作成する
            command_template_file = Path(str(command_file).replace(".txt", "_template.txt"))
            command_template = f"# Write machine IDs to {command_file} in 'machine_id' format.\n"
            command_template += "# Example:\n"
            command_template += "# 0\n"
            command_template += "# 1\n"
            command_template += "# 2\n"
            with open(command_template_file, "w") as file:
                file.write(command_template)

            print(f"Created template file {command_template_file}.")
            print(command_template)
            return
        except Exception as e:
            print(f"Error reading command file: {e}")
            return


    def mode_show_machines(self):
        """
        Show the list of machines.
        """
        # マシンの表示メソッド
        try: 
            self.experiments.show_machines()
        except Exception as e:
            print(f"Error showing machines: {e}")

    


    def mode_proceed(self):
        """
        Proceed to the next step.
        """
        self.proceed_to_next_step()

    def mode_stop(self):
        """
        Stop processing and stay idle.
        """
        print("Rest...zzz")

    def mode_exit(self):
        """
        Exit the plugin manager.
        """
        print("Exiting PluginManager...")
        sys.exit()

    def mode_eof(self):
        """
        Exit the plugin manager.
        """
        self.mode_exit()

import tempfile

def main():
    UNIX_2024_11_13_00_00_00_IN_JP = 1731423600
    
    dir = tempfile.mkdtemp()
    experiments = Experiments(parent_dir_path=Path(dir), reference_time = UNIX_2024_11_13_00_00_00_IN_JP//60)
    plugin_manager = PluginManager(experiments)

    # Ask whether to reload experiments before starting the plugin manager.
    reload_choice = 'y' #input("Do you want to reload experiments? (y/n): ").strip().lower()
    if reload_choice == 'y':
        try:
            step = '' #input("Enter the step to reload. Leave blank to reload up to the latest step automatically: ").strip()
            if step == '':
                step = None
            else:
                step = int(step)
            experiments = experiments.reload(step)
        except ValueError:
            print("Invalid step number. Skipping reload.")
        except Exception as err:
            print(f"Error occurred during reload: {err}. Skipping reload.")

    plugin_manager = PluginManager(experiments)
    plugin_manager.run()


if __name__ == '__main__':
    main()
