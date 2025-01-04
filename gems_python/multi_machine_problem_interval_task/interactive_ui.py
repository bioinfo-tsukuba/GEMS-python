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
        """特定のファイルパスからプラグインをロードまたはリロードします。"""
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
        """指定ディレクトリ内のすべてのPythonプラグインをスキャンしてロードします。"""
        for file_path in Path(self.module_path).glob('*.py'):
            last_modified = os.path.getmtime(file_path)
            if file_path not in self.plugin_timestamps or self.plugin_timestamps[file_path] < last_modified:
                # 新しいか更新されたプラグインのみをロード
                self.load_plugin(file_path)
                self.plugin_timestamps[file_path] = last_modified

    def get_mode(self):
        """mode.txt を読み込んで現在のモードを取得します。"""
        mode_file = self.mode_path / "mode.txt"
        try:
            with open(mode_file, "r") as file:
                mode = file.read().strip().lower()
                return mode
        except FileNotFoundError:
            print(f"Mode file {mode_file} not found. 継続中のモード {self.mode} を使用します。")
            return self.mode

    def run(self, interval=5):
        """N秒ごとにモードに応じて処理を実行するメインループです。"""
        print("PluginManagerが起動しました。モードを待機しています...")
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
                    print(f"モード '{mode}' を実行中...")
                    mode_method()
                    self.mode = mode  # 有効なモードの場合のみ現在のモードを更新
                else:
                    print(f"Unknown mode: {mode}")

            # インターバルの間隔を待機
            print(f"{interval}秒ごとにモードを確認します...")
            time.sleep(interval)

    def display_help(self):
        """利用可能なすべてのモードとその説明を表示します。"""
        print("利用可能なモードと説明:")
        # クラス内のすべてのメソッドを調べ、'mode_'で始まるものを探す
        for name, method in inspect.getmembers(self, predicate=inspect.ismethod):
            if name.startswith("mode_"):
                # メソッド名から 'mode_' を取り除いてモード名を取得
                mode_name = name[5:]
                # メソッドのdocstringから説明を取得
                description = inspect.getdoc(method) or "説明なし"
                print(f" - {mode_name}: {description}")

    def proceed_to_next_step(self):
        """次のステップに進みます。"""
        print("Proceeding to next step...")
        self.experiments.proceed_to_next_step()

    # モードごとの処理を以下に定義します

    def mode_loop(self):
        """
        自動ロードを実行します。
        """
        print("Running auto_load...")
        self.experiments.auto_load()

    def mode_module_load(self):
        """
        すべてのプラグインをロードまたはリロードします。
        """
        print("Loading all plugins...")
        self.load_all_plugins()

    # TODO: reschedule+proceed_nextstepを一回行うようにする # 現状は毎回リスケジュールさせている
    def mode_add_experiment(self):
        """
        'mode_add_experiment.txt' ファイルからコマンドを読み取り、実験を追加します。
        コマンドは 'module.class' 形式で記述されている必要があります。
        'mode_add_experiment.txt' は読み取り後に自動的に削除されます。
        例: 'my_module.MyExperimentClass'
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
        'delete_experiment' モードで実行されるメソッド。
        'mode_delete_experiment.txt' ファイルからコマンドを読み取り、指定された実験を削除します。
        コマンドは実験のUUIDである必要があります。
        'mode_delete_experiment.txt' は読み取り後に自動的に削除されます。
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
        'mode_add_experiments.txt' ファイルから複数のコマンドを読み取り、実験を追加します。
        各コマンドは 'module.class' 形式で記述されています。
        ファイルは読み取り後に自動的に削除されます。
        例:
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
                    print(f"Add experiments command file {command_file} は空です。")
                    return
                print(f"追加する実験コマンド: {commands}")
            # ファイルを読み取ったら削除
            os.remove(command_file)
        except FileNotFoundError:
            print(f"追加実験コマンドファイル {command_file} が見つかりません。")
            # templateファイルを作成する
            command_template_file = Path(str(command_file).replace(".txt", "_template.txt"))
            command_template = f"# 'module.class' 形式で{command_file}に記述してください。\n"
            command_template += "# 例:\n"
            command_template += "# my_module.ExperimentClass1\n"
            command_template += "# other_module.ExperimentClass2\n"
            with open(command_template_file, "w") as file:
                file.write(command_template)

            print(f"Created template file {command_template_file}.")
            print(command_template)
            return
        except Exception as e:
            print(f"コマンドファイルの読み取り中にエラーが発生しました: {e}")
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
                            print(f"モジュール {module_name} のクラス {class_name} を実験として追加しました。")
                        except Exception as e:
                            print(f"クラス {class_name} のインスタンス化中にエラーが発生しました: {e}")
                    else:
                        print(f"モジュール {module_name} にクラス {class_name} が見つかりません。")
                else:
                    print(f"モジュール {module_name} がロードされていません。実験を追加する前にモジュールをロードしてください。")
            else:
                print(f"無効なコマンド形式: '{command}'。 'module.class' 形式を使用してください。")

    # TODO: reschedule+proceed_nextstepを一回行うようにする # 現状は毎回リスケジュールさせている
    def mode_delete_experiments(self):
        """
        'delete_experiments' モードで実行されるメソッド。
        'mode_delete_experiments.txt' ファイルから複数のUUIDを読み取り、指定された実験を削除します。
        各UUIDは改行区切りで記述されています。
        ファイルは読み取り後に自動的に削除されます。
        例:
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
                    print(f"削除コマンドファイル {command_file} は空です。")
                    return
                print(f"削除する実験のUUID一覧: {uuids}")
            # ファイルを読み取ったら削除
            os.remove(command_file)
        except FileNotFoundError:
            print(f"削除コマンドファイル {command_file} が見つかりません。")
            # templateファイルを作成する
            command_template_file = Path(str(command_file).replace(".txt", "_template.txt"))
            command_template = f"# UUIDを{command_file}に記述してください。\n"
            command_template += "# 例:\n"
            command_template += "# uuid1\n"
            command_template += "# uuid2\n"
            with open(command_template_file, "w") as file:
                file.write(command_template)

            print(f"Created template file {command_template_file}.")
            print(command_template)
            return
        except Exception as e:
            print(f"コマンドファイルの読み取り中にエラーが発生しました: {e}")
            return

        for uuid in uuids:
            try:
                self.experiments.delete_experiment_with_experiment_uuid(uuid)
                print(f"UUID {uuid} の実験が削除されました。")
            except Exception as e:
                print(f"UUID {uuid} の実験を削除中にエラーが発生しました: {e}")

    def mode_show_experiments(self):
        """
        実験クラスのリストを表示します。
        """
        # 実験クラスの表示メソッド
        if hasattr(self.experiments, 'list'):
            self.experiments.list()
        else:
            print("No experiments to show.")

    def mode_add_machines(self):
        """
        'mode_add_machines.txt' ファイルからコマンドを読み取り、マシンを追加します。
        コマンドは 'machine_type(,description) の形式で記述されている必要があります。
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
                    print(f"Add machines command file {command_file} は空です。")
                    return
                print(f"追加するマシンコマンド: {commands}")
            # ファイルを読み取ったら削除
            os.remove(command_file)

            for command in commands:
                parts = command.split(',')
                if len(parts) == 2:
                    machine_type, description = parts
                    try:
                        machine_type = int(machine_type)
                        self.experiments.add_machine(machine_type, description)
                        print(f"Machine {machine_type} added with description '{description}'.")
                    except Exception as e:
                        print(f"Error adding machine {machine_type}: {e}")
                elif len(parts) == 1:
                    machine_type = parts[0]
                    try:
                        machine_type = int(machine_type)
                        self.experiments.add_machine(machine_type)
                        print(f"Machine {machine_type} added.")
                    except Exception as e:
                        print(f"Error adding machine {machine_type}: {e}")
                else:
                    print(f"Invalid command format: '{command}'. Use 'machine_type,description'.")

            

        except FileNotFoundError:
            print(f"Add machines command file {command_file} not found.")
            # templateファイルを作成する
            command_template_file = Path(str(command_file).replace(".txt", "_template.txt"))
            command_template = f"# 'machine_type,description' 形式で{command_file}に記述してください。\n"
            command_template += "# 例:\n"
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
        'mode_delete_machines.txt' ファイルからコマンドを読み取り、指定されたマシンを削除します。
        コマンドは 'machine_id' の形式で記述されている必要があります。
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
                    print(f"Delete machines command file {command_file} は空です。")
                    return
                print(f"削除するマシンID一覧: {machine_ids}")
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
            command_template = f"# 'machine_id' 形式で{command_file}に記述してください。\n"
            command_template += "# 例:\n"
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
        マシンのリストを表示します。
        """
        # マシンの表示メソッド
        try: 
            self.experiments.show_machines()
        except Exception as e:
            print(f"Error showing machines: {e}")

    


    def mode_proceed(self):
        """
        次のステップに進みます。
        """
        self.proceed_to_next_step()

    def mode_stop(self):
        """
        処理を停止し、休息状態になります。
        """
        print("Rest...zzz")

    def mode_exit(self):
        """
        プラグインマネージャーを終了します。
        """
        print("Exiting PluginManager...")
        sys.exit()

    def mode_eof(self):
        """
        プラグインマネージャーを終了します。
        """
        self.mode_exit()

import tempfile

def main():
    UNIX_2024_11_13_00_00_00_IN_JP = 1731423600
    
    dir = tempfile.mkdtemp()
    experiments = Experiments(parent_dir_path=Path(dir), reference_time = UNIX_2024_11_13_00_00_00_IN_JP//60)
    plugin_manager = PluginManager(experiments)

    # プラグインマネージャーの開始前にリロードの選択を促す
    reload_choice = 'y' #input("実験をリロードしますか？ (y/n): ").strip().lower()
    if reload_choice == 'y':
        try:
            step = '' #input("リロードするステップを入力してください。空白の場合は自動的に最大ステップまでリロードします。: ").strip()
            if step == '':
                step = None
            else:
                step = int(step)
            experiments = experiments.reload(step)
        except ValueError:
            print("無効なステップ番号です。リロードをスキップします。")
        except Exception as err:
            print(f"リロード中にエラーが発生しました: {err}. リロードをスキップします。")

    plugin_manager = PluginManager(experiments)
    plugin_manager.run()


if __name__ == '__main__':
    main()
