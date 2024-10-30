import sys
import time
import threading
from importlib import import_module, reload
from pathlib import Path

from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers.polling import PollingObserver as Observer
import cmd2

from gems_python.one_machine_problem_interval_task.transition_manager import Experiments, Experiment  # 必要なインポートを確認してください


class PluginManager:

    class Handler(PatternMatchingEventHandler):

        def __init__(self, manager: 'PluginManager', *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.manager = manager

        def on_created(self, event: FileSystemEvent):
            print(f"Created: {event.src_path}")
            if event.src_path.endswith('.py'):
                self.manager.load_plugin(Path(event.src_path))

        def on_modified(self, event):
            print(f"Modified: {event.src_path}")
            if event.src_path.endswith('.py'):
                self.manager.load_plugin(Path(event.src_path))


    def __init__(self, experiments: Experiments, path: Path = "experimental_setting/"):
        self.plugins = {}
        self.experiments = experiments
        self.path = self.experiments.parent_dir_path / path
        self.observer = Observer()

        self.path: str = str(self.path)
        sys.path.append(self.path)

    def start(self):
        self.scan_plugin()

        # Ensure patterns is a list
        patterns = ['*.py']
        self.observer.schedule(self.Handler(self, patterns=patterns), self.path)
        self.observer.start()

    def stop(self):
        self.observer.stop()
        self.observer.join()

    def scan_plugin(self):
        for file_path in Path(self.path).glob('*.py'):
            self.load_plugin(file_path)

    def load_plugin(self, file_path):
        module_name = file_path.stem
        if module_name not in self.plugins:
            print('{} loading.'.format(module_name))
            self.plugins[module_name] = import_module(module_name)
            print('{} loaded.'.format(module_name))
        else:
            print('{} reloading.'.format(module_name))
            self.plugins[module_name] = reload(self.plugins[module_name])
            print('{} reloaded.'.format(module_name))

    def add_experiment_cmd(self, experiment_generator_function):
        parts = experiment_generator_function.split('.')
        if len(parts) == 2:
            module_name, experiment_generator_function = parts
            if module_name in self.plugins:
                module = self.plugins[module_name]
                cls = getattr(module, experiment_generator_function, None)
                if cls:
                    self.experiments.add_experiment(cls())
                    print(f"Class {experiment_generator_function} added.")
                else:
                    print(f"Class {experiment_generator_function} not found in module {module_name}.")
            else:
                print(f"Module {module_name} not loaded.")
        else:
            print("Invalid command format. Use 'module.class'.")

    def show_possible_commands(self):
        possible_commands = ["add <module.class>", "show", "module_name.class_name", "stop", "reloop"]
        for cmd in possible_commands:
            print(cmd)

    def show_classes(self):
        # 実験クラスの表示メソッド
        if hasattr(self.experiments, 'show_experiment_directed_graph'):
            self.experiments.show_experiment_directed_graph()
        else:
            print("No experiments to show.")

    def show_experiments(self):
        # 実験クラスの表示メソッド
        if hasattr(self.experiments, 'list'):
            self.experiments.list()
        else:
            print("No experiments to show.")


class PluginCmd(cmd2.Cmd):

    def __init__(self, plugin_manager):
        super().__init__()
        self.plugin_manager = plugin_manager
        self.prompt = "plugin_manager> "

        # スレッド間で共有するフラグの保護
        self.lock = threading.Lock()

        # タイマー関連の初期化
        self.last_command_time = time.time()
        self.auto_load_enabled = True  # 自動ロードが有効かどうか
        self.stop_event = threading.Event()
        self.proceed_to_next_step()

        # バックグラウンドスレッドの開始（self.lock を先に定義）
        self.monitor_thread = threading.Thread(target=self.monitor_inactivity, daemon=True)
        self.monitor_thread.start()

    def monitor_inactivity(self):
        while not self.stop_event.is_set():
            with self.lock:
                if self.auto_load_enabled:
                    current_time = time.time()
                    if (current_time - self.last_command_time) > 1:
                        print("\nNo command received for 1 second. Running auto_load(). If you want to stop, type 'stop'.")
                        self.plugin_manager.experiments.auto_load()
                        # 自動ロードを一度実行したら再度カウントするため、last_command_timeを更新
                        self.last_command_time = current_time
            time.sleep(0.1)  # 監視のインターバル

    def reset_timer(self):
        with self.lock:
            self.last_command_time = time.time()

    def enable_auto_load(self):
        with self.lock:
            self.auto_load_enabled = True
            print("Auto-load has been reenabled.")

    def disable_auto_load(self):
        with self.lock:
            self.auto_load_enabled = False
            print("Auto-load has been disabled.")

    def preloop(self):
        """cmd2 の preloop をオーバーライドして、必要な初期化を行います。"""
        super().preloop()

    def postcmd(self, stop, line):
        """各コマンド実行後にタイマーをリセットします。"""
        self.reset_timer()
        return super().postcmd(stop, line)
    
    def proceed_to_next_step(self):
        """次のステップに進むためのメソッド"""
        self.plugin_manager.experiments.proceed_to_next_step()

    def do_add(self, class_name):
        """Add a class to the list."""
        self.plugin_manager.add_experiment_cmd(class_name)
        self.plugin_manager.experiments.proceed_to_next_step()

    def do_delete(self, experiment_uuid: str):
        """Delete an experiment by UUID."""
        self.plugin_manager.experiments.delete_experiment_with_experiment_uuid(experiment_uuid)

    def do_show(self, _):
        """Show all added classes."""
        # self.plugin_manager.show_classes()
        self.do_show_experiments(_)
        
    def do_show_experiments(self, _):
        """Show all added classes."""
        self.plugin_manager.show_experiments()

    def do_schedule(self, _):
        """Empty the list of classes."""
        pass

    def do_cmdlist(self, _):
        """Show all possible commands."""
        self.plugin_manager.show_possible_commands()

    def do_stop(self, _):
        """Disable auto-load functionality."""
        self.disable_auto_load()

    def do_reloop(self, _):
        """Enable auto-load functionality."""
        self.enable_auto_load()

    def do_exit(self, _):
        """Exit the plugin manager."""
        return True

    def do_EOF(self, _):
        """Handle EOF to exit."""
        print("Exiting.")
        return True

    def cmdloop(self, intro=None):
        """Override cmdloop to handle graceful shutdown."""
        try:
            super().cmdloop(intro=intro)
        except KeyboardInterrupt:
            print("\nInterrupted.")
        finally:
            self.stop_event.set()
            self.monitor_thread.join()


def main():
    experiments = Experiments(parent_dir_path=Path("volatile"))
    plugin_manager = PluginManager(experiments)
    print("Plugin Manager started.")
    print(f"{plugin_manager=}")
    print(f"{plugin_manager.path=}")
    print(sys.path)

    plugin_manager.start()

    plugin_cmd = PluginCmd(plugin_manager)
    plugin_cmd.cmdloop()

    plugin_manager.stop()


if __name__ == '__main__':
    main()
