from transition_manager import Experiments, Experiment

import sys
import time
from importlib import import_module, reload
from pathlib import Path

from watchdog.events import FileSystemEvent, PatternMatchingEventHandler
from watchdog.observers.polling import PollingObserver as Observer
import cmd2

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
        self.classes_list = []


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

    def execute_command(self, cmd):
        parts = cmd.split('.')
        if len(parts) == 2:
            module_name, class_name = parts
            if module_name in self.plugins:
                module = self.plugins[module_name]
                cls = getattr(module, class_name, None)
                if cls:
                    return cls()
                else:
                    print(f"Class {class_name} not found in module {module_name}.")
            else:
                print(f"Module {module_name} not loaded.")
        else:
            print("Invalid command format. Use 'module.class'.")

    def add_class(self, class_name):
        self.classes_list.append(class_name)
        print(f"Added class: {class_name}")

    def show_classes(self):
        for cls in self.classes_list:
            print(cls)

    def show_possible_commands(self):
        possible_commands = ["add <class_name>", "show", "module_name.class_name"]
        for cmd in possible_commands:
            print(cmd)


class PluginCmd(cmd2.Cmd):

    def __init__(self, plugin_manager):
        super().__init__()
        self.plugin_manager = plugin_manager
        self.prompt = "plugin_manager> "

    def do_add(self, class_name):
        """Add a class to the list."""
        self.plugin_manager.add_class(class_name)

    def do_show(self, _):
        """Show all added classes."""
        self.plugin_manager.show_classes()

    def do_void(self, _):
        """Empty the list of classes."""
        print("Void")
        
    def do_run(self, cmd):
        """Run a command in the format module_name.class_name."""
        result = self.plugin_manager.execute_command(cmd)
        if result:
            print(f"Executed command {cmd}: {result}")

    def do_cmdlist(self, _):
        """Show all possible commands."""
        self.plugin_manager.show_possible_commands()

    def do_exit(self, _):
        """Exit the plugin manager."""
        return True


def main():
    experiments = Experiments(parent_dir_path=Path("experiment_test"))
    plugin_manager = PluginManager(experiments)
    print("Plugin Manager started.")
    print(f"{plugin_manager=}")
    print(f"{plugin_manager.path=}")
    print(sys.path)

    plugin_manager.start()

    print("Plugin Manager started.")
    plugin_cmd = PluginCmd(plugin_manager)
    plugin_cmd.cmdloop()

    plugin_manager.stop()

if __name__ == '__main__':
    main()
