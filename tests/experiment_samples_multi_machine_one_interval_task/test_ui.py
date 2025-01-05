from datetime import datetime
import time
import os
from pathlib import Path
import sys
from importlib import import_module, reload
import inspect
import tempfile

from enum import Enum
import unittest
from typing import List, Dict

from gems_python.multi_machine_problem_interval_task.interactive_ui import PluginManager
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments

class TestInteractiveUI(unittest.TestCase):
    def test_initialise(self):
        current_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
        print(current_dir)
        with tempfile.TemporaryDirectory(dir=current_dir) as dir:
            current_unixtime = datetime.now()
            print(f"current_unixtime: {current_unixtime.timestamp()} ({current_unixtime.astimezone().isoformat()})")
            current_unixtime = int(current_unixtime.timestamp())

            experiments = Experiments(parent_dir_path=Path(dir), reference_time = current_unixtime)
            plugin_manager = PluginManager(experiments)

            # Copy ./minimum.py to dir/experimental_setting/minimum.py
            import shutil
            shutil.copy(f"{current_dir}/minimum.py", f"{dir}/experimental_setting/minimum.py")




            machines = \
r"""
0,Pippeting machine 1
0,Pippeting machine 2
1,Heating machine 1
2,Centrifuge machine 1

"""
            with open(f"{dir}/mode/mode_add_machines.txt", "w") as f:
                f.write(machines)

            mode = "add_machines"
            with open(f"{dir}/mode/mode.txt", "w") as f:
                f.write(mode)


            ex = "minimum.gen_standard_experiment"
            with open(f"{dir}/mode/mode_add_experiments.txt", "w") as f:
                f.write(ex)

            with open(f"{dir}/mode/mode_add_experiments.txt", "w") as f:
                f.write(ex)

            # mode = "add_experiments"
            # with open(f"{dir}/mode/mode.txt", "w") as f:
            #     f.write(mode)


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



