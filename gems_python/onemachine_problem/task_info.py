import copy
from dataclasses import asdict, dataclass, field
import json
from typing import List, Type
from pathlib import Path

from gems_python.onemachine_problem.penalty.base import PenaltyType        
        
@dataclass
class OneMachineTaskLocalInformation:
    """
    OneMachineTaskLocalInformation class.
    This class is used as a data class for the task.
    """
    optimal_timing: int
    processing_time: int
    penalty_type: Type[PenaltyType]
    experiment_operation: str

from simanneal import Annealer
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import random
@dataclass
class OneMachineTask:
    """
    OneMachineTask class.
    This class is used as a data class for the task.
    """
    optimal_timing: int
    processing_time: int
    penalty_type: Type[PenaltyType]
    experiment_operation: str
    experiment_name: str
    experiment_uuid: str
    task_id: int = field(default=None)
    scheduled_timing: int = field(default=None)

    def to_dict(self) -> dict:
        """
        Converts the task object to a dictionary.
        :return: Dictionary representation of the task object.
        """
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'OneMachineTask':
        """
        Creates a task object from a dictionary.
        :param data: Dictionary containing the task data.
        :return: Task object.
        """
        return cls(**data)

    def to_json(self) -> str:
        """
        Converts the task object to a JSON string.
        :return: JSON string representation of the task object.
        """
        data = self.to_dict()
        data["penalty_type"] = self.penalty_type.to_json()
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'OneMachineTask':
        """
        Creates a task object from a JSON string.
        :param json_str: JSON string containing the task data.
        :return: Task object.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)

    @classmethod
    def simulated_annealing_schedule(cls, tasks: List['OneMachineTask']) -> List['OneMachineTask']:
        """
        Simulated annealing scheduler.
        :param tasks: List of tasks to schedule.
        :return: Scheduled tasks.
        """
        print("SA_schedule:")
        class TaskAnnealer(Annealer):

            def __init__(self, state):
                super().__init__(state)
                self.step_count: int = 0

            def move(self):
                self.step_count += 1

                """
                random move or swap
                """
                temp = max(1, int(self.step_count/self.steps * (self.Tmax - self.Tmin) + self.Tmin))
                for i in range(min(temp, len(self.state))):
                  a = random.randint(0, len(self.state) - 1)
                  self.state[a].scheduled_timing += random.randint(-int(temp), int(temp))
                  self.state[a].scheduled_timing = max(0, self.state[a].scheduled_timing)

                if random.random() < 0.5:
                    a = random.randint(0, len(self.state) - 1)
                    b = random.randint(0, len(self.state) - 1)
                    self.state[a].scheduled_timing, self.state[b].scheduled_timing = self.state[b].scheduled_timing, self.state[a].scheduled_timing

            def energy(self):
                """Calculates the total penalty for the current state."""
                total_penalty = 0
                for task in self.state:
                    total_penalty += task.penalty_type.calculate_penalty(
                        task.scheduled_timing, task.optimal_timing
                    )

                # Overlapping penalty
                sorted_tasks = sorted(self.state, key=lambda x: x.scheduled_timing)
                for i in range(len(sorted_tasks) - 1):
                    task1 = sorted_tasks[i]
                    task2 = sorted_tasks[i + 1]
                    overlap = task1.scheduled_timing + task1.processing_time - task2.scheduled_timing
                    if overlap > 0:
                        total_penalty += overlap * 100000
                return total_penalty

        # Initialize the tasks with some initial schedule (e.g., their optimal timings)
        time = 0
        tasks = sorted(tasks, key=lambda x: x.optimal_timing + x.processing_time)
        for task in tasks:
            task.scheduled_timing = max(time, task.optimal_timing)
            time = task.scheduled_timing + task.processing_time

        # Create an instance of the annealer with the initial state
        annealer = TaskAnnealer(tasks)
        # Set the annealing parameters as needed
        annealer.steps = 100000
        annealer.Tmax = 25000.0
        annealer.Tmin = 1.0

        # Run the annealing process
        state, _ = annealer.anneal()

        return state
    
    @classmethod
    def get_earliest_scheduled_task(cls, tasks: List['OneMachineTask']) -> 'OneMachineTask':
        earliest_scheduled_task_index = 0
        earliest_scheduled_task_scheduled_timing = tasks[0].scheduled_timing

        for i in range(len(tasks)):
            if earliest_scheduled_task_scheduled_timing < tasks[i].scheduled_timing:
                earliest_scheduled_task_index = i
                earliest_scheduled_task_scheduled_timing = tasks[i].scheduled_timing

        return copy.deepcopy(tasks[earliest_scheduled_task_index])


    @classmethod
    def vis(cls, tasks: List['OneMachineTask'], save_path: Path = None):
        """
        Visualizes the scheduled tasks as a Gantt chart.
            - tasks: List of scheduled tasks to visualize.
            - save_path: Path to save the visualization. If None, the visualization is shown.
        """
        fig, ax = plt.subplots()
        tasks = sorted(tasks, key=lambda x: x.scheduled_timing)

        # ガントチャートのデータを準備
        for task in tasks:
            start_time = task.scheduled_timing
            end_time = start_time + task.processing_time
            ax.barh(task.experiment_operation, end_time - start_time, left=start_time, edgecolor='black')

        # 軸のラベルを設定
        ax.set_xlabel('Time')
        ax.set_ylabel('Task')

        # time軸の最大値と最小値を設定
        max_time = max(task.scheduled_timing + task.processing_time for task in tasks)
        max_time = max(max(task.optimal_timing + task.processing_time for task in tasks), max_time) + 5
        min_time = min(task.scheduled_timing for task in tasks)
        min_time = min(min(task.optimal_timing for task in tasks), min_time) - 5
        ax.set_xlim(min_time, max_time)

        # タスクごとに異なる色を設定
        colors = plt.cm.get_cmap('tab10', len(tasks))
        for i, task in enumerate(tasks):
            start_time = task.scheduled_timing
            end_time = start_time + task.processing_time
            ax.barh(task.experiment_operation, end_time - start_time, left=start_time, color=colors(i), edgecolor='black')

        # 凡例を設定
        patches = [mpatches.Patch(color=colors(i), label=task.experiment_operation) for i, task in enumerate(tasks)]
        plt.legend(handles=patches)

        # グラフを表示
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()


    @classmethod
    def vis_with_diff(cls, tasks: List['OneMachineTask'], save_path: Path = None):
        """
        Visualizes the scheduled tasks as a Gantt chart.
            - tasks: List of scheduled tasks to visualize.
            - save_path: Path to save the visualization. If None, the visualization is shown.
        """
        fig, ax = plt.subplots()
        tasks = sorted(tasks, key=lambda x: x.scheduled_timing)

        # ガントチャートのデータを準備
        for task in tasks:
            start_time = task.scheduled_timing
            end_time = start_time + task.processing_time
            optimal_start = task.optimal_timing
            ax.barh(task.experiment_operation, end_time - start_time, left=start_time, edgecolor='black', alpha=0.7)

            # optimal_timingとscheduled_timingの差を表す線を追加
            if start_time != optimal_start:
                ax.plot([optimal_start, start_time], [task.experiment_operation, task.experiment_operation], 'r--', linewidth=2)

        # 軸のラベルを設定
        ax.set_xlabel('Time')
        ax.set_ylabel('Task')

        # time軸の最大値と最小値を設定
        max_time = max(task.scheduled_timing + task.processing_time for task in tasks)
        max_time = max(max(task.optimal_timing + task.processing_time for task in tasks), max_time) + 5
        min_time = min(task.scheduled_timing for task in tasks)
        min_time = min(min(task.optimal_timing for task in tasks), min_time) - 5
        ax.set_xlim(min_time, max_time)

        # タスクごとに異なる色を設定
        colors = plt.cm.get_cmap('tab10', len(tasks))
        for i, task in enumerate(tasks):
            start_time = task.scheduled_timing
            end_time = start_time + task.processing_time
            ax.barh(task.experiment_operation, end_time - start_time, left=start_time, color=colors(i), edgecolor='black')

        # 凡例を設定
        patches = [mpatches.Patch(color=colors(i), label=task.experiment_operation) for i, task in enumerate(tasks)]
        plt.legend(handles=patches)

        # グラフを表示
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()