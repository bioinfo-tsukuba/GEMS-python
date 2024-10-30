from enum import Enum
from dataclasses import field, asdict
from pathlib import Path
import random

from simanneal import Annealer
from gems_python.common.class_dumper import auto_dataclass as dataclass
import json
from typing import List, Tuple, Type

from gems_python.one_machine_problem_interval_task.penalty.penalty_class import NonePenalty, PenaltyType

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import matplotlib.path as mpath


@dataclass
class Task:
    """
    A class to represent a task with scheduling and status management.
    Attributes:
    ----------
    processing_time : int
        The processing time of the task.
    interval : int
        The interval between the task and the previous task. There is no interval for the first task.
    experiment_operation : str
        The experiment operation associated with the task.
    completed : bool
        Whether the task has been completed.
    scheduled_time : int
        The start time of the task. Defaults to None.
    task_id : int
        The unique identifier for the task. Defaults to None.
    """
    processing_time: int  # タスクの処理時間
    experiment_operation: str
    interval: int = field(default=0)        # タスク間のインターバル、最初のタスクにはインターバルはない
    completed: bool = False  # タスクが終了したかどうか
    scheduled_time: int = field(default=None)  # タスクの開始時刻
    task_id: int = field(default=None)

    @classmethod
    def find_task(cls, tasks: List['Task'], task_id: int) -> int:
        """
        Find the index of the task with the given ID in the list of tasks.
        :param tasks: A list of tasks.
        :param task_id: The ID of the task to find.
        :return: The index of the task with the given ID in the list of tasks.
        """
        for index, task in enumerate(tasks):
            if task.task_id == task_id:
                return index
        return None

class TaskGroupStatus(Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    ERROR = "Error"

@dataclass
class TaskGroup:
    optimal_start_time: int      # 最適な開始時刻
    penalty_type: Type[PenaltyType] # ペナルティの種類
    tasks: List[Task] = field(default_factory=list)  # タスクのリスト
    status: TaskGroupStatus = TaskGroupStatus.NOT_STARTED  # デフォルトで未開始
    task_group_id: int = field(default=None) # グループ番号. Experiments 生成時に割り当て
    experiment_name: str = field(default=None)  # 実験の名前. Experiment 生成時に割り当て
    experiment_uuid: str = field(default=None)  # 実験のUUID. Experiment 生成時に割り当て


    def __post_init__(self):
        # タスクのIDを割り当て
        self._allocate_task_id()

    def is_completed(self) -> bool:
        # タスク群が完了しているかどうかを確認
        return self.status == TaskGroupStatus.COMPLETED
    
    def is_in_progress(self) -> bool:
        # タスク群が進行中かどうかを確認
        return self.status == TaskGroupStatus.IN_PROGRESS
    
    def is_not_started(self) -> bool:
        # タスク群が未開始かどうかを確認
        return self.status == TaskGroupStatus.NOT_STARTED
    
    def is_error(self) -> bool:
        # タスク群がエラーかどうかを確認
        return self.status == TaskGroupStatus.ERROR
    
    def status_update(self):
        # タスク群のステータスを更新
        if all(task.completed for task in self.tasks):
            self.status = TaskGroupStatus.COMPLETED
        elif any(task.completed for task in self.tasks):
            self.status = TaskGroupStatus.IN_PROGRESS
        else:
            self.status = TaskGroupStatus.NOT_STARTED
    
    def schedule_tasks(self, start_time: int):
        """_summary_

        Args:
            start_time (int): _description_
        """
        if start_time is None:
            start_time = self.optimal_start_time
        # タスクのスケジュールを計算
        if self.status != TaskGroupStatus.NOT_STARTED:
            print(f"タスク群 {self.task_group_id} はすでに進行しています。")
            return

        # 最適な開始時刻に合わせて、タスクの開始時刻を設定
        current_time = start_time
        for task in self.tasks:
            current_time += task.interval
            task.scheduled_time = current_time
            current_time += task.processing_time

    def _allocate_task_id(self):
        """
        Allocate unique IDs to the tasks in the group.
        """
        # Not None id
        used_task_ids_set = {task.task_id for task in self.tasks}
        new_task_id = 0
        for task_index in range(len(self.tasks)):
            if self.tasks[task_index].task_id is None:
                while new_task_id in used_task_ids_set:
                    new_task_id += 1
                self.tasks[task_index].task_id = new_task_id
                used_task_ids_set.add(new_task_id)

    def configure_task_group_settings(self, experiment_name: str, experiment_uuid: str):
        self.experiment_name = experiment_name
        self.experiment_uuid = experiment_uuid

    def get_ealiest_task(self) -> Task:
        """
        Get the task with the earliest scheduled time.
        """
        earliest_task = None
        for task in self.tasks:
            if task.completed:
                continue
            if earliest_task is None or task.scheduled_time < earliest_task.scheduled_time:
                earliest_task = task
        return earliest_task

    # --以下のメソッドは、scheduling に関するものである。

    @classmethod
    def set_task_group_ids(cls, task_groups: List['TaskGroup']) -> List['TaskGroup']:
        """
        Set the task group IDs for a list of task groups.
        """
        for group_index in range(len(task_groups)):
            task_groups[group_index].task_group_id = group_index

        return task_groups

    @classmethod
    def find_task_group(cls, task_groups: List['TaskGroup'], group_id: int) -> int:
        # 指定されたグループ番号に一致するタスク群のインデックスを探す
        for index, group in enumerate(task_groups):
            if group.task_group_id == group_id:
                return index
        return None
    
    @classmethod
    def add_task_group(cls, task_groups: List['TaskGroup'], group: 'TaskGroup') -> List['TaskGroup']:
        """_summary_

        Args:
            task_groups (List['TaskGroup']): Current task groups.
            group ('TaskGroup'): New task group to add.

        Returns:
            List['TaskGroup']: Updated task groups.
        """        

        # タスク群を追加
        task_groups.append(group)
        # タスク群のidを割り当て
        task_groups = cls.set_task_group_ids(task_groups)
        # タスク群のスケジュールを更新
        task_groups = cls.schedule_task_groups(task_groups, reference_time=0)
        
        return task_groups

    
    @classmethod
    def delete_task_group(cls, task_groups: List['TaskGroup'], group_id: int) -> List['TaskGroup']:
        """
        Delete a task group from the list of task groups.
        :param task_groups: A list of task groups.
        :param group_id: The ID of the task group to delete.
        :return: A list of task groups with the specified task group deleted.
        """
        group_index = cls.find_task_group(task_groups, group_id)
        if group_index is None:
            print(f"タスク群 {group_id} が存在しません。")
            return
        
        task_groups.pop(group_index)
        print(f"タスク群 {group_id} を削除しました。")
        return task_groups
    
    @classmethod
    def find_task(cls, task_groups: List['TaskGroup'], group_id: int, task_id: int) -> Tuple[int, int]:
        """
        Find the index of the task with the given ID in the list of tasks.
        :param task_groups: A list of task groups.
        :param group_id: The ID of the task group.
        :param task_id: The ID of the task to find.
        :return: The index of the task with the given ID in the list of tasks.
        """
        group_index = cls.find_task_group(task_groups, group_id)
        if group_index is None:
            print(f"タスク群 {group_id}が存在しません。")
            return

        task_index = Task.find_task(task_groups[group_index].tasks, task_id)
        if task_index is None:
            print(f"タスク群 {group_id}にタスク {task_id}が存在しません。")
            return
        
        return group_index, task_index

    @classmethod
    def complete_task(cls, task_groups: List['TaskGroup'], group_id: int, task_id: int) -> List['TaskGroup']:
        """
        Complete a task in the task group.

        :param task_groups: A list of task groups.
        :param group_id: The ID of the task group.
        :param task_id: The ID of the task to complete.

        :return: A list of task groups with the completed task.
        """
        group_index, task_index = cls.find_task(task_groups, group_id, task_id)

        task = task_groups[group_index].tasks[task_index]
        if task.completed:
            print(f"タスク群 {group_id} のタスク {task_id} は既に終了しています。")
            return

        # タスクを終了としてマーク
        task_groups[group_index].tasks[task_index].completed = True
        print(f"タスク群 {group_id} のタスク {task_id} が終了しました")

        # タスク群のステータスを更新
        task_groups[group_index].status_update()

        # 全てのタスクが終了しているかをチェック
        if task_groups[group_index].is_completed():
            print(f"タスク群 {group_id} が全て終了しました")

        return task_groups

    @classmethod
    def schedule_task_groups(cls, task_groups: List['TaskGroup'], reference_time: int) -> List['TaskGroup']:
        """
        Schedule a list of task groups.

        :param task_groups: A list of task groups to schedule.
        :param reference_time: The reference time for scheduling the task groups.

        :return: A list of scheduled task groups.
        """

        """
        Schedule all the groups one by one group.
        """
        # すでに終わっているタスクも含めてゼロの位置を決める
        # スケジュールするやつの基準はそれはそれで決める
        # 例えば、available_reference_time

        # タスク群をNOT_STARTEDを最後にして、開始時刻の昇順にソート
        task_groups = task_groups.copy()
        task_groups.sort(key=lambda group: (group.is_not_started(), group.optimal_start_time))
        
        # タスク群を順番にスケジュール
        scheduled_groups = list()
        for group in task_groups:
            diff = 0
            scheduled_groups.append(group.task_group_id)
            while True:
                # もし、スケジュール基準時刻がタスク群の最適な開始時刻よりも遅い場合、スケジュール基準時刻を最適な開始時刻に設定
                # そうでない場合、スケジュール基準時刻を最適な開始時刻に設定
                # スケジュール基準時刻より遅い範囲で、タスク群のスケジュールを試行（0, 1, -1, 2, -2, ...）
                # ちょっと遅いが、とりあえず実装はこれでいいかも
                # TODO: 実装を軽くするために、無駄なdiffの計算をなくす。すなわち、group.optimal_start_time + diff >= self.schedule_reference_timeを満たすdiffだけを試す
                time_candidate = max(reference_time, group.optimal_start_time + diff)
                group.schedule_tasks(time_candidate)
                if cls.eval_machine_penalty(task_groups, scheduled_groups) == 0:
                    break

                if diff <= 0:
                    diff *= -1
                    diff += 1
                else:
                    diff *= -1

        return task_groups
    


    @classmethod
    def schedule_task_groups_simulated_annealing(cls, task_groups: List['TaskGroup'], reference_time: int) -> List['TaskGroup']:
        """

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
                # PASS: ここで、タスクのスケジュールを変更する
                for i in range(min(temp, len(self.state))):
                  a = random.randint(0, len(self.state) - 1)
                  scheduled_time = self.state[a].tasks[0].scheduled_time - self.state[a].tasks[0].interval
                  scheduled_time += random.randint(-int(temp), int(temp))
                  scheduled_time = max(reference_time, scheduled_time)
                  self.state[a].schedule_tasks(scheduled_time)

            def energy(self):
                """Calculates the total penalty for the current state."""
                # PASS: ここで、タスクのペナルティを計算する
                total_penalty = 0
                for task_group in self.state:
                    total_penalty += task_group.penalty_type.calculate_penalty(
                        task_group.tasks[0].scheduled_time, task_group.optimal_start_time
                    )

                # Overlapping penalty
                overlap = cls.eval_machine_penalty(self.state)
                total_penalty += overlap * 100000
                return total_penalty

        # Initialize the tasks with some initial schedule (e.g., their optimal timings)
        time = 0
        task_groups = cls.schedule_task_groups(task_groups=task_groups, reference_time=reference_time)

        # Create an instance of the annealer with the initial state
        annealer = TaskAnnealer(task_groups)
        # Set the annealing parameters as needed
        annealer.steps = 100000
        annealer.Tmax = 25000.0
        annealer.Tmin = 1.0

        # Run the annealing process
        state, _ = annealer.anneal()

        return state

    @classmethod
    def eval_machine_penalty(cls, task_groups: List['TaskGroup'], eval_group_ids: List[int] = None) -> int:
        """
        Evaluate the penalty for the machine.
        """
        occupied_time = list()

        if eval_group_ids is None:
            eval_group_ids = [group.task_group_id for group in task_groups]
        for group in task_groups:
            if group.task_group_id in eval_group_ids:
                for task in group.tasks:
                    occupied_time.append((task.scheduled_time, True))
                    occupied_time.append((task.scheduled_time + task.processing_time, False))

        occupied_time.sort(key=lambda x: (x[0], x[1]))
        penalty = 0
        count = 0
        privous_time = occupied_time[0][0]
        for time, is_start in occupied_time:
            if count >= 2:
                penalty += (time - privous_time)*(count - 1)

            if is_start:
                count += 1
            else:
                count -= 1
            privous_time = time

        assert count == 0

        return penalty
    


    @classmethod
    def eval_schedule_penalty(cls, task_groups: List['TaskGroup'], eval_group_ids: List[int] = None) -> int:
        """
        Evaluate the penalty for the schedule.
        """
        penalty = 0
        for group in task_groups:
            scheduled_time = group.tasks[0].scheduled_time
            penalty += group.penalty_type.calculate_penalty(scheduled_time=scheduled_time, optimal_time=group.optimal_start_time)

        return penalty
    
    @classmethod
    def get_ealiest_task_in_task_groups(cls, task_groups: List['TaskGroup']) -> Tuple['Task', int]:
        """
        Get the task with the earliest scheduled time in the task groups.
        :param task_groups: A list of task groups.
        :return: The task with the earliest scheduled time and the ID of the group it belongs to.
        """
        earliest_task = None
        group_id = None
        for group in task_groups:
            task = group.get_ealiest_task()
            if earliest_task is None or (task is not None and task.scheduled_time < earliest_task.scheduled_time):
                earliest_task = task
                group_id = group.task_group_id
        return earliest_task, group_id
    
    @classmethod
    def generate_gantt_chart(cls, task_groups: List['TaskGroup'], save_dir: Path = None, file_name:str = "gantt_chart"):

        """
        Generates a Gantt chart from a list of TaskGroups, reflecting their statuses.
        
        Parameters:
        ----------
        task_groups : List[TaskGroup]
            The list of TaskGroup instances to visualize.
        """
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Define colors based on TaskGroupStatus
        status_colors = {
            'NOT_STARTED': 'gray',
            'IN_PROGRESS': 'blue',
            'COMPLETED': 'green',
            'ERROR': 'red'
        }
        
        # Y positions
        yticks = []
        yticklabels = []
        
        for idx, tg in enumerate(task_groups):
            # Determine color based on status
            status = tg.status.name
            color = status_colors.get(status, 'black')  # Default to black if status not found
            y = idx * 10  # Space out each TaskGroup vertically by 10 units
            yticks.append(y + 5)
            yticklabels.append(f"Group {tg.task_group_id} - {tg.experiment_name} ({tg.status.value})")
            
            # Plot optimal_start_time as a vertical dotted line
            ax.axvline(x=tg.optimal_start_time, color='orange', linestyle='dotted', linewidth=1, label='Optimal Start Time' if idx == 0 else "")
            
            # Sort tasks based on scheduled_time
            sorted_tasks = sorted(tg.tasks, key=lambda t: t.scheduled_time if t.scheduled_time is not None else 0)
            
            for t_idx, task in enumerate(sorted_tasks):
                if task.scheduled_time is None:
                    continue  # Skip tasks without a scheduled_time
                
                start = task.scheduled_time
                duration = task.processing_time
                ax.barh(y, duration, left=start, height=4, align='center', color=color, edgecolor='black')
                ax.text(start + duration/2, y, f"Task {task.task_id}", va='center', ha='center', color='white', fontsize=8)
                
                # If not the first task, draw interval
                if t_idx > 0:
                    prev_task = sorted_tasks[t_idx - 1]
                    prev_end = prev_task.scheduled_time + prev_task.processing_time
                    interval = task.interval
                    interval_start = prev_end
                    interval_end = task.scheduled_time
                    # Draw a curved dotted line representing the interval
                    control_offset = 2  # Control point offset for the curve
                    verts = [
                        (interval_start, y + 2),  # Start point
                        (interval_start + (interval_end - interval_start)/2, y + 2 + control_offset),  # Control point
                        (interval_end, y + 2)  # End point
                    ]
                    codes = [mpath.Path.MOVETO, mpath.Path.CURVE3, mpath.Path.CURVE3]
                    path = mpath.Path(verts, codes)
                    patch = mpatches.PathPatch(path, facecolor='none', edgecolor='black', linestyle='dotted', linewidth=1)
                    ax.add_patch(patch)
        
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticklabels)
        ax.set_xlabel('Time')
        ax.set_title('Gantt Chart of Task Groups')
        ax.grid(True, axis='x', linestyle='--', alpha=0.5)
        
        # Create custom legend
        status_patches = [
            mpatches.Patch(color=color, label=status.replace('_', ' ').title()) 
            for status, color in status_colors.items()
        ]
        optimal_patch = mlines.Line2D([], [], color='orange', linestyle='dotted', label='Optimal Start Time')
        ax.legend(handles=status_patches + [optimal_patch], bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()


        if save_dir is not None:
            save_dir = Path(save_dir)
            plt.savefig(save_dir / f"{file_name}.png")
            plt.savefig(save_dir / f"{file_name}.pdf")
            plt.savefig(save_dir / f"{file_name}.svg")
        else:
            plt.show()