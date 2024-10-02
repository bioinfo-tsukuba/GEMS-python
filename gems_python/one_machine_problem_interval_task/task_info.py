from enum import Enum
from dataclasses import dataclass, field, asdict
import json
from typing import List, Tuple, Type

from gems_python.one_machine_problem_interval_task.penalty.penalty_class import NonePenalty, PenaltyType



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
    interval: int        # タスク間のインターバル、最初のタスクにはインターバルはない
    experiment_operation: str
    completed: bool = False  # タスクが終了したかどうか
    scheduled_time: int = field(default=None)  # タスクの開始時刻
    task_id: int = field(default=None)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> 'Task':
        return cls(**data)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict())
    
    @classmethod
    def from_json(cls, json_str: str) -> 'Task':
        data = json.loads(json_str)
        return cls.from_dict(data)
    


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
    """
    A class to represent a group of tasks with scheduling and status management.
    Attributes:
    ----------
    optimal_start_time : int
        The optimal start time for the task group.
    penalty_type : Type[PenaltyType]
        The type of penalty associated with the task group.
    tasks : List[Task]
        A list of tasks in the task group.
    status : TaskGroupStatus
        The current status of the task group. Defaults to NOT_STARTED.
    group_id : int
        The unique identifier for the task group. Defaults to None.
    experiment_name : str
        The name of the experiment associated with the task group. Defaults to None.
    experiment_uuid : str
        The unique identifier for the experiment associated with the task group. Defaults to None.
    Methods:
    -------
    __post_init__():
        Allocates task IDs after initialization.
    to_dict() -> dict:
        Converts the task group to a dictionary.
    from_dict(data: dict) -> 'TaskGroup':
        Creates a TaskGroup instance from a dictionary.
    to_json() -> str:
        Converts the task group to a JSON string.
    from_json(json_str: str) -> 'TaskGroup':
        Creates a TaskGroup instance from a JSON string.
    is_completed() -> bool:
        Checks if the task group is completed.
    is_in_progress() -> bool:
        Checks if the task group is in progress.
    is_not_started() -> bool:
        Checks if the task group has not started.
    is_error() -> bool:
        Checks if the task group is in an error state.
    status_update():
        Updates the status of the task group based on the status of its tasks.
    schedule_tasks(start_time: int):
        Schedules the tasks in the group starting from the given start time.
    allocate_task_id():
        Allocates unique IDs to tasks in the group.
    configure_task_group_settings(experiment_name: str, experiment_uuid: str):
        Configures the settings for the task group with the given experiment name and UUID.
    """

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

    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TaskGroup':
        # Recreate the penalty type from its dictionary
        return cls(**data)

    def to_json(self) -> str:
        data = self.to_dict()
        data["penalty_type"] = self.penalty_type.to_json()
        data["status"] = self.status.value
        return json.dumps(data)
    
    @classmethod
    def from_json(cls, json_str: str) -> 'TaskGroup':
        data = json.loads(json_str)
        # Recreate the penalty type from its JSON string
        data["penalty_type"] = PenaltyType.from_json(data["penalty_type"])
        data["status"] = TaskGroupStatus(data["status"])
        data["tasks"] = [Task.from_dict(task_data) for task_data in data["tasks"]]
        return cls.from_dict(data)

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
        """
        Schedule the tasks in the group.
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
        """
        Add a task group to the list of task groups.
        :param task_groups: A list of task groups.
        :param group: The task group to add.
        :return: A list of task groups with the specified task group added.
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
    

    
    

@dataclass
class TaskScheduler:
    # TODO: TaskGroupのclassmethodにする
    task_groups: List[TaskGroup] = field(default_factory=list)
    schedule_reference_time: int = 0

    def __post_init__(self):
        # タスク群のIDを割り当て
        self.allocate_task_group_id()


    def to_dict(self):
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: dict) -> 'TaskScheduler':
        # Recreate the task groups from their dictionaries
        return cls(**data)

    def to_json(self) -> str:
        data = self.to_dict()
        data["task_groups"] = [group.to_json() for group in self.task_groups]
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'TaskScheduler':
        data = json.loads(json_str)
        # Recreate the task groups from their JSON strings
        data["task_groups"] = [TaskGroup.from_json(group_data) for group_data in data["task_groups"]]
        return cls.from_dict(data)

    def set_schedule_reference_time(self, time: int):
        self.schedule_reference_time = time

    def allocate_task_group_id(self):
        # TODO: TaskGroupのclassmethodにする
        # Not None id
        TaskGroup.set_task_group_ids(self.task_groups)

    def add_task_group(self, group: TaskGroup):
        # タスク群を追加
        self.task_groups.append(group)
        # タスク群のidを割り当て
        self.allocate_task_group_id()
        # タスク群のスケジュールを更新
        self.schedule_greedy()

    def _add_task_group_manual_style(
        self,
        group_id: int,
        T0: int,
        M: int,
        processing_times: List[int],
        intervals: List[int],
        penalty_type: Type[PenaltyType],
        experiment_operations: List[str],
        experiment_name: str,
        experiment_uuid: str,
    ):
        """
        Add a task group to the scheduler.

        :param group_id: タスク群のID
        :param T0: タスク群の最適な開始時刻
        :param M: タスクの数
        :param processing_times: タスクの処理時間のリスト、長さはM
        :param intervals: タスク間のインターバル（前タスクの終了時刻から次のタスクの開始時刻までの時間）のリスト、最初のタスクにはインターバルはないためその長さはM-1
        :param penalty_type: ペナルティの種類
        :param experiment_operations: 各タスクの実験操作のリスト、長さはM
        :param experiment_name: 実験の名前
        :param experiment_uuid: 実験のUUID
        """
        assert len(processing_times) == M, "processing_timesの長さがMと一致しません。"
        assert len(intervals) == M - 1, "intervalsの長さがM-1と一致しません。"
        assert len(experiment_operations) == M, "experiment_operationsの長さがMと一致しません。"

        # タスク群がすでに存在するかどうかを確認
        existing_group = self.find_task_group(group_id)
        if existing_group:
            print(f"タスク群 {group_id} はすでに存在します。")
            return

        # 新しいタスク群を追加
        tasks = [
            Task(
                processing_time=processing_times[i],
                interval=intervals[i - 1] if i > 0 else 0,
                experiment_operation=experiment_operations[i],
                scheduled_time=0,
                task_id=None
            )
            for i in range(M)
        ]

        new_group = TaskGroup(
            task_group_id=group_id,
            optimal_start_time=T0,
            penalty_type=penalty_type,
            tasks=tasks,
            experiment_name=experiment_name,
            experiment_uuid=experiment_uuid
        )

        self.task_groups.append(new_group)

        # タスク群のスケジュールを更新
        self.schedule_greedy()


    def complete_task(self, group_id: int, task_id: int):
        group = self.find_task_group(group_id)
        if not group or task_id >= len(group.tasks):
            print(f"タスク群 {group_id} またはタスク {task_id} が存在しません。")
            return

        task = group.tasks[task_id]
        if task.completed:
            print(f"タスク群 {group_id} のタスク {task_id} は既に終了しています。")
            return

        # タスクを終了としてマーク
        task.completed = True
        print(f"タスク群 {group_id} のタスク {task_id} が終了しました")

        # タスク群のステータスを更新
        group.status_update()

        # 全てのタスクが終了しているかをチェック
        if group.is_completed():
            print(f"タスク群 {group_id} が全て終了しました")

    def find_task_group(self, group_id: int) -> TaskGroup:
        # 指定されたグループ番号に一致するタスク群を探す
        for group in self.task_groups:
            if group.task_group_id == group_id:
                return group
        return None
    
    def eval_machine_penalty(self, eval_group_id: List[int] = None) -> int:
        """
        Evaluate the penalty for the machine.
        """

        occupied_time = list()

        if eval_group_id is None:
            eval_group_id = [group.task_group_id for group in self.task_groups]
        for group in self.task_groups:
            if group.task_group_id in eval_group_id:
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


    def schedule_greedy(self)  -> List[TaskGroup]:
        """
        Schedule all the groups one by one group.
        """
        # すでに終わっているタスクも含めてゼロの位置を決める
        # スケジュールするやつの基準はそれはそれで決める
        # 例えば、available_reference_time

        # タスク群をNOT_STARTEDを最後にして、開始時刻の昇順にソート
        self.task_groups.sort(key=lambda group: (group.is_not_started(), group.optimal_start_time))
        
        # タスク群を順番にスケジュール
        scheduled_groups = list()
        for group in self.task_groups:
            diff = 0
            scheduled_groups.append(group.task_group_id)
            while True:
                # もし、スケジュール基準時刻がタスク群の最適な開始時刻よりも遅い場合、スケジュール基準時刻を最適な開始時刻に設定
                # そうでない場合、スケジュール基準時刻を最適な開始時刻に設定
                # スケジュール基準時刻より遅い範囲で、タスク群のスケジュールを試行（0, 1, -1, 2, -2, ...）
                # ちょっと遅いが、とりあえず実装はこれでいいかも
                # TODO: 実装を軽くするために、無駄なdiffの計算をなくす。すなわち、group.optimal_start_time + diff >= self.schedule_reference_timeを満たすdiffだけを試す
                time_candidate = max(self.schedule_reference_time, group.optimal_start_time + diff)
                group.schedule_tasks(time_candidate)
                if self.eval_machine_penalty(scheduled_groups) == 0:
                    break

                if diff <= 0:
                    diff *= -1
                    diff += 1
                else:
                    diff *= -1

        return scheduled_groups
    

def main():
    task_scheduler = TaskScheduler()

    task_group_1 = TaskGroup(
        optimal_start_time=0,
        penalty_type=NonePenalty(),
        tasks=[
            Task(processing_time=2, interval=0, experiment_operation="A"),
            Task(processing_time=3, interval=15, experiment_operation="B"),
            Task(processing_time=4, interval=20, experiment_operation="C")
        ]
    )

    task_group_2 = TaskGroup(
        optimal_start_time=2,
        penalty_type=NonePenalty(),
        tasks=[
            Task(processing_time=2, interval=0, experiment_operation="A"),
            Task(processing_time=3, interval=1, experiment_operation="B"),
            Task(processing_time=4, interval=2, experiment_operation="C")
        ]
    )
    
    print(task_group_1.tasks)

    task_scheduler.add_task_group(task_group_1)
    print(task_scheduler.task_groups)
    task_scheduler.add_task_group(task_group_2)

    desired_task_schedule_1 = [0, 17, 40]
    desired_task_schedule_2 = [2, 5, 10]

    for group in task_scheduler.task_groups:
        print(group.tasks)

    for group, desired_task_schedule in zip(task_scheduler.task_groups, [desired_task_schedule_1, desired_task_schedule_2]):
        for task, desired_start_time in zip(group.tasks, desired_task_schedule):
            assert task.scheduled_time == desired_start_time



    # タスクをスケジュール
    # 本来できない操作
    task_scheduler.task_groups[1].schedule_tasks(0)
    for task in task_scheduler.task_groups[1].tasks:
        print(task.scheduled_time)
    # タスクを終了
    # task_scheduler.complete_task(1, 0)
    # task_scheduler.complete_task(1, 1)
    # task_scheduler.complete_task(1, 2)
    task_scheduler.complete_task(2, 0)
    task_scheduler.set_schedule_reference_time(2)

    # タスクをスケジュール
    task_scheduler.schedule_greedy()
    for group in task_scheduler.task_groups:
        print(group.task_group_id)
        print(group.tasks)


    original = task_scheduler.task_groups[0]
    js = original.to_json()
    js_file_name = "volatile_task_scheduler.json"

    # JSON形式で保存
    with open(js_file_name, "w") as f:
        f.write(js)

    # JSON形式で読み込み
    with open(js_file_name, "r") as f:
        js = f.read()

    # JSON形式からTaskSchedulerオブジェクトを復元
    read = original.from_json(js)

    print(f"ori:\n{original}")
    print(f"read:\n{read}")

    assert original == read


    original = task_scheduler
    js = original.to_json()
    js_file_name = "volatile_task_scheduler.json"

    # JSON形式で保存
    with open(js_file_name, "w") as f:
        f.write(js)

    # JSON形式で読み込み
    with open(js_file_name, "r") as f:
        js = f.read()

    # JSON形式からTaskSchedulerオブジェクトを復元
    read = original.from_json(js)

    print(f"ori:\n{original}")
    print(f"read:\n{read}")

    assert original == read

    

if __name__ == "__main__":
    main()
