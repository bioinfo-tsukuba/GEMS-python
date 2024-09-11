from enum import Enum
from dataclasses import dataclass, field, asdict
import json
from typing import List, Type

from gems_python.one_machine_problem.penalty.penalty_class import NonePenalty, PenaltyType



class TaskGroupStatus(Enum):
    NOT_STARTED = "Not Started"
    IN_PROGRESS = "In Progress"
    COMPLETED = "Completed"
    ERROR = "Error"

@dataclass
class Task:
    start_time: int    # タスクの開始時刻
    processing_time: int  # タスクの処理時間
    interval: int        # タスク間のインターバル、最初のタスクにはインターバルはない
    completed: bool = False  # タスクが終了したかどうか

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
    


@dataclass
class TaskGroup:
    group_id: int               # グループ番号
    optimal_start_time: int      # 最適な開始時刻
    penalty_type: Type[PenaltyType] # ペナルティの種類
    tasks: List[Task] = field(default_factory=list)  # タスクのリスト
    status: TaskGroupStatus = TaskGroupStatus.NOT_STARTED  # デフォルトで未開始

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
            print(f"タスク群 {self.group_id} はすでに進行しています。")
            return

        # 最適な開始時刻に合わせて、タスクの開始時刻を設定
        current_time = start_time
        for task in self.tasks:
            current_time += task.interval
            task.start_time = current_time
            current_time += task.processing_time

@dataclass
class TaskScheduler:
    task_groups: List[TaskGroup] = field(default_factory=list)
    schedule_reference_time: int = 0


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

    def add_task_group(self, group_id: int, T0: int, M: int, processing_times: List[int], intervals: List[int], penalty_type: Type[PenaltyType]):
        """
        Add a task group to the scheduler.

        :param group_id: タスク群のID
        :param T0: タスク群の最適な開始時刻
        :param M: タスクの数
        :param processing_times: タスクの処理時間のリスト、長さはM
        :param intervals: タスク間のインターバル（前タスクの終了時刻から次のタスクの開始時刻までの時間）のリスト、 最初のタスクにはインターバルはないためその長さはM-1
        :param penalty_type: ペナルティの種類
        """
        assert len(processing_times) == M
        assert len(intervals) == M - 1
        # タスク群がすでに存在するかどうかを確認
        existing_group = self.find_task_group(group_id)
        if existing_group:
            print(f"タスク群 {group_id} はすでに存在します。")
            return

        # 新しいタスク群を追加
        tasks = [Task(start_time=0, processing_time=processing_times[i], interval=intervals[i-1] if i > 0 else 0) for i in range(M)]
        new_group = TaskGroup(group_id=group_id, optimal_start_time=T0, tasks=tasks, penalty_type=penalty_type)

        self.task_groups.append(new_group)

        # タスク群のスケジュール
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
            if group.group_id == group_id:
                return group
        return None
    
    def eval_machine_penalty(self, eval_group_id: List[int] = None) -> int:
        """
        Evaluate the penalty for the machine.
        """

        occupied_time = list()

        if eval_group_id is None:
            eval_group_id = [group.group_id for group in self.task_groups]
        for group in self.task_groups:
            if group.group_id in eval_group_id:
                for task in group.tasks:
                    occupied_time.append((task.start_time, True))
                    occupied_time.append((task.start_time + task.processing_time, False))

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
            scheduled_groups.append(group.group_id)
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
    task_scheduler.add_task_group(1, 0, 3, [2, 3, 4], [15, 20], NonePenalty())
    task_scheduler.add_task_group(2, 0, 3, [2, 3, 4], [1, 2], NonePenalty())

    desired_task_schedule_1 = [0, 17, 40]
    desired_task_schedule_2 = [2, 5, 10]

    for group in task_scheduler.task_groups:
        print(group.tasks)

    for group, desired_task_schedule in zip(task_scheduler.task_groups, [desired_task_schedule_1, desired_task_schedule_2]):
        for task, desired_start_time in zip(group.tasks, desired_task_schedule):
            assert task.start_time == desired_start_time



    # タスクをスケジュール
    # 本来できない操作
    task_scheduler.task_groups[1].schedule_tasks(0)
    for task in task_scheduler.task_groups[1].tasks:
        print(task.start_time)
    # タスクを終了
    # task_scheduler.complete_task(1, 0)
    # task_scheduler.complete_task(1, 1)
    # task_scheduler.complete_task(1, 2)
    task_scheduler.complete_task(2, 0)
    task_scheduler.set_schedule_reference_time(2)

    # タスクをスケジュール
    task_scheduler.schedule_greedy()
    for group in task_scheduler.task_groups:
        print(group.group_id)
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


        

        





