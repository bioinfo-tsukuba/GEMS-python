"""Importable, finite protocols; no random observations or wall-clock input."""
from pathlib import Path

from gems_python.common import dataframes as frames
from gems_python.multi_machine_problem_interval_task.lifecycle import ChildSpec, Finish, Split
from gems_python.multi_machine_problem_interval_task.task_info import Machine, MachineList, Task, TaskGroup
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from gems_python.multi_machine_problem_interval_task.transition_manager import State, Experiment, Experiments


class FiniteState(State):
    def __init__(self, limit=2, machine_type=0, duration=1):
        self.limit, self.machine_type, self.duration = limit, machine_type, duration

    def task_generator(self, history):
        time = 0 if not len(history) else int(frames.last_value(history, "time"))
        return TaskGroup(time, LinearPenalty(1), [Task(self.duration, "observe", self.machine_type)])

    def transition_function(self, history):
        return self.state_name

    def lifecycle_action(self, history, context):
        if len(history) and frames.last_value(history, "value") >= self.limit:
            return Finish("protocol complete")


class EmptyState(FiniteState):
    def task_generator(self, history):
        return TaskGroup(0, LinearPenalty(1), [])

    def lifecycle_action(self, history, context):
        return Finish("zero-task protocol complete")


class LegacyEmptyState(EmptyState):
    def lifecycle_action(self, history, context):
        return None


class SplitState(FiniteState):
    def lifecycle_action(self, history, context):
        if len(history) and frames.last_value(history, "value") >= self.limit:
            return Split(child_specs(frames.backend_of(history), context.logical_time))


def child_specs(backend="pandas", time=0):
    return [
        ChildSpec(f"child-{i}", f"child-{i}", [FiniteState(limit=2+i)], "FiniteState",
                  frames.make_frame({"time": [time], "value": [0]}, backend),
                  {"volume": 5.0, "unit": "virtual-volume", "container_id": f"container-{i}"},
                  {"volume": "protocol:equal-aliquot", "unit": "protocol:test-unit",
                   "container_id": "protocol:new-container"})
        for i in range(2)
    ]


def manager(directory, backend="pandas", split=False):
    state = SplitState() if split else FiniteState()
    exp = Experiment("parent", [state], state.state_name,
                     experiment_uuid="parent", dataframe_backend=backend)
    return Experiments([exp], Path(directory), MachineList([Machine(0, "m0")]),
                       enable_split=split)


def observe(state, history, group, task, time):
    value = 0 if not len(history) else frames.last_value(history, "value")
    return frames.make_frame({"time": [time], "value": [value + 1]},
                             frames.backend_of(history))


def advance(lab):
    result = lab.simulate_one(observation_provider=observe)
    for event in list(lab.split_requests.values()):
        if event.status == "PENDING":
            lab.confirm_split(event.request_id)
    return result
