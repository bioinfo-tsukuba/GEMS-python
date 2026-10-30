"""Small virtual protocol: two roots, two generations, two shared machines.

Run through ``python -m scripts.run_experiment_retrospective_example`` so the
State class retains this importable module name in saved snapshots.
"""
from pathlib import Path

from gems_python.common import dataframes as frames
from gems_python.multi_machine_problem_interval_task.lifecycle import ChildSpec, Finish, Split
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from gems_python.multi_machine_problem_interval_task.task_info import Machine, MachineList, Task, TaskGroup
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, Experiments, State


CHILDREN = {"A": ("A1", "A2"), "A2": ("A2a", "A2b"), "B": ("B1", "B2")}


class ReviewState(State):
    def __init__(self, name):
        self.name = name

    def task_generator(self, history):
        born = int(frames.last_value(history, "time"))
        return TaskGroup(born + 2, LinearPenalty(1), [
            Task(3, "capture", 1), Task(4, "process", 2, interval=2),
            Task(2, "capture", 1, interval=2)])

    def transition_function(self, history):
        return self.state_name

    def lifecycle_action(self, history, context):
        children = CHILDREN.get(self.name)
        if not children:
            return Finish("complete")
        backend = frames.backend_of(history)
        return Split([
            ChildSpec(name, name, [ReviewState(name)], "ReviewState",
                      frames.make_frame({"time": [context.logical_time], "value": [0]}, backend),
                      {"container": name}, {"container": "virtual:new-container"})
            for name in children])


def observe(state, history, group, task, time):
    return frames.make_frame({"time": [time],
                              "value": [int(frames.last_value(history, "value")) + 1]},
                             frames.backend_of(history))


def run(directory, backend="pandas"):
    parents = [Experiment(name, [ReviewState(name)], "ReviewState",
                          frames.make_frame({"time": [0], "value": [0]}, backend),
                          experiment_uuid=f"review-{name}", dataframe_backend=backend)
               for name in ("A", "B")]
    lab = Experiments(parents, Path(directory), MachineList([
        Machine(1, "imager", "Imager"), Machine(2, "processor", "Processor")]),
        enable_split=True, solver_time_limit_seconds=1)
    events = []
    for _ in range(40):
        event = lab.simulate_one(observation_provider=observe)
        events.append(event)
        for request in list(lab.split_requests.values()):
            if request.status == "PENDING":
                lab.confirm_split(request.request_id)
        if event["status"] != "ok":
            break
    if lab.experiments or events[-1]["status"] != "no_tasks":
        raise RuntimeError("Review example did not finish")
    if len(lab.ended_experiments) != 8 or len(lab.retrospective_trace["tasks"]) != 24:
        raise RuntimeError("Review example lost experiment/task records")
    return lab, events
