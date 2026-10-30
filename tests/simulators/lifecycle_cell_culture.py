"""Finite culture fixture with explicit virtual aliquots, not a physical model."""
from pathlib import Path

from gems_python.common import dataframes as frames
from gems_python.multi_machine_problem_interval_task.lifecycle import ChildSpec, Finish, Split
from gems_python.multi_machine_problem_interval_task.transition_manager import State, Experiment, Experiments
from gems_python.multi_machine_problem_interval_task.task_info import Machine, MachineList, Task, TaskGroup
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from tests.simulators.cell_culture import logistic_growth, OBSERVATION_INTERVAL_MINUTES


class CultureState(State):
    def __init__(self, generation=0):
        self.generation = generation

    def task_generator(self, history):
        time = int(frames.last_value(history, "time"))
        return TaskGroup(time + OBSERVATION_INTERVAL_MINUTES, LinearPenalty(1),
                         [Task(10, "observe-culture", 1)])

    def transition_function(self, history):
        return self.state_name

    def lifecycle_action(self, history, context):
        rows = frames.convert(history, "polars")
        own = rows.filter((rows["generation"] == self.generation) & (rows["event"] == "observation"))
        if len(own) < 2:
            return None
        if self.generation:
            return Finish("two child observations completed")
        children = [
            ChildSpec(f"aliquot-{i}", f"Aliquot{i}", [CultureState(1)], "CultureState",
                      frames.make_frame({"time": [context.logical_time], "confluence": [seed],
                                         "generation": [1], "event": ["initial"]},
                                        frames.backend_of(history)),
                      {"volume": 5.0, "unit": "virtual-volume", "container_id": f"vessel-{i}",
                       "confluence": seed},
                      {"volume": "protocol:equal-aliquot", "unit": "fixture",
                       "container_id": "protocol:new-vessel", "confluence": "protocol:reseeding"})
            for i, seed in enumerate((0.12, 0.18))
        ]
        return Split(children)


def observe(state, history, group, task, time):
    previous = int(frames.last_value(history, "time"))
    value = logistic_growth(float(frames.last_value(history, "confluence")), (time - previous) / 60)
    return frames.make_frame({"time": [time], "confluence": [value],
                              "generation": [state.generation], "event": ["observation"]},
                             frames.backend_of(history))


def build(directory, backend="pandas"):
    history = frames.make_frame({"time": [0], "confluence": [0.12],
                                 "generation": [0], "event": ["initial"]}, backend)
    parent = Experiment("CultureParent", [CultureState()], "CultureState", history,
                        experiment_uuid="culture-parent", dataframe_backend=backend,
                        initial_conditions={"volume": 10.0, "unit": "virtual-volume"},
                        initial_condition_origins={"volume": "fixture", "unit": "fixture"})
    return Experiments([parent], Path(directory), MachineList([Machine(1, "imager")]),
                       enable_split=True)


def run(lab, steps=20):
    logs = []
    for _ in range(steps):
        result = lab.simulate_one(observation_provider=observe)
        logs.append(result)
        for event in list(lab.split_requests.values()):
            if event.status == "PENDING":
                # Virtual protocol explicitly confirms its successful aliquoting.
                lab.confirm_split(event.request_id)
        if result["status"] != "ok":
            break
    return logs
