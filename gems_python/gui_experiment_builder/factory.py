import polars as pl

from gems_python.gui_experiment_builder.models import ExperimentDraft, StateDraft
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import NonePenalty
from gems_python.multi_machine_problem_interval_task.task_info import Task, TaskGroup
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, State


class DraftState(State):
    def __init__(self, draft: StateDraft):
        super().__init__()
        self.draft = draft

    @property
    def state_name(self) -> str:
        return self.draft.name

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        task = Task(
            processing_time=self.draft.processing_time,
            interval=self.draft.interval,
            experiment_operation=self.draft.operation,
            optimal_machine_type=self.draft.machine_type,
        )
        return TaskGroup(
            optimal_start_time=0,
            penalty_type=NonePenalty(),
            tasks=[task],
        )

    def transition_function(self, df: pl.DataFrame) -> str:
        return self.draft.next_state

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        if df.height == 0 or "time" not in df.columns:
            next_time = 0
        else:
            next_time = int(df["time"].max()) + 1
        return pl.DataFrame(
            {
                "time": [next_time],
                "state": [self.state_name],
                "measurement": [float(next_time)],
                "task_group_id": [task_group_id],
                "task_id": [task_id],
            }
        )


def build_experiment_from_draft(draft: ExperimentDraft) -> Experiment:
    states = [DraftState(state) for state in draft.states]
    shared_variable_history = pl.DataFrame(
        {
            "time": [0],
            "state": [draft.initial_state],
            "measurement": [0.0],
        }
    )
    return Experiment(
        experiment_name=draft.experiment_name,
        states=states,
        current_state_name=draft.initial_state,
        shared_variable_history=shared_variable_history,
    )
