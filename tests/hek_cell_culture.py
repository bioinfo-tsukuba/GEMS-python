from dataclasses import dataclass
import polars as pl
from gems_python.onemachine_problem.penalty import *
from gems_python.onemachine_problem.transition_manager import Experiment, OneMachineTaskLocalInformation, State
from dataclasses import dataclass
import polars as pl

@dataclass
class ExpireState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "ExpireState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        last_row = df.tail(1)
        current_time = last_row["time"][0]
        return OneMachineTaskLocalInformation(
            optimal_timing=current_time,  # 現在の時間
            processing_time=5,  # 定義済みの処理時間（例: 5分）
            penalty_type=LinearWithRange(lower=0, lower_coefficient=0, upper=0, upper_coefficient=0),
            experiment_operation="End culture"
        )


@dataclass
class PassageState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "GetImageState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        last_row = df.tail(1)
        current_time = last_row["time"][0]
        return OneMachineTaskLocalInformation(
            optimal_timing=current_time,  # 現在の時間
            processing_time=15,  # 定義済みの処理時間（例: 15分）
            penalty_type=LinearWithRange(lower=-10, lower_coefficient=100, upper=10, upper_coefficient=100),
            experiment_operation="Passage"
        )


@dataclass
class GetImageState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        passage_count = int(df[-1]['passage_count'])
        time_since_last_passage = int(df[-1]['time']) - int(df[-1]['last_passage_time'])

        if passage_count <= 1:
            if time_since_last_passage >= 12 * 60:
                return "GetImageJustBeforePassageState"
            else:
                return "GetImageState"
        else:
            if time_since_last_passage >= 12 * 60:
                return "GetImageJustBeforeSamplingState"
            else:
                return "GetImageState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        passage_count = int(df[-1]['passage_count'])
        if passage_count <= 1:
            optimal_timing = int(df[-1]['time']) + 12 * 60
        else:
            optimal_timing = int(df[-1]['time']) + 12 * 60

        return OneMachineTaskLocalInformation(
            optimal_timing=optimal_timing,
            processing_time=10,  # 定義済みの処理時間（例: 10分）
            penalty_type=LinearWithRange(lower=-60, lower_coefficient=1, upper=60, upper_coefficient=1),
            experiment_operation="Get image"
        )


@dataclass
class SamplingState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "ExpireState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        return OneMachineTaskLocalInformation(
            optimal_timing=int(df[-1]['time']),
            processing_time=20,  # 定義済みの処理時間（例: 20分）
            penalty_type=LinearWithRange(lower=-10, lower_coefficient=10, upper=10, upper_coefficient=10),
            experiment_operation="Sampling"
        )


@dataclass
class GetImageJustBeforePassageState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "PassageState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        optimal_timing = int(df[-1]['passage_target_time']) - 10
        return OneMachineTaskLocalInformation(
            optimal_timing=optimal_timing,
            processing_time=10,  # 定義済みの処理時間（例: 10分）
            penalty_type=LinearWithRange(lower=-10, lower_coefficient=100, upper=0, upper_coefficient=100),
            experiment_operation="Get image just before passage"
        )


@dataclass
class GetImageJustBeforeSamplingState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "SamplingState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        optimal_timing = int(df[-1]['sampling_target_time']) - 10
        return OneMachineTaskLocalInformation(
            optimal_timing=optimal_timing,
            processing_time=10,  # 定義済みの処理時間（例: 10分）
            penalty_type=LinearWithRange(lower=-10, lower_coefficient=10, upper=0, upper_coefficient=10),
            experiment_operation="Get image just before sampling"
        )


@dataclass
class HekCellCulture(Experiment):
    def __init__(self):
        # Define the experiment using the states

        # Create a shared variable history DataFrame (empty for this example)
        shared_variable_history = pl.DataFrame(
            {
                "time": [0],
                "passage_count": [0],
                "last_passage_time": [0],
                "passage_target_time": [0],
                "sampling_target_time": [0]
            }
        )

        # Define the initial state and experiment
        super().__init__(
            experiment_name="HekCellCulture",
            states=[
                ExpireState(),
                PassageState(),
                GetImageState(),
                SamplingState(),
                GetImageJustBeforePassageState(),
                GetImageJustBeforeSamplingState()
                ],
            current_state_name="ExpireState",
            shared_variable_history=shared_variable_history
        )