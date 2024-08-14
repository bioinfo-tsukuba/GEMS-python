from dataclasses import dataclass
import numpy as np
import polars as pl
from gems_python.onemachine_problem.penalty import *
from gems_python.onemachine_problem.transition_manager import Experiment, OneMachineTaskLocalInformation, State
from dataclasses import dataclass
import polars as pl
from scipy.optimize import curve_fit

# /// The processing time of each state, the unit is minute
# pub(crate) static HEK_CULTURE_PROCESSING_TIME:[common_param_type::ProcessingTime; 4] = [
#     0, // EXPIRE
#     60, // PASSAGE
#     15, // GET_IMAGE
#     40, // SAMPLING
# ];

# TODO: 増殖曲線の予測（Passageによるリセットも考慮に入れたもの）を実装する
# def logistic(t, r, N0, K=1):
#     return K / (1 + (K/N0 - 1) * np.exp(-r * t))

# def def_piecewise_logistic(pieces, K=1):
#     pieces = [0] + pieces

#     def pf(t, r, *n0_list):
#         assert len(n0_list) == len(pieces)
#         n0 = np.zeros_like(t)
#         t0 = np.zeros_like(t)
#         for i, p in enumerate(pieces):
#             n0[p <= t] = n0_list[i]
#             t0[p <= t] = p
#         return logistic(t-t0, r, n0, K)
#     return pf

# def fit_param(self):
#     if len(self.density_log) <= 3:
#         return False
#     passage_time = self.list_passage_time()
#     piecewise_logistic = def_piecewise_logistic(passage_time, self.K)

#     x = self.time2spentdays(self.density_log.index)
#     y = self.density_log.values
#     p0 = [1] + [0.1] * (len(passage_time) + 1)
#     popt, _ = curve_fit(piecewise_logistic, x, y, p0=p0)
#     self.fitted_r = popt[0]
#     self.fitted_s = popt[1:]
#     return True

PROCESSING_TIME = {
    "ExpireState": 0,
    "PassageState": 60,
    "GetImageState": 15,
    "SamplingState": 40,
    "GetImageJustBeforePassageState": 15,
    "GetImageJustBeforeSamplingState": 15,
}

OPERATION_INTERVAL = (12 * 60)


@dataclass
class ExpireState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "ExpireState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        last_row = df.tail(1)
        current_time = last_row["time"][0]
        return OneMachineTaskLocalInformation(
            optimal_timing=current_time,  # 現在の時間
            processing_time=PROCESSING_TIME["ExpireState"], 
            penalty_type=LinearWithRange(lower=0, lower_coefficient=0, upper=0, upper_coefficient=0),
            experiment_operation="Getimage"
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
            processing_time=PROCESSING_TIME["PassageState"],
            penalty_type=LinearPenalty(penalty_coefficient=100),
            experiment_operation="Passage"
        )


@dataclass
class GetImageState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        passage_count = int(df[-1]['passage_count'])
        time_since_last_passage = int(df[-1]['time']) - int(df[-1]['last_passage_time'])

        if passage_count <= 1:
            if time_since_last_passage >= OPERATION_INTERVAL:
                return "GetImageJustBeforePassageState"
            else:
                return "GetImageState"
        else:
            if time_since_last_passage >= OPERATION_INTERVAL:
                return "GetImageJustBeforeSamplingState"
            else:
                return "GetImageState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        passage_count = int(df[-1]['passage_count'])
        if passage_count <= 1:
            optimal_timing = int(df[-1]['time']) + OPERATION_INTERVAL
        else:
            optimal_timing = int(df[-1]['time']) + OPERATION_INTERVAL

        return OneMachineTaskLocalInformation(
            optimal_timing=optimal_timing,
            processing_time=10,  # 定義済みの処理時間（例: 10分）
            penalty_type=LinearPenalty(penalty_coefficient=1),
            experiment_operation="Getimage"
        )


@dataclass
class SamplingState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "ExpireState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        return OneMachineTaskLocalInformation(
            optimal_timing=int(df[-1]['time']),
            processing_time=PROCESSING_TIME["SamplingState"],
            penalty_type=LinearPenalty(penalty_coefficient=10),
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
            processing_time=PROCESSING_TIME["GetImageJustBeforePassageState"],
            penalty_type=LinearPenalty(penalty_coefficient=100),
            experiment_operation="Getimage"
        )


@dataclass
class GetImageJustBeforeSamplingState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "SamplingState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        optimal_timing = int(df[-1]['sampling_target_time']) - 10
        return OneMachineTaskLocalInformation(
            optimal_timing=optimal_timing,
            processing_time=PROCESSING_TIME["GetImageJustBeforeSamplingState"],
            penalty_type=LinearPenalty(penalty_coefficient=100),
            experiment_operation="Getimage"
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
            current_state_name="PassageState",
            shared_variable_history=shared_variable_history
        )