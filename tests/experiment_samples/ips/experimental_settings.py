from dataclasses import dataclass
import numpy as np
import polars as pl
from gems_python.onemachine_problem.penalty import *
from gems_python.onemachine_problem.transition_manager import Experiment, OneMachineTaskLocalInformation, State
from dataclasses import dataclass
import polars as pl
from scipy.optimize import curve_fit

from tests.experiment_samples.curve import calculate_optimal_time_from_df


IPS_CULTURE_STATE_TIMES = {
    "EXPIRE": 0,
    "PASSAGE": 120,
    "GET_IMAGE_1": 10,
    "MEDIUM_CHANGE_1": 20,
    "GET_IMAGE_2": 10,
    "MEDIUM_CHANGE_2": 20,
    "PLATE_COATING": 20,
}



PROCESSING_TIME = {
    "EXPIRE": 0,
    "PASSAGE": 120,
    "GET_IMAGE_1": 10,
    "MEDIUM_CHANGE_1": 20,
    "GET_IMAGE_2": 10,
    "MEDIUM_CHANGE_2": 20,
    "PLATE_COATING": 20,
}

def get_current_time(df: pl.DataFrame) -> float:
    return df["time"].max()


OPERATION_INTERVAL = (24 * 60)
PASSAGE_DENSITY = 0.7
SAMPLING_DENSITY = 0.4



@dataclass
class ExpireState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "ExpireState"

    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        current_time = df["time"].max()
        optimal_time = int(current_time)
        return OneMachineTaskLocalInformation(
            optimal_time=optimal_time,
            processing_time=PROCESSING_TIME["EXPIRE"], 
            penalty_type=LinearWithRange(lower=0, lower_coefficient=0, upper=0, upper_coefficient=0),
            experiment_operation="Getimage"
        )


@dataclass
class PassageState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "GetImage1State"
        return "ExpireState"
    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        optimal_time: float = calculate_optimal_time_from_df(df, target_density=PASSAGE_DENSITY)
        return OneMachineTaskLocalInformation(
            optimal_time=int(optimal_time),  # 現在の時間
            processing_time=PROCESSING_TIME["PASSAGE"],
            penalty_type=LinearPenalty(penalty_coefficient=100),
            experiment_operation="Passage"
        )


@dataclass
class GetImage1State(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        passage_count = len(df.filter(pl.col("operation") == "Passage"))

        if passage_count <= 1:
            optimal_time = calculate_optimal_time_from_df(df, target_density=SAMPLING_DENSITY)

            if optimal_time >= OPERATION_INTERVAL:
                return "MediumChange1State"
            else:
                return "GetImage1State"
            

        return "ExpireState"
    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        current_time = df["time"].max()
        optimal_time = int(current_time) + OPERATION_INTERVAL
        return OneMachineTaskLocalInformation(
            optimal_time=optimal_time,
            processing_time=PROCESSING_TIME["GET_IMAGE_1"],
            penalty_type=LinearPenalty(penalty_coefficient=1),
            experiment_operation="Getimage"
        )

@dataclass
class MediumChange1State(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "GetImage2State"

        return "ExpireState"
    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        # DUMMY
        return OneMachineTaskLocalInformation(
            optimal_time=int(df["time"].max()),
            processing_time=PROCESSING_TIME["MEDIUM_CHANGE_1"],
            penalty_type=NonePenalty(),
            experiment_operation="MediumChange"
        )

@dataclass
class GetImage2State(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        optimal_time = calculate_optimal_time_from_df(df, target_density=SAMPLING_DENSITY)

        if optimal_time >= OPERATION_INTERVAL:
            return "PlateCoatingState"
        else:
            return "MediumChange2State"

        return "ExpireState"
    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:

        # DUMMY
        return OneMachineTaskLocalInformation(
            optimal_time=int(df["time"].max()),
            processing_time=PROCESSING_TIME["MEDIUM_CHANGE_1"],
            penalty_type=NonePenalty(),
            experiment_operation="MediumChange"
        )

@dataclass
class MediumChange2State(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "GetImage2State"

        return "ExpireState"
    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:

        # DUMMY
        return OneMachineTaskLocalInformation(
            optimal_time=int(df["time"].max()),
            processing_time=PROCESSING_TIME["MEDIUM_CHANGE_1"],
            penalty_type=NonePenalty(),
            experiment_operation="MediumChange"
        )

@dataclass
class PlateCoatingState(State):
    def transition_function(self, df: pl.DataFrame) -> str:
        return "PassageState"

        return "ExpireState"
    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:

        # DUMMY
        return OneMachineTaskLocalInformation(
            optimal_time=int(df["time"].max()),
            processing_time=PROCESSING_TIME["MEDIUM_CHANGE_1"],
            penalty_type=NonePenalty(),
            experiment_operation="MediumChange"
        )


class IPSExperiment(Experiment):
    def __init__(self, current_state_name, shared_variable_history=None):
        # Define the experiment using the states

        # Create a shared variable history DataFrame (empty for this example)
        if shared_variable_history is None:
            # │ 34568 ┆ null     ┆ Passage   │
            # │ 36008 ┆ 0.411028 ┆ GetImage  │
            # │ 37448 ┆ 0.534656 ┆ GetImage  │
            # │ 38888 ┆ 0.632666 ┆ GetImage
            shared_variable_history = pl.DataFrame(
                {
                    "time": [0, 1440, 2880, 4320],
                    "density": [None, 0.411028, 0.534656, 0.632666],
                    "operation": ["Passage", "GetImage", "GetImage", "GetImage"]
                }
            )
            
        # Define the initial state and experiment
        super().__init__(
            experiment_name="IPSExperiment",
            states=[
                ExpireState(),
                PassageState(),
                GetImage1State(),
                MediumChange1State(),
                GetImage2State(),
                MediumChange2State(),
                PlateCoatingState(),
            ],
            current_state_name=current_state_name,
            shared_variable_history=shared_variable_history
        )



if __name__ == "__main__":
    # Create an instance of the experiment
    experiment = IPSExperiment(current_state_name="ExpireState")

    
    