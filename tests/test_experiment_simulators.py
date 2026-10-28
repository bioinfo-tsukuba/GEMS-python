"""Reproducible synthetic experiment trajectories using the live scheduler."""

import contextlib
import io
import tempfile
import unittest
from pathlib import Path

import polars as pl
from gems_python.common import dataframes

from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import LinearPenalty
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task, TaskGroup,
)
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    Experiment, Experiments, State,
)
from tests.simulators import cell_culture, reaction_search
from tests.simulators.core import SimulationRunner


class TimedState(State):
    def __init__(self, duration: int, machine_type: int, first_start: int):
        super().__init__()
        self.duration = duration
        self.machine_type = machine_type
        self.first_start = first_start

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        start = self.first_start if dataframes.row_count(df) == 0 else int(dataframes.last_value(df, "time")) + 100
        return TaskGroup(
            optimal_start_time=start,
            penalty_type=LinearPenalty(1),
            tasks=[Task(self.duration, "work", self.machine_type)],
        )

    def transition_function(self, df: pl.DataFrame) -> str:
        return "TimedState"


class TestExperimentSimulators(unittest.TestCase):
    def test_cell_growth_observation_and_passage(self):
        with tempfile.TemporaryDirectory() as directory:
            with contextlib.redirect_stdout(io.StringIO()):
                experiments = cell_culture.build_experiments(Path(directory))
                results = SimulationRunner(
                    experiments, cell_culture.observe
                ).run(8)
        history = dataframes.convert(experiments.experiments[0].shared_variable_history, "polars")
        self.assertEqual(history["event"].to_list().count("passage"), 2)
        self.assertEqual(results[6]["to_state"], "PassageState")
        self.assertEqual(history["confluence"][-1], cell_culture.SEED_CONFLUENCE)
        self.assertAlmostEqual(
            history["confluence"][1],
            cell_culture.logistic_growth(
                cell_culture.SEED_CONFLUENCE,
                (history["time"][1] - history["time"][0]) / 60,
            ),
        )
        self.assertEqual(history["time"].to_list(), sorted(history["time"].to_list()))

    def test_reaction_search_repeats_and_has_known_oracle(self):
        candidates = reaction_search.candidate_order("random", seed=7)
        histories = []
        for _ in range(2):
            with tempfile.TemporaryDirectory() as directory:
                with contextlib.redirect_stdout(io.StringIO()):
                    experiments = reaction_search.build_experiments(
                        Path(directory), candidates
                    )
                    SimulationRunner(
                        experiments, reaction_search.ReactionObserver(candidates)
                    ).run(9)
            histories.append(
                dataframes.convert(experiments.experiments[0].shared_variable_history, "polars")
                .drop(["task_group_id", "task_id"]).to_dict(as_series=False)
            )
        self.assertEqual(histories[0], histories[1])
        yields = histories[0]["yield"]
        best = reaction_search.best_so_far(yields)
        self.assertEqual(best, sorted(best))
        self.assertEqual(best[-1], max(yields))
        self.assertEqual(
            max(map(reaction_search.synthetic_yield, reaction_search.CONDITIONS)),
            reaction_search.BEST_POSSIBLE_YIELD,
        )
        self.assertNotEqual(
            candidates, reaction_search.candidate_order("grid")
        )

    def test_parallel_tasks_report_results_in_completion_order(self):
        with tempfile.TemporaryDirectory() as directory:
            machines = MachineList([
                Machine(0, machine_id="slow_machine"),
                Machine(1, machine_id="fast_machine"),
            ])
            with contextlib.redirect_stdout(io.StringIO()):
                experiments = Experiments(
                    experiments=[
                        Experiment("slow", [TimedState(20, 0, 0)], "TimedState"),
                        Experiment("fast", [TimedState(1, 1, 1)], "TimedState"),
                    ],
                    machine_list=machines,
                    parent_dir_path=Path(directory),
                )
                results = SimulationRunner(
                    experiments,
                    lambda state, history, group, task, minute: pl.DataFrame({
                        "time": [minute], "value": [1]
                    }),
                ).run(2)
        self.assertEqual(
            [(result["experiment_name"], result["simulated_time"]) for result in results],
            [("fast", 2), ("slow", 20)],
        )


if __name__ == "__main__":
    unittest.main()
