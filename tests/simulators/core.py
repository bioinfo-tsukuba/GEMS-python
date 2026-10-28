"""Reusable simulation runner driven by scheduled completion times."""

from dataclasses import dataclass
from typing import Callable

import polars as pl

from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments


ObservationProvider = Callable[[object, pl.DataFrame, object, object, int], pl.DataFrame]


@dataclass
class SimulationRunner:
    experiments: Experiments
    observation_provider: ObservationProvider

    def run(self, steps: int) -> list[dict]:
        results = self.experiments.simulate(
            max_steps=steps, observation_provider=self.observation_provider
        )
        if len(results) != steps or any(result["status"] != "ok" for result in results):
            raise RuntimeError(f"Simulation stopped before {steps} steps: {results}")
        times = [result["simulated_time"] for result in results]
        if times != sorted(times):
            raise RuntimeError(f"Simulation time went backwards: {times}")
        return results
