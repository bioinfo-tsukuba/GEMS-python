
from __future__ import annotations
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import List, Optional

import math
import polars as pl

# Multi-machine interval task APIs
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task, TaskGroup
)
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import (
    LinearPenalty,
)
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    State, Experiment, Experiments,
)

# =====================================================
# Minimal HEK-like flow with growth simulation & estimation
# States: Passage -> Observation (loop)
# - Passage: reseed (reset density)
# - Observation: simulate logistic growth and estimate r in-state
# =====================================================

@dataclass
class PassageState(State):
    """Reseeding state: resets cell density to a low starting value."""
    seed_confluence: float = 0.12  # initial fraction of confluence after passage (0-1)
    K: float = 1.0                 # carrying capacity (confluence)

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # Bench work (type 0), simple 20 min
        return TaskGroup(
            optimal_start_time=0,
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(processing_time=20, interval=0,
                     experiment_operation="Passage (reseeding)", optimal_machine_type=0),
            ]
        )

    def transition_function(self, df: pl.DataFrame) -> str:
        # After a passage record appears, move to Observation
        if "event" in df.columns and df.filter(pl.col("event") == "passage").height > 0:
            return "ObservationState"
        return "PassageState"

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        # Emit a row that sets confluence to seed_confluence
        now = datetime.now().isoformat()
        return pl.DataFrame({
            "time": [now],
            "task_group_id": [task_group_id],
            "task_id": [task_id],
            "state": [self.state_name],
            "event": ["passage"],
            "confluence": [min(max(self.seed_confluence, 1e-6), self.K - 1e-6)],  # clip (0, K)
        })


@dataclass
class ObservationState(State):
    """Observation with simple logistic growth simulation and in-state r estimation."""
    # Growth parameters (used by the simulator)
    K: float = 1.0                    # carrying capacity (confluence)
    true_r_per_hour: float = 0.035    # intrinsic growth rate (per hour) used for simulation
    delta_hours: float = 6.0          # interval between observations assumed by the simulator
    # Control: when to passage again
    target_confluence: float = 0.30   # once reached, we return to PassageState

    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        # Imaging/measurement (type 1), short 10 min
        return TaskGroup(
            optimal_start_time=60,
            penalty_type=LinearPenalty(penalty_coefficient=10),
            tasks=[
                Task(processing_time=10, interval=0,
                     experiment_operation="Observation (imaging/measurement)", optimal_machine_type=1),
            ]
        )

    # ---- helpers for growth & estimation ----
    def _last_confluence(self, df: pl.DataFrame) -> Optional[float]:
        if "confluence" not in df.columns or df.height == 0:
            return None
        # use the last row's value
        return df.select(pl.col("confluence").last()).item()

    def _second_last_confluence(self, df: pl.DataFrame) -> Optional[float]:
        if "confluence" not in df.columns or df.height < 2:
            return None
        return df.select(pl.col("confluence").slice(-2, 1)).item()

    def _simulate_next_confluence(self, c: float) -> float:
        """One logistic step: c_{t+Δ} = c + r*c*(1-c/K)*Δt (Euler step)."""
        dt = self.delta_hours
        r = self.true_r_per_hour
        K = self.K
        c_next = c + r * c * (1.0 - c / K) * dt
        # clamp to (0, K)
        eps = 1e-6
        return float(min(max(c_next, eps), K - eps))

    def _estimate_r_from_history(self, c_prev: float, c_curr: float) -> Optional[float]:
        """Estimate r using the logistic linearisation with known K:
           logit(c/K) grows linearly with slope r * Δt.
        """
        if c_prev <= 0 or c_curr <= 0 or c_prev >= self.K or c_curr >= self.K:
            return None
        def logit(x: float) -> float:
            x = min(max(x, 1e-9), self.K - 1e-9)
            xk = x / self.K
            return math.log(xk / (1.0 - xk))
        try:
            dt = self.delta_hours
            r_hat = (logit(c_curr) - logit(c_prev)) / dt
            return float(r_hat)
        except Exception:
            return None

    # ----------------------------------------

    def transition_function(self, df: pl.DataFrame) -> str:
        # If we've reached target confluence, go back to Passage
        last = self._last_confluence(df)
        if last is not None and last >= self.target_confluence:
            return "PassageState"
        # Otherwise remain observing
        return "ObservationState"

    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        # Determine current confluence (seed from last passage if available)
        last_c = self._last_confluence(df)
        # If absent, attempt to seed from a passage row in df; otherwise default small value
        if last_c is None:
            if "event" in df.columns:
                # use the most recent 'passage' confluence if present
                sub = df.filter(pl.col("event") == "passage")
                if sub.height > 0 and "confluence" in sub.columns:
                    last_c = sub.select(pl.col("confluence").last()).item()
        if last_c is None:
            last_c = 0.1  # fallback

        # Simulate next observation
        next_c = self._simulate_next_confluence(last_c)

        # Estimate r from last two points (if available)
        second_last_c = self._second_last_confluence(df)
        r_hat = None
        if second_last_c is not None:
            r_hat = self._estimate_r_from_history(second_last_c, next_c)

        now = datetime.now().isoformat()
        out = {
            "time": [now],
            "task_group_id": [task_group_id],
            "task_id": [task_id],
            "state": [self.state_name],
            "event": ["observation"],
            "confluence": [next_c],
        }
        if r_hat is not None:
            out["est_r_per_hour"] = [r_hat]
        return pl.DataFrame(out)


# ===================
# Build utilities
# ===================

def build_experiment_two_state(
    experiment_name: str = "HEK_TwoState_GrowthDemo",
    current_state_name: str = "PassageState",
) -> Experiment:
    states: List[State] = [
        PassageState(),
        ObservationState(),
    ]
    return Experiment(
        experiment_name=experiment_name,
        states=states,
        current_state_name=current_state_name,
        shared_variable_history=pl.DataFrame(),
    )


def build_experiments_two_state(
    parent_dir_path: str = "experiments_dir_growth_demo",
) -> Experiments:
    ml = MachineList()
    # 0: bench, 1: imager
    ml.add_machine(Machine(machine_type=0, description="Bench / biosafety cabinet"))
    ml.add_machine(Machine(machine_type=1, description="Imager / reader"))
    exps = Experiments(
        experiments=[build_experiment_two_state()],
        parent_dir_path=parent_dir_path,
        machine_list=ml,
    )
    return exps


# ==============
# Quick self-test
# ==============
if __name__ == "__main__":
    exps = build_experiments_two_state()
    # Run several simulated steps
    logs = exps.simulate(max_steps=8, save_each_step=True)
    for i, r in enumerate(logs, 1):
        print(f"[simulate step {i}] {r}")
