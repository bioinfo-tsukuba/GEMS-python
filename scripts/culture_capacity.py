"""ROUND3 normal protocols and empirical virtual growth, separate from policy.

State methods see observations only. Sampled growth parameters are held by the
observation provider, never by the protocol or prediction function.
"""

from functools import lru_cache
import math

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.special import expit, logit

from gems_python.common import dataframes
from gems_python.multi_machine_problem_interval_task.lifecycle import Finish
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import (
    CyclicalRestPenaltyWithLinear, LinearPenalty,
)
from gems_python.multi_machine_problem_interval_task.task_info import (
    Machine, MachineList, Task, TaskGroup,
)
from gems_python.multi_machine_problem_interval_task.transition_manager import (
    Experiment, Experiments, State,
)
from scripts.calibrate_round3_growth import curve, crossing_hours


@lru_cache(maxsize=10000)
def _predict_from_observations(kind, target, passages, observations):
    """Stable equivalent of ROUND3's shared-rate, per-passage logistic fit.

    Removes the source's unused k parameter and null-density pseudo-observations.
    Positive bounds avoid nonphysical fits. Source recent-passage sigma weights
    and the already-over-target rule are retained.
    """
    if len(observations) < 3:
        return math.inf
    current = [(time, value) for time, value in observations if time > passages[-1]]
    if not current:
        return math.inf
    largest = max(current, key=lambda pair: pair[1])
    if largest[1] > target:
        return float(largest[0])
    grouped = []
    for index, passage in enumerate(passages):
        end = passages[index + 1] if index + 1 < len(passages) else math.inf
        values = [(time, value) for time, value in observations if passage < time < end]
        if values:
            grouped.append((index, passage, values))
    if not grouped:
        return math.inf
    slopes = []
    for _, passage, values in grouped:
        if len(values) >= 2:
            t = np.array([(time - passage) / 60 for time, _ in values])
            y = np.array([value for _, value in values])
            slopes.append(float(np.polyfit(t, logit(np.clip(y, 1e-8, 1-1e-8)), 1)[0]))
    rate = float(np.clip(np.median(slopes) if slopes else .01, 1e-5, .9))
    guesses = [rate]
    prepared = []
    for index, passage, values in grouped:
        t = np.array([(time - passage) / 60 for time, _ in values])
        y = np.array([value for _, value in values])
        intercept = float(np.clip(logit(np.clip(y[0], 1e-8, 1-1e-8)) - rate*t[0],
                                  -19, 9))
        guesses.append(intercept)
        prepared.append((t, y, 2.0 ** (index - grouped[-1][0])))

    def residual(params):
        return np.concatenate([(expit(params[i+1] + params[0]*t) - y)*weight
                               for i, (t, y, weight) in enumerate(prepared)])

    fitted = least_squares(residual, guesses,
                           bounds=([1e-5] + [-20]*len(grouped),
                                   [1.0] + [10]*len(grouped)),
                           max_nfev=150, ftol=1e-7, xtol=1e-7, gtol=1e-7)
    if not fitted.success:
        return math.inf
    remaining = (logit(target) - fitted.x[-1]) / fitted.x[0]
    return float(passages[-1] + remaining * 60)


def predict(history, kind, target):
    passages = tuple(int(row.time) for row in history.itertuples()
                     if row.operation == kind + "Passage")
    observations = tuple((int(row.time), float(row.density)) for row in history.itertuples()
                         if row.operation == kind + "GetImage" and pd.notna(row.density))
    if not passages:
        return math.inf
    return _predict_from_observations(kind, float(target), passages, observations)


class Round3State(State):
    def __init__(self, kind, protocol):
        self.kind = kind
        self.protocol = protocol

    @property
    def settings(self):
        return self.protocol[self.kind]

    def task(self, operation, interval=0):
        return Task(self.settings["processing_minutes"][operation],
                    self.kind + operation, 0, interval=int(interval))

    def linear(self, start, operations, coefficient):
        return TaskGroup(int(start), LinearPenalty(coefficient), operations)

    def prediction(self, history, sampling=False):
        return predict(history, self.kind,
                       self.settings["sampling_density" if sampling else "passage_density"])

    def terminal(self, history):
        return int((history.operation == self.kind + "Passage").sum()) >= (
            self.settings["passages_including_initial"])

    def sampling_group(self, history):
        optimal = self.prediction(history, sampling=True)
        if not math.isfinite(optimal):
            raise ValueError("Sampling state entered without a finite observation forecast")
        duration = self.settings["processing_minutes"]
        penalty = CyclicalRestPenaltyWithLinear(
            self.protocol["sampling_rest_cycle_start"], 1440,
            [tuple(x) for x in self.protocol["sampling_rest_ranges"]],
            self.protocol["sampling_penalty_coefficient"])
        return TaskGroup(int(optimal-duration["Sampling"]-duration["GetImage"]),
                         penalty, [self.task("GetImage"), self.task("Sampling")])

    def decide(self, history, waiting, passage, sampling):
        terminal = self.terminal(history)
        predicted = self.prediction(history, sampling=terminal)
        if math.isfinite(predicted) and predicted - int(history.time.iloc[-1]) <= (
                self.settings["observation_interval_minutes"]):
            return sampling if terminal else passage
        return waiting


class HEKFirstImage(Round3State):
    def task_generator(self, history):
        return self.linear(int(history.time.iloc[-1])+self.settings["first_observation_minutes"],
                           [self.task("GetImage")], 1)

    def transition_function(self, history):
        return "HEKObservation"


class HEKObservation(Round3State):
    def task_generator(self, history):
        return self.linear(int(history.time.iloc[-1]) +
                           self.settings["observation_interval_minutes"],
                           [self.task("GetImage")], 1)

    def transition_function(self, history):
        return self.decide(history, "HEKObservation", "HEKPassage", "HEKSampling")


class HEKPassage(Round3State):
    def task_generator(self, history):
        optimal = self.prediction(history)
        duration = self.settings["processing_minutes"]
        return self.linear(optimal-duration["Passage"]-duration["GetImage"], [
            self.task("GetImage"), self.task("Passage"),
            self.task("GetImage", self.settings["first_observation_minutes"])], 1000)

    def transition_function(self, history):
        return "HEKObservation"


class HEKSampling(Round3State):
    def task_generator(self, history):
        return self.sampling_group(history)

    def transition_function(self, history):
        return "HEKSampling"

    def lifecycle_action(self, history, context):
        return Finish("ROUND3 sampling completed")


class IPSInitialPassage(Round3State):
    def task_generator(self, history):
        # Source wall clock is replaced by a common virtual start; no negative
        # processing is credited before enrollment.
        return self.linear(0, [self.task("PlateCoating"),
                           self.task("GetImage", self.settings["coating_interval_minutes"]),
                           self.task("Passage")], 1000)

    def transition_function(self, history):
        return "IPSMediumChange1"


class IPSMediumChange1(Round3State):
    def task_generator(self, history):
        interval = self.settings["observation_interval_minutes"]
        return self.linear(int(history.time.iloc[-1])+interval, [
            self.task("GetImage"), self.task("MediumChange"),
            self.task("GetImage", interval)], 100)

    def transition_function(self, history):
        return "IPSMediumChange2"


class IPSMediumChange2(Round3State):
    def task_generator(self, history):
        return self.linear(int(history.time.iloc[-1]) +
                           self.settings["observation_interval_minutes"],
                           [self.task("GetImage"), self.task("MediumChange")], 100)

    def transition_function(self, history):
        return self.decide(history, "IPSMediumChange2", "IPSPassage", "IPSSampling")


class IPSPassage(Round3State):
    def task_generator(self, history):
        optimal = self.prediction(history)
        duration = self.settings["processing_minutes"]
        return self.linear(optimal-self.settings["coating_interval_minutes"] -
                           duration["PlateCoating"]-duration["GetImage"], [
            self.task("PlateCoating"),
            self.task("GetImage", self.settings["coating_interval_minutes"]),
            self.task("Passage")], 1000)

    def transition_function(self, history):
        return "IPSMediumChange1"


class IPSSampling(Round3State):
    def task_generator(self, history):
        return self.sampling_group(history)

    def transition_function(self, history):
        return "IPSSampling"

    def lifecycle_action(self, history, context):
        return Finish("ROUND3 sampling completed")


class Refill(State):
    def __init__(self, protocol):
        self.protocol = protocol

    def task_generator(self, history):
        p = self.protocol
        tasks = [Task(0, "LabwareRefill", 0)]
        for index in range(p["refill_tasks_per_group_including_marker"]-1):
            interval = (p["refill_interval_minutes"] if index == 0 else
                        p["refill_interval_minutes"]-p["refill_processing_minutes"])
            tasks.append(Task(p["refill_processing_minutes"], "LabwareRefill", 0,
                              interval=interval))
        return TaskGroup(p["refill_first_marker_minutes"],
                         LinearPenalty(p["refill_penalty_coefficient"]), tasks)

    def transition_function(self, history):
        return "Refill"


def draw_cultures(model, config, hek, ips, repetition):
    """Shared random-number prefixes across count conditions, joint block draws."""
    cultures = {}
    for kind_number, (kind, count) in enumerate([("HEK", hek), ("iPS", ips)]):
        pool = [p for p in model["parameter_pool"] if p["cell_type"] == kind]
        donors = sorted({p["donor"] for p in pool})
        for index in range(count):
            rng = np.random.default_rng(np.random.SeedSequence([
                config["biological_seed_namespace"], repetition, kind_number, index]))
            donor = str(rng.choice(donors))
            blocks, provenance = [], []
            for generation in range(3):
                # HEK's pre-reset older passage is retained for validation but
                # the main protocol uses the three generations after that reset.
                source_generation = generation+1 if kind == "HEK" else generation
                match = [p for p in pool if p["donor"] == donor and
                         p["generation"] == source_generation]
                fallback = False
                if not match:
                    match = [p for p in pool if p["generation"] == source_generation]
                    fallback = True
                if not match:
                    raise ValueError(f"No empirical {kind} generation {source_generation} pool")
                record = dict(match[int(rng.integers(len(match)))])
                blocks.append(record)
                provenance.append({"source_donor": record["donor"],
                                   "source_generation": source_generation,
                                   "missing_generation_fallback": fallback})
            cultures[f"{kind}-{index:02d}"] = {
                "cell_type": kind, "donor": donor, "parameters": blocks,
                "draw_provenance": provenance,
            }
    return cultures


class CultureObservations:
    def __init__(self, cultures, config):
        self.cultures = cultures
        self.config = config
        self.events = []
        self.origins = {key: {"generation": 0 if value["cell_type"] == "HEK" else -1,
                             "passage_end": 0} for key, value in cultures.items()}

    def density(self, identity, time):
        culture = self.cultures[identity]
        origin = self.origins[identity]
        if origin["generation"] < 0:
            # Startup passage transfers an existing parent culture. It has no
            # target-density decision in the source and is excluded from scoring.
            return self.config["round3"]["iPS"]["passage_density"]
        params = culture["parameters"][origin["generation"]]
        return float(curve([(time-origin["passage_end"])/60], params)[0])

    def __call__(self, state, history, group, task, event_time):
        identity = group.experiment_uuid
        operation = task.experiment_operation
        kind = self.cultures[identity]["cell_type"] if identity in self.cultures else "refill"
        density = self.density(identity, task.scheduled_time) if kind != "refill" else None
        target, crossing = None, None
        if operation.endswith(("Passage", "Sampling")) and kind != "refill":
            origin = self.origins[identity]
            if origin["generation"] >= 0:
                settings = self.config["round3"][kind]
                target = settings["sampling_density" if operation.endswith("Sampling")
                                  else "passage_density"]
                crossing = origin["passage_end"] + crossing_hours(
                    target, self.cultures[identity]["parameters"][origin["generation"]])*60
        group_start = group.tasks[0].scheduled_time-group.tasks[0].interval
        event = {
            "culture_id": identity, "cell_type": kind, "operation": operation,
            "group_uuid": group.task_group_uuid, "task_id": task.task_id,
            "group_start": group_start, "group_optimal_start": group.optimal_start_time,
            "group_state": state.state_name,
            "group_operations": [t.experiment_operation for t in group.tasks],
            "start": task.scheduled_time, "end": event_time,
            "duration": task.processing_time, "interval": task.interval,
            "machine": task.allocated_machine_id,
            "density": density, "target": target, "target_crossing_time": crossing,
            "generation": self.origins[identity]["generation"] if kind != "refill" else None,
        }
        if target is not None:
            nominal_start = group.optimal_start_time + task.scheduled_time-group_start
            nominal_density = self.density(identity, nominal_start)
            event["nominal_action_start"] = nominal_start
            event["nominal_action_density"] = nominal_density
            event["schedule_shift_minutes"] = group_start-group.optimal_start_time
            event["nominal_density_error"] = nominal_density-target
            event["schedule_density_change"] = density-nominal_density
            event["error"] = density-target
            event["primary_pass"] = abs(density-target) <= (
                self.config["primary_density_half_width"]+1e-12)
            band = min(self.config["primary_density_half_width"],
                       self.config["supplementary_relative_half_width"]*target)
            event["supplementary_pass"] = abs(density-target) <= band+1e-12
        self.events.append(event)
        if kind != "refill" and operation.endswith("Passage"):
            self.origins[identity]["generation"] += 1
            self.origins[identity]["passage_end"] = event_time
        # Images are acquired at task start and delivered at completion; passage
        # resets at completion. Resource occupancy always uses full configured time.
        history_time = task.scheduled_time if operation.endswith("GetImage") else event_time
        return dataframes.make_frame({
            "time": [history_time], "operation": [operation],
            "density": [density if operation.endswith("GetImage") else None],
        }, backend="pandas")


def build_manager(cultures, config, directory):
    p = config["round3"]
    experiments = []
    for identity, culture in cultures.items():
        kind = culture["cell_type"]
        if kind == "HEK":
            states = [HEKFirstImage(kind, p), HEKObservation(kind, p),
                      HEKPassage(kind, p), HEKSampling(kind, p)]
            initial = "HEKFirstImage"
            history = pd.DataFrame({"time": [0], "operation": ["HEKPassage"],
                                    "density": [None]})
        else:
            states = [IPSInitialPassage(kind, p), IPSMediumChange1(kind, p),
                      IPSMediumChange2(kind, p), IPSPassage(kind, p), IPSSampling(kind, p)]
            initial = "IPSInitialPassage"
            history = pd.DataFrame({"time": [0], "operation": ["Dummy"], "density": [None]})
        # Initial metadata contains no latent growth parameters.
        experiments.append(Experiment(identity, states, initial, history,
                                      experiment_uuid=identity))
    experiments.append(Experiment("Refill", [Refill(p)], "Refill",
                                  experiment_uuid="refill"))
    return Experiments(experiments, directory,
                       MachineList([Machine(0, "ROUND3-shared-resource")]),
                       solver_time_limit_seconds=config["solver_time_limit_seconds"],
                       unknown_policy=config["unknown_policy"])
