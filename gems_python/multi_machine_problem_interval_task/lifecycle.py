"""Explicit experiment lifecycle and bounded planning for the multi-machine manager.

All mutations run in the existing single-threaded manager. Requests contain
data, not callbacks, and can be saved before confirmation. No physical operation
is retried by this module.
"""

import copy
from dataclasses import dataclass, field
import hashlib
import json
import logging
from uuid import NAMESPACE_URL, uuid5

from gems_python.common import dataframes as frames
from gems_python.common.planning_errors import PlanningError, SolverFailure
from gems_python.common.serialization import encode
from gems_python.common.scheduling import validate_groups, validate_inputs, validate_time_limit

LOGGER = logging.getLogger(__name__)
FEATURE_VERSION = 1


def attempt_record(report):
    # Wall-clock metrics belong to structured logs, not scientific state.
    return {k: report[k] for k in ("attempt", "solver_time_limit_seconds", "status")
            if k in report}


def fingerprint(value):
    encoded = encode(value)
    definitions = {}
    def collect(item):
        if isinstance(item, dict):
            if "object_id" in item:
                definitions[item["object_id"]] = item
            for child in item.values():
                collect(child)
        elif isinstance(item, list):
            for child in item:
                collect(child)
    collect(encoded)
    def semantic(item):
        if isinstance(item, dict):
            if item.get("kind") == "reference" and "_gems_serialization" in item:
                return semantic(definitions[item["target"]])
            return {k: semantic(v) for k, v in item.items() if k != "object_id"}
        if isinstance(item, list):
            return [semantic(child) for child in item]
        return item
    return hashlib.sha256(json.dumps(semantic(encoded), sort_keys=True,
                                     ensure_ascii=False).encode()).hexdigest()


@dataclass(frozen=True)
class LifecycleContext:
    experiment_uuid: str
    task_group_uuid: str
    request_id: str
    logical_time: int


@dataclass
class Finish:
    reason: str


@dataclass
class ChildSpec:
    """Scientific initial values and their origins must be supplied by the protocol."""
    key: str
    experiment_name: str
    states: list
    current_state_name: str
    initial_history: object
    initial_conditions: dict
    initial_condition_origins: dict


@dataclass
class Split:
    children: list[ChildSpec]


@dataclass
class EndRecord:
    experiment_uuid: str
    reason: str
    logical_time: int
    request_id: str
    state_name: str
    _history: object
    initial_conditions: dict
    initial_condition_origins: dict
    parent_uuid: str | None
    _settings: dict = field(default_factory=dict)
    feature_version: int = FEATURE_VERSION

    @property
    def history(self):
        return frames.copy_frame(self._history)

    @property
    def settings(self):
        return copy.deepcopy(self._settings)


@dataclass
class SplitRecord:
    request_id: str
    parent_uuid: str
    specification_hash: str
    children: list[ChildSpec]
    child_uuids: list[str]
    status: str = "PENDING"
    logical_time: int | None = None
    failure_reason: str | None = None
    feature_version: int = FEATURE_VERSION


@dataclass
class PlanningProgress:
    input_hash: str
    base_limit: float
    method: str
    next_attempt: int = 0
    reports: list = field(default_factory=list)
    complete: bool = False


class LifecycleManager:
    """Mixin for Experiments; persistent fields are declared on the dataclass."""

    def _experiment(self, uuid, *, optional=False):
        experiment = next((e for e in self.experiments if e.experiment_uuid == uuid), None)
        if experiment is None and not optional:
            raise ValueError(f"No registered Experiment: {uuid}")
        return experiment

    def runnable_groups(self):
        if self.manager_status == "STOPPED":
            return []
        return [g for g in self.task_groups if g.tasks and not g.is_completed()
                and (self._experiment(g.experiment_uuid, optional=True) is None or
                     self._experiment(g.experiment_uuid).lifecycle_status not in
                     ("WAITING_MACHINE", "WAITING_PLAN"))
                and all(t.completed or (t.scheduled_time is not None and
                                        t.allocated_machine_id is not None) for t in g.tasks)]

    def _diagnostic(self, kind, reason, **details):
        try:
            inputs = encode((self.task_groups, self.machine_list))
        except Exception as err:
            # An unsupported value must not prevent STOP from being recorded.
            inputs = {"serialization_error": f"{type(err).__name__}: {err}",
                      "task_groups": repr(self.task_groups), "machines": repr(self.machine_list)}
        return dict(kind=kind, reason=str(reason), reference_time=self.reference_time,
                    input=inputs, **details)

    def _stop(self, kind, reason, **details):
        self.manager_status = "STOPPED"
        self.stop_diagnostic = self._diagnostic(kind, reason, **details)
        LOGGER.error("%s: %s", kind, reason, extra={"planning_error": self.stop_diagnostic})
        return PlanningError(self.stop_diagnostic)

    def _validate_policy(self):
        if self.unknown_policy not in ("retry", "stop"):
            raise ValueError("unknown_policy must be 'retry' or 'stop'")
        if type(self.max_unknown_retries) is not int or self.max_unknown_retries < 0:
            raise ValueError("max_unknown_retries must be a nonnegative integer")
        if self.lifecycle_version != FEATURE_VERSION:
            raise ValueError(f"Unsupported lifecycle_version: {self.lifecycle_version}")
        if type(self.enable_split) is not bool:
            raise ValueError("enable_split must be boolean")
        if self.manager_status not in ("RUNNING", "STOPPED"):
            raise ValueError("Invalid manager_status")

    def _check_empty_actions(self):
        for experiment in list(self.experiments):
            if (not experiment.current_task_group.tasks
                    and experiment.lifecycle_status != "SPLIT_PENDING"
                    and experiment.current_task_group.task_group_uuid not in self.handled_groups):
                self._complete_group(experiment.current_task_group)

    def plan_tasks(self, scheduling_method="o", reference_time=None, *,
                   solver_time_limit_seconds=None):
        from gems_python.common.cp_sat_scheduler import schedule_task_groups
        from gems_python.multi_machine_problem_interval_task.task_info import TaskGroup
        limit = validate_time_limit(self.solver_time_limit_seconds if solver_time_limit_seconds is None
                                    else solver_time_limit_seconds)
        self._validate_policy()
        if self.manager_status == "STOPPED":
            return {"status": "STOPPED", "diagnostic": self.stop_diagnostic}
        if reference_time is not None:
            self.reference_time = reference_time
        try:
            self._check_empty_actions()
        except Exception as err:
            raise self._stop("LIFECYCLE_ERROR", err) from err
        self.set_task_group_ids()
        eligible, waiting = [], []
        machine_types = {m.machine_type for m in self.machine_list.machines}
        try:
            # Malformed held groups must still stop the manager.
            for group in self.task_groups:
                if not group.tasks:
                    continue
                try:
                    validate_inputs([group], self.reference_time)
                except ValueError as err:
                    raise ValueError(f"Experiment {group.experiment_uuid}, "
                                     f"TaskGroup {group.task_group_uuid}: {err}") from err
            if len({m.machine_id for m in self.machine_list.machines}) != len(self.machine_list.machines):
                raise ValueError("Duplicate machine ID")
            for group in self.task_groups:
                if not group.tasks or group.is_completed():
                    continue
                missing = sorted({t.optimal_machine_type for t in group.tasks
                                  if not t.completed and t.optimal_machine_type not in machine_types})
                experiment = self._experiment(group.experiment_uuid, optional=True)
                if missing and group.is_not_started():
                    if experiment is None:
                        raise ValueError(f"No machine of type {missing}; group has no registered Experiment")
                    reason = f"Required machine types not registered: {missing}"
                    if experiment.hold_reason != reason:
                        LOGGER.warning("Experiment %s: %s", experiment.experiment_uuid, reason)
                    experiment.lifecycle_status, experiment.hold_reason = "WAITING_MACHINE", reason
                    waiting.append(experiment.experiment_uuid)
                else:
                    eligible.append(group)
            fixed = [g for g in eligible if not g.is_not_started()]
            # Check fixed reservations before constructing a new model.
            validate_groups(fixed, self.reference_time, self.machine_list)
        except ValueError as err:
            raise self._stop("INVALID_INPUT_OR_FIXED_CONFLICT", err) from err

        input_hash = fingerprint((eligible, self.machine_list, self.reference_time))
        progress = self.planning_progress
        if (progress is None or progress.complete or progress.input_hash != input_hash
                or progress.base_limit != limit or progress.method != scheduling_method):
            progress = self.planning_progress = PlanningProgress(input_hash, limit, scheduling_method)
        while True:
            attempt = progress.next_attempt
            effective_limit = limit * 2 ** attempt
            report = {"attempt": attempt, "solver_time_limit_seconds": effective_limit,
                      "input_hash": input_hash}
            try:
                if scheduling_method == "o":
                    schedule_task_groups(eligible, self.reference_time, self.machine_list,
                                         solver_time_limit_seconds=effective_limit,
                                         diagnostics=report)
                elif scheduling_method == "s":
                    TaskGroup.schedule_task_groups_simulated_annealing(
                        eligible, self.machine_list, self.reference_time)
                    report["status"] = "FINISHED"
                elif scheduling_method == "f":
                    TaskGroup.schedule_task_groups(eligible, self.machine_list, self.reference_time)
                    report["status"] = "FINISHED"
                else:
                    raise ValueError(f"Unexpected scheduling_method: {scheduling_method}")
                progress.reports.append(attempt_record(report))
                progress.complete = True
                progress.input_hash = fingerprint((eligible, self.machine_list, self.reference_time))
                for group in eligible:
                    e = self._experiment(group.experiment_uuid, optional=True)
                    if e is None:
                        continue
                    if e.lifecycle_status != "SPLIT_PENDING":
                        e.lifecycle_status = "ACTIVE"
                    e.hold_reason = None
                return {"status": "READY", "waiting_experiments": waiting}
            except SolverFailure as err:
                report.update(err.diagnostics)
                report["attempt"] = attempt
                progress.reports.append(attempt_record(report))
                if err.status != "UNKNOWN":
                    progress.complete = True
                    reason = str(err)
                    if err.status == "INFEASIBLE":
                        reason += "; conflicting constraints were not identified by the solver; inspect diagnostic input"
                    raise self._stop(err.status, reason, solver=err.diagnostics) from err
                LOGGER.warning("UNKNOWN: attempt %s, search limit %ss", attempt, effective_limit)
                for group in eligible:
                    e = self._experiment(group.experiment_uuid, optional=True)
                    if e is None:
                        continue
                    if e.lifecycle_status != "SPLIT_PENDING":
                        e.lifecycle_status = "WAITING_PLAN"
                    e.hold_reason = "UNKNOWN: no valid plan"
                if self.unknown_policy == "stop":
                    progress.complete = True
                    self._stop("UNKNOWN", "Configured to stop without retry", solver=err.diagnostics)
                    return {"status": "STOPPED", "diagnostic": self.stop_diagnostic}
                if attempt >= self.max_unknown_retries:
                    progress.complete = True
                    raise self._stop("UNKNOWN_RETRY_LIMIT",
                                     f"No plan after {attempt + 1} attempts; retry limit reached",
                                     solver=err.diagnostics) from err
                progress.next_attempt = attempt + 1
            except Exception as err:
                progress.complete = True
                raise self._stop("PLANNING_ERROR", err) from err

    def resume(self):
        """Explicitly start a new bounded planning session after correcting STOP."""
        self._validate_policy()
        self.manager_status = "RUNNING"
        self.stop_diagnostic = None
        self.planning_progress = None
        try:
            for group in list(self.task_groups):
                if group.is_completed():
                    self._complete_group(group)
            return self.plan_tasks()
        except Exception as err:
            if self.manager_status != "STOPPED":
                raise self._stop("RESUME_ERROR", err) from err
            raise

    def start_task(self, task_group_id=None, task_id=None, *, task_group_uuid=None):
        from gems_python.multi_machine_problem_interval_task.task_info import TaskGroup
        group = self._resolve_group(task_group_id, task_group_uuid)
        experiment = self._experiment(group.experiment_uuid)
        if self.manager_status == "STOPPED" or experiment.lifecycle_status in (
                "WAITING_MACHINE", "WAITING_PLAN"):
            raise RuntimeError(f"Task start blocked: {self.manager_status}, "
                               f"{experiment.lifecycle_status}: {experiment.hold_reason}")
        task = next(t for t in group.tasks if t.task_id == task_id)
        if task.scheduled_time is None or task.allocated_machine_id is None:
            raise RuntimeError("Task has no valid schedule")
        TaskGroup.start_task(self.task_groups, group.task_group_id, task_id)

    def _resolve_group(self, group_id=None, group_uuid=None):
        if group_uuid is not None:
            return next(g for g in self.task_groups if g.task_group_uuid == group_uuid)
        return next(g for g in self.task_groups if g.task_group_id == group_id)

    def _end_record(self, experiment, request_id, reason, history=None):
        return EndRecord(experiment.experiment_uuid, reason, self.reference_time, request_id,
                         experiment.current_state_name,
                         experiment.history_for_state() if history is None else history,
                         copy.deepcopy(experiment.initial_conditions),
                         copy.deepcopy(experiment.initial_condition_origins),
                         experiment.parent_uuid,
                         {"protocol_states": copy.deepcopy(experiment.states),
                          "dataframe_backend": experiment.dataframe_backend,
                          "solver_time_limit_seconds": self.solver_time_limit_seconds,
                          "unknown_policy": self.unknown_policy,
                          "max_unknown_retries": self.max_unknown_retries,
                          "enable_split": self.enable_split})

    def request_finish(self, experiment_uuid, request_id, reason):
        if not isinstance(request_id, str) or not request_id or not isinstance(reason, str) or not reason:
            raise ValueError("Finish request requires a nonempty request_id and reason")
        specification = fingerprint((experiment_uuid, reason))
        if request_id in self.finish_requests:
            saved_uuid, saved_hash = self.finish_requests[request_id]
            if saved_hash != specification:
                raise ValueError(f"Conflicting finish request ID: {request_id}")
            return self.ended_experiments[saved_uuid]
        if request_id in self.split_requests:
            raise ValueError(f"Request ID already used for split: {request_id}")
        experiment = self._experiment(experiment_uuid)
        if any(g.experiment_uuid == experiment_uuid and g.tasks and not g.is_completed()
               for g in self.task_groups):
            raise ValueError(f"Experiment {experiment_uuid} still has unfinished tasks")
        if any(r.parent_uuid == experiment_uuid and r.status == "PENDING"
               for r in self.split_requests.values()):
            raise ValueError(f"Experiment {experiment_uuid} has a pending split")
        record = self._end_record(experiment, request_id, reason)
        self.ended_experiments[experiment_uuid] = record
        self.finish_requests[request_id] = (experiment_uuid, specification)
        self.experiments = [e for e in self.experiments if e.experiment_uuid != experiment_uuid]
        self.task_groups = [g for g in self.task_groups if g.experiment_uuid != experiment_uuid]
        return record

    def request_split(self, parent_uuid, request_id, children):
        if not self.enable_split:
            raise ValueError("Experimental splitting is disabled")
        if not isinstance(request_id, str) or not request_id:
            raise ValueError("Split request requires a nonempty request_id")
        specification = fingerprint((parent_uuid, children))
        if request_id in self.split_requests:
            record = self.split_requests[request_id]
            if record.specification_hash != specification:
                raise ValueError(f"Conflicting split request ID: {request_id}")
            return record
        if request_id in self.finish_requests:
            raise ValueError(f"Request ID already used for finish: {request_id}")
        parent = self._experiment(parent_uuid)
        if any(r.parent_uuid == parent_uuid and r.status == "PENDING"
               for r in self.split_requests.values()):
            raise ValueError(f"Parent {parent_uuid} already has a pending split")
        try:
            if len(children) < 2 or any(not isinstance(s, ChildSpec) for s in children):
                raise ValueError("A split requires at least two ChildSpec records")
            if len({s.key for s in children}) != len(children):
                raise ValueError("Child keys must be unique")
            for spec in children:
                if not isinstance(spec.key, str) or not spec.key:
                    raise ValueError("Child key must be a nonempty string")
                names = [s.state_name for s in spec.states]
                if len(set(names)) != len(names) or spec.current_state_name not in names:
                    raise ValueError(f"Invalid initial state for child {spec.key}")
                if any(s is p for s in spec.states for p in parent.states):
                    raise ValueError("Child states must be fresh, not parent State instances")
                frames.serialize(spec.initial_history)
                if not isinstance(spec.initial_conditions, dict) or not isinstance(
                        spec.initial_condition_origins, dict):
                    raise ValueError("Initial conditions and origins must be dictionaries")
                if set(spec.initial_conditions) != set(spec.initial_condition_origins):
                    raise ValueError(f"Every initial condition needs an origin: {spec.key}")
            uuids = [str(uuid5(NAMESPACE_URL, f"gems:{parent_uuid}:{request_id}:{s.key}"))
                     for s in children]
            if any(u in self.ended_experiments or any(e.experiment_uuid == u for e in self.experiments)
                   for u in uuids):
                raise ValueError("Child UUID collision")
            # Preview protocol-defined task generation before accepting a request.
            # These temporary children are never registered or executed.
            self._build_children(parent, children, uuids, parent.history_for_state())
        except Exception as err:
            raise self._stop("INVALID_CHILD_SPECIFICATION", err) from err
        record = SplitRecord(request_id, parent_uuid, specification,
                             copy.deepcopy(children), uuids)
        self.split_requests[request_id] = record
        parent.lifecycle_status = "SPLIT_PENDING"
        return record

    def _build_children(self, parent, specifications, uuids, history):
        from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment
        children = [
            Experiment(spec.experiment_name, copy.deepcopy(spec.states), spec.current_state_name,
                       shared_variable_history=frames.copy_frame(spec.initial_history),
                       experiment_uuid=uuid, dataframe_backend=parent.dataframe_backend,
                       _inherited_history=history, parent_uuid=parent.experiment_uuid,
                       initial_conditions=copy.deepcopy(spec.initial_conditions),
                       initial_condition_origins=copy.deepcopy(spec.initial_condition_origins))
            for spec, uuid in zip(specifications, uuids)
        ]
        for child in children:
            if child.current_task_group.tasks:
                validate_inputs([child.current_task_group], self.reference_time)
        return children

    def confirm_split(self, request_id, succeeded=True, *, reason=None, replan=True):
        """Confirm a completed virtual operation. Never invokes a device operation."""
        record = self.split_requests[request_id]
        if record.status != "PENDING":
            if (record.status == "COMMITTED") != bool(succeeded):
                raise ValueError(f"Conflicting split confirmation: {request_id}")
            return record
        parent = self._experiment(record.parent_uuid)
        if not succeeded:
            record.status, record.failure_reason = "FAILED", reason or "Split completion failed"
            parent.hold_reason = record.failure_reason
            LOGGER.warning("Split %s: %s", request_id, record.failure_reason)
            return record
        if self.manager_status == "STOPPED":
            raise RuntimeError("Cannot confirm a split while STOPPED; call resume after correction")
        if any(g.experiment_uuid == parent.experiment_uuid and g.tasks and not g.is_completed()
               for g in self.task_groups):
            raise ValueError("Cannot confirm split before parent tasks finish")
        history = parent.history_for_state()
        before = (self.experiments, self.task_groups, self.ended_experiments,
                  record.status, record.logical_time)
        try:
            if fingerprint((record.parent_uuid, record.children)) != record.specification_hash:
                raise ValueError(f"Split specification changed after request: {request_id}")
            children = self._build_children(parent, record.children, record.child_uuids, history)
            ending = self._end_record(parent, request_id, "split", history)
            new_experiments = [e for e in self.experiments if e is not parent] + children
            new_groups = [g for g in self.task_groups if g.experiment_uuid != parent.experiment_uuid]
            new_groups += [e.current_task_group for e in children if e.current_task_group.tasks]
            # Allocate containers before the commit; no user code runs in the swap.
            endings = dict(self.ended_experiments, **{parent.experiment_uuid: ending})
            self._commit_split(new_experiments, new_groups, endings, record)
        except Exception as err:
            (self.experiments, self.task_groups, self.ended_experiments,
             record.status, record.logical_time) = before
            raise self._stop("INVALID_CHILD_SPECIFICATION", err) from err
        if replan:
            self.plan_tasks()
        return record

    def _commit_split(self, experiments, groups, endings, record):
        before = (self.experiments, self.task_groups, self.ended_experiments,
                  record.status, record.logical_time)
        try:
            self.experiments, self.task_groups, self.ended_experiments = experiments, groups, endings
            record.status, record.logical_time = "COMMITTED", self.reference_time
        except BaseException:
            (self.experiments, self.task_groups, self.ended_experiments,
             record.status, record.logical_time) = before
            raise

    def _complete_group(self, group):
        if group.task_group_uuid in self.handled_groups:
            return
        experiment = self._experiment(group.experiment_uuid)
        pending = next((r for r in self.split_requests.values()
                        if r.parent_uuid == experiment.experiment_uuid and r.status == "PENDING"), None)
        context = LifecycleContext(experiment.experiment_uuid, group.task_group_uuid,
                                   f"lifecycle:{group.task_group_uuid}", self.reference_time)
        action = None if pending else experiment.states[experiment.current_state_index].lifecycle_action(
            experiment.history_for_state(), context)
        if isinstance(action, Finish):
            self.request_finish(experiment.experiment_uuid, context.request_id, action.reason)
        elif isinstance(action, Split):
            self.request_split(experiment.experiment_uuid, context.request_id, action.children)
        elif action is not None:
            raise ValueError(f"Unsupported lifecycle action: {type(action).__name__}")
        elif not pending and group.tasks:
            old = (experiment.current_state_name, experiment.current_state_index,
                   experiment.current_task_group, copy.deepcopy(experiment.states),
                   experiment.task_group_sequence)
            try:
                new = experiment.execute_one_step()
            except Exception:
                (experiment.current_state_name, experiment.current_state_index,
                 experiment.current_task_group, experiment.states,
                 experiment.task_group_sequence) = old
                raise
            if new.tasks:
                self.task_groups.append(new)
        self.task_groups = [g for g in self.task_groups if g.task_group_uuid != group.task_group_uuid]
        self.handled_groups[group.task_group_uuid] = context.request_id

    def record_task_result(self, task_group_uuid, task_id, result_id, result, *, update_type="a"):
        return self.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
            task_group_id=None, task_id=task_id, new_result_of_experiment=result,
            update_type=update_type, task_group_uuid=task_group_uuid, result_id=result_id)
