"""Independent schedule checks, physical-machine reservations and diagnostics."""

from functools import wraps
import inspect
import logging
import math
from numbers import Integral, Real
from time import perf_counter


LOGGER = logging.getLogger(__name__)
DEFAULT_SOLVER_TIME_LIMIT_SECONDS = 5.0


def validate_time_limit(value):
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError("solver_time_limit_seconds must be a finite positive number")
    if not math.isfinite(value) or value <= 0:
        raise ValueError("solver_time_limit_seconds must be a finite positive number")
    return float(value)


def _integer(value, name, *, nonnegative=False):
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} must be an integer")
    if nonnegative and value < 0:
        raise ValueError(f"{name} must be nonnegative")


def validate_inputs(groups, reference_time, machines=None):
    _integer(reference_time, "reference_time")
    machine_by_id = {}
    if machines is not None:
        for machine in machines.machines:
            if machine.machine_id in machine_by_id:
                raise ValueError(f"Duplicate machine ID: {machine.machine_id}")
            machine_by_id[machine.machine_id] = machine
    for group in groups:
        if not group.tasks and not group.is_completed():
            raise ValueError("An active task group must contain tasks")
        if group.is_not_started() and any(
            task.completed or getattr(getattr(task, "task_status", None), "name", None) == "IN_PROGRESS"
            for task in group.tasks
        ):
            raise ValueError("Unstarted group contains started tasks; call status_update()")
        if group.is_completed() and any(not task.completed for task in group.tasks):
            raise ValueError("Completed group contains unfinished tasks; call status_update()")
        for task in group.tasks:
            _integer(task.processing_time, "processing_time", nonnegative=True)
            _integer(task.interval, "interval", nonnegative=True)
            if machines is not None and not task.completed and not group.is_completed():
                if not any(m.machine_type == task.optimal_machine_type
                           for m in machine_by_id.values()):
                    raise ValueError(f"No machine of type {task.optimal_machine_type}")
                if not group.is_not_started():
                    if task.allocated_machine_id is None and getattr(
                        getattr(task, "task_status", None), "name", None
                    ) == "NOT_STARTED":
                        continue  # Assign a future task while preserving its fixed time.
                    machine = machine_by_id.get(task.allocated_machine_id)
                    if machine is None or machine.machine_type != task.optimal_machine_type:
                        raise ValueError(f"Invalid fixed machine ID: {task.allocated_machine_id}")
    return machine_by_id


def fixed_work(groups):
    """Use task identity, rather than IDs that managers may renumber."""
    return {
        id(task): (task.scheduled_time, getattr(task, "allocated_machine_id", None))
        for group in groups if not group.is_not_started()
        for task in group.tasks
    }


def _check_intervals(by_machine):
    for machine_id, intervals in by_machine.items():
        end = None
        for start, finish in sorted(intervals):
            if finish == start:
                continue  # Empty half-open intervals occupy no machine time.
            if end is not None and start < end:
                raise ValueError(f"Overlapping tasks on machine {machine_id}")
            end = finish


def validate_groups(groups, reference_time, machines=None, fixed=None):
    machine_by_id = validate_inputs(groups, reference_time, machines)
    intervals = {}
    for group in groups:
        if group.is_completed():
            continue
        group_start = group.tasks[0].scheduled_time
        _integer(group_start, "scheduled_time")
        group_start -= group.tasks[0].interval
        if group.is_not_started() and group_start < reference_time:
            raise ValueError("Group starts before reference_time")
        elapsed = 0
        for task in group.tasks:
            elapsed += task.interval
            _integer(task.scheduled_time, "scheduled_time")
            if task.scheduled_time != group_start + elapsed:
                raise ValueError("Task order or fixed interval changed")
            if fixed is not None and id(task) in fixed:
                current = (task.scheduled_time, getattr(task, "allocated_machine_id", None))
                time, machine_id = fixed[id(task)]
                if current[0] != time or (machine_id is not None and current[1] != machine_id):
                    raise ValueError("Fixed task time or machine ID changed")
            elapsed += task.processing_time
            if task.completed:
                continue
            machine_id = None
            if machines is not None:
                machine_id = task.allocated_machine_id
                machine = machine_by_id.get(machine_id)
                if machine is None or machine.machine_type != task.optimal_machine_type:
                    raise ValueError(f"Invalid assigned machine: {machine_id}")
            intervals.setdefault(machine_id, []).append(
                (task.scheduled_time, task.scheduled_time + task.processing_time)
            )
    _check_intervals(intervals)


def validate_simple(tasks, reference_time):
    intervals = []
    for task in tasks:
        _integer(task.processing_time, "processing_time", nonnegative=True)
        _integer(task.scheduled_time, "scheduled_time")
        if task.scheduled_time < reference_time:
            raise ValueError("Task starts before reference_time")
        intervals.append((task.scheduled_time, task.scheduled_time + task.processing_time))
    _check_intervals({None: intervals})


def measured_schedule(function):
    """Validate before returning, roll back task edits on failure, and emit a record.

    Diagnostics are delivered as structured logging metadata, never saved on an
    experiment or recovered by parsing human-readable exception messages.
    """
    signature = inspect.signature(function)

    @wraps(function)
    def measured(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        values = bound.arguments
        simple = "tasks" in values
        items = values["tasks" if simple else "task_groups"]
        tasks = items if simple else [task for group in items for task in group.tasks]
        saved = [(task, task.scheduled_time, getattr(task, "allocated_machine_id", None))
                 for task in tasks]
        fixed = {} if simple else fixed_work(items)
        started = perf_counter()
        external_report = values.get("diagnostics")
        report = {"scheduler": function.__name__, "status": "NO_TASKS",
                  "build_seconds": 0.0, "search_seconds": 0.0,
                  "apply_seconds": 0.0, "validation_seconds": 0.0}
        values["diagnostics"] = report
        try:
            if "solver_time_limit_seconds" in values:
                values["solver_time_limit_seconds"] = validate_time_limit(
                    values["solver_time_limit_seconds"])
                report["solver_time_limit_seconds"] = values["solver_time_limit_seconds"]
            if simple:
                _integer(values["reference_time"], "reference_time")
                for task in tasks:
                    _integer(task.processing_time, "processing_time", nonnegative=True)
            else:
                validate_inputs(items, values["reference_time"], values.get("machines"))
            result = function(**values)
            checking = perf_counter()
            if simple:
                validate_simple(result, values["reference_time"])
            else:
                validate_groups(result, values["reference_time"], values.get("machines"), fixed)
            report["validation_seconds"] = perf_counter() - checking
            report["valid"] = True
            return result
        except Exception as error:
            for task, time, machine in saved:
                task.scheduled_time = time
                if hasattr(task, "allocated_machine_id"):
                    task.allocated_machine_id = machine
            report["valid"] = False
            report["error"] = f"{type(error).__name__}: {error}"
            if report["status"] in ("NO_TASKS", "FIXED_ONLY"):
                report["status"] = "INVALID_INPUT" if isinstance(error, ValueError) else "ERROR"
            elif report["status"] in ("OPTIMAL", "FEASIBLE", "FINISHED"):
                report["status"] = "INVALID_OUTPUT"
            raise
        finally:
            report["total_seconds"] = perf_counter() - started
            report["other_seconds"] = max(0.0, report["total_seconds"] - sum(
                report[key] for key in
                ("build_seconds", "search_seconds", "apply_seconds", "validation_seconds")))
            if external_report is not None:
                external_report.update(report)
            LOGGER.info("Scheduling status=%s valid=%s total_seconds=%.6f",
                        report["status"], report.get("valid"), report["total_seconds"],
                        extra={"scheduling_report": report})
    return measured


def _free(reservations, start, end):
    return start == end or all(first == last or end <= first or last <= start
                              for first, last in reservations)


def assign_machine_ids(groups, machines):
    """Reserve known fixed IDs, then color unassigned half-open intervals.

    This is a deterministic allocator, not a proof that no alternative coloring
    exists when fixed reservations make the greedy coloring fail.
    """
    calendars = {m.machine_id: [] for m in machines.machines}
    assignments = []
    unassigned = []
    for index, group in enumerate(groups):
        if group.is_completed() or group.is_not_started():
            continue
        for task_index, task in enumerate(group.tasks):
            if task.completed:
                continue
            if task.scheduled_time is None:
                raise ValueError("Fixed task must have a scheduled time and machine ID")
            if task.allocated_machine_id is None and task.task_status.name == "NOT_STARTED":
                unassigned.append((task.scheduled_time, index, task_index, task))
                continue
            if task.allocated_machine_id not in calendars:
                raise ValueError("Fixed task must have a scheduled time and machine ID")
            start, end = task.scheduled_time, task.scheduled_time + task.processing_time
            calendar = calendars[task.allocated_machine_id]
            if not _free(calendar, start, end):
                raise ValueError("Conflicting fixed machine reservations")
            calendar.append((start, end))
    movable = [(task.scheduled_time, index, task_index, task)
               for index, group in enumerate(groups) if group.is_not_started()
               for task_index, task in enumerate(group.tasks) if not task.completed]
    for start, _, _, task in sorted(unassigned + movable, key=lambda item: item[:3]):
        end = start + task.processing_time
        compatible = [m for m in machines.machines if m.machine_type == task.optimal_machine_type]
        machine = next((m for m in compatible if _free(calendars[m.machine_id], start, end)), None)
        if machine is None:
            raise ValueError(f"Cannot assign task to a machine of type {task.optimal_machine_type}")
        calendars[machine.machine_id].append((start, end))
        assignments.append((task, machine.machine_id))
    for task, machine_id in assignments:
        task.allocated_machine_id = machine_id


@measured_schedule
def schedule_sequential_groups(task_groups, machines, reference_time, *, diagnostics=None):
    """Place whole groups, preserving physical reservations and fixed offsets."""
    started = perf_counter()
    groups = sorted(task_groups, key=lambda group: (group.is_not_started(), group.optimal_start_time))
    placed = [group for group in groups if not group.is_not_started()]
    assign_machine_ids(placed, machines)
    validate_groups(placed, reference_time, machines)
    for group in groups:
        if not group.is_not_started():
            continue
        # Once all reserved work ends a serial placement is always available.
        latest_end = max([reference_time] + [
            task.scheduled_time + task.processing_time
            for prior in placed for task in prior.tasks if not task.completed
        ])
        preferred = max(reference_time, group.optimal_start_time)
        upper = max(preferred, latest_end)
        def candidates():
            yield preferred
            for delta in range(1, max(upper - preferred, preferred - reference_time) + 1):
                if preferred + delta <= upper:
                    yield preferred + delta
                if preferred - delta >= reference_time:
                    yield preferred - delta
        for candidate in candidates():
            group.schedule_tasks(candidate)
            try:
                assign_machine_ids(placed + [group], machines)
            except ValueError:
                continue
            placed.append(group)
            break
        else:
            raise RuntimeError("No serial placement found")
    diagnostics.update(status="FINISHED", apply_seconds=perf_counter() - started)
    return groups
