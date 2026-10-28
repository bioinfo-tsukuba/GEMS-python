"""CP-SAT scheduling shared by the three task representations."""

from importlib import import_module
import logging

from ortools.sat.python import cp_model


LOGGER = logging.getLogger(__name__)
SOLVER_TIME_LIMIT_SECONDS = 5.0


def _rest_maximum(penalty) -> int:
    module = type(penalty).__module__.rsplit(".", 1)[0]
    return import_module(f"{module}.config").PENALTY_MAXIMUM


def _cost(model, start, lower, upper, optimal, penalty, name):
    kind = type(penalty).__name__
    distance_bound = max(abs(lower - optimal), abs(upper - optimal))
    if kind == "NonePenalty":
        return 0, 0

    if kind in ("LinearPenalty", "CyclicalRestPenaltyWithLinear"):
        coefficient = penalty.penalty_coefficient
        if coefficient < 0:
            raise ValueError("Penalty coefficients must be nonnegative")
        distance = model.new_int_var(0, distance_bound, f"{name}_distance")
        model.add_abs_equality(distance, start - optimal)

    if kind == "LinearPenalty":
        maximum = coefficient * distance_bound
        cost = model.new_int_var(0, maximum, f"{name}_cost")
        model.add(cost == coefficient * distance)
        return cost, maximum

    if kind == "LinearWithRangePenalty":
        if penalty.lower > penalty.upper or min(
            penalty.lower_coefficient, penalty.upper_coefficient
        ) < 0:
            raise ValueError("Invalid linear range penalty")
        early_bound = max(0, optimal + penalty.lower - lower)
        late_bound = max(0, upper - optimal - penalty.upper)
        early = model.new_int_var(0, early_bound, f"{name}_early")
        late = model.new_int_var(0, late_bound, f"{name}_late")
        model.add_max_equality(early, [0, optimal + penalty.lower - start])
        model.add_max_equality(late, [0, start - optimal - penalty.upper])
        maximum = (
            early_bound * penalty.lower_coefficient
            + late_bound * penalty.upper_coefficient
        )
        cost = model.new_int_var(0, maximum, f"{name}_cost")
        model.add(
            cost
            == early * penalty.lower_coefficient
            + late * penalty.upper_coefficient
        )
        return cost, maximum

    if kind not in ("CyclicalRestPenalty", "CyclicalRestPenaltyWithLinear"):
        raise ValueError(f"Unsupported penalty type for OR-Tools: {kind}")

    duration = penalty.cycle_duration
    if duration <= 0:
        raise ValueError("cycle_duration must be positive")
    active = model.new_bool_var(f"{name}_cycle_active")
    model.add(start >= penalty.cycle_start_time).only_enforce_if(active)
    model.add(start < penalty.cycle_start_time).only_enforce_if(active.Not())
    elapsed = model.new_int_var(
        0, max(0, upper - penalty.cycle_start_time), f"{name}_elapsed"
    )
    model.add(elapsed == start - penalty.cycle_start_time).only_enforce_if(active)
    model.add(elapsed == 0).only_enforce_if(active.Not())
    phase = model.new_int_var(0, duration - 1, f"{name}_phase")
    model.add_modulo_equality(phase, elapsed, duration)

    # calculate_penalty() includes both endpoints of every rest range.
    rest_table = [
        int(any(first <= minute <= last for first, last in penalty.rest_time_ranges))
        for minute in range(duration)
    ]
    phase_is_rest = model.new_bool_var(f"{name}_phase_is_rest")
    model.add_element(phase, rest_table, phase_is_rest)
    is_rest = model.new_bool_var(f"{name}_is_rest")
    model.add(is_rest <= active)
    model.add(is_rest <= phase_is_rest)
    model.add(is_rest >= active + phase_is_rest - 1)

    maximum = _rest_maximum(penalty)
    if kind == "CyclicalRestPenalty":
        cost = model.new_int_var(0, maximum, f"{name}_cost")
        model.add(cost == maximum * is_rest)
        return cost, maximum

    maximum = max(maximum, coefficient * distance_bound)
    cost = model.new_int_var(0, maximum, f"{name}_cost")
    model.add(cost == 0).only_enforce_if(active.Not())
    model.add(cost == _rest_maximum(penalty)).only_enforce_if(is_rest)
    model.add(cost == coefficient * distance).only_enforce_if(
        [active, is_rest.Not()]
    )
    return cost, maximum


def _solve(model, costs, cost_bounds, starts, reference_time, horizon):
    # The second term chooses earlier times among equal-penalty schedules.
    span = horizon - reference_time
    weight = len(starts) * span + 1
    if sum(cost_bounds) * weight + len(starts) * span >= 2**63:
        raise ValueError("Scheduling objective exceeds CP-SAT integer range")
    model.minimize(sum(costs) * weight + sum(start - reference_time for start in starts))
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = SOLVER_TIME_LIMIT_SECONDS
    solver.parameters.num_search_workers = 1
    solver.parameters.random_seed = 0
    status = solver.solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"OR-Tools failed to find a schedule: {solver.status_name(status)}")
    LOGGER.info(
        "OR-Tools status=%s penalty=%s objective=%s best_bound=%s wall_time_seconds=%.3f",
        solver.status_name(status),
        sum(solver.value(cost) if not isinstance(cost, int) else cost for cost in costs),
        solver.objective_value,
        solver.best_objective_bound,
        solver.wall_time,
    )
    return solver


def schedule_simple_tasks(tasks, reference_time: int):
    """Schedule independent tasks on one machine using absolute minute times."""
    if not tasks:
        return tasks
    if any(task.processing_time < 0 for task in tasks):
        raise ValueError("processing_time must be nonnegative")
    horizon = max(reference_time, *(task.optimal_time for task in tasks))
    horizon += sum(task.processing_time for task in tasks) + sum(
        max(0, getattr(task.penalty_type, "cycle_duration", 0)) for task in tasks
    ) + 1
    model = cp_model.CpModel()
    starts, costs, cost_bounds, intervals = [], [], [], []
    for index, task in enumerate(tasks):
        start = model.new_int_var(reference_time, horizon, f"task_{index}_start")
        end = model.new_int_var(
            reference_time, horizon + task.processing_time, f"task_{index}_end"
        )
        intervals.append(
            model.new_interval_var(
                start, task.processing_time, end, f"task_{index}_interval"
            )
        )
        cost, bound = _cost(
            model, start, reference_time, horizon, task.optimal_time,
            task.penalty_type, f"task_{index}",
        )
        starts.append(start)
        costs.append(cost)
        cost_bounds.append(bound)
    model.add_no_overlap(intervals)
    solver = _solve(model, costs, cost_bounds, starts, reference_time, horizon)
    for task, start in zip(tasks, starts):
        task.scheduled_time = solver.value(start)
    return tasks


def schedule_task_groups(task_groups, reference_time: int, machines=None):
    """Schedule fixed-offset task groups on one or multiple physical machines."""
    if not task_groups:
        return task_groups
    movable = [group for group in task_groups if group.is_not_started()]
    if not movable:
        return task_groups
    all_tasks = [task for group in task_groups for task in group.tasks]
    if any(task.processing_time < 0 or task.interval < 0 for task in all_tasks):
        raise ValueError("processing_time and interval must be nonnegative")
    fixed_end = [
        task.scheduled_time + task.processing_time
        for group in task_groups if not group.is_not_started()
        for task in group.tasks
        if not task.completed and task.scheduled_time is not None
    ]
    horizon = max(
        reference_time,
        *(group.optimal_start_time for group in movable),
        *fixed_end,
    )
    horizon += sum(
        task.processing_time + task.interval
        for group in movable for task in group.tasks
    ) + sum(
        max(0, getattr(group.penalty_type, "cycle_duration", 0))
        for group in movable
    ) + 1
    model = cp_model.CpModel()
    starts, costs, cost_bounds = [], [], []
    assignments = []
    if machines is None:
        machine_intervals = {None: []}
        by_type = None
    else:
        machine_intervals = {machine.machine_id: [] for machine in machines.machines}
        by_type = {}
        for machine in machines.machines:
            by_type.setdefault(machine.machine_type, []).append(machine)

    def add_task(task, start, end, label, fixed=False):
        if by_type is None:
            machine_intervals[None].append(
                model.new_interval_var(start, task.processing_time, end, label)
            )
            return
        compatible = by_type.get(task.optimal_machine_type, [])
        if not compatible:
            raise ValueError(f"No machine of type {task.optimal_machine_type}")
        if fixed and task.allocated_machine_id is not None:
            compatible = [
                machine for machine in compatible
                if machine.machine_id == task.allocated_machine_id
            ]
            if not compatible:
                raise ValueError(f"Unknown assigned machine: {task.allocated_machine_id}")
        choices = []
        for machine in compatible:
            chosen = model.new_bool_var(f"{label}_on_{machine.machine_id}")
            interval = model.new_optional_interval_var(
                start, task.processing_time, end, chosen,
                f"{label}_interval_{machine.machine_id}",
            )
            machine_intervals[machine.machine_id].append(interval)
            choices.append((chosen, machine.machine_id))
        model.add(sum(chosen for chosen, _ in choices) == 1)
        assignments.append((task, choices))

    for group_index, group in enumerate(task_groups):
        if group.is_completed():
            continue
        if group.is_not_started():
            group_start = model.new_int_var(
                reference_time, horizon, f"group_{group_index}_start"
            )
            cost, bound = _cost(
                model, group_start, reference_time, horizon,
                group.optimal_start_time, group.penalty_type,
                f"group_{group_index}",
            )
            starts.append(group_start)
            costs.append(cost)
            cost_bounds.append(bound)
            elapsed = 0
            for task_index, task in enumerate(group.tasks):
                elapsed += task.interval
                start = model.new_int_var(
                    reference_time + elapsed, horizon + elapsed,
                    f"group_{group_index}_task_{task_index}_start",
                )
                model.add(start == group_start + elapsed)
                end = model.new_int_var(
                    reference_time + elapsed,
                    horizon + elapsed + task.processing_time,
                    f"group_{group_index}_task_{task_index}_end",
                )
                add_task(task, start, end, f"group_{group_index}_task_{task_index}")
                assignments.append((task, start))
                elapsed += task.processing_time
        else:
            for task_index, task in enumerate(group.tasks):
                if task.completed:
                    continue
                if task.scheduled_time is None:
                    raise ValueError("In-progress task has no scheduled_time")
                start = task.scheduled_time
                end = start + task.processing_time
                add_task(
                    task, start, end, f"group_{group_index}_task_{task_index}",
                    fixed=True,
                )
    for intervals in machine_intervals.values():
        model.add_no_overlap(intervals)
    solver = _solve(model, costs, cost_bounds, starts, reference_time, horizon)
    for item in assignments:
        task, value = item
        if isinstance(value, list):
            task.allocated_machine_id = next(
                machine_id for chosen, machine_id in value if solver.value(chosen)
            )
        else:
            task.scheduled_time = solver.value(value)
    return task_groups
