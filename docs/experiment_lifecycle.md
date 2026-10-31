# Experiment completion, splitting, and holds

[README](../README.md#experiment-lifecycle) · [Post-run visualization](experiment_retrospective.md) · [Saving and reloading](persistence.md)

Issue #24 covers virtual experiments in `multi_machine_problem_interval_task`.
The existing `transition_function(history) -> str` contract remains in place.
Completion is not inferred from State names or empty Task groups.

## Declaring completion and splitting

A State can optionally implement `lifecycle_action(history, context)`.
It is called when a Task group completes, or when an empty group is registered
or entered on a State transition. Returning `None` continues normal progression.
`context` contains the Experiment UUID, persistent Task group UUID, request ID,
and logical time.

```python
from gems_python.multi_machine_problem_interval_task.lifecycle import Finish, Split, ChildSpec

def lifecycle_action(self, history, context):
    if protocol_has_finished(history):
        return Finish("protocol complete")
    return None
```

Completion first checks that no required Task remains unfinished, then removes
the Experiment from active execution management.
`lab.ended_experiments[uuid]` retains its history, completion reason, logical time,
request ID, initial conditions, and provenance.
`EndRecord.settings` retains copies of State settings and final progress, the
history backend, and the manager's search, retry, and splitting settings.
Physical samples are not discarded, and history is retained.
`EndRecord.history` returns a copy. When all experiments finish, `no_tasks` is
a normal result.

Splitting is an experimental feature enabled by
`Experiments(..., enable_split=True)`.
Request it with `Split(children)` or
`lab.request_split(parent_uuid, request_id, children)`.
Each `ChildSpec` requires a child key, name, independent States, initial State
name, initial observations, an initial-conditions dictionary, and a dictionary
describing each condition's provenance.
These dictionaries can store quantities, units, and container IDs. The protocol
defines scientific copying, conversion, and conservation rules; the framework
does not infer values or distribute quantities automatically.

A request puts the parent in `SPLIT_PENDING`; children are not yet active.
After the parent's required Tasks finish, confirm with
`lab.confirm_split(request_id, succeeded=True)`.
This API does not operate devices. The virtual executor calls it after confirming
success. Record a failure with
`confirm_split(..., succeeded=False, reason="...")`; no children are registered.
Use a new request ID when submitting corrected specifications.

Confirmation removes the parent, registers all children, and records provenance
events together, then replans the eligible experiments.
A child with a missing machine type is still registered, with only that child
waiting for a machine. A planning failure after confirmation does not restore
the parent.
Child UUIDs are derived from the parent UUID, request ID, and child key, so
repeated requests do not create additional children. Reusing a request ID with
different specifications raises an error.
A split requires at least two children. If the remaining sample should continue,
declare it as another child.

## History and repeated result delivery

`shared_variable_history` holds the Experiment's own initial and subsequent
observations. A child separately retains a snapshot of its parent's history at
the split.
`history_for_state()` returns a copy combining parent history, child initial
observations, and child observations in that order. State transitions, Task
generation, and virtual observations receive the same input.
`inherited_history` also returns a copy, leaving parent and sibling histories
unchanged. Inherited observations are not duplicated across multiple generations.

Numeric `task_group_id` values are reassigned when the list changes.
Use persistent `task_group_uuid` and `result_id` values for repeated delivery.

```python
lab.start_task(task_group_uuid=group.task_group_uuid, task_id=task.task_id)
lab.record_task_result(group.task_group_uuid, task.task_id, result_id, result_frame)
```

Repeated delivery of the same result ID, Task, and observations does not add
history or events. Reusing a result ID with different content raises an error.
Result files can also include `task_group_uuid` and `result_id`.
The legacy numeric-ID API still operates on the current list. Persistent IDs
are required for repeated delivery across deletion or reassignment.

## Holds, search retries, and STOP

Inputs are checked for the entire first Task group and whenever a subsequent
group is generated. If a required machine type is unregistered, the affected
Experiment enters `WAITING_MACHINE` and emits a warning.
It remains registered, and its unstarted group is excluded from the search;
other experiments continue.
Registering a machine with `lab.add_machine(...)` triggers revalidation and
replanning. A busy machine does not cause this hold when a valid future
reservation is available.

`OPTIMAL` and `FEASIBLE` plans are accepted.
`UNKNOWN` emits a warning, holds unplanned starts, and doubles the search limit.

| Setting | Default | Basis |
| --- | --- | --- |
| `solver_time_limit_seconds` | 5 seconds | Existing setting and per-call overrides are preserved |
| `unknown_policy` | `"retry"` | Issue #24 policy |
| `max_unknown_retries` | 3 retries, excluding the first attempt | Implementation plan |
| `enable_split` | `False` | Issue #24 experimental-feature policy |

The defaults allow up to four attempts: 5 → 10 → 20 → 40 seconds.
Actual elapsed time also includes model construction, validation, and other work.
Retries leave the manager's initial setting, registrations, and confirmed splits
unchanged. `unknown_policy="stop"` warns and stops on the first `UNKNOWN`.
Exhausted retries, `INFEASIBLE`, invalid specifications, and conflicting fixed
reservations stop the whole manager with a diagnostic error.
Diagnostics include reproducible inputs for the affected Task groups, machines,
and settings. When the solver cannot identify the conflicting constraint, the
`INFEASIBLE` status and inputs are retained without guessing the cause.
Search, construction, validation, and application times are recorded in structured
logs, separately from scientific state.

`STOPPED` prevents new Task starts without forcibly interrupting running operations.
Results from running Tasks are accepted and recorded, while State progression
and new Task generation wait until the stop is lifted.
Correct the cause, then explicitly revalidate and replan with `lab.resume()`.
A normal snapshot can continue execution after loading; a stopped snapshot
remains stopped. Loading alone does not start Tasks or replan.

## Snapshot generations and legacy saves

`save_snapshot` writes JSON and pickle into the same generation under
`.snapshot_generations/<id>/`, synchronizes files and directories, then switches
`snapshot_manifest.json` atomically.
Loading selects the manifest once and checks that generation's hashes.
An unfinished generation is not selected; the existing committed generation
remains available. If neither format is usable, loading reports the reasons
without combining generations.
If only one format is valid, `source="auto"` warns and loads it.
A committed generation remains loadable even if writing the compatibility outputs
fails after manifest publication. A save exception alone does not establish that
the old generation was selected; inspect the manifest and reload result.

Root-level `experiments.json` and `experiments.pkl` remain as compatibility outputs.
When a manifest exists, they are not the authoritative reload inputs.
Direct edits by external tools are not implicitly incorporated into a manifested
snapshot. Legacy saves without a manifest use the previous two-file loading path.
Generations are retained for reproduction and investigation without automatic
deletion. This change applies to all three managers using the shared persistence
implementation.

The general serialization format remains version 1; lifecycle data has its own
version 1. Only legacy saves without a lifecycle version receive disabled
splitting, empty records, and default retry settings.
Missing progress fields in new saves raise an error. State and Task progress
are not regenerated.
The code defining saved classes must be available when loading. Solver search
internals are not saved. Resuming a search starts a new search with the remaining
retry count and next time limit.

## Verification and exclusions

Acceptance tests are `test_experiment_lifecycle`, `test_experiment_scheduling_policy`,
`test_lifecycle_persistence`, and `test_retrospective`.
The finite culture and explicit sample-splitting example can be reproduced with
`build` / `run` in `tests/simulators/lifecycle_cell_culture.py`.
Existing reservation-validation, DataFrame, persistence, and virtual-execution
tests also cover compatibility.
The archived [capacity evaluation after splitting](https://github.com/bioinfo-tsukuba/GEMS-python/blob/8ca9080bf62a4e6c2b70fe73a0f830c5f627b183/docs/benchmarks/experiment_lifecycle_capacity_2026-10-30/README.md)
and [implementation and verification record](https://github.com/bioinfo-tsukuba/GEMS-python/blob/cd78405f3c66061c3b71b72623a83540aaca83fb/docs/benchmarks/issue24_verification.md)
document historical results and reproduction steps.
The feature targets a single-process manager. Physical device splitting,
exactly-once guarantees for irreversible operations, forced interruption,
physical recovery, and concurrent writes from multiple processes are outside
its scope.

## Visualization after all experiments finish

`lab.render_retrospective("review", font="/path/to/font.ttf", time_unit="min")`
exports a lineage timeline, family-specific lineage and timeline pages, and an
interactive view linking lineage selection to machine reservations.
Call it after all experiments finish, using a new output directory.

`retrospective_trace` stores registered names and logical registration times,
plus completed Tasks' final reservations and machines, so plots can be regenerated
after Task groups are removed.
Bars show reservations, not measured physical execution intervals.
Missing times or Tasks in legacy saves produce a partial view with warnings;
they are not reconstructed.
See [post-run visualization](experiment_retrospective.md) for usage and a runnable example.
