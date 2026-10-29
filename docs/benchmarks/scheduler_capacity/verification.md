# Issue #23 verification, 2026-10-29

## Requirements and environment

The user approved a change from whole-plan budgets to fixed CP-SAT search limits:
1.0/60.0 seconds are passed directly; other work is neither subtracted nor used
as a success gate. The normal default remains 5.0 seconds. Limits are saved per
manager and can be overridden per call without changing the saved setting.
The study ceiling is 512 experiments, boundary conditions have five repetitions,
and the total local study allowance is six hours.

Baseline: `aee423c0d7c78fd2a34062036669c1dad6164931`, latest main at task start.
Work branch: `fix/23-scheduler-capacity`. Issue #23 records the revised criterion,
approach, progress and remaining work. Python 3.14.4 / OR-Tools 9.15.6755 use the
existing `/tmp/gems-venv` environment; no dependency or lock-file change.

## Correctness and compatibility

`overlap_regression.json` preserves the original four-group/two-machine input
and before/after outputs from `scripts.compare_schedulers.run_case`:

| Method / seed | Before overlap minutes | After overlap minutes |
| --- | ---: | ---: |
| f / 0 | 4 | 0 |
| s / 0 | 4 | 0 |
| s / 1 | 4 | 0 |
| s / 2 | 4 | 0 |
| o / 0 | 0 | 0 |

Physical-machine assignment and group-interval errors are also zero after the
fix. These overlap values are virtual experimental minutes, not computation time.

Focused verification:

```bash
MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg /tmp/gems-venv/bin/python \
  -m unittest tests.test_scheduler_capacity tests.test_capacity_measurement \
  tests.test_cp_sat_scheduler tests.test_experiment_simulators \
  tests.test_persistence_roundtrip tests.test_dataframe_backends \
  tests.test_scheduler_pattern_study -q

VIRTUAL_ENV=/tmp/gems-venv POETRY_CACHE_DIR=/tmp/gems-poetry-cache \
  /tmp/gems-venv/bin/poetry check --lock
```

All 54 focused tests passed. Lock checking passed with pre-existing Poetry
metadata deprecation warnings. The CLI smoke study exercised fresh worker
processes, paired snapshots, scan repetition, experiment additions, structured
call collection and reconstruction of summaries.

The complete suite ran in separate disposable source copies, with stdin closed
and a 360-second safeguard:

```bash
MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg timeout 360 \
  /tmp/gems-venv/bin/python -m unittest discover -s tests -p 'test_*.py' \
  < /dev/null
```

Baseline: 87 tests, five errors. Fixed version: 98 tests, the same five errors:
three HEK fixtures omit `current_state_name`; an interactive test encounters EOF;
a legacy assertion tries to unpack `OneMachineTaskLocalInformation`.
No additional failure remains. Generated files stayed in disposable copies.

## Failures found and repaired

- **Old saved data:** adding the setting initially caused v1 records without it
  to fail with `missing saved fields solver_time_limit_seconds`. The decoder
  now defaults only this documented new field to 5.0; missing progress fields
  still fail. Old JSON/pickle and existing legacy fixtures are verified.
- **Simultaneous completions:** two same-duration tasks ended together, but the
  task whose completion had not yet been reported was treated as unstarted and
  moved from minute 0 to 3. All tasks that have begun by the event time now keep
  their group times and IDs. Completion order is used for both dummy/provider
  paths. A two-experiment regression verifies both completions at minute 3 and
  then advances to normal termination.
- **Failure compatibility:** an intermediate eager fixed-overlap check changed
  the existing CP-SAT `INFEASIBLE` RuntimeError into ValueError. Removing that
  eager check restored the solver-proven path while keeping independent output
  validation, explicit legacy failures and rollback. The original regression
  and full-suite comparison verify the repair.
- **Annealing finalization:** the original allocator released unrelated machine
  IDs and annealing reused assignments from its initial timing. Reservations
  now use physical IDs; final assignments are recomputed. Failed finalization
  returns the independent validated initial plan and records the reason.
  The deterministic greedy allocator can reject a coloring that a more complete
  assignment search would find; rejection is not an infeasibility proof.

## Limits of these checks

Small tests establish physical constraints, fixed work, API propagation and
persistence, not general optimality or arbitrary-scale capacity. `FEASIBLE` is
never presented as optimal. Operational benchmarks use fixed synthetic
observations, and do not model real computation delaying real equipment.
The remaining legacy suite errors are outside Issue #23.
