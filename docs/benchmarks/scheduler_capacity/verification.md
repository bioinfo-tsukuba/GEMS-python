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

All 58 focused tests passed. Lock checking passed with pre-existing Poetry
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

Baseline: 87 tests, five errors. Final version: 102 tests, the same five errors:
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

- **Inconsistent progress:** a running task inside a group still marked
  NOT_STARTED could move during replanning. Such contradictory progress now
  raises a clear input error before task edits; callers can use `status_update()`.
  Completed groups with unfinished work are rejected too. Explicit IN_PROGRESS
  groups with future pending work remain fixed reservations.
- **Unassigned future work:** an intermediate guard rejected a pending task with
  a fixed time but no machine ID, which the original CP scheduler accepted.
  Pending future IDs can now be assigned while preserving their time and any
  known ID. Running work still requires a known compatible ID. Fixed-only inputs
  are also handled and a second scheduling call preserves the first assignment.
- **Zero duration:** native no-overlap and the legacy allocator treated empty
  intervals inconsistently. A fixed zero-length task at minute 5 pushed a
  two-minute task away from its preferred minute 4. Empty half-open intervals now
  occupy no capacity in every CP variant and the physical allocator, while their
  times and compatible assignments are retained.
- **Benchmark CLI pickle:** changing a main-script class's `__module__` alone
  made newly added live states differ from the importable class. JSON succeeded
  but pickle failed with `not the same object`. The CLI now uses the importable
  class itself. A subprocess regression adds an experiment and round-trips the
  resulting manager through pickle. All four new regressions failed on the
  intermediate measured source and pass on the final source; logs are preserved.
- **Commit identity:** the initial commit attempt lacked a configured identity.
  Commits use per-command `user.name=Codex` and `user.email=codex@openai.com`,
  consistent with earlier repository commits. No persistent git configuration
  was changed.

## Limits of these checks

Small tests establish physical constraints, fixed work, API propagation and
persistence, not general optimality or arbitrary-scale capacity. `FEASIBLE` is
never presented as optimal. Operational benchmarks use fixed synthetic
observations, and do not model real computation delaying real equipment.
The remaining legacy suite errors are outside Issue #23.

## Measured records and final source

The primary study evaluated `9e269d41bd7ce83b80a99955856d14d46a018e3e`,
with a clean tree, from 2026-10-29 04:00:36 UTC. All 193 planned trials completed
in 10,088.052 seconds; none were skipped. Configuration and environment are
preserved in `config.json`, `environment.json` and `completion.json`.
The hardware was an Intel Core Ultra 7 265KF with 20 logical CPUs and about
62 GiB RAM, on Linux/WSL2. Solver worker count was 1 and seed was 0.
There was no CPU affinity or exclusive-host isolation.

The final executable source is
`b08dbe83d1addf14dc40a845d1e5c52c9d58d48e`; later artifact/documentation commits
do not change executable code. Its 58 focused tests passed in 18.608 seconds.
Its archived complete suite ran 102 tests in 46.612 seconds, with the same
five baseline errors. The primary study's 45 paired static conditions have
identical typed inputs across limits and repetitions.

`scripts/verify_scheduler_capacity.py` verified:

- 193 restored input hashes and final input guards;
- 333 scheduling-call records with the requested per-call limit;
- 161 valid output schedules;
- 153 static/IO penalty values and fixed time/ID comparisons;
- 67 retained update inputs, including those preceding failed updates;
- five saved JSON/pickle round-trips and nonempty PNG/PDF/SVG outputs;
- 32 preserved failures, all classified as NO_SOLUTION.

The 32 failures comprise 30 static UNKNOWN runs and two operational UNKNOWN
runs. `many_candidates` at 128 experiments / 1 second completed one update
before failing in the second; `mixed_penalties` at 128 / 1 second failed in
the first update's second scheduling call. These are observations at the
configured limit, not proofs of infeasibility. All five 60-second operational
conditions completed eight updates and eighteen scheduling calls. Static
success at a size is not an operational reliability guarantee.

Model capture without solving verified identical model text hashes and solver
parameters for the largest 512-experiment input of every pattern between the
measured and final sources. `model_equivalence.json` records those fingerprints
and both sources' file hashes; the archive includes the exported model text.
This checks representative retained states, not every internal operational
model individually. New input checks and the restored-input hash can add
overhead: the primary timing/memory figures belong to the measured commit.
Final-source confirmation is recorded separately.

The clean final source ran 17 confirmation conditions: 15 VALID and two
NO_SOLUTION, with 106 scheduling calls. All ten representative static conditions
were valid. The original `many_candidates` 128 / 1 second and `mixed_penalties`
128 / 1 second operational failures recurred, with respectively one and zero
successful updates. Supplemental 64-experiment / 1-second operations completed
eight updates and eighteen calls for both patterns. Contention 256,
heterogeneous 512 and replanning 256 also completed their eight-update
1-second confirmations. These single sequences are not reliability estimates
or proven maxima. See `confirmation.json` and `confirmation_summary.md`;
full requests, results and failed updates are in the archive.

The initial measurement recorded the request hash as `actual_input_hash`.
The independent restoration check confirms that all primary inputs restore
to exactly that hash. Final workers instead calculate this field directly from
the restored manager before modifying its output path.

The [overview](measurement_summary.md) includes large virtual delays despite
valid plans. No delay/quality acceptance threshold was imposed. Five boundary
repetitions and one eight-update sequence per operational condition do not
establish a sustainable facility throughput or a precise operational p95.
Fresh-worker peak memory ranged from 168.113 to 741.855 MiB; this includes
imports and cumulative retained records. Two-experiment IO saves took
0.0027–0.0031 seconds and drawing 0.2700–0.2765 seconds. These IO samples
are reference costs, not large-scale extrapolations. The success-rate figure
and an actual generated Gantt PNG were visually inspected.

Verification logs preserve the baseline, intermediate failures and repairs,
final focused/full suites, model comparison and artifact checks. Their commands
appear above and in the accompanying README. The branch is retained while its
pull request is awaiting review; merging was not authorized.

The compressed archive was extracted into a separate directory. Its archive
SHA-256 and all 992 member-file hashes passed; the restored study passed the
same input/output checks above. `archive_verification.json` records the result.
