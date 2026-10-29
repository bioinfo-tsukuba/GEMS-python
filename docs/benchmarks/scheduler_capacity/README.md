# Issue #23: fixed-search-limit capacity evaluation

The user changed the original Issue's time criterion: **1 / 60 seconds is the
fixed CP-SAT search limit**, applied directly to `max_time_in_seconds`. Model
construction, application, checks and update overhead do not consume that limit
or determine success. Early optimality ends a search without waiting.
Each search in an update has its own limit; an update can contain multiple searches.

Success means `OPTIMAL` or `FEASIBLE` with a valid physical schedule. `UNKNOWN`
means no solution was found by that run, not proven infeasibility. The study
does not impose delay or quality thresholds and does not establish a total
one-second / sixty-second response guarantee.

## Configuration and reproduction

```bash
MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg poetry run python \
  scripts/measure_scheduler_capacity.py \
  --config scripts/scheduler_capacity_config.json \
  --output /tmp/gems-capacity-study

# Rebuild tables from saved records, without solving again:
poetry run python scripts/measure_scheduler_capacity.py \
  --summarize /tmp/gems-capacity-study

# Replay a saved trial (preserve the original directory first):
poetry run python scripts/measure_scheduler_capacity.py \
  --worker /tmp/gems-capacity-study/trials/<trial>/request.json
```

The main study refuses to reuse an existing output directory. A worker replay
writes `result.json`, `output.json` and any error in its request's directory.
Use a copy to preserve the original measured output.

The tracked configuration fixes worker 1, seed 0, three tasks per group
(durations 5/3/4 minutes, preceding intervals 2/2/1 minutes), three group cycles,
eight operational updates and one new experiment at updates 2 and 4.
No experimental noise, sample splitting, real equipment or calibrated scientific
response model is used. CPU/OS interference can still vary search outcomes.

| Pattern | Machines / timing |
| --- | --- |
| contention | Two machines of one type; preferred starts 10–13 minutes. |
| heterogeneous | Three successive types with 2/2/1 machines. |
| many_candidates | Eight same-type machines, increasing assignment choices. |
| mixed_penalties | Two machines; Linear, Range and Rest+Linear rotated by experiment index; rest cycle 120 minutes, inclusive ranges 20–30 and 90–105. |
| replanning | Two machines; every fourth group has completed first work and fixed remaining work. One task is already running at the reference time; future fixed tasks wait after a deterministic long interval. |

Sizes are 2/4/8/16/32/64/128/256/512 experiments. Each limit's scan stops after
the next size above its first failure; this is not an assumption of monotonicity.
The first-failure neighborhood, or the last two sizes if all passed, is repeated
to five runs per condition. Operational trials use the smaller of the two
limits' largest all-success measured sizes. The controller's total execution
allowance is six hours, independently of individual search limits. The worker
timeout is a separate emergency safeguard and is reported as `WORKER_TIMEOUT`.

## Records and interpretation

Each trial retains the exact typed input snapshot and SHA-256, result snapshot,
structured per-search records, exceptions and fresh-worker peak resident memory.
Only the output path is normalized for input hashing. Paired static conditions
have the same input hash; closed-loop inputs can diverge because schedules change
completion order. Updates retain their own inputs and every scheduling call,
including experiment additions.

CSV/JSON summarize median, linearly interpolated p95 and maximum search/total
times, successes, statuses and memory. Raw records also retain weighted solver
objective and bound in identical units, penalty, delay distribution, pending
schedule utilization, group/task/machine/type/assignment/fixed/history counts.
Five repetitions do not constitute a precise operational p95 guarantee.
Memory includes imports; operational memory is the peak up to that update.
Saving/drawing is disabled in operational timing. Artifact serialization occurs
outside per-search timing. Separate two-experiment `io` trials record snapshot
saving and Gantt drawing times as reference costs.

Delay and schedule span use virtual minutes. Computation uses real seconds.
Low delay or high utilization does not demonstrate scientific validity or a
facility's sustainable throughput. Success at 512 means the tested upper limit
was reached, not that the maximum supported scale is 512.

## Verification

```bash
poetry run python -m unittest tests.test_scheduler_capacity \
  tests.test_capacity_measurement tests.test_cp_sat_scheduler \
  tests.test_experiment_simulators tests.test_persistence_roundtrip \
  tests.test_dataframe_backends tests.test_scheduler_pattern_study -q
```

The regressions cover the original f/s overlap, small counts and zero-duration
tasks, fixed future reservations, simultaneous events, missing machines, rejected
annealing fallback, all three limit interfaces and old saved data, UNKNOWN
classification, finite updates, absence of internal f calls in the o path, and
paired workload fingerprints. Results and observed failures are recorded in the
verification report accompanying the measured artifacts.
