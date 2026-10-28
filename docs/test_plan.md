# Test plan for local version 0.1.4

## Acceptance checks

Run with Python 3.14.4, project dependencies installed, and `MPLCONFIGDIR`
pointing to a writable directory:

```bash
poetry check --lock
poetry run python -m unittest tests.test_cp_sat_scheduler tests.test_experiment_simulators -q
poetry run python -m unittest discover -s tests -p 'test_*.py'
```

Inspect solver status and the resulting task start/end times, machine IDs,
state transitions, generated histories, and `git status` after each run. The
full legacy suite writes files into the checkout and contains interactive
tests; run it in a disposable checkout. Redirecting stdout does not prevent
these files from being written.

## Scheduler scenarios

| Scenario | Acceptance criterion |
| --- | --- |
| All three task variants | Omitted method selects OR-Tools; explicit `"s"` remains available, and interval variants retain `"f"`. |
| One machine | Tasks do not overlap; a small case agrees with exhaustive search on minimum penalty. |
| Interval group | Each task starts after its specified interval; the group penalty uses group start, including when the first interval is nonzero. |
| Multiple machines | Every task receives a compatible machine ID; tasks on each machine do not overlap; parallel machines can run tasks concurrently. |
| In-progress work | Existing non-completed task times and assigned machine IDs stay fixed. |
| Cyclical rest | Cost matches `calculate_penalty` before cycle start and at both inclusive boundaries, immediately after a boundary, and in the next cycle. |
| Failures | Missing compatible machines, invalid parameters, and no feasible solver result report explicit errors. |
| Time limit | Record `OPTIMAL` or `FEASIBLE`, penalty, objective, and solve time; never label an unproven feasible solution optimal. |
| Concurrent simulation | A shorter task that starts later reports its observation first; the longer task keeps its original start and completion times. |

## Old/new scheduler comparison

The reproducible comparison, all five timing-penalty formulas, three fixed
workloads, three annealing seeds, physical-machine overlap metric, CSV/JSON
records, and PNG plots are documented in
[scheduler comparison](benchmarks/scheduler_comparison.md). The focused tests
check the 390 formula/CP-SAT pairs and all 15 OR-Tools public-API cases.
The comparison calls the existing `"s"` and `"f"` implementations without
changing their annealing parameters.

## Synthetic experiment scenarios

**Cell culture:** start at confluence 0.12, use logistic growth with rate
0.035 per hour and carrying capacity 1, observe every 360 minutes, and passage
once confluence reaches 0.30. Check monotonic growth between passages, the
reset after passage, state transitions, and advancing virtual time. These
numbers follow the repository's earlier demonstration fixture and are not
calibrated to measured HEK cells. Logistic equations are used in published
mammalian-cell growth models [1].

**Reaction-condition search:** use a 27-point grid of temperatures
`40/60/80`, concentrations `0.25/0.50/0.75`, and durations `30/60/90`. A
bounded synthetic quadratic response has known maximum yield 1 at
`(60, 0.50, 60)`. Compare deterministic grid and seeded-random candidate
orders at the same trial budget using best-so-far yield. Verify repeated runs
produce identical histories; do not assert one baseline always beats the
other. This tests the comparison workflow used in reaction-optimization
research [2], without claiming the synthetic response models a real reaction.

## References

1. *Computer programs for modeling mammalian cell batch and fed-batch cultures
   using logistic equations* (2012).
   https://pmc.ncbi.nlm.nih.gov/articles/PMC3397109/
2. *A Multi-Objective Active Learning Platform and Web App for Reaction
   Optimization*, Journal of the American Chemical Society (2022),
   DOI: 10.1021/jacs.2c08592.
   https://pubs.acs.org/doi/10.1021/jacs.2c08592

## Verification on 2026-10-27

- Python 3.14.4, Poetry 2.5.1, OR-Tools 9.15.6755: locked dependency install
  and all 12 focused tests passed. Both `tests/test_simulate.py` and
  `tests/test_simulate_one.py` completed using the new default.
- The final full suite reported 56 tests with 10 errors. An untouched archive of
  the starting commit, run with the same Python environment, reported 44
  tests with the same 10 errors. They come from missing HEK fixture arguments
  (3), UUID JSON serialization (1), an interactive input call (1), an old
  Matplotlib API (4), and an old tuple-unpacking assertion (1). These remain
  outside this change.
- A concurrent simulation test initially showed a running task being moved
  after another task finished. The simulator now fixes started tasks before
  rescheduling; the reproduction passes with completion times 2 and 20.

## Verification on 2026-10-28

- The 70-run scheduler comparison finished with 15 `OPTIMAL` CP-SAT results.
  All 390 formula checks passed. All 15 CP-SAT comparison cases had zero
  physical-machine overlap, invalid assignments, and group-interval errors.
- The legacy annealer produced physical-machine ID overlaps in 14 of 45 runs;
  the sequential method did so in 5 of 10 runs. The exact schedules, seeds,
  and diagrams are preserved in the comparison report. This is a finding in
  the existing optional methods, not a new CP-SAT failure.
- The 16 focused scheduler and simulator tests passed. The final
  disposable-checkout full suite reported 60 tests with the same 10 errors
  as on 2026-10-27:
  missing HEK arguments (3), UUID JSON (1), interactive input (1),
  old Matplotlib API (4), and tuple-unpacking (1). `poetry check --lock`
  completed with only pre-existing metadata deprecation warnings.
