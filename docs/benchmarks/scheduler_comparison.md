# Scheduler comparison for local version 0.1.4

On 2026-10-28, the previous default, simulated annealing (`s`), was compared
with the new default, OR-Tools CP-SAT (`o`). Sequential placement (`f`) was
also included as a reference for both group variants. Results use fixed,
small inputs and do not guarantee general runtime or solution quality.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/gems-mpl poetry run python scripts/compare_schedulers.py \
  --output-dir /tmp/gems-scheduler-recheck --seeds 0 1 2
MPLCONFIGDIR=/tmp/gems-mpl poetry run python -m unittest \
  tests.test_cp_sat_scheduler tests.test_experiment_simulators -q
```

The original output directory contained per-run CSV
(`scheduler_2026-10-28/runs.csv`), input and per-run JSON
(`scheduler_2026-10-28/runs.json`), timing penalties
(`scheduler_2026-10-28/penalties.png`), individual machine-ID overlap
(`scheduler_2026-10-28/overlap.png`), runtime
(`scheduler_2026-10-28/runtime.png`), and a Gantt chart for linear penalties
(`scheduler_2026-10-28/gantt.png`). These generated files are not tracked.
The reproduction command writes the same filenames under `--output-dir`.
In the figures, `Rest` means `CyclicalRestPenalty` and `Rest+Linear` means
`CyclicalRestPenaltyWithLinear`. Penalty values use a symlog vertical axis;
runtimes use a logarithmic vertical axis.

## Comparison conditions

- Python 3.14.4 and the repository's locked dependencies. Detailed versions
  and input values were recorded in JSON.
- Reference time was 0 minutes in all cases. Simple tasks had optimal starts
  at `10, 12, 15, 18` minutes and processing times of `6, 5, 4, 5` minutes.
  Single-machine groups had optimal starts at `10, 12, 14` minutes, each with
  `(preceding interval 2 minutes, processing 5 minutes) → (preceding interval 3 minutes, processing 4 minutes)`.
  Multi-machine groups had optimal starts at `10, 11, 12, 13` minutes, two
  type-0 machines (`m0`, `m1`), and each group had
  `(preceding interval 2 minutes, processing 5 minutes) → (preceding interval 2 minutes, processing 3 minutes)`.
- Simulated annealing retained the existing implementation's 100,000 steps,
  Tmax 25000 and Tmin 1, with Python random seeds `0, 1, 2`. CP-SAT used the
  existing five-second limit, one worker and seed 0. Each method received
  fresh, identical input for every run.
- **Timing penalty** was recomputed from scheduled start times by summing each
  type's `calculate_penalty()`. Group evaluation time was
  `first task start time − first preceding interval`. **Overlap time** summed
  excess concurrent task counts in minutes for each assigned machine ID when
  two or more tasks ran simultaneously. Missing assignments and incorrect
  within-group intervals were counted separately.
- Simulated annealing's internal score uses a soft overlap penalty multiplied
  by 100,000, while OR-Tools prohibits overlap. The common metrics above were
  therefore used instead of directly comparing internal objectives. The old
  multi-machine `eval_machine_penalty()` checks concurrent usage by machine
  type, rather than overlap on individual machine IDs.

## Definitions and verification of all penalty types

`t` is the evaluation time above, `t*` the optimal time, and `c` a coefficient.
This comparison used linear coefficient 3; range bounds `−2` / `2` with lower /
upper coefficients 4 / 2; and cyclical start 0, period 20 minutes and rest range
`[10, 14]` minutes. Both endpoints of the cyclical rest range are **inclusive**.

| Type | Evaluation by `calculate_penalty()` |
| --- | --- |
| `NonePenalty` | Always 0 |
| `LinearPenalty` | `c × \|t − t*\|` |
| `LinearWithRangePenalty` | If `t − t*` is below the lower bound: `lower coefficient × (lower bound − (t − t*))`; above the upper bound: `upper coefficient × ((t − t*) − upper bound)`; otherwise 0 |
| `CyclicalRestPenalty` | From the cyclical start onward, 1,000,000 if the evaluation-time phase is within the rest range; otherwise 0 |
| `CyclicalRestPenaltyWithLinear` | Before the cyclical start: 0; within the rest range: 1,000,000; otherwise `c × \|t − t*\|` |

Separately from these five types, the old annealing score adds a resource-overlap
penalty of `100,000 × overlap quantity in the old implementation`. This
comparison recomputed overlap using the actual assigned machine IDs.
With two machines of the same type, assigning tasks at 0–5 and 1–6 minutes to
the same `m0` still makes the old multi-machine `eval_machine_penalty()` return
0. Individual machine-ID overlap is four minutes; an automated test fixes this
case as a regression check.

Across all five types in three variants, 26 evaluation times each, covering
before the cyclical start and more than two cycles, gave 390 conditions where
existing `calculate_penalty()` and CP-SAT `_cost()` agreed. Another 15 cases
through the public API checked CP-SAT solution status, overlap, assignments
and within-group intervals. Cyclical rest follows the existing definition,
which evaluates **start times only**. It does not prohibit execution throughout
a processing interval that overlaps a rest period.

## Observed results

Lower timing penalties are better. `s` shows the minimum–maximum across three
seeds. Parentheses give overlap minutes on individual machine IDs. `f` is
unavailable for simple tasks. Values with overlap do not represent schedules
feasible on individual machine IDs.

| Variant | Type | OR-Tools `o` | Simulated annealing `s` | Sequential placement `f` |
| --- | --- | ---: | ---: | ---: |
| Simple task | None | 0 (0) | 0 (0) | — |
| Simple task | Linear | 27 (0) | 51 (0) | — |
| Simple task | Range | 10 (0) | 22 (0) | — |
| Simple task | Rest | 0 (0) | 1,000,000 (0) | — |
| Simple task | Rest+Linear | 30 (0) | 1,000,051 (0) | — |
| Single-machine group | None | 0 (0) | 0 (0) | 0 (0) |
| Single-machine group | Linear | 60 (0) | 75–90 (0) | 90 (0) |
| Single-machine group | Range | 44 (0) | 46–52 (0) | 52 (0) |
| Single-machine group | Rest | 0 (0) | 0 (0) | 0 (0) |
| Single-machine group | Rest+Linear | 69 (0) | 81 (0) | 81 (0) |
| Multi-machine group | None | 0 (0) | 0 (4) | 0 (4) |
| Multi-machine group | Linear | 42 (0) | 48 (4) | 48 (4) |
| Multi-machine group | Range | 16 (0) | 24 (4) | 24 (4) |
| Multi-machine group | Rest | 0 (0) | 0 (0–5) | 2,000,000 (4) |
| Multi-machine group | Rest+Linear | 45 (0) | 84–93 (3) | 2,000,048 (4) |

All 15 `o` runs had solver status `OPTIMAL`, zero machine-ID overlap minutes,
zero assignments to the wrong machine type, and zero within-group interval
violations. Machine IDs overlapped in 14 of 45 `s` runs and 5 of 10 `f` runs.
All overlap occurred in the multi-machine group variant. In one linear case,
the first tasks of groups 2 and 3, assigned to `m0` by `f` and `s`, ran at
22–27 and 23–28 minutes, overlapping for four minutes. This fits the total
capacity of two type-0 machines but cannot run concurrently on the individual
`m0`. The old machine-type evaluation function and machine-ID assignment during
sequential placement cause this difference. Annealing does not reassign machine
IDs after changing times from the initial placement.

Median wall-clock time per call was 0.00631 seconds for 15 `o` runs,
2.98115 seconds for 45 `s` runs, and 0.00013 seconds for 10 `f` runs.
These measurements depend on input size, the solver's five-second limit
and annealing's 100,000 steps.

## Limitations

Each type used one fixed, small input configuration. Three random seeds do
not estimate the full distribution of annealing results. Runtime is wall-clock
time on one development machine and does not establish general performance
for any method.
