# Scheduler-only pattern study

This document reports results aggregated by `scripts/study_scheduler_patterns.py`.
The previous method is simulated annealing `s` through the public API; the new method is OR-Tools `o`.
Only scheduling was run; the simulator was not called.

## Reproduction

```bash
MPLCONFIGDIR=/tmp/gems-mpl poetry run python scripts/study_scheduler_patterns.py \
  --output-dir /tmp/gems-pattern-study-recheck --seeds 0 1 2
```

The script generates CSV/JSON, this summary and figures in a new output directory.
To reaggregate only, using saved per-run JSON:

```bash
MPLCONFIGDIR=/tmp/gems-mpl poetry run python scripts/study_scheduler_patterns.py \
  --from-runs /tmp/gems-pattern-study-recheck/runs.json \
  --output-dir /tmp/gems-pattern-study-reanalysis
```

[Score scatter plot for feasible pairs](paired_scores.png) and [feasibility and runtime figure](feasibility_runtime.png).

## Execution conditions

- Python 3.14.4, OR-Tools 9.15.6755, simanneal 0.5.0, Matplotlib 3.11.2.
- Execution environment: Linux-6.18.54.1-microsoft-standard-WSL2-x86_64-with-glibc2.43.
- 64 cases. Simulated annealing seeds: [0, 1, 2].
- CP-SAT: 5.0-second limit, one worker, seed 0. Simulated annealing: 100,000 steps, Tmax 25000, Tmin 1 (existing implementation).
- Inputs are rebuilt for each method and checked for identical SHA-256 hashes before scheduling.
- The simple-task variant has 4/8 tasks and one machine. Group variants have 3/6 groups, each containing two tasks. The multi-machine variant has 1/2 machines of the same type.
- Optimal start times begin at 12 minutes with fixed spacing. Clustered/spread spacing is 2/8 minutes for simple tasks, 3/14 minutes for single-machine groups, and 2/8 minutes for multi-machine groups. Processing-time and preceding-interval generation formulas are fixed in the script.
- Penalty patterns use Linear, Range or Rest+Linear alone, or Mixed, cycling through all five types. Coefficients and rest cycles are fixed in `make_penalty()` in `scripts/compare_schedulers.py`.
- These pattern counts, intervals and seeds are synthetic conditions selected for comparison; they are not estimated from operational experiment data.

## Metrics

- Scores sum the existing `calculate_penalty()` for each task/group. Lower is better. Score comparisons include only runs satisfying resource, assignment and preceding-interval constraints.
- `CP lower/equal/SA lower` counts feasible pairs for the same case and seed. The score gap is `SA − CP`; a positive value means CP is lower. The 90th percentile uses the nearest-rank value.
- CP `OPTIMAL` and `FEASIBLE` are distinguished. Runtime is wall-clock time for the scheduling call.

## Results

- CP: 64 proven optimal, 0 feasible within the time limit, 0 infeasible or failed.
- Simulated annealing: 168/192 runs feasible at the individual machine-ID level.
- Among 168 feasible pairs, CP lower 110, equal 58, SA lower 0.
- Per case, across 3 simulated annealing seeds, compared with the best feasible result, among 56 cases: CP lower 35, equal 21, SA lower 0. The remaining 8 cases have no feasible simulated annealing solution.
- Median score gap 17.00, 90th percentile 1,000,021.
- Median wall-clock time: CP 0.006637 seconds, simulated annealing 3.587091 seconds.

### By variant

| Condition | Cases | CP proven optimal | SA feasible/all runs | CP lower/equal/SA lower | Median score gap | 90th percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| multi_interval | 32 | 32 | 72/96 | 51/21/0 | 16.50 | 155 |
| simple | 16 | 16 | 48/48 | 36/12/0 | 500,067.50 | 2,000,078 |
| single_interval | 16 | 16 | 48/48 | 23/25/0 | 0.00 | 93 |

### By penalty pattern

| Condition | Cases | CP proven optimal | SA feasible/all runs | CP lower/equal/SA lower | Median score gap | 90th percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| linear | 16 | 16 | 42/48 | 24/18/0 | 6.00 | 57 |
| mixed | 16 | 16 | 42/48 | 29/13/0 | 27.00 | 1,999,982 |
| range | 16 | 16 | 42/48 | 24/18/0 | 4.00 | 50 |
| rest_linear | 16 | 16 | 42/48 | 33/9/0 | 73.50 | 1,999,982 |

### By clustering of preferred start times

| Condition | Cases | CP proven optimal | SA feasible/all runs | CP lower/equal/SA lower | Median score gap | 90th percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| clustered | 32 | 32 | 72/96 | 71/1/0 | 46.00 | 1,000,024 |
| spread | 32 | 32 | 96/96 | 39/57/0 | 0.00 | 999,991 |

### By task/group count

| Condition | Cases | CP proven optimal | SA feasible/all runs | CP lower/equal/SA lower | Median score gap | 90th percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 24 | 24 | 60/72 | 34/26/0 | 3.50 | 33 |
| 4 | 8 | 8 | 24/24 | 18/6/0 | 500,010.50 | 1,000,024 |
| 6 | 24 | 24 | 60/72 | 40/20/0 | 22.00 | 183 |
| 8 | 8 | 8 | 24/24 | 18/6/0 | 1,000,063.00 | 2,000,096 |

### By machine count

| Condition | Cases | CP proven optimal | SA feasible/all runs | CP lower/equal/SA lower | Median score gap | 90th percentile |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 16 | 16 | 48/48 | 47/1/0 | 24.00 | 217 |
| 2 | 16 | 16 | 24/48 | 4/20/0 | 0.00 | 3 |

## Observations by condition

- Clustered conditions: among 72 feasible pairs, CP is lower in 71; spread conditions: among 96 feasible pairs, CP is lower in 39.
- Multi-machine simulated annealing feasibility is 48/48 with one machine of the type, and 24/48 with two machines.
- The overall 90th-percentile score gap is affected by the 1,000,000-point cyclical rest penalty applied when a start time falls within the rest range. See the penalty-pattern table above for separate linear and range results.

## Interpretation and limitations

Low simulated annealing scores from schedules with overlapping individual machine IDs do not count as feasible wins. Where CP-SAT cannot prove optimality within the time limit, only the score of the obtained solution is compared.
Results use synthetic inputs and few random seeds in one environment; they do not guarantee general speed or solution quality, or predict actual cell culture.

When rerunning, save CSV/JSON in a new output directory and compare generated results with this summary and figures. Without adding raw data to Git, the generation script, input SHA-256 hashes, conditions and aggregation formulas remain traceable.
