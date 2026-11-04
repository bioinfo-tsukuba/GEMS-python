# Issue #23: Evaluation conclusions and archived records

Search limits of 1 or 60 seconds were set directly on CP-SAT. Construction,
application and validation time are excluded from the success criterion.
Success means returning a plan that satisfies constraints. FEASIBLE means
optimality is unproven; UNKNOWN means no solution was found.

## Observed scale

These measurements confirmed returned plans at a scale of **tens to hundreds
of experiments and hundreds to about 1,500 operations on 2–8 machines**.
Operation counts include current unfinished work, excluding future task groups.

- **Initial planning with a 1-second search**: the largest measured scales with
  every repetition successful ranged from 128 to 512 experiments by workload.
- **Continued updates with a 1-second search**: many assignment candidates and
  complex penalties reached UNKNOWN during updates at 128 experiments.
  Supplementary runs at 64 experiments completed eight updates for both workloads
  (one sequence each). 64 is not a reliable capacity limit or guarantee.
- **60-second search**: initial planning succeeded at 512 experiments for all
  five workloads. Continued updates completed eight updates at scales of
  128–512 experiments by workload, rather than 512 for every workload.

Returned plans for hundreds of experiments were observed, while confirmed
scales for continued replanning within a one-second search varied by workload.
512 experiments is the measurement ceiling, not a capacity limit or a guarantee
for physical equipment.

## Summary of evaluated problems

The common base configuration has one current task group per experiment,
with three operations lasting 5, 3 and 4 minutes. Fixed intervals are two
minutes from group start to the first operation, two minutes from the first
operation's end to the second, and one minute from the second operation's end
to the third. Planning selects group starts and machine assignments for each
operation. It minimizes start-time penalties, preferring earlier starts for
equal penalties. Counts below are unfinished operations passed to the search,
excluding future task groups.

| Workload | Problem summary | 1 second: largest measured scale with all repetitions successful (experiments/operations) | 60 seconds: same measure |
| --- | --- | ---: | ---: |
| Same-type machine contention | Assign all three operations to two same-type machines. Evaluate contention among many groups with nearby preferred starts competing for the same machines. | 256/768 | 512/1,536 |
| Heterogeneous operations | The first, second and third operations use different machine types with 2, 2 and 1 machines respectively. Evaluate per-type contention and coordination with fixed intervals. | 512/1,536 | 512/1,536 |
| Many assignment candidates | Choose among eight same-type machines for all three operations. Evaluate problems with many assignment choices. | 128/384 | 512/1,536 |
| Complex penalties | Use two same-type machines and mix three group start-time penalties: deviation from preferred time, a tolerance range, and cyclical rest ranges. Evaluate differing time-dependent costs. | 128/384 | 512/1,536 |
| Replanning with in-progress work | Use two same-type machines; one quarter of groups have started and completed their first operation. Fix remaining times and machine IDs in started groups, preserving future reservations across long waits while placing unstarted groups. | 256/704 | 512/1,408 |

Started groups in replanning have a longer interval before the second operation,
so they differ from the base intervals above. Fixed unfinished operations number
128 at the 1-second scale and 256 at the 60-second scale. Rest ranges in complex
penalties are costs on group start times, rather than constraints on machine
availability. Inputs are reproducible using
[`scheduler_capacity_config.json`](../../../scripts/scheduler_capacity_config.json)
and [`make_case` / `CapacityState`](../../../scripts/measure_scheduler_capacity.py).

Boundary conditions for each value succeeded 5/5 times. 512 is the measurement
ceiling, not the true capacity limit. Static success does not guarantee continued
updates. At 1 second and 128 experiments, many assignment candidates / complex
penalties reached UNKNOWN during updates. At 60 seconds, all five workloads
completed eight updates at the **1-second scales** in the table. Final source
reproduced both failures; supplementary 64-experiment, 1-second runs completed
eight updates for both workloads (one sequence each).

Of 39 conditions where both limits succeeded on identical input, the 60-second
median penalty improved in 23 and was equal in 16. Maximum delay improved in
20, was equal in 17 and increased in 2. These are descriptive statistics from
few repetitions with identical deterministic input and seed. They do not
guarantee success rates for arbitrary inputs, optimality, response time or
equipment throughput.

## Response time after result receipt (2026-10-30)

Additional timing covered passing a received result DataFrame through fixing
work, updating history, replanning and generating full schedule JSON. Each
response performs one search. Previous operational times (about 2/120 seconds)
also include search before receipt and are treated as a separate metric.
With three repetitions at the same five workloads and scales, median successful
response times were 1.016–1.035 seconds at the 1-second setting and
60.014–60.025 seconds at the 60-second setting. Complex penalties at
128 experiments / 1 second gave 3/3 UNKNOWN responses (median failed response
0.550 seconds); all other conditions succeeded 3/3 times.

These runs replay the first result partway through a task group. They exclude
responses at group completion with state transitions / next-group generation,
complex experiment analysis, hardware communication, rendering and disk saving.
They do not guarantee continued updates. Figures, full condition tables, CSV
and raw inputs/outputs are separated into the
[archived records](https://github.com/bioinfo-tsukuba/GEMS-python/blob/747902191e754bb79de902145b2ac6bd9cca456e/docs/benchmarks/result_update_latency_2026-10-30/README.md).
Archive branch `archive/23-result-update-latency` and pinned commit
`747902191e754bb79de902145b2ac6bd9cca456e` remain retained.
Cross-checks covered 30 inputs, 30 searches, 27 valid outputs and 3 UNKNOWN
responses. All 65 relevant tests passed, as did statistical regeneration and
archive hash checks for 165 files.

## Detailed records retained in history

Figures, statistical tables and raw data were separated from the implementation
PR's final diff and retained on `archive/23-scheduler-capacity` at pinned commit
`bf457334a0bddc3a75ff6560726a80e3c42dbf80`. This branch remains after merge
for continued reference. Figures are also visible in Issue #23 / PR #25 comments.

- [Statistics and four figures (PNG/SVG)](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/statistics/README.md)
- [Statistics CSV for all conditions](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/statistics/condition_statistics.csv)
- [Raw data and restoration instructions](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/README.md)
- [Statistics verification](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/statistics/verification.json)
- [Implementation verification, five existing errors and limitations](https://github.com/bioinfo-tsukuba/GEMS-python/blob/bf457334a0bddc3a75ff6560726a80e3c42dbf80/docs/benchmarks/scheduler_capacity/verification.md)

Primary evaluation used `9e269d4` (193 trials); final-source confirmation used
`b08dbe8` (17 conditions). Sources are aggregated separately, retaining input
hashes, every call, failures and environment information. Additional analysis
did not rerun searches; agreement with the original 105-condition aggregation
and reproduction in a separate output directory were verified. All 63 relevant
tests passed.

## Reproduction

Extract archived records into a separate directory and regenerate statistics
and figures with scripts from the current branch. The following only reads
saved files and aggregates results; it does not run searches.

```bash
mkdir -p /tmp/gems-capacity-records
git archive bf457334a0bddc3a75ff6560726a80e3c42dbf80 \
  docs/benchmarks/scheduler_capacity |
  tar -x -C /tmp/gems-capacity-records
MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg poetry run python \
  scripts/analyze_scheduler_capacity.py \
  /tmp/gems-capacity-records/docs/benchmarks/scheduler_capacity \
  --output /tmp/gems-capacity-statistics
```
