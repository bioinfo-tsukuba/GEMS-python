# ROUND3 cell-culture capacity study: proposed plan

Planning date: 2026-11-20. Tracking issue: [#30](https://github.com/bioinfo-tsukuba/GEMS-python/issues/30). Related work: #21, #23 and #24.

This document plans a study of concurrent HEK and iPS cultures using the ROUND3 protocol and measured growth observations. The current request covers inspection, planning, an Issue and questions for the user. Model calibration, implementation, simulation and capacity measurement require a subsequent instruction after the consequential conditions are resolved.

## Requirements and evidence

The user requested multiple cultures of both cell types registered from the beginning, no injected failures, and growth simulation based on the actual ROUND3 data. An equal HEK/iPS ratio, identical initial growth phases, random individual variation, measurement noise, equipment limits and a success threshold have **not** been specified.

The previous [scheduler capacity study](benchmarks/scheduler_capacity/README.md) evaluated synthetic scheduling workloads. Its initial-plan and limited-update results do not establish sustainable cell-culture counts. Its search limits, maximum experiment count, repetition count and runtime budget are specific to that evaluation and are not inherited by this proposal.

Source references inspected for planning:

| Resource | Fixed reference and relevant paths |
| --- | --- |
| Actual experiment repository | `CAB314/GEMS-python-ROUND3`, branch `HEK5_iPS5_ROUND3`, commit `26bd4552e08a3a3b8c1c654107d8b3198cb988f8` |
| Candidate experimental run | `iPSsimulation/results/2024-11-13_real_round3/` |
| Protocol and prediction | That run's `experimental_setting/`, including normal and restart definitions, refill definitions and `curve.py` |
| Observations and progress | Per-step `experiments/*_shared_variable_history.csv`, `experiments.json`, and operation records; inspect all relevant steps during extraction |
| Culture-to-container mapping | `iPSsimulation/samples/experiments/hek293a.py` and `ips_253g1.py`, then verify against the actual run |
| GEMS implementation | Planning base `dd5ae4011f4b7475b9506f7f1c38f06b00c885df` |

The source contains normal and restart definitions, cumulative histories, configured processing times, observation-driven predictions and refill reservations. These provide inputs to the study, but do not establish which settings were active throughout the run, why an individual trajectory changed, or whether configured times represent measured resource occupancy. Those questions need an audit before fitting.

The experiment repository is private; the GEMS repository and Issue are public. This planning change includes source references and proposed methods, without copying raw measurements, images or source credentials. Decide where extracted records, fitted parameters and results may be stored and published before producing those artifacts. Reproduction of private inputs requires access to the source repository.

## Decisions to resolve

The alternatives below are proposals, not approved defaults. Questions about the primary design, endpoint, resources, usable observations and the meaning of sampling were sent to the user before implementation.

| Decision | Proposed starting point | Consequence or unresolved detail |
| --- | --- | --- |
| Culture count and unit | HEK N + iPS N; count each independently operated culture | Confirm whether one culture corresponds to an Experiment, a vessel, a plate or a well. Always report both per-type and total counts. |
| Enrollment and initial state | Register every culture at virtual time zero; compare aligned phases with empirical phase variation | Initial registration does not imply simultaneous seeding. Specify prior passage age, initial density and whether startup operations compete for the resource. Do not give cultures free pre-run processing unless that is the agreed initial condition. |
| Endpoint | Complete the ROUND3-equivalent passage-and-sampling sequence | Alternatively evaluate maintenance over a specified duration. Agree on the passage count, whether initial passage counts, the virtual horizon and how incomplete cultures are classified. Finite completion does not establish indefinite maintenance capacity. |
| Biological acceptance | Operation-specific timing and density criteria | Specify allowed early/late passage and sampling, density ranges, maximum medium-change gaps, observation delays, and whether every culture must pass. Evaluate the relevant operation start or end, rather than only task-group start. |
| Equipment and durations | ROUND3's single shared operation resource and configured processing times | Measured-duration calibration or future equipment configurations are separate alternatives. An operation history timestamp alone does not give actual start-to-end occupancy. |
| Availability and other resources | Inventory refill reservations, work hours and culture storage limits | Decide whether human work, consumables, storage and refill demand constrain the main study. A limit measured without these constraints applies only to operation scheduling. Increasing culture counts must not silently assume unlimited refill capacity. |
| Growth-data inclusion | Use confirmed normal growth intervals; review interruptions, restarts and anomalous intervals | Retain usable normal intervals before a failure where appropriate. Report exclusions and sensitivity to inclusion choices. Survivor-only fitting can bias capacity estimates. |
| Growth variation | Validate a representative model, then sample empirical individual parameters | Confirm whether “sample from actual data” means individual growth-rate/reseeding variation, a fixed representative curve, or additional observation noise. Biological variation and fitted-parameter uncertainty need separate interpretation. |
| Online prediction | Reproduce the normal ROUND3 observation-based prediction policy | Keep latent simulator parameters and future observations unavailable to the planner. Document any corrected prediction behavior separately from a faithful baseline. |
| Solver and execution budget | Use the current default OR-Tools scheduler | Explicitly choose search limit, retry policy, seeds, repetitions, count ceiling and runtime budget before capacity execution. Existing implementation defaults are five seconds, one worker, solver seed zero and bounded UNKNOWN retries; these are not automatically study settings. |
| Model quality | Validate density forecasts and target-density arrival forecasts | Set acceptable predictive error before interpreting a count as actionable, or label it exploratory. The user supplies biological criteria; model-form selection can follow documented validation. |
| Output access | Keep private source observations in an access-controlled location | Decide access and publication of calibrated parameters and derived results. Public reproduction instructions can identify private prerequisites without embedding them. |

If biological acceptance criteria cannot yet be supplied, report curves of delay, density, completion and utilization. Do not replace missing criteria with a chosen “maximum feasible culture count.”

## Stage 1: audit inputs and reconstruct the normal protocol

Pin the source commit, inventory the relevant input files and record hashes. Reconstruct culture identities and setting changes from histories, snapshots and operation records. Final histories alone are insufficient to confirm whether earlier records were revised or an ID was reused at restart.

Extract observations without concatenating cumulative snapshots as independent measurements. Preserve source step, row and revision provenance. Separate exact repetitions, additions and corrections; do not collapse distinct observations merely because their timestamps or values match.

Confirm the meaning, scale and units of density, and whether each timestamp denotes acquisition, task start, task end or result receipt. Convert to a documented relative virtual clock, retaining original timestamps and Japan time for work-hour interpretation. Use elapsed time since the actual passage boundary for growth fitting. Treat missing passage densities as missing, not as observed zero density.

Split trajectories at passages and identify initialization, reseeding, interruptions and restarts. Produce an inclusion/exclusion manifest with reason codes and unresolved cases for review. A falling density is evidence to inspect, not by itself proof of a failure. Confirm whether medium changes, recovery after passage or different passage generations need distinct effects that the observations can identify.

Inventory operation durations, fixed intervals, target densities, penalties, prediction rules, refill reservations, sampling and termination. Distinguish configured durations from actual resource occupancy. Cyclical rest penalties in GEMS act on group starts; they do not prohibit work from crossing a rest interval. Turning those preferences into hard availability constraints would change the problem and requires an explicit decision.

Output: a private data manifest, a normalized protocol specification and a report of units, identity mapping, active settings and unresolved discrepancies. Do not begin calibration with unresolved consequential mappings.

## Stage 2: calibrate and validate growth

Use a cell-type-specific logistic model as the first candidate, consistent with the existing prediction approach, rather than an assumed scientific conclusion. Estimate growth rate and post-passage initial density. Compare with a simpler exponential model over the observed range or an identifiable recovery-phase model when validation supports that comparison. Do not add parameters that the available observations cannot distinguish.

Fixing carrying capacity at a unit-normalized density or fixing a recovery period requires evidence about measurement scale and a sensitivity analysis. Estimate culture-level and passage-level variation only to the extent supported by the data. Sparse repeated measurements from the same culture are not independent biological replicates.

Evaluate withheld cultures where practical, and predict future observations from earlier observations within a passage. Report density error, target-density crossing-time error, residual structure, coverage and failed fits. Keep final evaluation observations separate from fitting and model selection. If a target was not bracketed by observations, identify its crossing time as uncertain rather than treating an interpolation or fitted curve as ground truth.

First establish a fixed representative case that makes protocol debugging reproducible. If empirical sampling is selected, sample correlated parameter sets or culture trajectories together, retain within-culture dependencies across passages where supported, and record the empirical pool and random seeds. Use culture-level resampling to study fit uncertainty; report it separately from individual biological variation. A small experiment does not justify an arbitrary precise population distribution.

Represent growth as a function of actual elapsed virtual time. Delayed observation or passage must let the culture continue growing. Passage resets density according to the fitted reseeding rule; coating and waiting do not freeze cell growth. Record the simulated density at biologically relevant operation starts and ends, including hidden threshold crossings between observations.

The observation provider may access latent growth parameters. The prediction and scheduling policy may access only past observations and agreed prior information. Calibration can learn priors from historical ROUND3 data, but a capacity run cannot expose its sampled truth or future trajectory to the planner.

Output: candidate fits, validation plots, crossing-time errors, parameter provenance, sensitivity results and a justified model choice. Report limitations before claiming that the model can support a practical capacity estimate.

## Stage 3: verify small complete simulations

The proposed integration uses the current multi-machine interval manager with one shared machine for the main single-resource case, and `observation_provider` for virtual measurements. This reuses the supported simulation/update path and resource validation without introducing physical robot or API dependencies.

Start with 1 HEK + 1 iPS, then the 5 + 5 reference scale. Reconstruct the agreed normal sequence: passage, imaging, medium change, coating and sampling as appropriate. Separate occupied processing time from nonoccupying culture/coating intervals. Register all cultures initially and disable later enrollment and sample splitting for this study.

Replace source wall-clock-dependent scheduling with the agreed virtual clock. Preserve the intended normal protocol and document any differences between source and current APIs. Use an explicit finish after the agreed endpoint; do not count post-sampling placeholder waits or an endless expired state as continued culture.

Verify actual outputs: observations in completion order, correct state transitions, growth during delays, density reset at passage, fixed started work, resource-compatible assignments and no overlap on the shared resource. Check the accepted operation timing and density criteria against an independent event/schedule inspection.

No injected failures means no simulated contamination, equipment breakdown or recovery/restart event. It does not suppress overload indicators, excessive density or missed timing criteria. If those occur because of congestion, record a capacity-condition violation without silently adding a recovery workflow.

Output: complete event histories and schedule snapshots for both reference scales, independent verification and documented discrepancies. Agreement in intended protocol is required; a new scheduler is not expected to reproduce the historical schedule exactly.

## Stage 4: measure count-dependent behavior

Freeze the accepted protocol, growth model, criteria, solver settings and execution budget. Sweep counts coarsely, then refine around observed transitions. Use fresh output directories and preserve unsuccessful conditions and inputs. Choose repetitions after a small cost/variance pilot and before the main comparison.

Run observation, prediction and replanning through the agreed endpoint for every tested condition. Do not substitute successful initial planning or a fixed small number of updates for completion. Use common empirical draws where possible when comparing configurations, and distinguish solver timing variability from differences in simulated cultures.

Report per-type and total initial counts, active counts over time, completed cultures, acceptance rate, operation-specific delay and medium-change gaps, passage/sampling densities, horizon, queue growth, utilization and planning status. Include the first violated criterion and affected culture/operation. For long maintenance runs, increasing queue delay is evidence of overload under the tested policy.

Classify outcomes separately: accepted completed runs, biological/timing violations, solver UNKNOWN, proven solver infeasibility, invalid input/output, software/resource errors, and execution-budget censoring. UNKNOWN is not proof of infeasibility. FEASIBLE is not proof of optimality. A violated criterion under this scheduling policy does not prove that every possible scheduling policy would fail.

Report the largest accepted tested scale with its repetition count and variation, nearby failed scales and the untested region. Do not assume success is monotonic in count or call a count ceiling the true limit. An analytic workload/utilization calculation can check plausibility and identify bottlenecks, but average load alone misses coincident passage demand.

Output: count-versus-acceptance, delay, density and utilization plots, timelines near the observed boundary, failure classifications, and conclusions that separate solver limitations, resource contention and growth/timing constraints.

## Implementation candidates and verification records

Use existing dependencies (NumPy/SciPy, Pandas/Polars, OR-Tools and Matplotlib) where sufficient. Avoid new services or dependencies before a demonstrated need. Relevant existing files:

- `gems_python/common/cp_sat_scheduler.py`
- `gems_python/multi_machine_problem_interval_task/transition_manager.py` and `task_info.py`
- `scripts/measure_scheduler_capacity.py` and `scripts/verify_scheduler_capacity.py`
- `tests/simulators/cell_culture.py` and `tests/test_experiment_simulators.py`

Proposed additions, to be finalized during implementation:

- `scripts/extract_round3_culture_data.py`: pinned source ingestion, revisions and provenance.
- `scripts/calibrate_round3_growth.py`: candidate fits and withheld-data evaluation.
- `scripts/measure_culture_capacity.py`: complete virtual runs and outcome records.
- `scripts/verify_culture_capacity.py`: independent input/event/schedule checks.
- An explicit study configuration, focused tests and a report under `docs/benchmarks/culture_capacity/`.

The existing cell-culture fixture is synthetic and explicitly not calibrated to measured HEK observations. Its numerical constants must not become empirical study parameters.

Save source and implementation commits, hashes, configuration provenance, package versions, commands, virtual time units, seeds, solver options, hardware information, raw run outcomes and verification results. Keep data validation, model validation and scheduling validation distinct. Document useful failures, fixes and whether each fix was verified.

## Current verification and completion status

Planning inspection used authenticated `gh` reads of the pinned private repository, local source inspection and the existing #21/#23 records. Normal and restart definitions, the prediction implementation, final histories, representative snapshots and the saved operation table were inspected. The full per-step audit, actual-duration reconstruction, model fitting, model validation, physical storage constraints and capacity runs have not been performed.

The restricted environment initially prevented GitHub access and Git branch creation. Authorized network reads and branch creation succeeded through the escalation mechanism. Shell inspection uses `python3` because the unqualified `python` executable is unavailable outside the managed environment.

The planning deliverable is complete when this document and Issue #30 agree, consequential decisions have been asked and their current status recorded, a dedicated branch/commit and planning PR exist, and documentation links and the scoped diff have been checked. A planning PR does not close the future evaluation Issue or approve unresolved study settings.

The future evaluation is complete only after the accepted settings are frozen, input and model audits pass to the documented extent, reference simulations pass, count/boundary runs are verified, and reproducible results and remaining limitations are recorded. Its conclusions will be estimates under the specified equipment, protocol, observations and model, not guarantees for actual cultures.
