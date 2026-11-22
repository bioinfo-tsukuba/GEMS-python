# ROUND3 cell-culture capacity study: proposed plan

Planning date: 2026-11-20. Tracking issue: [#30](https://github.com/bioinfo-tsukuba/GEMS-python/issues/30). Related work: #21, #23 and #24.

This document records the plan for concurrent HEK and iPS cultures using the ROUND3 protocol and measured growth observations. The initial request covered inspection, planning, an Issue and questions. The user's subsequent instruction authorized full implementation, calibration, evaluation and a report with appropriate visualizations on a dedicated branch. Current execution choices and source-audit findings are recorded in the [methodology](benchmarks/culture_capacity/methodology.md).

## Requirements and evidence

The user requested multiple cultures of both cell types registered from the beginning, no injected failures, and growth simulation informed by actual ROUND3 data. The follow-up decisions specify the full grid of HEK counts 3 through 10 and iPS counts 3 through 10, growth-curve variation, completion through sampling, ROUND3 settings including refill work, and 30 repetitions from the outset. This is 64 count combinations and 1,920 main evaluation runs, with 6 to 20 cultures per run.

For density acceptance, the user proposed target density ±0.1 as a provisional rule and asked whether equal absolute widths at targets 0.8 and 0.3 were comparable. After the recommendation to retain that primary rule and report relative errors as well, the user accepted the recommendation. The primary rule remains target ±0.1; the proposed alternative, relative ±20% capped at an absolute width of 0.1, is a supplementary sensitivity calculation from the same recorded outcomes. Neither rule has been biologically validated for these experimental conditions.

The user clarified that abnormal/interrupted courses should be omitted from the simulation and that growth modeling should use values before those events. Data extraction must retain normal observations preceding an interruption, rather than exclude every observation from an affected culture. The event boundary remains an audit question; do not infer it from the eventual outcome alone.

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

The experiment repository is private; the GEMS repository and Issue are public. Original observations, images and detailed extraction ledgers stay in ignored private directories. Derived fitted parameters, aggregate validation and simulated outcomes are prepared locally for reproducibility. Automatic approval review rejected publishing the derived model to the public remote without explicit approval of that payload and destination; code-only publication can proceed separately. Reproduction of raw input extraction requires access to the source repository.

## Recorded decisions and remaining details

The table distinguishes user decisions from implementation proposals and details to be audited. The initial decision responses were followed by explicit authorization to implement and execute the full study; see the methodology for choices fixed during preparation.

| Decision | Recorded choice or implementation proposal | Consequence or unresolved detail |
| --- | --- | --- |
| Culture count and unit | User: every pair `(HEK, iPS)` in `{3, ..., 10} × {3, ..., 10}` | Confirm the source mapping from independently operated culture to Experiment, vessel, plate or well. Report both per-type and total counts. Equal-count-only sweeps are superseded. |
| Enrollment and initial state | User: all cultures initially enrolled, with varying growth curves; implementation proposal: retain ROUND3 protocol entry states | “Growth phase” meant time since passage at enrollment, not the shape of the growth curve. A separately shifted-phase sweep is not requested. Audit initial history/time mapping and include resource-competing startup operations required by the source. |
| Endpoint | User: through sampling, using the ROUND3 sequence | Audit the source passage count and treatment of initial passage; end each culture after sampling. Keep incomplete runs visible. No long-term maintenance endpoint is selected. |
| Biological acceptance | User-accepted provisional recommendation: target ±0.1 as primary; report absolute and relative errors and supplementary relative ±20% with absolute width capped at 0.1 | Proposed run criterion: every culture reaches sampling and all targeted passage/sampling operations meet the density band. Evaluate density before the operation/reset, not a post-passage value. Additional timing thresholds are not selected; retain timing/gap diagnostics without inventing pass/fail limits. |
| Equipment and durations | User: ROUND3 settings, including refill processing time | Keep the source single shared operation resource, configured durations and intervals. Measured-duration substitution and added equipment are outside this selected main design. |
| Availability and other resources | User: include ROUND3 refill reservations; implementation proposal: reproduce source frequency, durations and work-hour preferences | Audit the active refill definition and source availability rules. Do not silently add count-dependent refill scaling or new inventory constraints. Report storage/consumable capacity as unverified where the source does not establish it. |
| Growth-data inclusion | User: ignore abnormal/interrupted courses and use earlier normal values | Audit cutoff boundaries, preserve preceding normal intervals and omit post-event/restart trajectories from the main fitting pool. Report exclusions and uncertainty in the boundary. |
| Growth variation | User: varying growth curves informed by real observations, with repeated runs | Proposed method: assign jointly drawn growth-rate/reseeding parameters to each simulated culture, preserving supported dependencies. “Parameter sampling” means selecting simulation parameters; it does not mean laboratory sampling. Additional observation noise is not requested and is not automatically introduced. |
| Online prediction | Reproduce the normal ROUND3 observation-based prediction policy | Keep latent simulator parameters and future observations unavailable to the planner. Document any corrected prediction behavior separately from a faithful baseline. |
| Repetitions | User: 30 per count combination from the outset | 64 × 30 = 1,920 main runs. Repetitions vary growth parameters/seeds; rerunning the same growth curves only measures computational variation. Pilots and debug runs are additional and separately labeled. |
| Solver and execution budget | Proposal: use the current default OR-Tools scheduler; execution budget still to be specified | Record search limit, retry policy, seeds and runtime budget before main execution. Existing implementation defaults are five seconds, one worker, solver seed zero and bounded UNKNOWN retries; these are not automatically study settings. |
| Model quality | Validate density forecasts and target-density arrival forecasts | Set acceptable predictive error before interpreting a count as actionable, or label it exploratory. The user supplies biological criteria; model-form selection can follow documented validation. |
| Output access | Keep private source observations in an access-controlled location | Decide access and publication of calibrated parameters and derived results. Public reproduction instructions can identify private prerequisites without embedding them. |

The accepted primary density interval is inclusive. For a target `d*`, accept the recorded pre-operation density `d` when `abs(d - d*) <= 0.1`. Record the target, actual latent density, observed/predicted density where available, signed error, absolute error and relative error `abs(d - d*) / d*` separately. Use latent simulator density to score a run; it remains unavailable to the prediction policy.

| Target | Primary absolute band | Relative size of primary half-width | Supplementary half-width `min(0.1, 0.20 × target)` | Supplementary band |
| --- | --- | --- | --- | --- |
| 0.8 | 0.7–0.9 | 12.5% | 0.1 | 0.7–0.9 |
| 0.3 | 0.2–0.4 | 33.3% | 0.06 | 0.24–0.36 |

These are arithmetic comparisons, not evidence that equal relative error is biologically equivalent across cell types. The supplementary choice of 20% is an exploratory sensitivity value recommended by the agent and accepted as part of the proposed comparison. If the two rules give different acceptance regions, show both rather than select a favorable threshold after seeing results.

Plot the outcome grid with HEK count and iPS count as its two axes, showing accepted runs out of 30, plus per-culture outcomes, density errors and scheduling failures. Describe an accepted region of count pairs, rather than force mixed-cell-type capacity into one maximum total count. A 30-run proportion is an empirical estimate under the fitted model, not a guaranteed probability for actual cultures.

## Stage 1: audit inputs and reconstruct the normal protocol

Pin the source commit, inventory the relevant input files and record hashes. Reconstruct culture identities and setting changes from histories, snapshots and operation records. Final histories alone are insufficient to confirm whether earlier records were revised or an ID was reused at restart.

Extract observations without concatenating cumulative snapshots as independent measurements. Preserve source step, row and revision provenance. Separate exact repetitions, additions and corrections; do not collapse distinct observations merely because their timestamps or values match.

Confirm the meaning, scale and units of density, and whether each timestamp denotes acquisition, task start, task end or result receipt. Convert to a documented relative virtual clock, retaining original timestamps and Japan time for work-hour interpretation. Use elapsed time since the actual passage boundary for growth fitting. Treat missing passage densities as missing, not as observed zero density.

Split trajectories at passages and identify initialization, reseeding, interruptions and restarts. In accordance with the follow-up decision, retain normal pre-event observations for fitting and omit abnormal/interrupted and restart courses from the main growth pool. Produce an inclusion/exclusion manifest with reason codes and unresolved cutoff boundaries. A falling density is evidence to inspect, not by itself proof of a failure. Confirm whether medium changes, recovery after passage or different passage generations need distinct effects that the observations can identify.

Inventory operation durations, fixed intervals, target densities, penalties, prediction rules, refill reservations, sampling and termination. Distinguish configured durations from actual resource occupancy. Cyclical rest penalties in GEMS act on group starts; they do not prohibit work from crossing a rest interval. Turning those preferences into hard availability constraints would change the problem and requires an explicit decision.

Output: a private data manifest, a normalized protocol specification and a report of units, identity mapping, active settings and unresolved discrepancies. Do not begin calibration with unresolved consequential mappings.

## Stage 2: calibrate and validate growth

Use a cell-type-specific logistic model as the first candidate, consistent with the existing prediction approach, rather than an assumed scientific conclusion. Estimate growth rate and post-passage initial density. Compare with a simpler exponential model over the observed range or an identifiable recovery-phase model when validation supports that comparison. Do not add parameters that the available observations cannot distinguish.

Fixing carrying capacity at a unit-normalized density or fixing a recovery period requires evidence about measurement scale and a sensitivity analysis. Estimate culture-level and passage-level variation only to the extent supported by the data. Sparse repeated measurements from the same culture are not independent biological replicates.

Evaluate withheld cultures where practical, and predict future observations from earlier observations within a passage. Report density error, target-density crossing-time error, residual structure, coverage and failed fits. Keep final evaluation observations separate from fitting and model selection. If a target was not bracketed by observations, identify its crossing time as uncertain rather than treating an interpolation or fitted curve as ground truth.

First establish a fixed representative case for protocol debugging; this is not a replacement for the selected varied-growth main evaluation. Then draw correlated parameter sets or culture trajectories together, retain within-culture dependencies across passages where supported, and record the empirical pool and random seeds. Infer variation from usable pre-event observations; do not invent a Gaussian spread or a fixed percentage jitter without reporting its basis. Use culture-level resampling to study fit uncertainty and report it separately from individual biological variation. A small experiment does not justify an arbitrary precise population distribution.

Represent growth as a function of actual elapsed virtual time. Delayed observation or passage must let the culture continue growing. Passage resets density according to the fitted reseeding rule; coating and waiting do not freeze cell growth. Record the simulated density at biologically relevant operation starts and ends, including hidden threshold crossings between observations.

The observation provider may access latent growth parameters. The prediction and scheduling policy may access only past observations and agreed prior information. Calibration can learn priors from historical ROUND3 data, but a capacity run cannot expose its sampled truth or future trajectory to the planner.

Output: candidate fits, validation plots, crossing-time errors, parameter provenance, sensitivity results and a justified model choice. Report limitations before claiming that the model can support a practical capacity estimate.

## Stage 3: verify small complete simulations

The proposed integration uses the current multi-machine interval manager with one shared machine for the main single-resource case, and `observation_provider` for virtual measurements. This reuses the supported simulation/update path and resource validation without introducing physical robot or API dependencies.

Start with 1 HEK + 1 iPS, then the 5 + 5 reference scale as correctness checks, separate from the 64-condition main evaluation. Reconstruct the normal ROUND3 sequence: passage, imaging, medium change, coating and sampling as appropriate. Include the source refill tasks and their processing time on the same shared operation resource. Separate occupied processing time from nonoccupying culture/coating intervals. Register all cultures initially and disable later enrollment and sample splitting for this study.

Replace source wall-clock-dependent scheduling with the agreed virtual clock. Preserve the intended normal protocol and document any differences between source and current APIs. Use an explicit finish after the agreed endpoint; do not count post-sampling placeholder waits or an endless expired state as continued culture.

Verify actual outputs: observations in completion order, correct state transitions, growth during delays, density reset at passage, fixed started work, resource-compatible assignments and no overlap on the shared resource. Check the accepted operation timing and density criteria against an independent event/schedule inspection.

No injected failures means no simulated contamination, equipment breakdown or recovery/restart event. It does not suppress overload indicators, excessive density or missed timing criteria. If those occur because of congestion, record a capacity-condition violation without silently adding a recovery workflow.

Output: complete event histories and schedule snapshots for both reference scales, independent verification and documented discrepancies. Agreement in intended protocol is required; a new scheduler is not expected to reproduce the historical schedule exactly.

## Stage 4: measure count-dependent behavior

Freeze the accepted protocol, growth model, criteria, solver settings and execution budget. Evaluate all 64 pairs in `{3, ..., 10} × {3, ..., 10}`, with 30 varied-growth repetitions per pair from the outset. Do not substitute a coarse sweep or give only boundary conditions 30 repetitions. Use fresh output directories and preserve unsuccessful conditions and inputs.

A small pilot estimates computational cost and verifies variability before main execution; it does not change the selected 30 main repetitions. Declare pilot cost, the estimated cost of all 1,920 runs, resource/time budget and censoring policy before starting the main study. Budget exhaustion must be reported as incomplete evaluation rather than used to silently reduce the selected grid or repetition count.

Run observation, prediction and replanning through the agreed endpoint for every tested condition. Do not substitute successful initial planning or a fixed small number of updates for completion. Use common empirical draws where possible when comparing configurations, and distinguish solver timing variability from differences in simulated cultures.

Report per-type and total initial counts, active counts over time, completed cultures, acceptance rate, operation-specific delay and medium-change gaps, passage/sampling densities, horizon, queue growth, utilization and planning status. Include the first violated criterion and affected culture/operation. For long maintenance runs, increasing queue delay is evidence of overload under the tested policy.

Classify outcomes separately: accepted completed runs, biological/timing violations, solver UNKNOWN, proven solver infeasibility, invalid input/output, software/resource errors, and execution-budget censoring. UNKNOWN is not proof of infeasibility. FEASIBLE is not proof of optimality. A violated criterion under this scheduling policy does not prove that every possible scheduling policy would fail.

Report the accepted region of count pairs with repetition counts and variation, nearby failed pairs and the untested region. If 10 + 10 succeeds, the range ceiling has been reached; a limit above it has not been measured and extension is not automatic. Do not assume success is monotonic in either count or call a count ceiling the true limit. An analytic workload/utilization calculation can check plausibility and identify bottlenecks, but average load alone misses coincident passage demand.

Output: a two-dimensional grid of counts versus acceptance, delay, density and utilization plots, timelines near the observed boundary, primary/supplementary density comparisons, failure classifications, and conclusions that separate solver limitations, resource contention and growth/timing constraints.

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

The follow-up decisions are recorded above: the full 3–10 by 3–10 grid, variable growth, 30 repetitions, sampling endpoint, ROUND3 settings including refill, pre-interruption normal data and provisional primary/supplementary density criteria. Implementation and calibration now proceed on `feat/30-round3-culture-capacity`. Completion and measured outcomes will be reported separately from this original plan.

The planning deliverable is complete when this document and Issue #30 agree, consequential decisions and their current status are recorded, a dedicated branch/commit and planning PR exist, and documentation links, grid arithmetic and the scoped diff have been checked. A planning PR does not close the future evaluation Issue or approve unresolved execution settings.

The future evaluation is complete only after the accepted settings are frozen, input and model audits pass to the documented extent, reference simulations pass, count/boundary runs are verified, and reproducible results and remaining limitations are recorded. Its conclusions will be estimates under the specified equipment, protocol, observations and model, not guarantees for actual cultures.
