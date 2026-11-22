# ROUND3 empirical concurrent-culture evaluation

This study implements [Issue #30](https://github.com/bioinfo-tsukuba/GEMS-python/issues/30) and the [accepted plan](../../round3_culture_capacity_plan.md). The user authorized data preparation, calibration, all evaluation runs and a report with visualizations. The implementation branch is `feat/30-round3-culture-capacity`, distinct from the earlier planning branch.

## Fixed design and configuration provenance

| Item | Value | Source |
| --- | --- | --- |
| Count grid | HEK 3–10 × iPS 3–10, all 64 pairs | User |
| Repetitions | 30 per pair, 1,920 main runs | User |
| Enrollment | All cultures present initially | User |
| Endpoint | ROUND3 sequence through sampling | User |
| Variation | Empirical growth-curve parameters per culture and passage | User; fitting/draw method selected by agent |
| Primary density band | Target ±0.1, inclusive | User's provisional criterion and accepted recommendation |
| Supplementary band | `min(0.1, 0.20 × target)` | Agent's exploratory suggestion accepted by user |
| Operation resource | One shared physical machine | ROUND3 setting |
| Refill | 120-minute occupation, daily cadence, latest active 40-task group | Observed ROUND3 reservations |
| Solver | CP-SAT, 5-second search, one worker, seed zero | Repository defaults |
| UNKNOWN policy | Stop and classify; no retry with a changed limit | Agent selection |
| Biological seeds | Namespace 30, repetition 0–29, type and culture index | Agent selection for reproducibility |
| Parallelism | 12 independent one-thread processes | Agent selection after inspecting 20 available CPUs and host memory |
| Watchdogs | 10,000 events, 120 virtual days, 1,800 wall seconds per run | Agent-selected execution censoring, not biological failure thresholds |

The tracked [`culture_capacity_config.json`](../../../scripts/culture_capacity_config.json) is the configuration source. No additional observation noise, equipment breakdowns, biological failure events, later enrollment or sample splitting are introduced. Configured durations, rather than measured execution durations, determine resource occupancy.

For a run to pass the primary criterion, every culture must be sampled and every subsequent targeted passage/sampling operation must be within its density band. Two subsequent passages and one sampling operation are scored for each culture; initial seeding is outside that density criterion. HEK begins post-passage, matching its source entry state. iPS performs the initial coating/image/passage workflow on the shared resource. Startup iPS has a nominal parent density 0.3; it is not a target-driven source operation and is not scored. Growth while that initial parent stock waits is unmodeled: the nominal density remains constant until passage.

All culture clocks are normalized to the common enrollment origin, with time zero at clock phase zero. This intentionally differs from the historical staggered addition of cell types. No extra study axis shifts initial passage ages or work-hour phase. The initial HEK passage is a completed initial condition, while no iPS processing is credited before time zero. These entry conditions do not represent simultaneous de novo seeding of both cell types.

## Source audit

Inputs are pinned to `CAB314/GEMS-python-ROUND3` commit `26bd4552e08a3a3b8c1c654107d8b3198cb988f8`, directory `iPSsimulation/results/2024-11-13_real_round3/`.

The ingestion downloaded and verified 498 Git blobs, including 477 numbered snapshots and the `step_current` snapshot. Numbered snapshots were replayed in order. They yielded 377 unique culture-operation rows, 210 density observations and 10 source culture identities. No value revisions were found. Cumulative repetitions are counted once. The final snapshot alone omits earlier observations after source history truncation/re-registration.

The old Polars history format is a JSON string containing per-column values. The first ingestion attempt failed on that encoding; the reader was repaired and tested against the legacy format, empty histories and ragged-column failure. Detailed provenance, hashes and observations stay in the ignored `volatile_round3/source/` directory. The derived model contains fitted parameters and aggregate validation, not original measurement rows or images, and remains local pending explicit publication approval.

The source associates each independent culture Experiment with a cell-plate ID. Counts in this study are independently scheduled culture units; the number of occupied wells per plate and physical storage capacity are not established by this evaluation.

No independently annotated abnormality onset was available in the inspected snapshots. Normal observed intervals before removal/interruption are retained. A removed culture is not retrospectively treated as abnormal throughout its history. Sparse intervals with insufficient points or no identifiable positive trend are excluded from parameter fitting, with reasons; that is a fitting decision, not a diagnosis of culture failure. Restart trajectories are not injected into the simulation.

The model interprets recorded `density` as an image-derived fraction and recorded `time` as epoch minutes. This interpretation follows the source, without recalibrating images against cell counts. The simulator acquires images at task start, delivers results at completion and resets after passage completion; original measurement/result timestamps are not a perfect record of actual resource occupancy.

## Growth calibration and individual variation

Candidates are logistic growth with carrying capacity 1, logistic growth with a post-passage lag, and capped exponential growth. Fitting uses only observed densities after a passage, not null passage-density rows. Time is hours since the recorded passage. Positive parameter bounds and convergence checks prevent nonphysical parameter records from entering the empirical pool.

Forward validation fits earlier observations and predicts a later observation excluded from that fit. Candidate comparisons use matched prefixes where possible. A lag model has an extra parameter; the agent selected a minimum evidence rule of at least ten matched forecasts from at least three source cultures before it can replace a two-parameter model. Without that support, the two-parameter candidates are compared over their eligible forward forecasts.

This rule selected logistic growth for HEK and lag-logistic growth for iPS. The initial unrestricted comparison preferred lag growth for HEK by a small error difference on only two matched forecasts; that selection was rejected as insufficient evidence, before any main runs. There are 34 observed passage intervals and 31 fitted records for the chosen models. Unfit sparse terminal intervals are explicitly retained in the private audit with exclusion reasons.

The selected fitted pool is the local artifact `volatile_round3/calibration/growth_model.json`. A source-culture block is drawn uniformly for each virtual culture. Generation-matched initial density, growth rate and lag are drawn jointly from that block. The internal `donor` field identifies a technical source culture, not a human or genetic donor. HEK's main sequence uses the three source generations following the history reset; its older initial interval remains available to validation. If a selected block lacks a fitted generation, a complete joint record from the same-type, same-generation pool is drawn and the fallback is recorded.

This preserves dependencies within a source culture where supported without recreating abnormal terminal courses. It does not claim to estimate a full population distribution from five cultures per type. Across count pairs, the same repetition/type/index seed reuses a culture-parameter prefix, making comparisons paired. The 1,920 outcomes therefore must not be interpreted as 1,920 independent biological replicates. Post-passage starting density is an empirical draw; the model does not conserve cell mass or change the next passage's growth parameters in response to an earlier density overshoot.

Cross-culture generalization is inspected by excluding one whole source culture and predicting its observations using median parameters from the remaining cultures. The resulting errors are reported separately from online forward errors. The data used for model selection are not an independent final validation dataset; calibration and validation are exploratory.

## Prediction, scheduling and scoring

The planner is not given sampled growth parameters or future observations. It sees completed observation histories. The adapter retains ROUND3's shared-rate, per-passage logistic prediction, recent-passage weighting and already-over-target rule. The source's unused fitted `k` parameter is removed, null densities are not treated as zero measurements, and positive bounds are added. These numerical changes are explicit; the historical nonlinear optimizer is not reproduced bit for bit.

The source task-group templates were audited across snapshots. Notably, active iPS sampling uses an image with zero initial interval, sampling penalty coefficient 100 and the same epoch-aligned rest ranges as HEK. The initial `iPSsetting.py` differs from those active sampling settings. The main study uses the actually observed terminal template, drops terminal waiting placeholders and finishes after sampling.

Rest ranges `(0,600)` and `(960,1440)` penalize task-group starts; they are not hard machine-availability constraints. Coating waits and culture waits do not occupy the machine. Refill processing occupies the same machine as culture operations. The source fixed refill group repeats if necessary until all cultures finish; future refill reservations beyond the final sampling time do not contribute to measured utilization.

Inter-group observation/medium-change target times are also preferences, not hard minimum intervals. The scheduler may move a not-yet-started group earlier when that reduces its total objective. Intra-group intervals remain fixed once the group starts. The first iPS medium-change group contains an image, medium change and another image after 24 hours; the following group prefers another 24-hour wait. Thus a roughly 48-hour medium gap is already possible in the source route. The report measures actual medium/passage-end to next medium-start gaps, including unexpectedly short gaps, without claiming they are biologically appropriate.

There is one initial schedule, followed by one production history/state update and replan per received result. This skips a redundant pre-receipt solve on an already valid plan. A focused test checks that this driver agrees with `simulate_one` on operation order, scheduled times and simulated observations in a deterministic reference case.

Density is scored immediately before the actual passage/sampling operation, prior to any passage reset. Cells continue to grow while awaiting processing. Simulator density is used for scoring, while the predictor receives image observations only. Errors at actual operation time include forecast error, scheduling displacement and processing-time conventions.

For interpretation, each targeted event also records latent density at the action time implied by its group's preferred start. This nominal-density error combines observation-based forecast error and the protocol's start/end convention. Its signed difference from actual-density error measures the change associated with scheduling displacement on that same curve. The two signed components sum to the recorded error; this is a descriptive decomposition, not an independent experiment or a universal causal proof. These diagnostic values are never sent back to the prediction policy.

| Target | Primary band | Primary relative half-width | Supplementary band |
| --- | --- | --- | --- |
| HEK passage 0.8 | 0.7–0.9 | 12.5% | 0.7–0.9 |
| HEK sampling 0.4 | 0.3–0.5 | 25% | 0.32–0.48 |
| iPS passage/sampling 0.3 | 0.2–0.4 | 33.3% | 0.24–0.36 |

Neither band is a biologically validated tolerance. Medium-change intervals and operation delays are reported without adding unspecified timing pass/fail thresholds. Failing a provisional band is an evaluation violation, not an injected biological failure.

## Independent verification and interpretation

Every recorded run retains input genomes, event records, solver records and its outcome. The offline verifier independently recomputes density from elapsed time and stored parameters, checks passage resets, machine occupancy, fixed group intervals, completion/target counts, error criteria, solver limits and artifact hashes. Verification does not rerun all optimizations.

Outcomes distinguish completed runs, density violations, UNKNOWN, proven infeasibility, execution/input/output errors and watchdog censoring. FEASIBLE results are usable validated schedules with unproven optimality. An UNKNOWN result is not proof that the physical workload is impossible.

Reports show completion and full-run acceptance separately. Full-run acceptance naturally becomes harder when there are more independently scored cultures; an observed reduction alone is not proof of resource saturation. Paired parameter prefixes, per-operation error distributions, occupancy and timing diagnostics help interpret that difference.

Successful 10+10 runs reach the requested range ceiling, not a demonstrated upper capacity limit. Storage, sufficient consumables, morphology, cell quality, long-term stability and actual hardware timing are not independently verified. Conclusions apply to the specified normal protocol, provisional criterion, empirical growth model and tested planning policy.

## Reproduce and review

Run these commands from the repository root using the project's locked dependencies. Paths under `volatile_round3/` are ignored and remain local. The source reader requires access to the private source repository.

```bash
.venv/bin/python scripts/extract_round3_culture_data.py \
  --download --output volatile_round3/source --workers 8
.venv/bin/python scripts/calibrate_round3_growth.py \
  --source volatile_round3/source \
  --output volatile_round3/calibration --private-output volatile_round3/calibration
.venv/bin/python scripts/measure_culture_capacity.py \
  --model volatile_round3/calibration/growth_model.json \
  --output volatile_round3/main --workers 12 --wall-budget-seconds 86400
.venv/bin/python scripts/verify_culture_capacity.py volatile_round3/main
.venv/bin/python scripts/report_culture_capacity.py volatile_round3/main \
  --model volatile_round3/calibration/growth_model.json \
  --output volatile_round3/report --reference volatile_round3/reference \
  --private-audit volatile_round3/calibration/fit_audit.json
```

The optional 1+1 reference uses `--case 1 1 N` for each `N` from 0 to 29, the same configuration/model, two independent processes, and a separate output directory. Verify it independently before passing it to the report. To inspect a pilot or an unfinished study, `--allow-partial` labels the HTML as a preview; never use that flag to certify the full study. Reports refuse to overwrite earlier outputs.

The standalone HTML includes count-pair selection, all recorded repetitions, and whole-run/day event timelines. Accompanying PNG/SVG figures and CSV tables contain completion, density acceptance, culture-level acceptance, paired HEK-00/iPS-00 comparisons, calibration checks, medium-change gaps and occupancy. Representative runs are selected near median completion time without reference to their acceptance.

`scripts/finalize_culture_capacity.py --wait` can run alongside a long study. It waits for the driver's final execution manifest, independently verifies every planned outcome, then creates the local report. It cannot mark an incomplete calculation complete and does not publish any artifact.

```bash
.venv/bin/python scripts/finalize_culture_capacity.py volatile_round3/main \
  --model volatile_round3/calibration/growth_model.json \
  --output volatile_round3/report --reference volatile_round3/reference \
  --private-audit volatile_round3/calibration/fit_audit.json --wait
```

Use a persistent execution environment for the driver and finalizer. In this session the dedicated `tmux` sessions are `gems30-main` and `gems30-finalize`. `volatile_round3/main/finalization.json` distinguishes waiting, a verified report, and a failure requiring inspection. Add `--browser` only in an environment with the optional browser tools configured.

Optional browser validation uses `scripts/verify_culture_capacity_report.py` and Playwright/Chromium, not runtime dependencies. The verification environment lacked `libnspr4`, `libnss3` and `libasound`; distribution packages were downloaded and extracted under `/tmp/gems30-browser-libs/`, without installing system packages. With the extracted library path and `PLAYWRIGHT_BROWSERS_PATH=/tmp/gems30-playwright`, the portable pilot HTML passed Chromium checks after those missing-library launch failures. Images, condition selection, repetition counts, day zoom and absence of external requests were checked.

## Execution and publication boundary

The initial foreground pilot was interrupted when tool execution lost its session. Completed records were preserved; its unfinished case was not counted as an outcome. Long calculations were moved to dedicated `tmux` sessions, and the 10+10 pilot was rerun in a fresh directory. This is an execution failure, not a culture failure. Core source hashes guard resumed runs against mixing implementations.

Automatic approval review rejected a push containing private-source-derived fitted parameters to the public implementation repository. The code-only push subsequently succeeded after the model was moved to an ignored local directory and the destination was verified. Original observations/images, fitted models, simulated records and generated reports remain local pending explicit publication approval. The report's optional private fit supplement contains source observations and is excluded from the portable HTML.
