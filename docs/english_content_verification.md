# Issue #28: Maintained English content verification (2026-10-31)

The implementation translates maintained explanations and diagnostics while
preserving execution, APIs, CLI flags, paths, serialization keys and scientific
conditions. Current evaluation prose is included by the user's latest decision.
Pinned historical commits, archive branches, raw results and figures remain
unchanged. `AGENTS.md` is explicitly excluded.

## Revisions and inventory

- Baseline: merged PR #27, main `63faac9662e28f027040a76e9dc9db067b1db6f8`.
- Implementation: `9c4d6dab79e9526b5aba840366677a513951892d`.
- Branch: `docs/28-english-maintained-content`.
- Initial inventory: 39 files with Japanese text, including two files containing
  only deliberate Unicode regression data. An initial Issue comment incorrectly
  counted 38; the complete inventory contains 39.
- Translation: seven Markdown guides/evaluations and 30 Python files. This
  verification record is an additional document.
- Retained data: Japanese paths at lines 75/280 in
  `tests/test_persistence_roundtrip.py`, and the experiment name plus its
  expected label at lines 126/130 in `tests/test_retrospective.py`.

The translated guides cover persistence, both simulation samples, persistence
verification, scheduler comparison, scheduler capacity and the pattern study.
The five affected scripts are `analyze_scheduler_capacity.py`,
`report_scheduler_capacity.py`, `study_scheduler_patterns.py`,
`measure_result_update_latency.py` and `report_result_update_latency.py`.
Library and test changes cover comments, docstrings and diagnostic text in
the three task managers, penalties, interactive UIs and samples.

## Environment

Python 3.14.4 was supplied by the environment. The previously documented
`/tmp/gems-venv`, Poetry and pip were absent. Initial venv creation failed
because `ensurepip` was unavailable; its interpreter was usable, and pip was
bootstrapped into `/tmp/gems-issue28-venv`. Poetry 2.5.1 was installed there as
verification tooling. No repository dependency files were changed.

Dependencies were installed with the unchanged `poetry.lock`:

```bash
VIRTUAL_ENV=/tmp/gems-issue28-venv \
POETRY_CACHE_DIR=/tmp/gems-issue28-poetry-cache \
POETRY_CONFIG_DIR=/tmp/gems-issue28-poetry-config \
/tmp/gems-issue28-venv/bin/poetry install --no-root --no-interaction
```

All 58 packages applicable to this platform/Python matched their locked versions.
These include Pandas 3.1.0, Polars 1.44.2, PyArrow 26.0.0, OR-Tools 9.15.6755,
NumPy 2.5.4, Matplotlib 3.11.2 and NetworkX 3.6. An initial naive audit compared
all lock entries, including platform/Python-inapplicable packages; evaluating
their markers resolved the apparent mismatches.

Lock SHA-256: `f83230caf91f70c0fa0c35434ce4c120262f8e48d907e24b4d802c335ba004b5`.
Tests used `MPLBACKEND=Agg` and temporary Matplotlib configuration directories.
Poetry wrappers were exercised, as well as direct interpreter commands.

## Tests and output inspection

Before/after full suites ran in separate `git archive` copies with stdin at EOF,
keeping generated JSON, CSV, images and experiment directories out of the checkout.
Both ran 148 tests: 143 passed, with the same five errors and no assertion failures
or skips. Test IDs and full tracebacks matched after normalizing copy paths.
The final implementation was also checked from its fixed commit.
Baseline/final full-suite runtimes were 66.934/66.618 seconds; these are check
durations, not performance comparisons.

The existing errors are three HEK initialization errors caused by omitted
`current_state_name`, an interactive-input EOF error, and unpacking
`OneMachineTaskLocalInformation` using an old tuple return contract. See the
[persistence verification record](benchmarks/persistence_roundtrip.md#five-remaining-out-of-scope-errors).

The 14 relevant modules below ran 85 tests, all passing before and after,
including Unicode paths/names and four dumper tests outside normal discovery.

```bash
python -m unittest \
  tests.test_persistence_roundtrip tests.test_dataframe_backends \
  tests.test_cp_sat_scheduler tests.test_experiment_simulators \
  tests.test_scheduler_pattern_study tests.test_capacity_statistics \
  tests.test_capacity_measurement tests.test_result_update_latency \
  tests.test_scheduler_capacity tests.test_retrospective \
  tests.test_class_dumper tests.test_penalty_class \
  tests.dumper_test.test_dumper_deserialise \
  tests.experiment_samples_multi_machine_one_interval_task.test_class_dumper
python -m unittest discover -s tests -p 'test_*.py'
```

For reproduction, use the dependency environment above and substitute either
revision listed under [revisions and inventory](#revisions-and-inventory):

```bash
revision=9c4d6dab79e9526b5aba840366677a513951892d
verification_dir=$(mktemp -d /tmp/gems-english-check-XXXXXX)
git archive "$revision" | tar -x -C "$verification_dir"
cd "$verification_dir"
MPLBACKEND=Agg MPLCONFIGDIR=/tmp/gems-issue28-mpl \
  /tmp/gems-issue28-venv/bin/python -m unittest discover -s tests -p 'test_*.py' </dev/null
```

Full discovery exits 1 because of the five existing errors.

Both documented simulation modules executed through `poetry run python -m`
in an isolated copy. They produced four single-machine and eleven multi-machine
snapshot manifests. Task-group diagnostics exercised missing groups/tasks,
completion, repeated completion and deletion. Both interactive sample reload
failure messages were exercised with the polling loop mocked.

Direct module execution records State classes as `__main__`. A fresh process
without those definitions could not reload the resulting snapshots. This was
reproduced on unchanged main. Importing each sample before building and saving
its collection allowed a separate process to reload at steps 3/10 with
`FinishState`. The guides now document this existing limitation.

## Reports without new searches

Saved capacity inputs came from `bf457334a0bddc3a75ff6560726a80e3c42dbf80`;
latency inputs came from `747902191e754bb79de902145b2ac6bd9cca456e`.
Archives and required inputs were extracted under `/tmp/gems-issue28-records`.
Before/after report generators used separate output copies. No large scheduler
study or latency measurement was rerun.

The commands exercised these interfaces with the project interpreter:

```bash
python scripts/analyze_scheduler_capacity.py CAPACITY_RECORDS --output NEW_STATISTICS
python scripts/report_scheduler_capacity.py CAPACITY_RAW_COPY
python scripts/report_result_update_latency.py LATENCY_RAW_COPY
python scripts/study_scheduler_patterns.py --from-runs FIXTURE_RUNS_JSON --output-dir NEW_PATTERNS
```

`CAPACITY_RECORDS` is the archived `docs/benchmarks/scheduler_capacity` directory.
`CAPACITY_RAW_COPY` uses its `config.json` and the primary `results.jsonl` from
`raw_artifacts.tar.gz`. `LATENCY_RAW_COPY` uses `results.jsonl` and `inputs.json`
from `raw_results.tar.gz`, plus archived `environment.json`. The callable
`scripts.measure_result_update_latency.summarize(Path(...))` was also exercised
on those saved latency records.

Nine generated CSV/JSON files and five copied input/configuration files were
byte-identical before/after. Eight generated PNGs had identical dimensions and
pixels. Numerical report rows were unchanged, and all generated report prose
was English. All five affected scripts' `--help` outputs were English and
identical before/after.

The original 64-case pattern `runs.json` is absent from tracked history.
Its reaggregation remains unverified. That report's generation path and plots
were exercised with four synthetic cases using values from the existing
aggregation regression test, including infeasible, lower, equal and higher
annealing scores. These fixtures do not constitute new measured results.

## Structure, links and scope checks

Recursive comparison of the 30 Python syntax trees permitted only translations
of Japanese-bearing strings. All other nodes and values matched, including
interpolation expressions, evaluation order, conversions and format specifiers.
The final comparison accounted for 134 translated literals. Intermediate checks
caught reordered interpolations; those translations were revised before commits.

Local Markdown links, images and heading anchors passed (34 targets before this
record was added, 36 including this record). Eleven pinned GitHub file references were checked against
local Git objects. Six pre-existing scheduler links pointed to untracked
generated files; the guide now identifies those output paths without broken
links. Literal pipes in penalty formulas were escaped for Markdown table
rendering, preserving the formulas.

Explanations were corrected from source where they disagreed with behavior:
simulation saving defaults, actual step counts and printed summaries, parent
directory creation, the first task's interval, completed-task marking comments,
State/task-return comments and curve-fit indexing comments.

Tracked-file scans include ignored paths and Japanese punctuation, avoiding an
initial plain `rg` query that omitted files in this environment. The only retained
Japanese is the deliberate Unicode regression data identified above.
`AGENTS.md`, raw data, figures, fixtures, dependency files and archived refs were
unchanged. `git diff --check` passed; generated outputs remain outside the checkout.

Physical experiment execution, the interactive polling loop and the absent
64-case raw pattern dataset were not verified in this language-only change.
