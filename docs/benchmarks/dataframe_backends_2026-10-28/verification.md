# Pandas / Polars verification — 2026-10-28

Issue: #17. Source commit: `d6acdef1a311ebccabd0d98068356cd2778fb29f`.
The comparison started with a clean working tree. No scheduler algorithm or
solver configuration changed.

## Decisions and affected boundaries

The user approved retaining the supplied DataFrame type automatically, using
Pandas when no history/backend is supplied, converting observations to the
history's backend, adding PyArrow, and treating row order/columns as data.
Pandas indexes are reset on conversion/append and omitted from CSV.

The affected boundaries are the three experiment managers (simple,
single-machine interval, multi-machine interval), the two interval schedule-table
factories, common and simple-task serialization, CSV loading/saving, and the
simulation callbacks. OR-Tools operates on task objects and has no DataFrame
dependency.

`Experiment.dataframe_backend` accepts `auto`, `pandas`, and `polars`. Selection
is resolved before initial task generation and retained across updates. Explicit
selection overrides input type. Mixed experiment backends are supported.
`TaskGroup.create_non_completed_tasks_df` has the same optional selector, with
Pandas as its no-input default. Existing argument positions are retained.

Compatible numeric columns are promoted on append; missing columns are filled
with null, existing columns precede newly introduced columns, and incompatible
types raise before task completion. JSON envelopes contain version 1, backend,
format `arrow-ipc`, and Base64 data. Arrow avoids decimal JSON floating-point
round-trip loss and retains the tested schemas. Concrete penalty types/modules
are also saved so restored experiments can be rescheduled.

## Environment and commands

Python 3.14.4, Pandas 3.1.0, Polars 1.44.2, PyArrow 26.0.0,
OR-Tools 9.15.6755. Dependencies are recorded in `poetry.lock`.
Solver limit: 5 seconds; seed: 0; workers: 1; scheduling method: `"o"`.
The limit/seed/worker settings are the existing repository configuration.

The session used `/tmp/gems-venv/bin/python` and
`/tmp/gems-venv/bin/poetry`, with `VIRTUAL_ENV=/tmp/gems-venv` and
`POETRY_CACHE_DIR=/tmp/gems-poetry-cache` during dependency installation:

```bash
poetry lock
poetry install --no-interaction
poetry check --lock
MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg poetry run python -m unittest \
  tests.test_dataframe_backends tests.test_cp_sat_scheduler \
  tests.test_experiment_simulators tests.test_scheduler_pattern_study -q
MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg poetry run python \
  scripts/compare_dataframe_backends.py \
  --output docs/benchmarks/dataframe_backends_2026-10-28
```

For reproduction, choose a new output directory; the script refuses to overwrite
existing results. The original comparison from source `ee0fff5` is also retained
locally at `/tmp/gems-dataframe-comparison-ee0fff-20261028`.

## Results and inspected outputs

All **9 pairs, 162 steps per backend, 324 steps total** matched exactly after
normalizing null representations. Compared actual input SHA-256 hashes,
history values and row/column order, transitions, task order/IDs, task starts and
ends, machine assignments, timing penalties, solver statuses, objectives and
bounds. Every one of the **594 solver calls was OPTIMAL**. Independent schedule
checks found zero overlap, invalid machine assignments, or group-interval errors.
Execution time is recorded but excluded from equality; this is not a performance
benchmark.

| Case | Steps per backend | Result |
| --- | ---: | --- |
| Cell culture | 16 | Equal; three passages with confluence resets and intervening growth |
| Reaction grid | 27 | Equal |
| Reaction random, seed 0 | 27 | Equal |
| Reaction random, seed 7 | 27 | Equal |
| Reaction random, seed 42 | 27 | Equal |
| Concurrent work | 2 | Equal; fast completion at minute 2, slow completion at minute 20 |
| Single-machine interval with observer | 12 | Equal |
| Single-machine interval with dummy output | 12 | Equal; opposite-backend results converted automatically |
| Simple-task update loop | 12 | Equal |

The **30 focused test methods passed**. They include automatic/explicit selection
in all three managers, empty/sparse histories, numeric promotion, nulls,
booleans, strings, UTC datetimes, overwrite, index handling, JSON/pickle round
trips, CSV ingestion, old history formats, incompatible results, invalid
providers, and a mixed Pandas/Polars manager saved and rescheduled after restore.
All five penalty kinds were independently saved/restored/rescheduled across
three task representations and two backends (30 subcases). Nonempty schedule
tables were also compared between backends.

`poetry check --lock` passed with the existing Poetry metadata deprecation
warnings. `git diff --check` passed.

Full discovery was run in disposable archives to isolate generated files and
interactive legacy tests:

```bash
git archive e1e5df6 | tar -x -C <baseline-directory>
git archive d6acdef | tar -x -C <final-directory>
# From each archive, with PYTHONPATH set to that archive:
MPLCONFIGDIR=/tmp/gems-mpl MPLBACKEND=Agg poetry run python \
  -m unittest discover -s tests -p 'test_*.py'
```

The actual executions used the shared `/tmp/gems-venv/bin/python` rather than
installing separate environments. Baseline: **63 tests, 10 errors**; final:
**74 tests, the same 10 errors**. Both test identifiers and exception causes
were checked for equality. Existing failures are missing HEK constructor
arguments (3), UUID JSON serialization in the old IPS fixture (1), interactive
input without a terminal (1), removed Matplotlib `get_cmap` API (4), and an old
tuple-unpacking assertion (1).

Tracked artifacts:

- `runs.json`: environment/source metadata and both complete runs per case.
- `summary.csv` / `summary.md`: equality and runtime summaries.
- `suite_failures.json`: baseline/final test counts and matching error causes.

Complete logs are retained locally at
`/tmp/gems-dataframe-baseline-20261028.log` and
`/tmp/gems-dataframe-final-fullsuite-20261028.log`.
The corresponding archives are
`/tmp/gems-dataframe-baseline-20261028-lby8t93n` and
`/tmp/gems-dataframe-final-fullsuite-20261028-r35_v7za`.

## Failures, fixes and limits

- Before installation, `pl.from_pandas` on nullable/string data and `to_pandas`
  failed because PyArrow was absent. The approved direct dependency fixes this.
- Simple-task JSON passed `current_state_index` (`init=False`) into its
  constructor. The reader now excludes it and reconstructs the current task.
- Group JSON previously restored penalties as dictionaries. The new metadata
  restores concrete classes, including their module; all five penalty kinds
  were verified by rescheduling after JSON restoration.
- A mixed CSV test initially omitted the existing multi-machine
  `task_response="Completed"` field. Correcting the test input enabled the
  intended ingestion path; this protocol was not changed.
- Git initially lacked a configured author. Commits use the prior implementation
  history's `Codex <codex@openai.com>` identity per command.
- The CSV writer initially used CRLF, which Git flagged as trailing whitespace.
  The writer now uses LF; the retained CSV was normalized without changing data.

Verified conversion types are ordinary numeric, boolean, string, null and
datetime columns. Arbitrary Python objects, every Polars/Pandas-specific type,
and the complete allowed Python/dependency version ranges are not verified.
CSV infers types and is not a substitute for schema-preserving JSON/pickle.
State functions must implement operations for the chosen backend, or use shared
helpers; their expressions are not translated automatically.

Legacy Polars history JSON and column dictionaries are supported. Name-only
simple State records require uniquely named, already imported classes with
constructors compatible with the old format. Old penalty records that never
saved a concrete type still lack enough information to reconstruct that type;
the new format prevents this loss. Custom State configuration still depends on
its existing serialization implementation.

The unrelated ten legacy suite errors remain. The branch is retained while the
PR is open; merging and subsequent branch deletion are separate operations.
