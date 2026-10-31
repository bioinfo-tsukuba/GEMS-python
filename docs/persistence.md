# Completing the previously unresolved persistence round trip

Issue #19 completes the previously unresolved save-and-restore round trip.
It was implemented on `fix/previously-unresolved-roundtrip`, starting from
main `3641e77817d8f4821b01026884be1a04543a7c1b`.

## Supported managers and saved content

The same persistence API is available in all three managers:

- `one_machine_problem_simple_task`
- `one_machine_problem_interval_task`
- `multi_machine_problem_interval_task`

Persistence records concrete classes and attributes for `Experiment`,
`Experiments`, State, Task, TaskGroup, penalties and Machine. This includes
required State configuration, dynamic progress, `init=False` attributes,
the current State name and index, history backend, column order, types and
values, task times, IDs and completion status, machine assignments, and the
manager's step and reference time. Shared references, such as a TaskGroup
referenced from multiple locations, are also restored.

Restoration does not execute constructors or `__post_init__`. It restores
saved values without generating UUIDs, recalculating schedules, regenerating
States or tasks, or creating output directories during loading.

## Usage

```python
from pathlib import Path
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments

# lab is an Experiments collection partway through scheduling or execution.
json_text = lab.to_json()
restored = Experiments.from_json(json_text)

lab.to_pickle(Path("experiments.pkl"))
restored = Experiments.from_pickle(Path("experiments.pkl"))
restored = Experiments.from_pickle(dumped=lab.to_pickle())

dictionary = lab.to_dict()
restored = Experiments.from_dict(dictionary)
```

The simple-task `Experiments` also provides these six methods.
`save_all()` saves the entire collection as `experiments.json` and
`experiments.pkl`, alongside the existing individual experiment and task
outputs. The interval-task managers' `save_results()` also saves both formats.

Since Issue #24, shared persistence writes and synchronizes both formats in the
same snapshot generation, then commits it by switching `snapshot_manifest.json`.
Reload checks the generation and hashes selected by the manifest, never selecting
an unfinished generation. The two root-level files remain as compatibility
outputs; editing them directly does not affect reload when a manifest exists.
Legacy saves without a manifest use the previous two-file loading path.
See [snapshot generations and legacy saves](experiment_lifecycle.md#snapshot-generations-and-legacy-saves).

## Automatic selection in reload

```python
# Interval tasks: automatically select the highest saved step.
restored = lab.reload()
restored = lab.reload(step=3, source="json")
restored = lab.reload(step=3, source="pickle")

# Simple tasks: load from parent_dir_path or the specified directory.
restored = simple_lab.reload()
restored = simple_lab.reload(save_dir=Path("saved"), source="json")
```

| source | Behavior |
| --- | --- |
| `"auto"` (default) | Read both. Return JSON if types, values, DataFrames and shared references match. If only one format is valid, warn and return that format |
| `"json"` | Read JSON only; raise an error on failure |
| `"pickle"` | Read pickle only; raise an error on failure |

If both formats load but differ, `auto` raises an error. Select a format
explicitly with `source`; loading never prompts for input. Neither successful
nor failed loading changes the calling lab or its step. Assign the returned
object to a variable to use it.

## Format and compatibility

The new JSON records `_gems_serialization: 1`, concrete `module:Class` names,
typed containers and shared references. The output structure of `to_dict()`
also changed. External code that directly accesses top-level entries in the
old dictionary must be updated. Use the public API for saving and loading.
DataFrames use the existing Arrow IPC format, with supplementary information
for RangeIndex columns in empty Pandas histories.

Supported values include nested dataclasses, list / dict / tuple / set /
frozenset, Enum, Path, UUID, bytes, dates, times, durations and NumPy scalars.
The new format restores saved concrete types even in nested values without
type annotations.

Loading supports legacy States with class information, identifiable penalties,
Polars JSON and column dictionaries, and existing Arrow histories. Legacy
tuples with type annotations are also restored. Input dictionaries are not
modified, so the same input can be restored repeatedly.

### Constraints and deliberate decisions

- Modules containing saved classes must be available in the loading environment.
  Function-local classes and deleted or renamed classes cannot be restored.
  Attributes are saved; class code, observation functions, external random
  generator state and internal solver state are not. Resume verification uses
  the same observer, inputs and solver settings.
- Cyclic references, file handles, functions and unsupported objects raise a
  `SerializationError` that identifies the field path. Shared references are
  supported when they do not form a cycle.
- When a legacy format lacks concrete class information, loading uses the class
  specified by the called API or existing type annotations. It does not infer
  a lost Experiment subtype. Missing required State configuration, concrete
  penalties or progress fields produce an error identifying the missing field,
  rather than resuming execution with default values.
  As an exception, only legacy multi-machine saves without a lifecycle version
  receive initial records with the additional features disabled and default retry
  settings. State and Task progress are not regenerated.
  If a legacy save lacks a State index, the name must identify one State
  uniquely; duplicate State names cause an error.
- Legacy JSON lacks shared-reference information, so object identity cannot be
  recovered. If a valid legacy pickle retains the original sharing, select it
  with `source="pickle"`. Using `auto` on JSON/pickle files with this difference
  raises a mismatch error.
- History follows the existing convention of ordered rows and named columns.
  Custom Pandas indexes are not saved. Column types must be representable in Arrow.
- Both JSON and pickle load the classes they use. Saved files must be trusted.
- Experiment IDs supplied as UUIDs are normalized to strings. Default UUIDs in
  IPS samples are generated per instance; explicitly supplied string IDs are retained.
- Old readers cannot read the new format. Preserve old files before writing back.

## Verification

Additional acceptance tests are in `tests/test_persistence_roundtrip.py`.
See the [verification record](benchmarks/persistence_roundtrip.md) for commands,
the environment and remaining errors in the full suite.
