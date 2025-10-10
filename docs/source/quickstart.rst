Quick Start Guide
=================

The GEMS Python toolkit offers two canonical entry points that you can adapt
to your experiments:

``gems_python.one_machine_problem_interval_task``
    A finite-state orchestration model for a single resource. The core
    abstractions (:class:`~gems_python.one_machine_problem_interval_task.transition_manager.State`,
    :class:`~gems_python.one_machine_problem_interval_task.transition_manager.Experiment`,
    and :class:`~gems_python.one_machine_problem_interval_task.transition_manager.Experiments`)
    allow you to model each stage, simulate transitions, and schedule work.

``gems_python.multi_machine_problem_interval_task``
    Extends the same state-machine model to multiple machines, handling
    resource allocation via :class:`~gems_python.multi_machine_problem_interval_task.task_info.Machine`
    and :class:`~gems_python.multi_machine_problem_interval_task.task_info.MachineList`.

Follow the steps below to bootstrap a local environment and render the full
documentation.

Set up the Environment
----------------------

.. code-block:: bash

   poetry install

This installs the project along with Sphinx and the extensions used by the docs.
Alternatively, use ``pip install -r requirements.txt``.

Build the Docs
--------------

.. code-block:: bash

   sphinx-build -b html docs/source docs/build/html

Open ``docs/build/html/index.html`` in your browser to explore the generated
documentation.

Next Steps
----------

* Review :doc:`api/index` for an overview of the modules that power the
  one-machine and multi-machine schedulers.
* Examine ``tests/test_simulate_one.py`` for a minimal runnable example
  of composing custom states and running ``Experiments.simulate``.
* Launch ``main.py`` to experiment with the plugin-driven multi-machine UI.

Worked Example
--------------

Follow the sequence below to take a one-machine simulation all the way into
the plugin-driven multi-machine environment.

1. **Create the experiment factory**

   Copy ``tests/test_simulate_one.py`` to ``examples/demo_states.py`` and tailor
   the state transitions or penalties to your workflow. Keep the
   ``build_experiment`` function as the entry point.

2. **Seed persistence artefacts**

   .. code-block:: bash

      python examples/demo_states.py

   Running the script writes the first simulation snapshots (CSV, PNG, JSON) to
   a directory such as ``experiments_dir_demo_one_machine/``.

3. **Wrap the factory in a plugin**

   .. code-block:: python

      # experimental_setting/demo_plugin.py
      from examples.demo_states import build_experiment

      def demo_plugin():
          return build_experiment()

   When the plugin manager starts, it scans this directory and makes the
   factory available.

4. **Launch the multi-machine loop**

   .. code-block:: bash

      python main.py

   * Set ``mode/mode.txt`` to ``add_experiment``.
   * Write ``demo_plugin.demo_plugin`` to
     ``mode/mode_add_experiment.txt`` so the manager registers the experiment.
   * Switch ``mode/mode.txt`` to ``loop`` to continue execution.

Inspect the generated ``schedule.csv``, Gantt chart, and experiment snapshots in
the step directory printed by the console to verify the workflow. These are the
same assets the README quick start references.

Designing States and Experiments
--------------------------------

Both interval-task packages depend on the same primitives. Use the checklist
below when introducing a new workflow.

1. **Implement custom states**

   * Subclass :class:`gems_python.one_machine_problem_interval_task.transition_manager.State`
     (or the multi-machine analogue) and implement:

     - ``task_generator(self, df: pl.DataFrame) -> TaskGroup`` to prepare the
       next batch of work with penalties and scheduling hints.
     - ``transition_function(self, df: pl.DataFrame) -> str`` to choose which
       state runs next.
     - ``dummy_output`` (optional) when you want to simulate measurements during
       offline testing.

   * Populate the returned :class:`~gems_python.one_machine_problem_interval_task.task_info.TaskGroup`
     with :class:`~gems_python.one_machine_problem_interval_task.task_info.Task`
     instances that record processing time, interval, and descriptive operation names.

2. **Provide experiment factories**

   * Create ``build_experiment()`` that returns an
     :class:`gems_python.one_machine_problem_interval_task.transition_manager.Experiment`
     with the state list, starting state name, and an initial
     ``pl.DataFrame()`` for the shared variable history.
   * Wrap one or more experiments with ``build_experiments()`` to produce an
     :class:`gems_python.one_machine_problem_interval_task.transition_manager.Experiments`
     collection compatible with the plugin manager.

3. **Persist and iterate**

   * Call ``simulate(..., save_each_step=True)`` during development to validate
     transitions.
   * Use ``proceed_to_next_step()`` to emit the directory structure that
     ``main.py`` expects when managing experiments interactively.

Refer to ``tests/test_simulate_one.py`` for a concrete template that stitches
these pieces together.
