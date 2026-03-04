import unittest

import polars as pl

from gems_python.gui_experiment_builder.factory import build_experiment_from_draft
from gems_python.gui_experiment_builder.models import ExperimentDraft, StateDraft


class TestGUIExperimentBuilderFactory(unittest.TestCase):
    def setUp(self):
        self.draft = ExperimentDraft(
            experiment_name="factory_test_experiment",
            initial_state="StateA",
            states=[
                StateDraft(name="StateA", next_state="StateB", machine_type=0, processing_time=2, interval=0),
                StateDraft(name="StateB", next_state="StateB", machine_type=0, processing_time=1, interval=0),
            ],
        )

    def test_build_experiment_from_draft(self):
        experiment = build_experiment_from_draft(self.draft)
        self.assertEqual(experiment.experiment_name, "factory_test_experiment")
        self.assertEqual(experiment.current_state_name, "StateA")
        self.assertEqual(sorted(experiment.get_all_state_names()), ["StateA", "StateB"])

    def test_generated_state_behaviour(self):
        experiment = build_experiment_from_draft(self.draft)
        first_state = experiment.states[0]

        task_group = first_state.task_generator(pl.DataFrame())
        self.assertEqual(task_group.tasks[0].optimal_machine_type, 0)
        self.assertEqual(task_group.tasks[0].processing_time, 2)
        self.assertEqual(first_state.transition_function(pl.DataFrame()), "StateB")

        dummy = first_state.dummy_output(pl.DataFrame(), task_group_id=1, task_id=2)
        self.assertTrue(set(["time", "state", "measurement", "task_group_id", "task_id"]).issubset(set(dummy.columns)))


if __name__ == "__main__":
    unittest.main()
