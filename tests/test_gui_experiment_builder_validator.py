import unittest

from gems_python.gui_experiment_builder.models import ExperimentDraft, StateDraft
from gems_python.gui_experiment_builder.validator import validate_experiment_draft


class TestGUIExperimentBuilderValidator(unittest.TestCase):
    def test_validate_success(self):
        draft = ExperimentDraft(
            experiment_name="valid_experiment",
            initial_state="StateA",
            states=[
                StateDraft(name="StateA", next_state="StateB", machine_type=0, processing_time=5, interval=0),
                StateDraft(name="StateB", next_state="StateB", machine_type=0, processing_time=3, interval=1),
            ],
        )
        errors, warnings = validate_experiment_draft(draft, available_machine_types={0})
        self.assertEqual(errors, [])
        self.assertEqual(warnings, [])

    def test_validate_duplicate_state_names(self):
        draft = ExperimentDraft(
            experiment_name="duplicate_state",
            initial_state="StateA",
            states=[
                StateDraft(name="StateA", next_state="StateA"),
                StateDraft(name="StateA", next_state="StateA"),
            ],
        )
        errors, _warnings = validate_experiment_draft(draft, available_machine_types={0})
        self.assertTrue(any("重複" in error for error in errors))

    def test_validate_missing_machine_type(self):
        draft = ExperimentDraft(
            experiment_name="missing_machine",
            initial_state="StateA",
            states=[StateDraft(name="StateA", next_state="StateA", machine_type=3)],
        )
        errors, _warnings = validate_experiment_draft(draft, available_machine_types={0, 1, 2})
        self.assertTrue(any("未登録" in error for error in errors))


if __name__ == "__main__":
    unittest.main()
