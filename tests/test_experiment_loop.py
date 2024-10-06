

import unittest

from tests.experiment_samples_simple_task.hek_cell_culture import HekCellCulture


class TestExperimentStructure(unittest.TestCase):
    def setUp(self):
        self.experiment = HekCellCulture()

    def test_experiment_structure(self):
        # Test the experiment structure
        self.assertEqual(self.experiment.experiment_name, "HekCellCulture")
        self.assertEqual(len(self.experiment.states), 6)

    def test_experiment_structure_graph(self):
        self.experiment.show_experiment_directed_graph()

    def test_experiment_structure_task_generation(self):
        self.experiment.show_experiment_name_and_state_names()