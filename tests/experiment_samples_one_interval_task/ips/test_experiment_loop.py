

from pathlib import Path
import unittest

from gems_python.one_machine_problem_interval_task.transition_manager import Experiments
from tests.experiment_samples_one_interval_task.ips.experimental_settings import IPSExperiment


class TestExperimentStructure(unittest.TestCase):
    def generate_experiments(self):
        experiment = IPSExperiment(
            current_state_name="GetImage1State"
        ) 
        dir = Path("volatile")
        return Experiments(
            experiments=[experiment],
            parent_dir_path=dir
        )

    def test_experiments_structure(self):
        experiments = self.generate_experiments()
        print(experiments)
