from pathlib import Path
import unittest

from gems_python.one_machine_problem_interval_task.transition_manager import Experiments
from tests.experiment_samples_one_interval_task.ips.experimental_settings import IPSExperiment


class TestDumpDeserialise(unittest.TestCase):
    def generate_experiments(self):
        experiment = IPSExperiment(
            current_state_name="GetImage1State"
        ) 
        dir = Path("volatile")
        return Experiments(
            experiments=[experiment],
            parent_dir_path=dir
        )

    def test_deserialise(self):
        lab = self.generate_experiments()
        lab.execute_scheduling(reference_time = 0)
        print(lab)

        lab_json = lab.to_json()

        print(f"{lab_json=}")
        lab_from_json = Experiments.from_json(lab_json)

        lab_re_to_json = lab_from_json.to_json()
        print(lab_re_to_json)
        assert lab_re_to_json == lab_json
