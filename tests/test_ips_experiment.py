import inspect


from pathlib import Path
import unittest
import uuid
import numpy as np
import polars as pl

from gems_python.onemachine_problem.transition_manager import Experiment, Experiments
from tests.experiment_samples.ips.experimental_settings import PROCESSING_TIME, IPSExperiment


class TestExperimentStructureIPS(unittest.TestCase):

    def logistic_function(self, t, r, N0, K=1):
        return K / (1 + (K/N0 - 1) * np.exp(-r * t))
    
    def calculate_optimal_time_simple(self, optimal_density, r, N0, K=1):
        # Binary search
        left = 0
        right = 10000

        while right - left > 1:
            right_value = self.logistic_function(right, r, N0, K)
            if right_value < optimal_density:
                right *= 2
            else:
                mid = (left + right) // 2
                mid_value = self.logistic_function(mid, r, N0, K)
                if mid_value < optimal_density:
                    left = mid
                else:
                    right = mid

        return right
    

    def setUp(self):
        self.experiment = IPSExperiment(current_state_name="GetImage2State")

    
    def test_get_image_1_state(self):
        print(f"~~~~~~~~~~~~~~\n{inspect.currentframe().f_code.co_name=}")
        optimal_density = 0.3
        r = 0.00025
        N0 = 0.05
        time = [0, 24*60]
        density = [self.logistic_function(t, r, N0) for t in time]
        density[0] = None
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage"]
            }
        )
        state_name = "GetImage1State"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        assert task.experiment_operation == "GetImage"
        assert task.optimal_time == 24*60+24*60

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "MediumChange1State"


        optimal_density = 0.3
        r = 0.00025
        N0 = 0.05
        time = [0, 24*60]
        density = [None, 0.04]
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage"]
            }
        )
        state_name = "GetImage1State"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        assert task.experiment_operation == "GetImage"
        assert task.optimal_time == 24*60+24*60

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "GetImage1State"



    def test_medium_change_1_state(self):
        print(f"~~~~~~~~~~~~~~\n{inspect.currentframe().f_code.co_name=}")
        optimal_density = 0.3
        r = 0.00025
        N0 = 0.05
        time = [0, 24*60]
        density = [self.logistic_function(t, r, N0) for t in time]
        density[0] = None
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage"]
            }
        )
        state_name = "MediumChange1State"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        assert task.experiment_operation == "MediumChange1"
        assert task.optimal_time == max(time)

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "GetImage2State"



    def test_get_image_2_state(self):
        print(f"~~~~~~~~~~~~~~\n{inspect.currentframe().f_code.co_name=}")
        optimal_density = 0.3
        r = 0.00025
        N0 = 0.05
        time = [0, 24*60, 24*3*60, 24*4*60]
        density = [self.logistic_function(t, r, N0) for t in time]
        density[0] = None
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage", "GetImage", "GetImage"]
            }
        )
        state_name = "GetImage2State"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        assert task.experiment_operation == "GetImage"
        assert task.optimal_time == max(time)+24*60

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "PlateCoatingState"



        optimal_density = 0.3
        r = 0.000025
        N0 = 0.05
        time = [0, 24*60, 24*3*60, 24*4*60]
        density = [self.logistic_function(t, r, N0) for t in time]
        density[0] = None
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage", "GetImage", "GetImage"]
            }
        )
        state_name = "GetImage2State"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        assert task.experiment_operation == "GetImage"
        assert task.optimal_time == max(time)+24*60

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "MediumChange2State"




        optimal_density = 0.3
        r = 0.000025
        N0 = 0.05
        time = [0, 24*60, 24*3*60, 24*4*60]
        density = [self.logistic_function(t, r, N0) for t in time]
        density[0] = None
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage", "GetImage", "MediumChange1"]
            }
        )
        state_name = "GetImage2State"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        assert task.experiment_operation == "GetImage"
        assert task.optimal_time == max(time)+24*60*2

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "MediumChange2State"



    def test_medium_change_2_state(self):
        print(f"~~~~~~~~~~~~~~\n{inspect.currentframe().f_code.co_name=}")
        optimal_density = 0.3
        r = 0.00025
        N0 = 0.05
        time = [0, 24*60]
        density = [self.logistic_function(t, r, N0) for t in time]
        density[0] = None
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage"]
            }
        )
        state_name = "MediumChange2State"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        assert task.experiment_operation == "MediumChange2"
        assert task.optimal_time == max(time)

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "GetImage2State"

    def test_plate_coating_state(self):
        print(f"~~~~~~~~~~~~~~\n{inspect.currentframe().f_code.co_name=}")
        optimal_density = 0.3
        r = 0.00025
        N0 = 0.05
        time = [0, 24*60, 24*3*60, 24*4*60]
        density = [self.logistic_function(t, r, N0) for t in time]
        density[0] = None
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage", "GetImage", "GetImage"]
            }
        )
        state_name = "PlateCoatingState"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        assert task.experiment_operation == "PlateCoating"
        assert np.abs(task.optimal_time-optimal_time+PROCESSING_TIME["PLATE_COATING"]) <= 10

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "PassageState"

    


    def test_passage_state(self):
        print(f"~~~~~~~~~~~~~~\n{inspect.currentframe().f_code.co_name=}")
        optimal_density = 0.3
        r = 0.00025
        N0 = 0.05
        time = [0, 24*60, 24*3*60, 24*4*60]
        density = [self.logistic_function(t, r, N0) for t in time]
        density[0] = None
        optimal_time = self.calculate_optimal_time_simple(optimal_density, r, N0)
        print(f"time: {time}")
        print(f"density: {density}")

        print(f"optimal_time: {optimal_time}")
        print(f"density at optimal_time: {self.logistic_function(optimal_time, r, N0)}")
        shared_variable_history = pl.DataFrame(
            {
                "time": time,
                "density": density,
                "operation": ["Passage", "GetImage", "GetImage", "GetImage"]
            }
        )
        state_name = "PassageState"


        ex = IPSExperiment(current_state_name=state_name, shared_variable_history=shared_variable_history)
        task = ex.generate_task_of_the_state()
        print(f"state: {state_name} -> task: {task}")

        next_state = ex.determine_next_state_name()
        print(f"current_state: {state_name} -> next_state: {next_state}")

        assert next_state == "GetImage1State"


    
    def test_experiment_structure(self):
        # Test the experiment structure
        self.assertEqual(self.experiment.experiment_name, "IPSExperiment")
        self.assertEqual(len(self.experiment.states), 7)

    def test_experiment_structure_graph(self):
        self.experiment.show_experiment_directed_graph()

    def test_show_experiment_directed_graph(self):
        self.experiment.show_experiment_with_tooltips(hide_nodes=[])

    def test_experiment_structure_task_generation(self):
        self.experiment.show_experiment_name_and_state_names()

    def test_task_generator_all_states(self):
        state_names = self.experiment.get_all_state_names()
        for state_name in state_names:
            print(state_name)

        for state_name in state_names:
            ex = IPSExperiment(current_state_name=state_name)
            task = ex.generate_task_of_the_state()
            print(f"state: {state_name} -> task: {task.experiment_operation}")
            # print(f"state: {state_name} -> task: {task}")

    def test_transition_function_all_states(self):
        state_names = self.experiment.get_all_state_names()
        for state_name in state_names:
            print(state_name)

        for state_name in state_names:
            ex = IPSExperiment(current_state_name=state_name)
            next_state = ex.determine_next_state_name()
            print(f"current_state: {state_name} -> next_state: {next_state}")



class TestRunExperimentIPS(unittest.TestCase):

    def test_run_experiment(self):
        parent_dir_path = Path(f"volatile_{uuid.uuid4()}")
        
        # Make directory
        parent_dir_path.mkdir(parents=True, exist_ok=True)
        

        shared_variable_history = pl.DataFrame(
            {
                "time": [0],
                "density": [None],
                "operation": ["Passage"],
            }
        )
        experiment = IPSExperiment(current_state_name="GetImage1State", shared_variable_history=shared_variable_history)
        experiments = Experiments(
            experiments=[],
            parent_dir_path=parent_dir_path,
        )

        experiments.add_experiment(experiment)
        
        print(f"{experiments=}")
        experiments.execute_scheduling()
        print(f"{experiments=}")

        experiments.save_all()

        test_experiment_path = parent_dir_path / "experiments/IPSExperiment.json"
        print(f"{test_experiment_path=}")

        experiments.start_experiments()



        # Sleep 10 seconds
        import time
        time.sleep(10)

        # Remove directory
        import shutil
        shutil.rmtree(parent_dir_path)