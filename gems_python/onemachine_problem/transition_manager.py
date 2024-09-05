import textwrap
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Arrow
import networkx as nx
import ast
import inspect
from abc import ABC, abstractmethod
import copy
from dataclasses import asdict, dataclass, field
import json
from typing import List, Type, Union
import uuid
import numpy as np
import polars as pl
from pathlib import Path
import os

from gems_python.onemachine_problem.task_info import OneMachineTask, OneMachineTaskLocalInformation        

"""MODULE: State
"""


@dataclass
class State(ABC):
    """State class.
    This class is used as just an superclass for the State class.
    Defining "State", you should inherit this class and implement the methods.
    Note:
    When inheriting this class, you should implement the following methods:
    - transition_function
    - task_generator
    """

    state_name: str = field(init=False, default=None)

    def __post_init__(self):
        self.state_name = self.__class__.__name__

    def extract_all_state_transition_candidates(self):
        func = self.transition_function
        """
        Extracts the return values from the function.
        :param func: Function to extract the return values from.
        :return: List of return values.
        """
        class ReturnVisitor(ast.NodeVisitor):
            def __init__(self):
                self.returns = []

            def visit_Return(self, node):
                if isinstance(node.value, ast.Constant):
                    if isinstance(node.value.value, str):
                        self.returns.append(node.value.value)
                elif isinstance(node.value, ast.Name):
                    self.returns.append(f"val:{node.value.id}")
                self.generic_visit(node)

        # Get the source code of the function
        source = inspect.getsource(func)
        
        # Adjust the indentation
        source = textwrap.dedent(source)
        
        tree = ast.parse(source)
        visitor = ReturnVisitor()
        visitor.visit(tree)
        return visitor.returns
        tree = ast.parse(source)
        visitor = ReturnVisitor()
        visitor.visit(tree)
        return visitor.returns

    @abstractmethod
    def transition_function(self, df: pl.DataFrame) -> str:
        """
        This function must return one of the state name in the 'states', state list of the parent experiment
        """
        pass

    @abstractmethod
    def task_generator(self, df: pl.DataFrame) -> OneMachineTaskLocalInformation:
        pass

"""MODULE: Experiment
"""

@dataclass
class Experiment:
    """
    Experiment class.

    Attributes:
        experiment_name (str): The name of the experiment.
        states (List[State]): The states of the experiment.
        current_state_name (str): The name of the current state.
        shared_variable_history (pl.DataFrame): The shared variable history of the experiment.
        current_task (OneMachineTask) (You can manually assign it or Automatically generated): The current task of the experiment.
        current_state_index (int) (Automatically generated): The index of the current state.
        experiment_uuid (str) (Automatically generated): The UUID of the experiment.
    """
    experiment_name: str
    states: List[Type[State]]
    current_state_name: str
    shared_variable_history: pl.DataFrame
    current_task: OneMachineTask = field(default=None)
    current_state_index: int = field(default=None, init=False)
    experiment_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))


    def to_dict(self) -> dict:
        """
        Converts the experiment object to a dictionary.
        :return: Dictionary representation of the experiment object.
        """
        data = asdict(self)
        data['states'] = [state.__class__.__name__ for state in self.states]  # Store class names instead of class objects
        data['shared_variable_history'] = self.shared_variable_history.to_dict(as_series=False)
        
        return data

    @classmethod
    def from_dict(cls, data: dict) -> 'Experiment':
        """
        Creates an experiment object from a dictionary.
        :param data: Dictionary containing the experiment data.
        :return: Experiment object.
        """
        data['states'] = [globals()[state_name] for state_name in data['states']]
        data['shared_variable_history'] = pl.DataFrame(data['shared_variable_history'])
        return cls(**data)

    def to_json(self) -> str:
        """
        Converts the experiment object to a JSON string.
        :return: JSON string representation of the experiment object.
        """
        data = self.to_dict()
        print(f"{data=}")
        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> 'Experiment':
        """
        Creates an experiment object from a JSON string.
        :param json_str: JSON string containing the experiment data.
        :return: Experiment object.
        """
        data = json.loads(json_str)
        return cls.from_dict(data)
    
    def define_all_node(self)-> set:
        all_node = set()
        for state in self.states:
            all_node.add(state.state_name)
            next_state_names = state.extract_all_state_transition_candidates()
            for next_state_name in next_state_names:
                all_node.add(next_state_name)

        return all_node
        
        

    def show_experiment_with_tooltips(self, save_path: Path = "./experiment_with_tooltips.png", hide_nodes: List[str] = ["ExpireState"]):
        """
        Show the directed graph of the experiment with tooltips for each state.

        Each state will have a tooltip that shows the content of its transition_function and task_generator.
        """

        plt.figure(figsize=(20, 20))
        
        # Filter out hidden nodes
        visible_states = [state for state in self.states if state.state_name not in hide_nodes]

        node_position_radius = 0.2
        tooltip_position_radius = 0.4 + 0.05*len(visible_states)/6
        angle_offset = np.pi / 6
        
        # Define node positions manually or via some simple layout logic
        node_positions = {}
        num_states = len(visible_states)
        for i, state in enumerate(visible_states):
            angle = 2 * np.pi * i / num_states
            x = 0.5 + node_position_radius * np.cos(angle + angle_offset)
            y = 0.5 + node_position_radius * np.sin(angle + angle_offset)
            node_positions[state.state_name] = (x, y)

        # Draw edges (transitions)
        for state in visible_states:
            next_state_names = state.extract_all_state_transition_candidates()
            for next_state_name in next_state_names:
                if next_state_name not in hide_nodes:
                    src_pos = node_positions[state.state_name]
                    dst_pos = node_positions[next_state_name]
                    if state.state_name == next_state_name:  # Self-loop
                        arc = Arc([src_pos[0], src_pos[1]], width=0.1, height=0.05, angle=0, theta1=0, theta2=180, color='gray', linewidth = 5)
                        plt.gca().add_patch(arc)
                    else:
                        arrow = Arrow(src_pos[0], src_pos[1], dst_pos[0] - src_pos[0], dst_pos[1] - src_pos[1], width=0.05, color='gray')
                        plt.gca().add_patch(arrow)
            
        # Draw nodes (states) with tooltips
        for state in visible_states:
            pos = node_positions[state.state_name]
            node_color = 'orange' if state.state_name == self.current_state_name else 'skyblue'
            plt.text(pos[0], pos[1], state.state_name, ha='center', va='center', fontsize=10, bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor=node_color))
            
            # Tooltip content
            tooltip_text = f"transition_function:\n{inspect.getsource(state.transition_function)}\ntask_generator:\n{inspect.getsource(state.task_generator)}"
            
            # Calculate tooltip position proportionally to node_position_radius
            tooltip_x = 0.35 + tooltip_position_radius * np.cos(np.arctan2(pos[1] - 0.5, pos[0] - 0.5) + angle_offset)
            tooltip_y = 0.5 + tooltip_position_radius * np.sin(np.arctan2(pos[1] - 0.5, pos[0] - 0.5) + angle_offset)

            # Add annotation for the tooltip
            plt.annotate(
                tooltip_text,
                xy=pos,  # The point being annotated
                xytext=(tooltip_x, tooltip_y),  # The position of the text
                textcoords='data',
                bbox=dict(boxstyle="round,pad=0.3", edgecolor="black", facecolor="lightyellow"),
                arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0.2"),
                fontsize=8,
                ha='left',  # Left align the text
                va='center'  # Align vertically to the top
            )

        # Add message for hidden nodes
        if hide_nodes and len(hide_nodes) > 0:
            plt.text(0.05, 0.95, f'Nodes {hide_nodes} are hidden', transform=plt.gca().transAxes, 
                    fontsize=8, verticalalignment='top', bbox=dict(facecolor='red', alpha=0.5))

        plt.xlim(0, 1)
        plt.ylim(0, 1)
        plt.axis('off')  # Hide the axes
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()
    
    def show_experiment_directed_graph(self, hide_nodes: List[str] = ["ExpireState"]):
        """
        Show the directed graph of the experiment.
        """
        G = nx.DiGraph()
        all_node = self.define_all_node()

        # Filter hide_nodes to include only those that are in all_node
        if hide_nodes is not None:
            hide_nodes = [node for node in hide_nodes if node in all_node]

        for node in all_node:
            if hide_nodes is None or node not in hide_nodes:
                G.add_node(node)
        
        for state in self.states:
            next_state_names = state.extract_all_state_transition_candidates()
            for next_state_name in next_state_names:
                if (hide_nodes is None or 
                    (state.state_name not in hide_nodes and next_state_name not in hide_nodes)):
                    G.add_edge(state.state_name, next_state_name)

        highlight_nodes = [self.current_state_name]

        # Draw the graph
        plt.figure(figsize=(8, 10))  # 画像サイズを指定
        pos = nx.spring_layout(G, seed=100, k=0.5)
        node_color = ["skyblue" if node not in highlight_nodes else "orange" for node in G.nodes()]
        node_size = [1000 if node not in highlight_nodes else 1500 for node in G.nodes()]

        nx.draw(G, pos, with_labels=True, node_size=node_size, node_color=node_color, font_size=10, font_weight="bold", edge_color="gray", width=1.0)

        # Add message for hidden nodes
        if hide_nodes and len(hide_nodes) > 0:
            plt.text(0.05, 0.95, f'Nodes {hide_nodes} are hidden', transform=plt.gca().transAxes, 
                    fontsize=8, verticalalignment='top', bbox=dict(facecolor='red', alpha=0.5))

        plt.show()

    
    def save_all(self, save_dir: Path = None):
        """
        Save the experiment and its states.
        """
        save_path = save_dir if save_dir is not None else Path("experiments")
        os.makedirs(save_path, exist_ok=True)
        with open(save_path / f"{self.experiment_name}.json", "w") as f:
            f.write(self.to_json())
        
        # for state in self.states:
        #     state.save_all(save_path)

    def __post_init__(self):
        """
        Initialize the experiment.
        """
        self.update_current_state_name_and_index(self.current_state_name)
        if self.current_task is None:
            self.current_task = self.generate_task_of_the_state()

        # if self.shared_variable_history != pl.DataFrame:
        #     pl.read_csv(self.shared_variable_history)

    def update_current_state_name_and_index(self, new_state_name: str):
        self.current_state_index = self.get_current_state_index_from_input_state_name(new_state_name)
        self.current_state_name = new_state_name


    def get_current_state_index_from_current_state_name(self):
        
        current_state_index = self.get_current_state_index_from_input_state_name(self.current_state_name)
        return current_state_index

    def get_current_state_index_from_input_state_name(self, input_state_name: str):
        current_state_index = -1
        for index in range(len(self.states)):
            if self.states[index].state_name == input_state_name:
                current_state_index = index
        
        if current_state_index == -1:
            RuntimeError(f"There is no state named {input_state_name}")
        
        return current_state_index
        

    def show_experiment_name_and_state_names(self):
        print(f"Experiment name: {self.experiment_name}")
        print("State names:")
        for state in self.states:
            print(f"  - {state.state_name}")

    def show_current_state_name(self):
        print(f"Current state: {self.current_state_name}")

    def get_current_state_name(self) -> str:
        return self.current_state_name
    
    def get_all_state_names(self) -> List[str]:
        return [state.state_name for state in self.states]

    def execute_one_step(self) -> OneMachineTask:
        """
        Execute one step. Determine the next state index and generate a task.
        Note: The shared variable history had to be updated
        Note: The current state index will be updated.
        Note: The task is generated by the task generator of the next state.
        :return: Generated task.
        """
        # Determine the next state index
        try:
            new_state_name: str = self.determine_next_state_name()
        except Exception as err:
            raise RuntimeError(f"Error determining the next state index: {err}")

        # Update the current state index
        self.update_current_state_name_and_index(new_state_name)
        
        # Generate a task
        try:
            task = self.generate_task_of_the_state()
        except Exception as err:
            raise RuntimeError(f"Error generating task: {err}")

        return task
    
    def generate_task_of_the_state(self) -> OneMachineTask:
        """
        Generate a task of the state.
        :return: Generated task.
        """
        # Generate a task
        state_index = self.current_state_index
        try:
            task_local_information = self.states[state_index].task_generator(self.shared_variable_history.clone())
            task = OneMachineTask(
                optimal_time=task_local_information.optimal_time,
                processing_time=task_local_information.processing_time,
                penalty_type=task_local_information.penalty_type,
                experiment_operation=task_local_information.experiment_operation,
                experiment_name=self.experiment_name,
                experiment_uuid=self.experiment_uuid,
            )
        except Exception as err:
            raise RuntimeError(f"Error generating task: {err}")

        return task
    
    def determine_next_state_name(self) -> str:
        """
        Determine the next state index.
        :return: Next state index.
        """
        # Determine the next state index
        try:
            next_state_name: str = self.states[self.current_state_index].transition_function(self.shared_variable_history)
        except Exception as err:
            raise RuntimeError(f"Error state transition: {err}")

        return next_state_name
 


"""MODULE: Experiments
"""

@dataclass
class Experiments:
    
    """
    Experiments class.
    This class is used as a data class for the experiments.
    """

    experiments: List[Experiment]
    parent_dir_path: Path
    # Automatically generated fields, not accept user input
    tasks: List[OneMachineTask] = field(default=None, init=False)

    def __post_init__(self):
        """
        Initialize the experiments.
        """
        if self.tasks is None:
            self.tasks = list()
            for experiment in self.experiments:
                self.tasks.append(copy.deepcopy(experiment.current_task))

            for index in range(len(self.tasks)):
                self.tasks[index].task_id = index

    def add_experiment(self, experiment: Experiment) -> Union[None, ValueError]:

        # Check the duplication of the experiment name and uuid
        for e in self.experiments:
            if e.experiment_name == experiment.experiment_name and e.experiment_uuid == experiment.experiment_uuid:
                raise ValueError(f"Experiment name and uuid must be unique: {experiment.experiment_name}, {experiment.experiment_uuid}")
            
        self.experiments.append(experiment)
        self.tasks.append(copy.deepcopy(experiment.current_task))
        self.tasks[-1].task_id = len(self.tasks) - 1


    def delete_experiment_with_experiment_uuid(self, experiment_uuid: str) -> Union[None, ValueError]:
        """
        Delete the experiment with the input experiment_uuid.
        """
        # Check the experiment exists
        is_exist = False
        for experiment in self.experiments:
            if experiment.experiment_uuid == experiment_uuid:
                is_exist = True
                break

        if not is_exist:
            raise ValueError(f"Experiment with uuid {experiment_uuid} does not exist")

        new_experiments = list()
        for experiment in self.experiments:
            if experiment.experiment_uuid != experiment_uuid:
                new_experiments.append(copy.deepcopy(experiment))
        
        self.experiments = new_experiments

    def list(self):
        """
        List the experiment names and uuids.
        """
        for experiment in self.experiments:
            print(f"Experiment name: {experiment.experiment_name}, Experiment uuid: {experiment.experiment_uuid}")

    def save_all(self, save_dir: Path = None, under_parent_dir: bool = True):
        if under_parent_dir:
            save_path = self.parent_dir_path / (save_dir if save_dir is not None else "")
        else:
            save_path = save_dir if save_dir is not None else self.parent_dir_path

        os.makedirs(save_path, exist_ok=True)
        tasks = [task.to_json() for task in self.tasks]
        print(f"{tasks=}")
        for task in self.tasks:
            print(f"{task.to_json()=}")
        with open(save_path / "tasks.json", "w") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        
        experiments_dir = save_path / "experiments"
        os.makedirs(experiments_dir, exist_ok=True)
        for experiment in self.experiments:
            experiment.save_all(experiments_dir)


    def assign_task_ids(self):
        """
        Overwrite the task ids in tasks list with identical index
        """
        for index in range(len(self.tasks)):
            self.tasks[index].task_id = index


    # TODO: スケジュール→実行→結果の受取→状態の更新→次のTaskを出力→スケジュール…(もっと早くできる)
    def delete_task_with_task_id(self, task_id: int):
        new_tasks = list()
        for task in range(len(self.tasks)):
            if self.tasks[task].task_id != task_id:
                new_tasks.append(copy.deepcopy(self.tasks[task]))

    def execute_scheduling(
            self,
            scheduling_method: str = 's',
            optimal_time_reference_time: int = 0
            ):
        """
        Execute the scheduling of the tasks.
        :param scheduling_method: The method of scheduling. 's' for simulated annealing.
        :param optimal_time_reference_time: The reference time for the optimal time.
        """

        self.assign_task_ids()
        # Reschedule
        tasks = self.tasks.copy()

        for task in tasks:
            task.scheduled_time = task.optimal_time - optimal_time_reference_time

        match scheduling_method:
            case 's':
                self.tasks = OneMachineTask.simulated_annealing_schedule(tasks)
            case _:
                AssertionError(f"Unexpected input: scheduling_method {scheduling_method}")

    def update_shared_variable_history_and_states_and_generate_task_and_reschedule(
            self,
            task_id: int,
            new_result_of_experiment: pl.DataFrame,
            update_type: str = 'a',
            scheduling_method = 's',
            optimal_time_reference_time: int = 0
            ) -> OneMachineTask:
        """
        TODO: explanation
        """
        experiment_uuid = 0
        for index in range(len(self.tasks)):
            if self.tasks[index].task_id == task_id:
                experiment_uuid = self.tasks[index].experiment_uuid
                break

        experiment_index = 0
        for index in range(len(self.experiments)):
            if self.experiments[index].experiment_uuid == experiment_uuid:
                experiment_index = index

        # Clean up the last task
        self.delete_task_with_task_id(task_id)

        # Update the shared variable history(dataframe) of the experiment of the task
        match update_type:
            case 'a':
                # Append the result to the shared variable history(dataframe)
                self.experiments[experiment_index].shared_variable_history = pl.concat([self.experiments[experiment_index].shared_variable_history, new_result_of_experiment], how="diagonal")
            case 'o':
                # Overwrite the shared variable history(dataframe) with new_result_of_experiment
                self.experiments[experiment_index].shared_variable_history = new_result_of_experiment
            case _:
                AssertionError(f"Unexpected input: update_type {update_type}")

        # Generate a new task of the updated experiment
        new_task = self.experiments[experiment_index].execute_one_step()
        
        self.tasks.append(new_task)
        self.assign_task_ids()
        
        self.execute_scheduling(scheduling_method, optimal_time_reference_time)

        return OneMachineTask.get_earliest_scheduled_task(self.tasks)
    
    def start_experiments(self):
        pass

