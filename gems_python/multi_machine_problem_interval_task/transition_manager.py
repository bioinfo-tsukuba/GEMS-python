from gems_python.common.class_dumper import SerializationMixin
from gems_python.common.persistence import load_snapshot, save_snapshot
from datetime import datetime
import textwrap
from matplotlib import pyplot as plt
from matplotlib.patches import Arc, Arrow
import networkx as nx
import ast
import inspect
from abc import ABC, abstractmethod
import copy
from dataclasses import field
import json
from typing import Callable, List, Tuple, Union
import uuid
import numpy as np
from gems_python.common import dataframes
from gems_python.common.dataframes import DataFrame
from pathlib import Path
import os
from gems_python.common.cp_sat_scheduler import schedule_task_groups
from gems_python.common.scheduling import DEFAULT_SOLVER_TIME_LIMIT_SECONDS, validate_time_limit
from gems_python.common.class_dumper import auto_dataclass as dataclass

from gems_python.multi_machine_problem_interval_task.task_info import Machine, MachineList, Task, TaskGroup
from gems_python.multi_machine_problem_interval_task.lifecycle import (
    LifecycleManager, FEATURE_VERSION, fingerprint,
)

"""MODULE: State
"""


class State(SerializationMixin, ABC):
    """State class.
    This class is used as just an superclass for the State class.
    Defining "State", you should inherit this class and implement the methods.
    Note:
    When inheriting this class, you should implement the following methods:
    - transition_function
    - task_generator
    """

    def __init__(self):
        print(f"{self.__class__.__module__=}.{self.__class__.__name__=}")
        pass

    @property
    def state_name(self) -> str:
        return str(self.__class__.__name__)

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
    

    @abstractmethod
    def task_generator(self, df: DataFrame) -> TaskGroup:
        pass

    @abstractmethod
    def transition_function(self, df: DataFrame) -> str:
        """
        This function must return one of the state name in the 'states', state list of the parent experiment
        """
        pass


    def dummy_output(self, df: DataFrame, task_group_id: int, task_id: int) -> DataFrame:
        """
        Return a dummy (simulated) result as a Pandas or Polars DataFrame.
        Override in each State subclass to provide the columns/values your transition_function expects.
        The default raises, so you notice missing implementations at simulation time.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.dummy_output(df, task_group_id, task_id) is not implemented."
        )

    def lifecycle_action(self, history, context):
        """Return Finish, Split or None at a completed group boundary."""
        return None


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
        shared_variable_history (DataFrame): The shared variable history of the experiment.
        current_task (TaskGroup) (You can manually assign it or Automatically generated): The current task of the experiment.
        current_state_index (int) (Automatically generated): The index of the current state.
        experiment_uuid (str) (Automatically generated): The UUID of the experiment.
    """
    experiment_name: str
    states: List[State]
    current_state_name: str
    shared_variable_history: DataFrame = field(default=None)
    current_task_group: TaskGroup = field(default=None)
    current_state_index: int = field(default=None, init=False)
    experiment_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    dataframe_backend: str = "auto"
    lifecycle_version: int = FEATURE_VERSION
    lifecycle_status: str = "ACTIVE"
    hold_reason: str | None = None
    parent_uuid: str | None = None
    _inherited_history: DataFrame | None = None
    initial_conditions: dict = field(default_factory=dict)
    initial_condition_origins: dict = field(default_factory=dict)
    task_group_sequence: int = 0

    def history_for_state(self):
        """Parent provenance followed by this experiment's own observations."""
        if self._inherited_history is None:
            return dataframes.copy_frame(self.shared_variable_history)
        inherited = dataframes.convert(self._inherited_history, self.dataframe_backend)
        return dataframes.append(inherited, self.shared_variable_history)

    @property
    def inherited_history(self):
        return (dataframes.make_frame(backend=self.dataframe_backend)
                if self._inherited_history is None else
                dataframes.convert(self._inherited_history, self.dataframe_backend))
    
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
    
    def show_experiment_directed_graph(self, save_path: Path = "./experiment.png", hide_nodes: List[str] = ["ExpireState"]):
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

        plt.axis('off')  # 軸を非表示に
        if save_path is not None:
            plt.savefig(save_path)
        else:
            plt.show()

    
    def save_all(self, save_dir: Path = None):
        """
        Save the experiment and its states.
        """
        save_path = save_dir if save_dir is not None else Path("experiments")
        os.makedirs(save_path, exist_ok=True)
        with open(save_path / f"{self.experiment_name}_{self.experiment_uuid}.json", "w") as f:
            f.write(self.to_json())

        # Save shared_variable_history
        dataframes.write_csv(self.shared_variable_history, save_path / f"{self.experiment_name}_{self.experiment_uuid}_shared_variable_history.csv")
        
        # for state in self.states:
        #     state.save_all(save_path)

    def __post_init__(self):
        """
        Initialize the experiment.
        """
        self.experiment_uuid = str(self.experiment_uuid)
        self.dataframe_backend = dataframes.resolve_backend(self.dataframe_backend, self.shared_variable_history)
        self.shared_variable_history = (
            dataframes.make_frame(backend=self.dataframe_backend)
            if self.shared_variable_history is None
            else dataframes.convert(self.shared_variable_history, self.dataframe_backend)
        )
        self.update_current_state_name_and_index(self.current_state_name)
        if self.current_task_group is None:
            self.current_task_group = self.generate_task_group_of_the_state()
        else:
            self.current_task_group.configure_task_group_settings(
                self.experiment_name, self.experiment_uuid)

        # if self.shared_variable_history != DataFrame:
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
            raise ValueError(f"There is no state named {input_state_name}")
        
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
    
    def update_task_group(self, new_task_group: TaskGroup):
        self.current_task_group = new_task_group

    def execute_one_step(self) -> TaskGroup:
        # TODO-DONE: TaskGroupに対応
        """
        Execute one step of the experiment.
        :return: Task 
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
            task_group = self.generate_task_group_of_the_state()
        except Exception as err:
            raise RuntimeError(f"Error generating task_group: {err}")

        return task_group
    
    def generate_task_group_of_the_state(self) -> TaskGroup:
        # TODO-DONE: TaskGroupに対応
        """
        Generate a task group of the current state.
        :return: Generated task group.
        """
        # Generate a task
        state_index = self.current_state_index
        try:
            task_group: TaskGroup = self.states[state_index].task_generator(self.history_for_state())
            task_group.task_group_uuid = str(uuid.uuid5(
                uuid.NAMESPACE_URL, f"gems:group:{self.experiment_uuid}:{self.task_group_sequence}"))
            self.task_group_sequence += 1
            self.current_task_group = task_group
            self.current_task_group.configure_task_group_settings(
                experiment_name = self.experiment_name,
                experiment_uuid = self.experiment_uuid
            )

        except Exception as err:
            raise RuntimeError(f"Error generating task: {err}")

        return task_group
    
    def determine_next_state_name(self) -> str:
        """
        Determine the next state index.
        :return: Next state index.
        """
        # Determine the next state index
        try:
            next_state_name: str = self.states[self.current_state_index].transition_function(self.history_for_state())
        except Exception as err:
            raise RuntimeError(f"Error state transition: {err}")

        return next_state_name
 


"""MODULE: Experiments
"""

@dataclass
class Experiments(LifecycleManager):
    
    """
    Experiments class.
    This class is used as a data class for the experiments.
    """

    experiments: List[Experiment] = field(default_factory=list)
    parent_dir_path: Path = field(default=Path("experiments_dir"))
    machine_list: MachineList = field(default_factory=MachineList)
    # Automatically generated fields, not accept user input
    task_groups: List[TaskGroup] = field(default=None)
    step: int = field(default=0)
    reference_time: int = field(default=0)
    solver_time_limit_seconds: float = DEFAULT_SOLVER_TIME_LIMIT_SECONDS
    lifecycle_version: int = FEATURE_VERSION
    manager_status: str = "RUNNING"
    stop_diagnostic: dict | None = None
    unknown_policy: str = "retry"
    max_unknown_retries: int = 3
    enable_split: bool = False
    planning_progress: object = None
    ended_experiments: dict = field(default_factory=dict)
    finish_requests: dict = field(default_factory=dict)
    split_requests: dict = field(default_factory=dict)
    completed_results: dict = field(default_factory=dict)
    handled_groups: dict = field(default_factory=dict)

    def __post_init__(self):
        self.solver_time_limit_seconds = validate_time_limit(self.solver_time_limit_seconds)
        self._validate_policy()
        self.parent_dir_path = Path(self.parent_dir_path)
        # TODO: TaskGroupに対応
        if self.parent_dir_path.exists():
            if not self.parent_dir_path.is_dir():
                raise ValueError(f"parent_dir_path must be a directory: {self.parent_dir_path}")
        else:
            os.makedirs(self.parent_dir_path, exist_ok=True)

        """
        Initialize the experiments.
        """
        if self.task_groups is None:
            self.task_groups = list()
            for experiment in self.experiments:
                self.task_groups.append(copy.deepcopy(experiment.current_task_group))

            self.set_task_group_ids()

    def reload(self, step: int = None, *, source: str = "auto"):
        """Return a saved step without changing this collection, even on failure."""
        if step is None:
            steps = [int(d.name[5:]) for d in self.parent_dir_path.iterdir()
                     if d.is_dir() and d.name.startswith("step_") and d.name[5:].isdigit()]
            if not steps:
                raise ValueError("No saved step directories were found.")
            step = max(steps)
        directory = self.parent_dir_path / f"step_{step:08d}"
        return load_snapshot(directory, type(self), source=source)

    def save_dir(self):
        return self.parent_dir_path / f"step_{str(self.step).zfill(8)}"

    def current_save_dir(self):
        return self.parent_dir_path / f"step_current"

    def set_reference_time(self, reference_time: int):
        # Confirm the reference time is int type
        if not isinstance(reference_time, int):
            print(f"reference_time must be int type: {reference_time}")
        elif reference_time < 0:
            print(f"reference_time must be non-negative: {reference_time}")
        else:
            self.reference_time = reference_time

    def add_experiment(self, experiment: Experiment, *,
                       solver_time_limit_seconds: float | None = None,
                       save_step: bool = True) -> Union[None, ValueError]:

        # Check the duplication of the experiment name and uuid
        solver_time_limit_seconds = validate_time_limit(
            self.solver_time_limit_seconds if solver_time_limit_seconds is None
            else solver_time_limit_seconds)
        for e in self.experiments:
            if e.experiment_uuid == experiment.experiment_uuid:
                raise ValueError(f"Experiment name and uuid must be unique: {experiment.experiment_name}, {experiment.experiment_uuid}")
        if experiment.experiment_uuid in self.ended_experiments:
            raise ValueError(f"Experiment UUID already ended: {experiment.experiment_uuid}")
            
        self.experiments.append(experiment)
        self.task_groups.append(copy.deepcopy(experiment.current_task_group))
        self.set_task_group_ids()
        self.execute_scheduling(solver_time_limit_seconds=solver_time_limit_seconds)
        if save_step:
            self.proceed_to_next_step()


    def delete_experiment_with_experiment_uuid(self, experiment_uuid: str) -> Union[None, ValueError]:
        # TODO: TaskGroupに対応
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

        new_task_groups = list()
        for task_group in self.task_groups:
            if task_group.experiment_uuid != experiment_uuid:
                new_task_groups.append(copy.deepcopy(task_group))

        self.task_groups = new_task_groups

        self.set_task_group_ids()
        self.execute_scheduling()

        self.proceed_to_next_step()
        

    def list(self):
        """
        List the experiment names and uuids.
        """
        for experiment in self.experiments:
            print(f"Experiment name: {experiment.experiment_name}, Experiment uuid: {experiment.experiment_uuid}\n{experiment}")

    def add_machine(self, machine: Union[Machine, None] = None,
                    machine_type: Union[int, None] = None,
                    description: Union[str, None] = "",
                    ) -> Union[None, ValueError]:
        """
        Add a machine to the machine list.
        """
        if machine is not None:
            pass
        elif machine_type is not None:
            description += f" Added at {datetime.now().astimezone().isoformat()}"
            machine = Machine(machine_type=machine_type, description=description)
        else:
            raise ValueError("Machine or machine_type and machine_name must be specified.")
        print(f"{machine=}")
        self.machine_list.add_machine(machine)
        print(f"{self.machine_list=}")
        self.set_task_group_ids()
        print(f"{self.task_groups=}")
        self.execute_scheduling()
        print(f"{self.task_groups=}")
        self.proceed_to_next_step()

    def delete_machine_with_machine_id(self, machine_id: str) -> Union[None, ValueError]:
        """
        Delete the machine with the input machine_id.
        """
        self.machine_list.delete_machine_with_machine_id(machine_id)
        self.set_task_group_ids()
        self.execute_scheduling()

        self.proceed_to_next_step()


    def show_machines(self):
        """
        Show the machine names and uuids.
        """
        print(f"{"*" * 10} Machines {"*" * 10}")        
        for machine in self.machine_list.machines:
            print(machine)
        print(f"{"*" * 30}")


    # @deprecated("Use to_json instead")
    def save_all(self, save_dir: Path = None, under_parent_dir: bool = True):
        if under_parent_dir:
            save_path = self.parent_dir_path / (save_dir if save_dir is not None else "")
        else:
            save_path = save_dir if save_dir is not None else self.parent_dir_path

        os.makedirs(save_path, exist_ok=True)
        save_snapshot(self, save_path)
        # TODO: TaskGroupに対応
        tasks = [task.to_json() for task in self.task_groups]
        # print(f"{tasks=}")
        # for task in self.tasks:
        #     print(f"{task.to_json()=}")
        with open(save_path / "tasks.json", "w") as f:
            json.dump(tasks, f, ensure_ascii=False, indent=2)
        
        experiments_dir = save_path / "experiments"
        os.makedirs(experiments_dir, exist_ok=True)
        for experiment in self.experiments:
            experiment.save_all(experiments_dir)


    def set_task_group_ids(self):
        self.task_groups = TaskGroup.set_task_group_ids(self.task_groups)

    def delete_task_with_task_id(self, task_id: int):
        new_tasks = list()
        for task in range(len(self.task_groups)):
            if self.task_groups[task].task_group_id != task_id:
                new_tasks.append(copy.deepcopy(self.task_groups[task]))

        self.task_groups = new_tasks

    def execute_scheduling(
            self,
            scheduling_method: str = 'o',
            reference_time: int = None,
            *, solver_time_limit_seconds: float | None = None,
            ):
        return self.plan_tasks(scheduling_method, reference_time,
                               solver_time_limit_seconds=solver_time_limit_seconds)

    def update_shared_variable_history(
            self,
            task_group_id: int,
            task_id: int,
            new_result_of_experiment: DataFrame,
            update_type: str = 'a',
            *, task_group_uuid: str | None = None, result_id: str | None = None,
            ) -> Tuple[TaskGroup, Task]:
        """
        TODO: explanation
        """
        if result_id is not None:
            if not task_group_uuid:
                raise ValueError("result_id requires a persistent task_group_uuid")
            result_hash = fingerprint((task_group_uuid, task_id,
                                       new_result_of_experiment, update_type))
            if result_id in self.completed_results:
                if self.completed_results[result_id]["fingerprint"] != result_hash:
                    raise ValueError(f"Conflicting result ID: {result_id}")
                return
        group = self._resolve_group(task_group_id, task_group_uuid)
        experiment = self._experiment(group.experiment_uuid)
        task = next(t for t in group.tasks if t.task_id == task_id)
        if task.completed:
            raise ValueError("Task already completed; resend with the original result_id")
        if self.manager_status == "STOPPED" and task.task_status.name != "IN_PROGRESS":
            raise RuntimeError("STOPPED manager only accepts results of in-progress tasks")
        new_history = dataframes.update(experiment.shared_variable_history,
                                        new_result_of_experiment, update_type)
        TaskGroup.complete_task(self.task_groups, group.task_group_id, task_id)
        experiment.shared_variable_history = new_history
        if result_id is not None:
            self.completed_results[result_id] = {
                "fingerprint": result_hash,
                "backend": dataframes.backend_of(new_result_of_experiment),
                "task_group_uuid": group.task_group_uuid, "task_id": task_id,
            }
        if group.is_completed() and self.manager_status != "STOPPED":
            try:
                self._complete_group(group)
            except Exception as err:
                raise self._stop("LIFECYCLE_ERROR", err) from err
        self.set_task_group_ids()

    def update_shared_variable_history_and_states_and_generate_task_and_reschedule(
            self,
            task_group_id: int,
            task_id: int,
            new_result_of_experiment: DataFrame,
            update_type: str = 'a',
            scheduling_method = 'o',
            optimal_time_reference_time: int = None,
            *, solver_time_limit_seconds: float | None = None,
            task_group_uuid: str | None = None, result_id: str | None = None,
            ) -> Tuple[TaskGroup, Task]:
        """
        TODO: explanation
        """
        solver_time_limit_seconds = validate_time_limit(
            self.solver_time_limit_seconds if solver_time_limit_seconds is None
            else solver_time_limit_seconds)
        # TODO-DONE: TaskGroupに対応
        # Update task_group

        if result_id is not None and result_id in self.completed_results:
            # Check identity before resolving a group which may have been deleted.
            digest = fingerprint((task_group_uuid, task_id, new_result_of_experiment, update_type))
            if digest != self.completed_results[result_id]["fingerprint"]:
                raise ValueError(f"Conflicting result ID: {result_id}")
            return None, None
        if optimal_time_reference_time is not None:
            self.reference_time = optimal_time_reference_time
        self.update_shared_variable_history(task_group_id, task_id, new_result_of_experiment, update_type,
                                            task_group_uuid=task_group_uuid, result_id=result_id)
        if self.manager_status == "STOPPED":
            return None, None
        self.set_task_group_ids()
        
        self.execute_scheduling(scheduling_method, solver_time_limit_seconds=solver_time_limit_seconds)
        print(f"{self.task_groups=}")

        earliest_task, eariest_group_id = TaskGroup.get_ealiest_task_in_task_groups(self.runnable_groups())
        if earliest_task is None:
            return None, None
        print(f"{earliest_task=}")
        print(f"{eariest_group_id=}")
        eariest_group_index = TaskGroup.find_task_group(self.task_groups, eariest_group_id)
        earliest_task_group = self.task_groups[eariest_group_index]


        return earliest_task_group, earliest_task
    

    def start_experiments(self):
        """
        Start the experiments.
        """
        mode = "autoload"

    def generate_gantt_chart(self, save_dir: Path = None):
        if save_dir is None:
            save_dir = self.save_dir()
        TaskGroup.generate_gantt_chart(self.runnable_groups(), save_dir = save_dir)

    def save_results(self, save_dir: Path = None):
        if save_dir is None:
            save_dir = self.save_dir()

        os.makedirs(save_dir, exist_ok=True)
        save_snapshot(self, save_dir)

        date_and_time = datetime.now().astimezone().isoformat()
        with open(save_dir / "date_and_time.txt", "w") as f:
            f.write(date_and_time)

        # Save the task groups
        schedule_df = TaskGroup.create_non_completed_tasks_df(self.runnable_groups())
        dataframes.write_csv(schedule_df, save_dir / "schedule.csv")

        # Save the gantt chart
        self.generate_gantt_chart(save_dir=save_dir)

        # Save each experiment
        experiments_dir = save_dir / "experiments"
        os.makedirs(experiments_dir, exist_ok=True)
        for experiment in self.experiments:
            experiment.show_experiment_directed_graph(save_path=experiments_dir / f"{experiment.experiment_name}_{experiment.experiment_uuid}.png")
            dataframes.write_csv(experiment.shared_variable_history, experiments_dir / f"{experiment.experiment_name}_{experiment.experiment_uuid}_shared_variable_history.csv")

    def proceed_to_next_step(self):
        self.step += 1
        next_step_dir = self.save_dir()
        current_step_dir = self.current_save_dir()
        # Once, delete current_step_dir
        if not next_step_dir.exists():
            os.makedirs(next_step_dir, exist_ok=True)
            print(f"Next step directory created: {next_step_dir}")

        else:
            print(f"Next step directory already exists: {next_step_dir}")
            print(f"Overwriting the existing directory: {next_step_dir}")

        # Save the results
        self.save_results(save_dir=next_step_dir)
        self.save_results(save_dir=current_step_dir)

        # Save the path of the current step in the current directory
        with open(current_step_dir / "current_step_dir_path.txt", "w") as f:
            f.write(str(next_step_dir.absolute()))

            

    def auto_load(self):
        """
        Automatically load an experiment result.
        Expected files:
        - experiment_result.json
        - result.csv

        Normally, 'Task' progresses like NOT_STARTED = "Not Started" -> "In Progress" -> "Completed"

        Example of
        experiment_result.json:
        if task is successfully started
        {
            "task_response": "In Progress",
            "task_group_id": 0,
            "task_id": 0,
            "schedule_reference_time": 0 (Updated time, normally current time)
        }
        Note: "schedule_reference_time" can be replaced with "optimal_time_reference_time" for backward compatibility.

        else if task is successfully completed
        {
            "task_response": "Completed",
            "task_group_id": 0,
            "task_id": 0,
            "schedule_reference_time": 0,
            "result_path": "result.csv"
        }

        else:
        {
            "task_response": "Error",
            "task_group_id": 0,
            "task_id": 0,
            "schedule_reference_time": 0
        }
        """
        # TODO-DONE: result typeの確認
        # 
        # Check that results are available
        result = self.save_dir() / "experiment_result.json"
        if not result.exists():
            print(f"Experiment result not found: {result}")
            return
        else:
            print(f"Experiment result found: {result}")
            # Load the result
            with open(result, "r") as f:
                result_data:dict = json.load(f)
            
            # Following fields are optional, add default values if not found
            task_response = result_data.get("task_response")
            update_type = result_data.get("update_type", "a")
            scheduling_method = result_data.get("scheduling_method", "o")
            task_group_id = result_data.get("task_group_id")
            task_group_uuid = result_data.get("task_group_uuid")
            result_id = result_data.get("result_id")
            task_id = result_data["task_id"]
            optimal_time_reference_time = result_data.get("schedule_reference_time")
            if optimal_time_reference_time is None:
                optimal_time_reference_time = result_data.get("optimal_time_reference_time")
            match task_response:
                case "In Progress":
                    # TODO-DONE 適切な処理
                    try:
                        self.start_task(task_group_id, task_id, task_group_uuid=task_group_uuid)
                    except Exception as err:
                        print(f"Error starting task: {err}")
                case "Completed":
                    duplicate = result_id is not None and result_id in self.completed_results
                    if duplicate:
                        backend = self.completed_results[result_id]["backend"]
                    else:
                        group = self._resolve_group(task_group_id, task_group_uuid)
                        backend = self._experiment(group.experiment_uuid).dataframe_backend
                    new_result_of_experiment = dataframes.read_csv(
                        result_data["result_path"], backend
                    )
                    self.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
                        task_group_id=task_group_id,
                        task_id=task_id,
                        new_result_of_experiment=new_result_of_experiment,
                        update_type=update_type,
                        scheduling_method=scheduling_method,
                        optimal_time_reference_time=optimal_time_reference_time,
                        task_group_uuid=task_group_uuid, result_id=result_id,
                    )
                    if duplicate:
                        return
                case "Error":
                    # TODO: 適切な処理
                    print(f"Task failed: {task_group_id}, {task_id}")
                    # Reschedule
                    self.execute_scheduling(scheduling_method, reference_time=optimal_time_reference_time)
                case _:
                    print(f"Unexpected input: task_response {task_response}")
                    self.execute_scheduling(scheduling_method, reference_time=optimal_time_reference_time)
            self.proceed_to_next_step()
            
    
    def experiments_loop(self):
        """
        Start the experiments.
        """
        mode = "autoload"

        while True:
            if mode == "autoload":
                pass
            elif mode == "add":
                pass
            elif mode == "delete":
                pass
            elif mode == "exit":
                break
            else:
                pass

    # ========================= 
    # Simulation (dummy outputs) 
    # ========================= 
    def simulate_one(
        self,
        scheduling_method: str = "o",
        optimal_time_reference_time: int | None = None,
        save_each_step: bool = False,
        observation_provider: Callable | None = None,
        *, solver_time_limit_seconds: float | None = None,
    ) -> dict:
        solver_time_limit_seconds = validate_time_limit(
            self.solver_time_limit_seconds if solver_time_limit_seconds is None
            else solver_time_limit_seconds)
        if optimal_time_reference_time is not None:
            self.reference_time = optimal_time_reference_time

        self.set_task_group_ids()
        self.execute_scheduling(scheduling_method=scheduling_method, reference_time=self.reference_time,
                                solver_time_limit_seconds=solver_time_limit_seconds)

        if self.manager_status == "STOPPED":
            return {"status": "stopped", "diagnostic": self.stop_diagnostic}
        if not self.runnable_groups():
            return {"status": "held" if any(e.lifecycle_status.startswith("WAITING") or
                                            e.lifecycle_status == "SPLIT_PENDING"
                                            for e in self.experiments) else "no_tasks"}

        try:
            group, earliest_task = min(
                (
                    (group, task)
                    for group in self.runnable_groups()
                    for task in group.tasks if not task.completed
                ),
                key=lambda item: (
                    item[1].scheduled_time + item[1].processing_time,
                    item[1].scheduled_time,
                    item[0].task_group_id,
                    item[1].task_id,
                ),
            )
            earliest_group_id = group.task_group_id
        except Exception as err:
            return {"status": "error", "error": f"failed_to_pick_task: {err!r}"}

        if earliest_task is None:
            return {"status": "no_tasks"}

        group_index = TaskGroup.find_task_group(self.task_groups, earliest_group_id)
        tg = self.task_groups[group_index]

        # 対応する Experiment を見つける
        exp_index = next(i for i, e in enumerate(self.experiments) if e.experiment_uuid == tg.experiment_uuid)
        exp = self.experiments[exp_index]

        # 現在の State からダミー出力を生成
        state_index = exp.get_current_state_index_from_current_state_name()
        state = exp.states[state_index]
        before_state_name = exp.current_state_name
        event_time = max(
            self.reference_time,
            earliest_task.scheduled_time + earliest_task.processing_time,
        )
        try:
            if observation_provider is None:
                dummy_df = state.dummy_output(
                    exp.history_for_state(),
                    task_group_id=earliest_group_id,
                    task_id=earliest_task.task_id,
                )
            else:
                dummy_df = observation_provider(
                    state, exp.history_for_state(), tg,
                    earliest_task, event_time,
                )
            if not dataframes.is_dataframe(dummy_df):
                raise TypeError("dummy_output/observation_provider must return a pandas.DataFrame or polars.DataFrame")
        except Exception as err:
            return {"status": "error", "error": f"dummy_output_failed: {err!r}", "state": before_state_name}

        # 本番と同経路で更新→遷移→再スケジュール
        try:
            for scheduled_group in self.runnable_groups():
                if scheduled_group.is_not_started():
                    for scheduled_task in scheduled_group.tasks:
                        if (
                            not scheduled_task.completed
                            and scheduled_task.scheduled_time <= event_time
                        ):
                            scheduled_task.task_status = type(
                                scheduled_task.task_status
                            ).IN_PROGRESS
                    scheduled_group.status_update()
            self.reference_time = event_time
            _next_tg, _next_task = self.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
                task_group_id=earliest_group_id,
                task_id=earliest_task.task_id,
                new_result_of_experiment=dummy_df,
                update_type="a",
                scheduling_method=scheduling_method,
                optimal_time_reference_time=self.reference_time,
                solver_time_limit_seconds=solver_time_limit_seconds,
                task_group_uuid=tg.task_group_uuid,
                result_id=f"simulation:{tg.task_group_uuid}:{earliest_task.task_id}",
            )
        except Exception as err:
            return {"status": "error", "error": f"simulation_update_failed: {err!r}"}

        after_state_name = exp.current_state_name

        if save_each_step:
            self.proceed_to_next_step()

        return {
            "status": "ok",
            "experiment_name": exp.experiment_name,
            "experiment_uuid": exp.experiment_uuid,
            "from_state": before_state_name,
            "to_state": after_state_name,
            "simulated_task_group_id": earliest_group_id,
            "simulated_task_id": earliest_task.task_id,
            "simulated_time": event_time,
            "step_after": self.step,
        }

    def simulate(
        self,
        max_steps: int = 10,
        scheduling_method: str = "o",
        optimal_time_reference_time: int | None = None,
        save_each_step: bool = False,
        stop_when_no_tasks: bool = True,
        observation_provider: Callable | None = None,
        *, solver_time_limit_seconds: float | None = None,
    ) -> List[dict]:
        results: list[dict] = []
        for _ in range(max_steps):
            res = self.simulate_one(
                scheduling_method=scheduling_method,
                optimal_time_reference_time=optimal_time_reference_time,
                save_each_step=save_each_step,
                observation_provider=observation_provider,
                solver_time_limit_seconds=solver_time_limit_seconds,
            )
            results.append(res)
            if res.get("status") != "ok" and stop_when_no_tasks:
                break
        return results
