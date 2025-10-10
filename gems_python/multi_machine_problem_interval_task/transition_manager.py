from datetime import datetime
import importlib
import warnings
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
from typing import Any, Dict, List, Tuple, Type, TypeVar, Union
import uuid
import numpy as np
import polars as pl
from pathlib import Path
import os
from gems_python.common.class_dumper import auto_dataclass as dataclass

from gems_python.multi_machine_problem_interval_task.task_info import Machine, MachineList, Task, TaskGroup

"""MODULE: State
"""


T = TypeVar('T', bound='State')

class State(ABC):
    """
    Template for experiment states used by the interval-scheduling framework.

    To introduce a new state:

    * Subclass :class:`State` and implement :meth:`task_generator` to return a
      :class:`~gems_python.multi_machine_problem_interval_task.task_info.TaskGroup`
      with the required tasks, penalties, and metadata.
    * Implement :meth:`transition_function` so the state can select the next
      state's name based on the shared variable history (:class:`polars.DataFrame`).
    * Optionally override :meth:`dummy_output` to emit synthetic measurements
      when running simulations without real hardware feedback.

    The worked example in ``tests/test_simulate_one.py`` demonstrates how the
    ``InitState``/``MeasureState``/``FinishState`` trio fits together. Mirror
    that structure when expanding your workflow.
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
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        pass

    @abstractmethod
    def transition_function(self, df: pl.DataFrame) -> str:
        """
        This function must return one of the state name in the 'states', state list of the parent experiment
        """
        pass


    def dummy_output(self, df: pl.DataFrame, task_group_id: int, task_id: int) -> pl.DataFrame:
        """
        Return a dummy (simulated) result as a Polars DataFrame.
        Override in each State subclass to provide the columns/values your transition_function expects.
        The default raises, so you notice missing implementations at simulation time.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__}.dummy_output(df, task_group_id, task_id) is not implemented."
        )

    def to_dict(self) -> dict:
        result = dict()
        result['_class'] = f"{self.__class__.__module__}.{self.__class__.__name__}"
        return result

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        if '_class' in data:
            class_path = data.pop('_class')
            module_name, class_name = class_path.rsplit('.', 1)
            module = importlib.import_module(module_name)
            cls = getattr(module, class_name)
        return cls(**data)

    @classmethod
    def from_json(cls: Type[T], json_str: str) -> T:
        data = json.loads(json_str)
        return cls.from_dict(data)



"""MODULE: Experiment
"""

@dataclass
class Experiment:
    """
    Container for a single experiment workflow.

    Parameters
    ----------
    experiment_name:
        Human-readable name of the experiment.
    states:
        Ordered list of :class:`State` instances that define the workflow.
    current_state_name:
        Name of the state that is currently active.
    shared_variable_history:
        Polars DataFrame that stores shared variables emitted during execution.
    current_task_group:
        Task group that is currently in progress. Assigned automatically.
    current_state_index:
        Derived index of the active state. Managed internally.
    experiment_uuid:
        Unique identifier generated for persistence and lookups.
    """
    experiment_name: str
    states: List[State]
    current_state_name: str
    shared_variable_history: pl.DataFrame = field(default_factory=pl.DataFrame)
    current_task_group: TaskGroup = field(default=None)
    current_state_index: int = field(default=None, init=False)
    experiment_uuid: str = field(default_factory=lambda: str(uuid.uuid4()))
    
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
        self.shared_variable_history.write_csv(save_path / f"{self.experiment_name}_{self.experiment_uuid}_shared_variable_history.csv")
        
        # for state in self.states:
        #     state.save_all(save_path)

    def __post_init__(self):
        """
        Initialize the experiment.
        """
        self.update_current_state_name_and_index(self.current_state_name)
        if self.current_task_group is None:
            self.current_task_group = self.generate_task_group_of_the_state()

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
            task_group: TaskGroup = self.states[state_index].task_generator(self.shared_variable_history.clone())
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
            next_state_name: str = self.states[self.current_state_index].transition_function(self.shared_variable_history)
        except Exception as err:
            raise RuntimeError(f"Error state transition: {err}")

        return next_state_name
 


"""MODULE: Experiments
"""

@dataclass
class Experiments:
    """
    Coordinate a collection of :class:`Experiment` instances and persist their progress.

    Parameters
    ----------
    experiments:
        List of active experiments, typically produced by a factory such as
        ``tests.test_simulate_one.build_experiment``.
    parent_dir_path:
        Root directory where simulation artefacts, schedule CSVs, and experiment
        snapshots are stored. The default ``"experiments_dir"`` is compatible
        with :class:`gems_python.multi_machine_problem_interval_task.interactive_ui.PluginManager`.
    machine_list:
        Shared :class:`MachineList` used during multi-machine scheduling.
    task_groups:
        Internal cache of task groups mirrored from the experiments.
    step:
        Current persistence step. Each call to :meth:`proceed_to_next_step`
        increments this counter and writes a new ``step_{n}`` directory.
    reference_time:
        Baseline timestamp (minutes) used to align schedules.

    Notes
    -----
    The directory layout produced here is the same structure that the
    ``PluginManager`` expects when you run ``python main.py``. That allows you
    to seed an experiment offline via ``Experiments.simulate`` and then continue
    manipulating it interactively.

    Examples
    --------
    Create a sandbox experiment, save its first step, and hand it off to the
    plugin manager::

        from pathlib import Path
        from tests.test_simulate_one import build_experiment
        from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
        from gems_python.multi_machine_problem_interval_task.interactive_ui import PluginManager

        exp = build_experiment()
        experiments = Experiments(
            experiments=[exp],
            parent_dir_path=Path("examples/multi_machine_demo")
        )
        experiments.simulate(max_steps=1, save_each_step=True)
        experiments.proceed_to_next_step()  # writes step_001 artefacts
        plugin = PluginManager(experiments)
        # Populate experimental_setting/demo_plugin.py and use mode files to register it
    """

    experiments: List[Experiment] = field(default_factory=list)
    parent_dir_path: Path = field(default=Path("experiments_dir"))
    machine_list: MachineList = field(default_factory=MachineList)
    # Automatically generated fields, not accept user input
    task_groups: List[TaskGroup] = field(default=None)
    step: int = field(default=0)
    reference_time: int = field(default=0)

    def __post_init__(self):
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

    def reload(self, step: int = None):
        """
        Reload the experiments from the saved directory.
        """
        if step is None:
            # 保存ディレクトリ内のすべてのstepディレクトリを取得
            step_dirs = [
                d for d in self.parent_dir_path.iterdir()
                if d.is_dir() and d.name.startswith('step_') and d.name[5:].isdigit()
            ]
            if not step_dirs:
                raise ValueError("リロード可能なステップディレクトリが見つかりません。")
            # ステップ番号を抽出し、最大値を選択
            step_numbers = [int(d.name[5:]) for d in step_dirs]
            step = max(step_numbers)
            print(f"ステップが指定されなかったため、最大のステップ {step} を選択しました。")
            
        old_step = self.step
        self.step = step

        experiments_js_path = self.save_dir() / "experiments.json"
        experiments_pkl_path = self.save_dir() / "experiments.pkl"
        experiments_json = None
        experiments_pkl = None
        # TODO: Jsonを読み込んでも、Penaltyの再構成がうまく行っていない
        try:
            json_str = ""
            with open(experiments_js_path, "r") as f:
                json_str = f.read()
            experiments_json = Experiments.from_json(json_str)
        except Exception as err:
            print(f"Error loading experiments from json: {err}")
        try:
            with open(experiments_pkl_path, "rb") as f:
                experiments_pkl = Experiments.from_pickle(experiments_pkl_path)
        except Exception as err:
            print(f"Error loading experiments from pickle: {err}")
            self.step = old_step

        if experiments_json is None and experiments_pkl is None:
            raise RuntimeError(f"Error loading experiments: {err}")
        elif experiments_json is None:
            experiments = experiments_pkl
            print(f"Experiments loaded from pickle: {experiments_pkl_path} because json file could not be loaded.")
        elif experiments_pkl is None:
            experiments = experiments_json
            print(f"Experiments loaded from json: {experiments_js_path} because pickle file could not be loaded.")
        else:
            if experiments_json.to_json() != experiments_pkl.to_json():
                # Warn that the loaded experiments are different using warnings
                warnings.warn("The loaded experiments from json and pickle are different. The experiments from json will be used.")
                while True:
                    ok = input("Do you want to use the experiments from json or pickle, or cancel the reload? (j/p/c): ").strip().lower()
                    if ok == 'j':
                        experiments = experiments_json
                        print(f"Experiments loaded from json: {experiments_js_path}")
                        break
                    elif ok == 'p':
                        experiments = experiments_pkl
                        print(f"Experiments loaded from pickle: {experiments_pkl_path}")
                        break
                    elif ok == 'c':
                        self.step = old_step
                        print("Reload canceled.")
                        raise RuntimeError("Reload canceled.")
                    else:
                        print("Invalid input. Please enter 'j', 'p', or 'c'.")
            else:
                experiments = experiments_pkl
                print(f"Experiments loaded from pickle because the json and pickle files are the same: {experiments_pkl_path}")
            
        return experiments

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

    def add_experiment(self, experiment: Experiment) -> Union[None, ValueError]:

        # Check the duplication of the experiment name and uuid
        for e in self.experiments:
            if e.experiment_name == experiment.experiment_name and e.experiment_uuid == experiment.experiment_uuid:
                raise ValueError(f"Experiment name and uuid must be unique: {experiment.experiment_name}, {experiment.experiment_uuid}")
            
        self.experiments.append(experiment)
        self.task_groups.append(copy.deepcopy(experiment.current_task_group))
        self.set_task_group_ids()
        self.execute_scheduling()
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
            scheduling_method: str = 's',
            reference_time: int = None
            ):
        # TODO-DONE: TaskGroupに対応
        """
        Execute the scheduling of the tasks.
        :param scheduling_method: The method of scheduling. 's' for simulated annealing.
        :param reference_time: The reference time for the optimal time.
        """
        if reference_time is not None:
            self.reference_time = reference_time
        
        reference_time = self.reference_time
        self.set_task_group_ids()
        # Reschedule
        task_groups = self.task_groups.copy()
        scheduled_task_groups = list()

        try:
            match scheduling_method:
                case 's':
                    scheduled_task_groups = TaskGroup.schedule_task_groups_simulated_annealing(task_groups = task_groups, machines = self.machine_list, reference_time = reference_time)
                case 'f':
                    scheduled_task_groups = TaskGroup.schedule_task_groups(task_groups = task_groups, machines = self.machine_list, reference_time = reference_time)
                case _:
                    AssertionError(f"Unexpected input: scheduling_method {scheduling_method}")
        except Exception as err:
            raise RuntimeError(f"Error scheduling the tasks with method {scheduling_method}: {err}")

        self.task_groups = scheduled_task_groups

    

    def update_shared_variable_history(
            self,
            task_group_id: int,
            task_id: int,
            new_result_of_experiment: pl.DataFrame,
            update_type: str = 'a',
            ) -> Tuple[TaskGroup, Task]:
        """
        TODO: explanation
        """
        # TODO-DONE: TaskGroupに対応
        # Update task_group
        self.task_groups = TaskGroup.complete_task(self.task_groups, group_id=task_group_id, task_id=task_id)
        task_group_index = TaskGroup.find_task_group(self.task_groups, task_group_id)
        experiment_uuid = self.task_groups[task_group_index].experiment_uuid
        experiment_index = 0
        for index in range(len(self.experiments)):
            if self.experiments[index].experiment_uuid == experiment_uuid:
                experiment_index = index

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

        if self.task_groups[task_group_index].is_completed():
            # Delete the task group
            self.task_groups = TaskGroup.delete_task_group(self.task_groups, task_group_id)
            # Transition and generate a new task group
            new_task_group = self.experiments[experiment_index].execute_one_step()
            # Add the new task group and reschedule
            self.task_groups = TaskGroup.add_task_group(self.task_groups, self.machine_list, new_task_group)

        self.set_task_group_ids()

    def update_shared_variable_history_and_states_and_generate_task_and_reschedule(
            self,
            task_group_id: int,
            task_id: int,
            new_result_of_experiment: pl.DataFrame,
            update_type: str = 'a',
            scheduling_method = 's',
            optimal_time_reference_time: int = None
            ) -> Tuple[TaskGroup, Task]:
        """
        TODO: explanation
        """
        # TODO-DONE: TaskGroupに対応
        # Update task_group

        if optimal_time_reference_time is not None:
            self.reference_time = optimal_time_reference_time
        self.update_shared_variable_history(task_group_id, task_id, new_result_of_experiment, update_type)
        self.set_task_group_ids()
        
        self.execute_scheduling(scheduling_method)
        print(f"{self.task_groups=}")

        earliest_task, eariest_group_id = TaskGroup.get_ealiest_task_in_task_groups(self.task_groups)
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
        TaskGroup.generate_gantt_chart(self.task_groups, save_dir = save_dir)

    def save_results(self, save_dir: Path = None):
        if save_dir is None:
            save_dir = self.save_dir()

        os.makedirs(save_dir, exist_ok=True)
        # Save the task groups
        experiment_js_path = save_dir / "experiments.json"
        with open(experiment_js_path, "w") as f:
            f.write(self.to_json())
        
        experiment_pickle_path = save_dir / "experiments.pkl"
        with open(experiment_pickle_path, "wb") as f:
            f.write(self.to_pickle())

        date_and_time = datetime.now().astimezone().isoformat()
        with open(save_dir / "date_and_time.txt", "w") as f:
            f.write(date_and_time)

        # Save the task groups
        schedule_df = TaskGroup.create_non_completed_tasks_df(self.task_groups)
        schedule_df.write_csv(save_dir / "schedule.csv")

        # Save the gantt chart
        self.generate_gantt_chart(save_dir=save_dir)

        # Save each experiment
        experiments_dir = save_dir / "experiments"
        os.makedirs(experiments_dir, exist_ok=True)
        for experiment in self.experiments:
            experiment.show_experiment_directed_graph(save_path=experiments_dir / f"{experiment.experiment_name}_{experiment.experiment_uuid}.png")
            experiment.shared_variable_history.write_csv(experiments_dir / f"{experiment.experiment_name}_{experiment.experiment_uuid}_shared_variable_history.csv")

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
        Automatically load experiment results saved on disk.

        Expected files
        --------------
        ``experiment_result.json``
            Metadata emitted by the external executor.
        ``result.csv``
            Measurement results associated with the completed task.

        Normal task status progression::

            NOT_STARTED -> In Progress -> Completed

        Example payload (task started)::

            {
                "task_response": "In Progress",
                "task_group_id": 0,
                "task_id": 0,
                "optimal_time_reference_time": 0
            }

        Example payload (task completed)::

            {
                "task_response": "Completed",
                "task_group_id": 0,
                "task_id": 0,
                "optimal_time_reference_time": 0,
                "result_path": "result.csv"
            }

        Failure payloads set ``task_response`` to ``"Error"`` and keep the same
        keys.
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
            scheduling_method = result_data.get("scheduling_method", "s")
            task_group_id = result_data["task_group_id"]
            task_id = result_data["task_id"]
            optimal_time_reference_time = result_data["optimal_time_reference_time"]
            
            match task_response:
                case "In Progress":
                    # TODO-DONE 適切な処理
                    try:
                        TaskGroup.start_task(self.task_groups, task_group_id, task_id)
                    except Exception as err:
                        print(f"Error starting task: {err}")
                case "Completed":
                    new_result_of_experiment = pl.read_csv(result_data["result_path"])
                    self.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
                        task_group_id=task_group_id,
                        task_id=task_id,
                        new_result_of_experiment=new_result_of_experiment,
                        update_type=update_type,
                        scheduling_method=scheduling_method,
                        optimal_time_reference_time=optimal_time_reference_time
                    )
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
        scheduling_method: str = "s",
        optimal_time_reference_time: int | None = None,
        save_each_step: bool = False,
    ) -> dict:
        if optimal_time_reference_time is not None:
            self.reference_time = optimal_time_reference_time

        self.set_task_group_ids()
        self.execute_scheduling(scheduling_method=scheduling_method, reference_time=self.reference_time)

        if len(self.task_groups) == 0:
            return {"status": "no_tasks"}

        try:
            earliest_task, earliest_group_id = TaskGroup.get_ealiest_task_in_task_groups(self.task_groups)
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
        try:
            dummy_df = state.dummy_output(
                exp.shared_variable_history.clone(),
                task_group_id=earliest_group_id,
                task_id=earliest_task.task_id,
            )
            if not isinstance(dummy_df, pl.DataFrame):
                raise TypeError("dummy_output must return a pl.DataFrame")
        except Exception as err:
            return {"status": "error", "error": f"dummy_output_failed: {err!r}", "state": before_state_name}

        # 本番と同経路で更新→遷移→再スケジュール
        try:
            _next_tg, _next_task = self.update_shared_variable_history_and_states_and_generate_task_and_reschedule(
                task_group_id=earliest_group_id,
                task_id=earliest_task.task_id,
                new_result_of_experiment=dummy_df,
                update_type="a",
                scheduling_method=scheduling_method,
                optimal_time_reference_time=self.reference_time,
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
            "step_after": self.step,
        }

    def simulate(
        self,
        max_steps: int = 10,
        scheduling_method: str = "s",
        optimal_time_reference_time: int | None = None,
        save_each_step: bool = False,
        stop_when_no_tasks: bool = True,
    ) -> List[dict]:
        results: list[dict] = []
        for _ in range(max_steps):
            res = self.simulate_one(
                scheduling_method=scheduling_method,
                optimal_time_reference_time=optimal_time_reference_time,
                save_each_step=save_each_step,
            )
            results.append(res)
            if res.get("status") != "ok" and stop_when_no_tasks:
                break
        return results
