import sys
import setuptools._distutils as _distutils
sys.modules['distutils'] = _distutils
import json
from pathlib import Path
from PyQt5 import QtWidgets, QtCore
from NodeGraphQt import NodeGraph, BaseNode


# Dialog to configure a State node
class StateDialog(QtWidgets.QDialog):
    def __init__(self, state_data, parent=None):
        super(StateDialog, self).__init__(parent)
        self.setWindowTitle(f"Edit State: {state_data['name']}")
        self.state_data = state_data

        # Layouts
        layout = QtWidgets.QVBoxLayout(self)

        # Tasks editor
        self.task_table = QtWidgets.QTableWidget(self)
        self.task_table.setColumnCount(4)
        self.task_table.setHorizontalHeaderLabels([
            'processing_time', 'interval', 'operation', 'machine_type'])
        layout.addWidget(QtWidgets.QLabel("Tasks:"))
        layout.addWidget(self.task_table)
        self._load_tasks()

        # Transition selector
        layout.addWidget(QtWidgets.QLabel("Next State:"))
        self.next_state_combo = QtWidgets.QComboBox(self)
        self.next_state_combo.addItems(state_data.get('all_states', []))
        if state_data.get('next_state'):
            idx = self.next_state_combo.findText(state_data['next_state'])
            if idx >= 0:
                self.next_state_combo.setCurrentIndex(idx)
        layout.addWidget(self.next_state_combo)

        # Buttons
        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel)
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def _load_tasks(self):
        tasks = self.state_data.get('tasks', [])
        self.task_table.setRowCount(len(tasks))
        for r, t in enumerate(tasks):
            self.task_table.setItem(r, 0, QtWidgets.QTableWidgetItem(str(t['processing_time'])))
            self.task_table.setItem(r, 1, QtWidgets.QTableWidgetItem(str(t['interval'])))
            self.task_table.setItem(r, 2, QtWidgets.QTableWidgetItem(t['operation']))
            self.task_table.setItem(r, 3, QtWidgets.QTableWidgetItem(str(t['machine_type'])))

    def get_data(self):
        tasks = []
        for r in range(self.task_table.rowCount()):
            tasks.append({
                'processing_time': int(self.task_table.item(r,0).text()),
                'interval': int(self.task_table.item(r,1).text()),
                'operation': self.task_table.item(r,2).text(),
                'machine_type': int(self.task_table.item(r,3).text())
            })
        return {
            'name': self.state_data['name'],
            'tasks': tasks,
            'next_state': self.next_state_combo.currentText()
        }

# Custom node representing a State
class StateNode(BaseNode):
    __identifier__ = 'gems'
    NODE_NAME = 'State'

    def __init__(self):
        super(StateNode, self).__init__()
        self.add_input('in', color=(200, 200, 200), multi_input=False)
        self.add_output('out', color=(200, 200, 200), multi_output=False)
        self.properties = {
            'state_name': 'State',
            'tasks': [],
            'next_state': ''
        }
        # double-click to edit
        self.set_node_double_clicked(self._edit)

    def _edit(self):
        # gather all state names
        graph = self.graph
        all_states = [n.get_property('state_name') for n in graph.all_nodes()]
        state_data = {
            'name': self.get_property('state_name'),
            'tasks': self.get_property('tasks'),
            'next_state': self.get_property('next_state'),
            'all_states': all_states
        }
        dlg = StateDialog(state_data)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            data = dlg.get_data()
            self.set_name(data['name'])
            self.set_property('state_name', data['name'])
            self.set_property('tasks', data['tasks'])
            self.set_property('next_state', data['next_state'])

# Main application window
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        self.setWindowTitle('Experiment Designer')
        self.resize(800, 600)

        # Node Graph
        self.graph = NodeGraph()
        widget = self.graph.widget()
        self.setCentralWidget(widget)

        # Menus
        menubar = self.menuBar()
        exp_menu = menubar.addMenu('Experiment')
        exp_menu.addAction('Save .py', self.save_py)

        # add state action
        node_menu = menubar.addMenu('Nodes')
        node_menu.addAction('Add State', self.add_state)

    def add_state(self):
        node = self.graph.create_node('gems.State')
        node.set_property('state_name', f'State{len(self.graph.all_nodes())}')
        node.set_name(node.get_property('state_name'))

    def save_py(self):
        path, _ = QtWidgets.QFileDialog.getSaveFileName(
            self, 'Save Experiment .py', filter='Python Files (*.py)')
        if not path:
            return
        # build data
        states = []
        for node in self.graph.all_nodes():
            states.append({
                'name': node.get_property('state_name'),
                'tasks': node.get_property('tasks'),
                'next_state': node.get_property('next_state')
            })
        # generate code
        code = self.generate_code(states)
        Path(path).write_text(code)
        QtWidgets.QMessageBox.information(self, 'Saved', f'Saved to {path}')

    def generate_code(self, states):
        template = '''from gems_python.multi_machine_problem_interval_task.transition_manager import Experiment, Experiments, State
from gems_python.common.class_dumper import auto_dataclass as dataclass
import polars as pl
from gems_python.multi_machine_problem_interval_task.task_info import TaskGroup, Task, Machine
from gems_python.multi_machine_problem_interval_task.penalty.penalty_class import NonePenalty

'''
        # State classes
        for s in states:
            tasks_list = ',\n                '.join([
                f"Task(processing_time={t['processing_time']}, interval={t['interval']}, experiment_operation=\"{t['operation']}\", optimal_machine_type={t['machine_type']})"
                for t in s['tasks']])
            template += f"class {s['name']}(State):\n"
            template += f"    def task_generator(self, df: pl.DataFrame) -> TaskGroup:\n"
            template += f"        return TaskGroup(\n            optimal_start_time=0,\n            penalty_type=NonePenalty(),\n            tasks=[\n                {tasks_list}\n            ]\n        )\n"
            template += f"    def transition_function(self, df: pl.DataFrame) -> str:\n"
            template += f"        return \"{s['next_state']}\"\n\n"
        # Experiment function
        template += "def gen_experiment(experiment_name='experiment') -> Experiment:\n"
        template += "    return Experiment(\n"
        template += "        experiment_name=experiment_name,\n"
        template += "        states=[\n"
        template += ''.join([f"            {s['name']}(),\n" for s in states])
        template += "        ],\n"
        template += f"        current_state_name=\"{states[0]['name']}\",\n"
        template += "        shared_variable_history=pl.DataFrame()\n"
        template += "    )\n"
        # Wrapper Experiments
        template += "def gen_experiments(temp_dir: str, experiment_name='experiment') -> Experiments:\n"
        template += "    exp = Experiments(\n"
        template += "        experiments=[gen_experiment(experiment_name=experiment_name)],\n"
        template += "        parent_dir_path=Path(temp_dir)\n"
        template += "    )\n"
        template += "    # Add machines here if needed\n"
        template += "    return exp\n"
        return template

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec_())
