import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QListWidget,
    QPushButton, QLabel, QLineEdit, QTextEdit, QMessageBox, QFileDialog, 
    QDialog, QFormLayout, QComboBox, QTableWidget, QTableWidgetItem
)
from PyQt5.QtCore import Qt
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from pathlib import Path
import json

# 既存のクラスをインポート
from gems_python.one_machine_problem_interval_task.transition_manager import Experiment, Experiments, State
from gems_python.one_machine_problem_interval_task.task_info import Task, TaskGroup
from gems_python.one_machine_problem_interval_task.penalty.penalty_class import NonePenalty

# 他の必要なインポート
import polars as pl

class MinimumState(State):
    def task_generator(self, df: pl.DataFrame) -> TaskGroup:
        return  TaskGroup(
            optimal_start_time=0,
            penalty_type=NonePenalty(), 
            tasks=[
                Task(processing_time=2, interval=0, experiment_operation="A"),
                Task(processing_time=3, interval=15, experiment_operation="B"),
                Task(processing_time=4, interval=20, experiment_operation="C")
            ]
        )
    
    def transition_function(self, df: pl.DataFrame) -> str:
        # return the state name
        return "MinimumState"
    

def gen_minimum_experiment(experiment_name = "minimum_experiment") -> Experiment:
    return Experiment(
        experiment_name=experiment_name,
        states=[
            MinimumState()
        ],
        current_state_name="MinimumState",
        shared_variable_history=pl.DataFrame()
    )
    
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Experiment Manager")
        self.setGeometry(100, 100, 1200, 800)
        
        # メインレイアウト
        main_layout = QHBoxLayout()
        
        # サイドバー
        self.state_list = QListWidget()
        self.state_list.itemClicked.connect(self.display_state_details)
        
        add_state_btn = QPushButton("Add State")
        add_state_btn.clicked.connect(self.add_state)
        remove_state_btn = QPushButton("Remove State")
        remove_state_btn.clicked.connect(self.remove_state)
        
        sidebar_layout = QVBoxLayout()
        sidebar_layout.addWidget(QLabel("States"))
        sidebar_layout.addWidget(self.state_list)
        sidebar_layout.addWidget(add_state_btn)
        sidebar_layout.addWidget(remove_state_btn)
        
        # 中央パネル
        self.state_detail = QTextEdit()
        self.state_detail.setReadOnly(True)
        
        central_layout = QVBoxLayout()
        central_layout.addWidget(QLabel("State Details"))
        central_layout.addWidget(self.state_detail)
        
        # 下部パネル（可視化）
        self.figure = plt.figure()
        self.canvas = FigureCanvas(self.figure)
        
        central_layout.addWidget(QLabel("State Transition Graph"))
        central_layout.addWidget(self.canvas)
        
        # メインレイアウトにサイドバーと中央パネルを追加
        main_layout.addLayout(sidebar_layout, 1)
        main_layout.addLayout(central_layout, 3)
        
        # 中央ウィジェット
        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)
        
        # Experimentの初期化
        self.experiments = Experiments(experiments=[], parent_dir_path=Path.cwd())
        self.current_experiment = None
        
        # メニューバー
        self.create_menu()
    
    def create_menu(self):
        menubar = self.menuBar()
        
        file_menu = menubar.addMenu("File")
        
        new_experiment_action = file_menu.addAction("New Experiment")
        new_experiment_action.triggered.connect(self.new_experiment)
        
        save_experiment_action = file_menu.addAction("Save Experiment")
        save_experiment_action.triggered.connect(self.save_experiment)
        
        load_experiment_action = file_menu.addAction("Load Experiment")
        load_experiment_action.triggered.connect(self.load_experiment)
        
        exit_action = file_menu.addAction("Exit")
        exit_action.triggered.connect(self.close)
    
    def new_experiment(self):
        dialog = NewExperimentDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            experiment_name, save_dir = dialog.get_values()
            if not experiment_name or not save_dir:
                QMessageBox.warning(self, "Warning", "Please provide both name and save directory.")
                return
            self.current_experiment = gen_minimum_experiment(experiment_name)
            self.experiments = Experiments(
                experiments=[self.current_experiment],
                parent_dir_path=Path(save_dir)
            )
            self.state_list.clear()
            self.update_graph()
            QMessageBox.information(self, "Success", "New experiment created.")
    
    def save_experiment(self):
        if self.current_experiment is None:
            QMessageBox.warning(self, "Warning", "No experiment to save.")
            return
        save_dir = QFileDialog.getExistingDirectory(self, "Select Save Directory")
        if not save_dir:
            return
        try:
            self.current_experiment.save_all(Path(save_dir))
            QMessageBox.information(self, "Success", "Experiment saved successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to save experiment: {str(e)}")
    
    def load_experiment(self):
        load_dir = QFileDialog.getExistingDirectory(self, "Select Experiment Directory")
        if not load_dir:
            return
        try:
            experiments_json_path = Path(load_dir) / "experiments.json"
            with open(experiments_json_path, "r") as f:
                experiments_data = json.load(f)
            self.experiments = Experiments.from_dict(experiments_data)
            self.current_experiment = self.experiments.experiments[0]  # 単一実験を想定
            self.state_list.clear()
            for state in self.current_experiment.states:
                self.state_list.addItem(state.state_name)
            self.update_graph()
            QMessageBox.information(self, "Success", "Experiment loaded successfully.")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load experiment: {str(e)}")
    
    def add_state(self):
        if self.current_experiment is None:
            QMessageBox.warning(self, "Warning", "Please create or load an experiment first.")
            return
        dialog = AddStateDialog(self)
        if dialog.exec_() == QDialog.Accepted:
            state = dialog.get_state()
            # Check for duplicate state names
            if any(s.state_name == state.state_name for s in self.current_experiment.states):
                QMessageBox.warning(self, "Warning", f"State '{state.state_name}' already exists.")
                return
            self.current_experiment.states.append(state)
            self.state_list.addItem(state.state_name)
            if not self.current_experiment.current_state_name:
                self.current_experiment.current_state_name = state.state_name
            self.update_graph()
            QMessageBox.information(self, "Success", f"State '{state.state_name}' added.")
    
    def remove_state(self):
        selected_items = self.state_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "Warning", "No state selected.")
            return
        for item in selected_items:
            state_name = item.text()
            # Remove from experiment
            self.current_experiment.states = [
                state for state in self.current_experiment.states if state.state_name != state_name
            ]
            self.state_list.takeItem(self.state_list.row(item))
            # Update current_state_name if necessary
            if self.current_experiment.current_state_name == state_name:
                self.current_experiment.current_state_name = ""
        self.update_graph()
        QMessageBox.information(self, "Success", "Selected state(s) removed.")
    
    def display_state_details(self, item):
        state_name = item.text()
        state = next((s for s in self.current_experiment.states if s.state_name == state_name), None)
        if state:
            details = f"State Name: {state.state_name}\n\n"
            details += "Tasks:\n"
            for task in state.task_generator(pl.DataFrame()).tasks:
                details += f"  - Operation: {task.experiment_operation}, Processing Time: {task.processing_time}, Interval: {task.interval}\n"
            details += f"\nNext State: {state.transition_function(pl.DataFrame())}"
            self.state_detail.setPlainText(details)
    
    def update_graph(self):
        if self.current_experiment is None:
            return
        self.figure.clear()
        G = nx.DiGraph()
        for state in self.current_experiment.states:
            G.add_node(state.state_name)
            next_state = state.transition_function(pl.DataFrame())
            if next_state:
                G.add_edge(state.state_name, next_state)
        pos = nx.spring_layout(G, seed=42)
        nx.draw(G, pos, with_labels=True, node_color='skyblue', node_size=2000, arrowstyle='->', arrowsize=20)
        self.canvas.draw()

class NewExperimentDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("New Experiment")
        self.setModal(True)
        
        layout = QFormLayout()
        
        self.name_input = QLineEdit()
        self.dir_input = QLineEdit()
        browse_btn = QPushButton("Browse")
        browse_btn.clicked.connect(self.browse_directory)
        
        dir_layout = QHBoxLayout()
        dir_layout.addWidget(self.dir_input)
        dir_layout.addWidget(browse_btn)
        
        layout.addRow("Experiment Name:", self.name_input)
        layout.addRow("Save Directory:", dir_layout)
        
        buttons_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(ok_btn)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addRow(buttons_layout)
        
        self.setLayout(layout)
    
    def browse_directory(self):
        directory = QFileDialog.getExistingDirectory(self, "Select Directory")
        if directory:
            self.dir_input.setText(directory)
    
    def get_values(self):
        return self.name_input.text(), self.dir_input.text()

class AddStateDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add State")
        self.setModal(True)
        
        layout = QVBoxLayout()
        
        form_layout = QFormLayout()
        
        self.state_name_input = QLineEdit()
        form_layout.addRow("State Name:", self.state_name_input)
        
        # Task Table
        self.task_table = QTableWidget()
        self.task_table.setColumnCount(3)
        self.task_table.setHorizontalHeaderLabels(["Operation", "Processing Time", "Interval"])
        self.task_table.setRowCount(0)
        add_task_btn = QPushButton("Add Task")
        add_task_btn.clicked.connect(self.add_task_row)
        remove_task_btn = QPushButton("Remove Task")
        remove_task_btn.clicked.connect(self.remove_task_row)
        
        task_btn_layout = QHBoxLayout()
        task_btn_layout.addWidget(add_task_btn)
        task_btn_layout.addWidget(remove_task_btn)
        
        form_layout.addRow(QLabel("Tasks:"), self.task_table)
        form_layout.addRow("", task_btn_layout)
        
        # Transition設定
        transition_layout = QHBoxLayout()
        transition_layout.addWidget(QLabel("Next State:"))
        self.next_state_input = QLineEdit()
        transition_layout.addWidget(self.next_state_input)
        form_layout.addRow("Transition Function:", transition_layout)
        
        # Penalty Type (for simplicity, fixed to NonePenalty)
        # 拡張性を持たせるためにペナルティタイプを選択できるようにすることも可能
        # ここではデフォルトでNonePenaltyを使用
        
        layout.addLayout(form_layout)
        
        # Buttons
        buttons_layout = QHBoxLayout()
        ok_btn = QPushButton("OK")
        ok_btn.clicked.connect(self.validate_and_accept)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        buttons_layout.addWidget(ok_btn)
        buttons_layout.addWidget(cancel_btn)
        
        layout.addLayout(buttons_layout)
        
        self.setLayout(layout)
    
    def add_task_row(self):
        row_position = self.task_table.rowCount()
        self.task_table.insertRow(row_position)
        self.task_table.setItem(row_position, 0, QTableWidgetItem(""))
        self.task_table.setItem(row_position, 1, QTableWidgetItem(""))
        self.task_table.setItem(row_position, 2, QTableWidgetItem(""))
    
    def remove_task_row(self):
        current_row = self.task_table.currentRow()
        if current_row >= 0:
            self.task_table.removeRow(current_row)
    
    def validate_and_accept(self):
        state_name = self.state_name_input.text()
        if not state_name:
            QMessageBox.warning(self, "Warning", "State name cannot be empty.")
            return
        # Ensure at least one task
        if self.task_table.rowCount() == 0:
            QMessageBox.warning(self, "Warning", "At least one task is required.")
            return
        # Ensure all task fields are filled
        for row in range(self.task_table.rowCount()):
            operation = self.task_table.item(row, 0).text()
            processing_time = self.task_table.item(row, 1).text()
            interval = self.task_table.item(row, 2).text()
            if not operation or not processing_time or not interval:
                QMessageBox.warning(self, "Warning", "All task fields must be filled.")
                return
            try:
                int(processing_time)
                int(interval)
            except ValueError:
                QMessageBox.warning(self, "Warning", "Processing Time and Interval must be integers.")
                return
        # Ensure next state is specified
        next_state = self.next_state_input.text()
        if not next_state:
            QMessageBox.warning(self, "Warning", "Next state must be specified.")
            return
        self.accept()
    
    def get_state(self) -> State:
        state_name = self.state_name_input.text()
        tasks = []
        for row in range(self.task_table.rowCount()):
            operation = self.task_table.item(row, 0).text()
            processing_time = int(self.task_table.item(row, 1).text())
            interval = int(self.task_table.item(row, 2).text())
            task = Task(
                processing_time=processing_time,
                interval=interval,
                experiment_operation=operation
            )
            tasks.append(task)
        penalty = NonePenalty()  # 固定
        task_group = TaskGroup(
            optimal_start_time=0,
            penalty_type=penalty,
            tasks=tasks
        )
        # Dynamically create a State subclass
        # ここでは単純にStateを継承したクラスを作成しますが、実際にはより複雑な設定が必要かもしれません
        class CustomState(State):
            def task_generator(self, df: pl.DataFrame) -> TaskGroup:
                return task_group
            
            def transition_function(self, df: pl.DataFrame) -> str:
                return self.next_state
        
        state = CustomState()
        state.next_state = self.next_state_input.text()
        return state

def main():
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
