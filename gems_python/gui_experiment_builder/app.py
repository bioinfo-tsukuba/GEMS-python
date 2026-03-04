from __future__ import annotations

from datetime import datetime
import math
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk

from gems_python.gui_experiment_builder.factory import build_experiment_from_draft
from gems_python.gui_experiment_builder.models import ExperimentDraft, StateDraft
from gems_python.gui_experiment_builder.validator import validate_experiment_draft
from gems_python.multi_machine_problem_interval_task.task_info import Machine
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments


class ExperimentBuilderApp(tk.Tk):
    def __init__(self, workspace_dir: Path | None = None):
        super().__init__()
        self.title("GEMS Experiment GUI")
        self.geometry("1260x840")
        self.minsize(1100, 760)

        base_dir = workspace_dir if workspace_dir is not None else Path("gui_workspace")
        self.workspace_dir = Path(base_dir)
        self.workspace_dir.mkdir(parents=True, exist_ok=True)

        self.experiments = Experiments(
            parent_dir_path=self.workspace_dir,
            reference_time=int(datetime.now().timestamp() // 60),
        )
        if not self.experiments.machine_list.machines:
            self.experiments.machine_list.add_machine(
                Machine(machine_type=0, description="Default machine for GUI")
            )

        self.state_drafts: list[StateDraft] = []

        self.experiment_name_var = tk.StringVar()
        self.initial_state_var = tk.StringVar()
        self.state_name_var = tk.StringVar()
        self.next_state_var = tk.StringVar()
        self.machine_type_var = tk.StringVar(value="0")
        self.processing_time_var = tk.StringVar(value="10")
        self.interval_var = tk.StringVar(value="0")
        self.operation_var = tk.StringVar(value="generic_operation")

        self.simulate_steps_var = tk.StringVar(value="5")
        self.machine_type_add_var = tk.StringVar(value="0")
        self.machine_description_var = tk.StringVar()
        self.experiment_filter_var = tk.StringVar(value="ALL")
        self.experiment_search_var = tk.StringVar()

        self.summary_var = tk.StringVar()

        self._build_layout()
        self.refresh_all_views()
        self.log_event("Main画面を初期化しました。")

    def _build_layout(self) -> None:
        container = ttk.Frame(self, padding=12)
        container.pack(fill=tk.BOTH, expand=True)

        self.notebook = ttk.Notebook(container)
        self.notebook.pack(fill=tk.BOTH, expand=True)

        self.main_tab = ttk.Frame(self.notebook, padding=10)
        self.wizard_tab = ttk.Frame(self.notebook, padding=10)
        self.machine_tab = ttk.Frame(self.notebook, padding=10)
        self.experiment_tab = ttk.Frame(self.notebook, padding=10)

        self.notebook.add(self.main_tab, text="Main")
        self.notebook.add(self.wizard_tab, text="Experiment作成")
        self.notebook.add(self.machine_tab, text="マシン管理")
        self.notebook.add(self.experiment_tab, text="実験一覧")

        self._build_main_tab()
        self._build_wizard_tab()
        self._build_machine_tab()
        self._build_experiment_tab()

    def _build_main_tab(self) -> None:
        header = ttk.Label(self.main_tab, text="GEMS Experiment GUI Main画面", font=("Helvetica", 16, "bold"))
        header.pack(anchor=tk.W)

        summary_frame = ttk.LabelFrame(self.main_tab, text="サマリー", padding=10)
        summary_frame.pack(fill=tk.X, pady=(10, 10))

        ttk.Label(summary_frame, textvariable=self.summary_var).pack(anchor=tk.W)
        ttk.Button(summary_frame, text="サマリー更新", command=self.refresh_all_views).pack(anchor=tk.W, pady=(8, 0))

        simulation_frame = ttk.LabelFrame(self.main_tab, text="シミュレーション", padding=10)
        simulation_frame.pack(fill=tk.X, pady=(0, 10))

        ttk.Button(simulation_frame, text="simulate_one 実行", command=self.run_simulate_one).grid(row=0, column=0, sticky=tk.W)
        ttk.Label(simulation_frame, text="simulate ステップ数").grid(row=0, column=1, padx=(16, 4), sticky=tk.W)
        ttk.Entry(simulation_frame, textvariable=self.simulate_steps_var, width=8).grid(row=0, column=2, sticky=tk.W)
        ttk.Button(simulation_frame, text="simulate 実行", command=self.run_simulate_many).grid(row=0, column=3, padx=(8, 0), sticky=tk.W)

        action_frame = ttk.LabelFrame(self.main_tab, text="ショートカット", padding=10)
        action_frame.pack(fill=tk.X, pady=(0, 10))
        ttk.Button(action_frame, text="Experiment作成タブを開く", command=lambda: self.notebook.select(self.wizard_tab)).pack(side=tk.LEFT)
        ttk.Button(action_frame, text="マシン管理タブを開く", command=lambda: self.notebook.select(self.machine_tab)).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(action_frame, text="実験一覧タブを開く", command=lambda: self.notebook.select(self.experiment_tab)).pack(side=tk.LEFT, padx=(8, 0))

        log_frame = ttk.LabelFrame(self.main_tab, text="イベントログ", padding=10)
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.event_log_text = tk.Text(log_frame, height=18)
        self.event_log_text.pack(fill=tk.BOTH, expand=True)

    def _build_wizard_tab(self) -> None:
        header_frame = ttk.LabelFrame(self.wizard_tab, text="1. Experiment作成ウィザード", padding=10)
        header_frame.pack(fill=tk.X)

        ttk.Label(header_frame, text="実験名").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(header_frame, textvariable=self.experiment_name_var, width=28).grid(row=0, column=1, padx=(4, 12), sticky=tk.W)
        ttk.Label(header_frame, text="初期状態").grid(row=0, column=2, sticky=tk.W)
        self.initial_state_combo = ttk.Combobox(header_frame, textvariable=self.initial_state_var, values=[])
        self.initial_state_combo.grid(row=0, column=3, padx=(4, 0), sticky=tk.W)

        state_editor_frame = ttk.LabelFrame(self.wizard_tab, text="2. 状態遷移の可視化エディタ", padding=10)
        state_editor_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

        form_frame = ttk.Frame(state_editor_frame)
        form_frame.pack(fill=tk.X)

        ttk.Label(form_frame, text="状態名").grid(row=0, column=0, sticky=tk.W)
        ttk.Entry(form_frame, textvariable=self.state_name_var, width=18).grid(row=0, column=1, padx=(4, 12), sticky=tk.W)
        ttk.Label(form_frame, text="遷移先状態").grid(row=0, column=2, sticky=tk.W)
        self.next_state_combo = ttk.Combobox(form_frame, textvariable=self.next_state_var, values=[], width=18)
        self.next_state_combo.grid(row=0, column=3, padx=(4, 12), sticky=tk.W)

        ttk.Label(form_frame, text="マシン種別").grid(row=1, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(form_frame, textvariable=self.machine_type_var, width=18).grid(row=1, column=1, padx=(4, 12), sticky=tk.W, pady=(8, 0))
        ttk.Label(form_frame, text="処理時間").grid(row=1, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Entry(form_frame, textvariable=self.processing_time_var, width=18).grid(row=1, column=3, padx=(4, 12), sticky=tk.W, pady=(8, 0))

        ttk.Label(form_frame, text="インターバル").grid(row=2, column=0, sticky=tk.W, pady=(8, 0))
        ttk.Entry(form_frame, textvariable=self.interval_var, width=18).grid(row=2, column=1, padx=(4, 12), sticky=tk.W, pady=(8, 0))
        ttk.Label(form_frame, text="操作名").grid(row=2, column=2, sticky=tk.W, pady=(8, 0))
        ttk.Entry(form_frame, textvariable=self.operation_var, width=18).grid(row=2, column=3, padx=(4, 12), sticky=tk.W, pady=(8, 0))

        button_row = ttk.Frame(form_frame)
        button_row.grid(row=3, column=0, columnspan=4, sticky=tk.W, pady=(10, 0))
        ttk.Button(button_row, text="状態を追加 / 更新", command=self.upsert_state).pack(side=tk.LEFT)
        ttk.Button(button_row, text="選択状態を削除", command=self.remove_selected_state).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(button_row, text="入力クリア", command=self.clear_state_form).pack(side=tk.LEFT, padx=(8, 0))

        transition_row = ttk.Frame(form_frame)
        transition_row.grid(row=4, column=0, columnspan=4, sticky=tk.W, pady=(8, 0))
        self.transition_source_var = tk.StringVar()
        self.transition_target_var = tk.StringVar()
        ttk.Label(transition_row, text="遷移元").pack(side=tk.LEFT)
        self.transition_source_combo = ttk.Combobox(transition_row, textvariable=self.transition_source_var, values=[], width=16)
        self.transition_source_combo.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(transition_row, text="遷移先").pack(side=tk.LEFT)
        self.transition_target_combo = ttk.Combobox(transition_row, textvariable=self.transition_target_var, values=[], width=16)
        self.transition_target_combo.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(transition_row, text="遷移を設定", command=self.set_transition).pack(side=tk.LEFT)

        table_frame = ttk.Frame(state_editor_frame)
        table_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.state_table = ttk.Treeview(
            table_frame,
            columns=("name", "next_state", "machine_type", "processing_time", "interval", "operation"),
            show="headings",
            height=8,
        )
        for column, label, width in [
            ("name", "状態名", 140),
            ("next_state", "遷移先", 140),
            ("machine_type", "マシン種別", 90),
            ("processing_time", "処理時間", 90),
            ("interval", "インターバル", 90),
            ("operation", "操作名", 260),
        ]:
            self.state_table.heading(column, text=label)
            self.state_table.column(column, width=width, anchor=tk.W)
        self.state_table.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self.state_table.bind("<<TreeviewSelect>>", self.on_state_selected)
        state_scroll = ttk.Scrollbar(table_frame, orient=tk.VERTICAL, command=self.state_table.yview)
        self.state_table.configure(yscrollcommand=state_scroll.set)
        state_scroll.pack(side=tk.RIGHT, fill=tk.Y)

        self.transition_canvas = tk.Canvas(state_editor_frame, bg="white", height=260)
        self.transition_canvas.pack(fill=tk.X, pady=(10, 0))

        validate_frame = ttk.LabelFrame(self.wizard_tab, text="3. 入力バリデーション", padding=10)
        validate_frame.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Button(validate_frame, text="下書きを検証", command=self.validate_current_draft).pack(anchor=tk.W)
        ttk.Button(validate_frame, text="Experimentを作成", command=self.create_experiment).pack(anchor=tk.W, pady=(6, 0))
        self.validation_result_text = tk.Text(validate_frame, height=8)
        self.validation_result_text.pack(fill=tk.BOTH, expand=True, pady=(8, 0))

    def _build_machine_tab(self) -> None:
        frame = ttk.LabelFrame(self.machine_tab, text="7. マシン管理画面", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        input_row = ttk.Frame(frame)
        input_row.pack(fill=tk.X)
        ttk.Label(input_row, text="マシン種別").pack(side=tk.LEFT)
        ttk.Entry(input_row, textvariable=self.machine_type_add_var, width=10).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(input_row, text="説明").pack(side=tk.LEFT)
        ttk.Entry(input_row, textvariable=self.machine_description_var, width=36).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(input_row, text="マシン追加", command=self.add_machine).pack(side=tk.LEFT)
        ttk.Button(input_row, text="選択マシン削除", command=self.delete_selected_machine).pack(side=tk.LEFT, padx=(8, 0))
        ttk.Button(input_row, text="一覧更新", command=self.refresh_machine_table).pack(side=tk.LEFT, padx=(8, 0))

        self.machine_table = ttk.Treeview(frame, columns=("machine_id", "machine_type", "description"), show="headings", height=16)
        self.machine_table.heading("machine_id", text="machine_id")
        self.machine_table.heading("machine_type", text="machine_type")
        self.machine_table.heading("description", text="description")
        self.machine_table.column("machine_id", width=340, anchor=tk.W)
        self.machine_table.column("machine_type", width=120, anchor=tk.W)
        self.machine_table.column("description", width=560, anchor=tk.W)
        self.machine_table.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def _build_experiment_tab(self) -> None:
        frame = ttk.LabelFrame(self.experiment_tab, text="8. 実験一覧とフィルタ", padding=10)
        frame.pack(fill=tk.BOTH, expand=True)

        toolbar = ttk.Frame(frame)
        toolbar.pack(fill=tk.X)
        ttk.Label(toolbar, text="ステータス").pack(side=tk.LEFT)
        self.experiment_filter_combo = ttk.Combobox(
            toolbar,
            textvariable=self.experiment_filter_var,
            values=["ALL", "NOT_STARTED", "IN_PROGRESS", "COMPLETED", "ERROR"],
            width=16,
            state="readonly",
        )
        self.experiment_filter_combo.pack(side=tk.LEFT, padx=(4, 8))
        ttk.Label(toolbar, text="検索").pack(side=tk.LEFT)
        ttk.Entry(toolbar, textvariable=self.experiment_search_var, width=30).pack(side=tk.LEFT, padx=(4, 8))
        ttk.Button(toolbar, text="一覧更新", command=self.refresh_experiment_table).pack(side=tk.LEFT)

        self.experiment_table = ttk.Treeview(
            frame,
            columns=("name", "uuid", "current_state", "status", "task_count"),
            show="headings",
            height=14,
        )
        for column, label, width in [
            ("name", "実験名", 220),
            ("uuid", "uuid", 320),
            ("current_state", "現在状態", 180),
            ("status", "ステータス", 120),
            ("task_count", "タスク数", 80),
        ]:
            self.experiment_table.heading(column, text=label)
            self.experiment_table.column(column, width=width, anchor=tk.W)
        self.experiment_table.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        self.experiment_table.bind("<<TreeviewSelect>>", self.show_experiment_detail)

        self.experiment_detail_text = tk.Text(frame, height=8)
        self.experiment_detail_text.pack(fill=tk.BOTH, expand=True, pady=(10, 0))

    def log_event(self, message: str) -> None:
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.event_log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.event_log_text.see(tk.END)

    def refresh_all_views(self) -> None:
        self.refresh_summary()
        self.refresh_machine_table()
        self.refresh_experiment_table()
        self.refresh_state_table()

    def refresh_summary(self) -> None:
        experiment_count = len(self.experiments.experiments)
        machine_count = len(self.experiments.machine_list.machines)
        status_counter = {"NOT_STARTED": 0, "IN_PROGRESS": 0, "COMPLETED": 0, "ERROR": 0}

        for experiment in self.experiments.experiments:
            if experiment.current_task_group is None:
                continue
            status_name = experiment.current_task_group.status.name
            if status_name in status_counter:
                status_counter[status_name] += 1

        self.summary_var.set(
            "実験数: {0} / マシン数: {1} / NOT_STARTED: {2}, IN_PROGRESS: {3}, COMPLETED: {4}, ERROR: {5}".format(
                experiment_count,
                machine_count,
                status_counter["NOT_STARTED"],
                status_counter["IN_PROGRESS"],
                status_counter["COMPLETED"],
                status_counter["ERROR"],
            )
        )

    def upsert_state(self) -> None:
        try:
            state = StateDraft(
                name=self.state_name_var.get().strip(),
                next_state=self.next_state_var.get().strip(),
                machine_type=int(self.machine_type_var.get().strip()),
                processing_time=int(self.processing_time_var.get().strip()),
                interval=int(self.interval_var.get().strip()),
                operation=self.operation_var.get().strip(),
            )
        except ValueError:
            messagebox.showerror("入力エラー", "マシン種別、処理時間、インターバルは整数で入力してください。")
            return

        existing_index = self._find_state_index(state.name)
        if existing_index is None:
            self.state_drafts.append(state)
            self.log_event(f"状態を追加しました: {state.name}")
        else:
            self.state_drafts[existing_index] = state
            self.log_event(f"状態を更新しました: {state.name}")

        if not self.initial_state_var.get().strip():
            self.initial_state_var.set(state.name)

        self.refresh_state_table()
        self.clear_state_form(keep_initial=True)

    def remove_selected_state(self) -> None:
        selected_item = self.state_table.selection()
        if not selected_item:
            messagebox.showwarning("選択なし", "削除する状態を選択してください。")
            return

        selected_state_name = self.state_table.item(selected_item[0], "values")[0]
        self.state_drafts = [state for state in self.state_drafts if state.name != selected_state_name]
        if self.initial_state_var.get() == selected_state_name:
            self.initial_state_var.set(self.state_drafts[0].name if self.state_drafts else "")
        self.log_event(f"状態を削除しました: {selected_state_name}")
        self.refresh_state_table()

    def clear_state_form(self, keep_initial: bool = False) -> None:
        self.state_name_var.set("")
        self.next_state_var.set("")
        self.machine_type_var.set("0")
        self.processing_time_var.set("10")
        self.interval_var.set("0")
        self.operation_var.set("generic_operation")
        if not keep_initial:
            self.initial_state_var.set("")

    def on_state_selected(self, _event: tk.Event) -> None:
        selected_item = self.state_table.selection()
        if not selected_item:
            return

        values = self.state_table.item(selected_item[0], "values")
        self.state_name_var.set(str(values[0]))
        self.next_state_var.set(str(values[1]))
        self.machine_type_var.set(str(values[2]))
        self.processing_time_var.set(str(values[3]))
        self.interval_var.set(str(values[4]))
        self.operation_var.set(str(values[5]))

    def set_transition(self) -> None:
        source = self.transition_source_var.get().strip()
        target = self.transition_target_var.get().strip()
        if not source or not target:
            messagebox.showwarning("入力不足", "遷移元と遷移先を指定してください。")
            return

        index = self._find_state_index(source)
        if index is None:
            messagebox.showerror("未定義状態", f"遷移元状態 '{source}' が存在しません。")
            return

        state = self.state_drafts[index]
        self.state_drafts[index] = StateDraft(
            name=state.name,
            next_state=target,
            machine_type=state.machine_type,
            processing_time=state.processing_time,
            interval=state.interval,
            operation=state.operation,
        )
        self.log_event(f"遷移を更新しました: {source} -> {target}")
        self.refresh_state_table()

    def refresh_state_table(self) -> None:
        for item in self.state_table.get_children():
            self.state_table.delete(item)

        for state in self.state_drafts:
            self.state_table.insert(
                "",
                tk.END,
                values=(
                    state.name,
                    state.next_state,
                    state.machine_type,
                    state.processing_time,
                    state.interval,
                    state.operation,
                ),
            )

        choices = [state.name for state in self.state_drafts]
        self.initial_state_combo["values"] = choices
        self.next_state_combo["values"] = choices
        self.transition_source_combo["values"] = choices
        self.transition_target_combo["values"] = choices

        self.draw_transition_graph()

    def draw_transition_graph(self) -> None:
        self.transition_canvas.delete("all")
        if not self.state_drafts:
            self.transition_canvas.create_text(12, 12, text="状態を追加すると遷移図を表示します。", anchor=tk.NW)
            return

        width = max(self.transition_canvas.winfo_width(), 780)
        height = max(self.transition_canvas.winfo_height(), 240)
        center_x = width // 2
        center_y = height // 2
        radius = min(width, height) * 0.34

        positions: dict[str, tuple[float, float]] = {}
        for index, state in enumerate(self.state_drafts):
            theta = (2.0 * math.pi * index) / max(1, len(self.state_drafts))
            x = center_x + radius * math.cos(theta)
            y = center_y + radius * math.sin(theta)
            positions[state.name] = (x, y)

        for state in self.state_drafts:
            src = positions.get(state.name)
            dst = positions.get(state.next_state)
            if src is None or dst is None:
                continue
            self.transition_canvas.create_line(src[0], src[1], dst[0], dst[1], arrow=tk.LAST, fill="#345995", width=2)

        for state_name, (x, y) in positions.items():
            fill = "#F7E08A" if state_name == self.initial_state_var.get().strip() else "#BFD7EA"
            self.transition_canvas.create_oval(x - 45, y - 24, x + 45, y + 24, fill=fill, outline="#1F2933", width=2)
            self.transition_canvas.create_text(x, y, text=state_name)

    def validate_current_draft(self) -> tuple[list[str], list[str]]:
        draft = self._current_draft()
        machine_types = {machine.machine_type for machine in self.experiments.machine_list.machines}
        errors, warnings = validate_experiment_draft(draft, machine_types)
        self.validation_result_text.delete("1.0", tk.END)

        if errors:
            self.validation_result_text.insert(tk.END, "Errors:\n")
            for err in errors:
                self.validation_result_text.insert(tk.END, f"- {err}\n")
        if warnings:
            self.validation_result_text.insert(tk.END, "Warnings:\n")
            for warning in warnings:
                self.validation_result_text.insert(tk.END, f"- {warning}\n")
        if not errors and not warnings:
            self.validation_result_text.insert(tk.END, "バリデーション結果: 問題なし\n")

        self.log_event(f"バリデーション実行: errors={len(errors)} warnings={len(warnings)}")
        return errors, warnings

    def create_experiment(self) -> None:
        errors, warnings = self.validate_current_draft()
        if errors:
            messagebox.showerror("バリデーションエラー", "入力エラーを解消してから作成してください。")
            return

        if warnings:
            proceed = messagebox.askyesno("警告", "警告があります。継続して作成しますか。")
            if not proceed:
                return

        draft = self._current_draft()
        try:
            experiment = build_experiment_from_draft(draft)
            self.experiments.add_experiment(experiment)
        except Exception as err:
            messagebox.showerror("作成失敗", f"Experiment作成に失敗しました: {err}")
            self.log_event(f"Experiment作成失敗: {err}")
            return

        self.log_event(f"Experimentを作成しました: {draft.experiment_name}")
        self.refresh_all_views()
        messagebox.showinfo("完了", f"Experiment '{draft.experiment_name}' を作成しました。")

    def _current_draft(self) -> ExperimentDraft:
        return ExperimentDraft(
            experiment_name=self.experiment_name_var.get().strip(),
            initial_state=self.initial_state_var.get().strip(),
            states=list(self.state_drafts),
        )

    def _find_state_index(self, name: str) -> int | None:
        for index, state in enumerate(self.state_drafts):
            if state.name == name:
                return index
        return None

    def add_machine(self) -> None:
        try:
            machine_type = int(self.machine_type_add_var.get().strip())
        except ValueError:
            messagebox.showerror("入力エラー", "マシン種別は整数で入力してください。")
            return

        description = self.machine_description_var.get().strip()
        try:
            self.experiments.add_machine(machine_type=machine_type, description=description)
        except Exception as err:
            messagebox.showerror("追加失敗", f"マシン追加に失敗しました: {err}")
            self.log_event(f"マシン追加失敗: {err}")
            return

        self.log_event(f"マシン追加: machine_type={machine_type}")
        self.machine_description_var.set("")
        self.refresh_all_views()

    def delete_selected_machine(self) -> None:
        selected_item = self.machine_table.selection()
        if not selected_item:
            messagebox.showwarning("選択なし", "削除するマシンを選択してください。")
            return

        machine_id = self.machine_table.item(selected_item[0], "values")[0]
        try:
            self.experiments.delete_machine_with_machine_id(machine_id)
        except Exception as err:
            messagebox.showerror("削除失敗", f"マシン削除に失敗しました: {err}")
            self.log_event(f"マシン削除失敗: {err}")
            return

        self.log_event(f"マシン削除: machine_id={machine_id}")
        self.refresh_all_views()

    def refresh_machine_table(self) -> None:
        for item in self.machine_table.get_children():
            self.machine_table.delete(item)

        for machine in self.experiments.machine_list.machines:
            self.machine_table.insert(
                "",
                tk.END,
                values=(machine.machine_id, machine.machine_type, machine.description),
            )

    def refresh_experiment_table(self) -> None:
        for item in self.experiment_table.get_children():
            self.experiment_table.delete(item)

        filter_status = self.experiment_filter_var.get().strip().upper()
        search_key = self.experiment_search_var.get().strip().lower()

        for experiment in self.experiments.experiments:
            current_task_group = experiment.current_task_group
            status = current_task_group.status.name if current_task_group is not None else "UNKNOWN"
            task_count = len(current_task_group.tasks) if current_task_group is not None else 0

            if filter_status and filter_status != "ALL" and status != filter_status:
                continue

            searchable = f"{experiment.experiment_name} {experiment.experiment_uuid}".lower()
            if search_key and search_key not in searchable:
                continue

            self.experiment_table.insert(
                "",
                tk.END,
                values=(
                    experiment.experiment_name,
                    experiment.experiment_uuid,
                    experiment.current_state_name,
                    status,
                    task_count,
                ),
            )

    def show_experiment_detail(self, _event: tk.Event) -> None:
        selected_item = self.experiment_table.selection()
        if not selected_item:
            return

        experiment_uuid = self.experiment_table.item(selected_item[0], "values")[1]
        experiment = next((item for item in self.experiments.experiments if item.experiment_uuid == experiment_uuid), None)
        if experiment is None:
            return

        lines = [
            f"実験名: {experiment.experiment_name}",
            f"UUID: {experiment.experiment_uuid}",
            f"現在状態: {experiment.current_state_name}",
            "状態一覧:",
        ]
        for state in experiment.states:
            lines.append(f"- {state.state_name}")
            if hasattr(state, "draft"):
                lines.append(f"  遷移先: {state.draft.next_state} / マシン種別: {state.draft.machine_type}")

        self.experiment_detail_text.delete("1.0", tk.END)
        self.experiment_detail_text.insert(tk.END, "\n".join(lines))

    def run_simulate_one(self) -> None:
        try:
            result = self.experiments.simulate_one(save_each_step=True)
        except Exception as err:
            messagebox.showerror("実行失敗", f"simulate_oneに失敗しました: {err}")
            self.log_event(f"simulate_one失敗: {err}")
            return

        self.log_event(f"simulate_one結果: {result}")
        self.refresh_all_views()

    def run_simulate_many(self) -> None:
        try:
            steps = int(self.simulate_steps_var.get().strip())
            if steps <= 0:
                raise ValueError("steps must be positive")
        except ValueError:
            messagebox.showerror("入力エラー", "ステップ数は1以上の整数で入力してください。")
            return

        try:
            results = self.experiments.simulate(max_steps=steps, save_each_step=True)
        except Exception as err:
            messagebox.showerror("実行失敗", f"simulateに失敗しました: {err}")
            self.log_event(f"simulate失敗: {err}")
            return

        self.log_event(f"simulate結果(steps={steps}): {results}")
        self.refresh_all_views()


def main() -> None:
    app = ExperimentBuilderApp()
    app.mainloop()


if __name__ == "__main__":
    main()
