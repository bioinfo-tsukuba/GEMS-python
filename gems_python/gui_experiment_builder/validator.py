from gems_python.gui_experiment_builder.models import ExperimentDraft


def validate_experiment_draft(
    draft: ExperimentDraft,
    available_machine_types: set[int] | None = None,
) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not draft.experiment_name.strip():
        errors.append("実験名は必須です。")

    if not draft.states:
        errors.append("状態を1件以上追加してください。")
        return errors, warnings

    state_names = [state.name.strip() for state in draft.states]
    state_name_set = set(state_names)

    if any(not name for name in state_names):
        errors.append("状態名が空の項目があります。")

    if len(state_names) != len(state_name_set):
        errors.append("状態名が重複しています。")

    if not draft.initial_state.strip():
        errors.append("初期状態は必須です。")
    elif draft.initial_state not in state_name_set:
        errors.append("初期状態が状態一覧に存在しません。")

    for state in draft.states:
        if not state.next_state.strip():
            errors.append(f"状態 '{state.name}' の遷移先が未設定です。")
        elif state.next_state not in state_name_set:
            errors.append(f"状態 '{state.name}' の遷移先 '{state.next_state}' が未定義です。")

        if state.processing_time <= 0:
            errors.append(f"状態 '{state.name}' の処理時間は1以上にしてください。")

        if state.interval < 0:
            errors.append(f"状態 '{state.name}' のインターバルは0以上にしてください。")

        if state.machine_type < 0:
            errors.append(f"状態 '{state.name}' のマシン種別は0以上の整数にしてください。")

    if available_machine_types is not None:
        if len(available_machine_types) == 0:
            warnings.append("マシンが未登録です。実験を追加してもスケジューリングに失敗する可能性があります。")
        for state in draft.states:
            if state.machine_type not in available_machine_types:
                errors.append(
                    f"状態 '{state.name}' が要求するマシン種別 {state.machine_type} が未登録です。"
                )

    if draft.initial_state in state_name_set:
        reachable = _collect_reachable_states(draft)
        unreachable = sorted(state_name_set - reachable)
        if unreachable:
            warnings.append(f"初期状態 '{draft.initial_state}' から到達できない状態があります: {', '.join(unreachable)}")

    return errors, warnings


def _collect_reachable_states(draft: ExperimentDraft) -> set[str]:
    next_state_map: dict[str, str] = {}
    for state in draft.states:
        next_state_map[state.name] = state.next_state

    visited: set[str] = set()
    cursor = draft.initial_state

    while cursor and cursor not in visited and cursor in next_state_map:
        visited.add(cursor)
        cursor = next_state_map[cursor]

    return visited
