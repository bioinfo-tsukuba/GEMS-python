from dataclasses import dataclass, field


@dataclass
class StateDraft:
    name: str
    next_state: str
    machine_type: int = 0
    processing_time: int = 10
    interval: int = 0
    operation: str = "generic_operation"


@dataclass
class ExperimentDraft:
    experiment_name: str
    initial_state: str
    states: list[StateDraft] = field(default_factory=list)

    def state_names(self) -> list[str]:
        return [state.name for state in self.states]
