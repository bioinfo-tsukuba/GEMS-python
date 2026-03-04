__all__ = ["ExperimentBuilderApp"]


def __getattr__(name: str):
    if name == "ExperimentBuilderApp":
        from gems_python.gui_experiment_builder.app import ExperimentBuilderApp

        return ExperimentBuilderApp
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
