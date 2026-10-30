"""Structured failures; callers must never infer solver status from text."""


class SolverFailure(RuntimeError):
    def __init__(self, status, diagnostics):
        self.status = status
        self.diagnostics = dict(diagnostics)
        super().__init__(f"OR-Tools failed to find a schedule: {status}")


class PlanningError(RuntimeError):
    def __init__(self, diagnostic):
        self.diagnostic = diagnostic
        super().__init__(f"{diagnostic['kind']}: {diagnostic['reason']}")
