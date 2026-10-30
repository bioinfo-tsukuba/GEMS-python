"""Export CP-SAT models without solving, for measured/final source comparison."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
from unittest.mock import patch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('source', type=Path)
parser.add_argument('study', type=Path)
parser.add_argument('output', type=Path)
args = parser.parse_args()
sys.path.insert(0, str(args.source.resolve()))
from ortools.sat.python import cp_model
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from scripts.measure_scheduler_capacity import fingerprint, snapshot

class Captured(Exception):
    pass

args.output.mkdir(parents=True, exist_ok=True)
records = []
for pattern in json.loads((args.study / 'config.json').read_text())['patterns']:
    trials = [p for p in (args.study / 'trials').glob(f'*-scan-{pattern}-512-60-0')]
    assert len(trials) == 1, trials
    request = json.loads((trials[0] / 'request.json').read_text())
    manager = Experiments.from_dict(request['input'])
    assert fingerprint(snapshot(manager)) == request['input_hash']
    def capture(solver, model, *unused, **keywords):
        text = str(model.proto)
        (args.output / f'{pattern}.pbtxt').write_text(text)
        records.append(dict(
            pattern=pattern, size=512, input_hash=request['input_hash'],
            model_sha256=hashlib.sha256(text.encode()).hexdigest(),
            variable_count=len(model.proto.variables), constraint_count=len(model.proto.constraints),
            solver_limit=solver.parameters.max_time_in_seconds,
            workers=solver.parameters.num_search_workers, seed=solver.parameters.random_seed,
        ))
        raise Captured()
    try:
        with patch.object(cp_model.CpSolver, 'solve', capture):
            manager.execute_scheduling(solver_time_limit_seconds=60)
    except RuntimeError as error:
        if not isinstance(error.__cause__, Captured):
            raise
    else:
        raise AssertionError('No model was built')
(args.output / 'model_fingerprints.json').write_text(json.dumps(records, indent=2))
print(json.dumps(records, indent=2))
