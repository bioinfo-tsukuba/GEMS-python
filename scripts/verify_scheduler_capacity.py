"""Verify preserved inputs, classifications and physical outputs without solving."""
import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems_python.common.scheduling import validate_groups, validate_inputs
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from scripts.measure_scheduler_capacity import classify, fingerprint, snapshot


def verify(root):
    rows = [json.loads(line) for line in (root / 'results.jsonl').read_text().splitlines()]
    config = json.loads((root / 'config.json').read_text())
    completion = json.loads((root / 'completion.json').read_text())
    assert completion['fully_measured'] and not completion['skipped'], completion
    assert completion['trial_count'] == len(rows)
    hashes, counts = defaultdict(set), Counter()
    for row in rows:
        directory = root / row['artifact_directory']
        request = json.loads((directory / 'request.json').read_text())
        manager = Experiments.from_dict(request['input'])
        digest = fingerprint(snapshot(manager))
        assert digest == request['input_hash'] == row['input_hash'] == row['actual_input_hash']
        validate_inputs(manager.task_groups, manager.reference_time, manager.machine_list)
        counts['verified_inputs'] += 1
        assert row['outcome'] == classify(row['reports'], row.get('error'))
        assert all(call['solver_time_limit_seconds'] == row['search_limit_seconds']
                   for call in row['reports'])
        counts['verified_calls'] += len(row['reports'])
        for update in row.get('updates', []):
            assert fingerprint(update['input']) == update['input_hash']
            prior = Experiments.from_dict(update['input'])
            validate_inputs(prior.task_groups, prior.reference_time, prior.machine_list)
            if update['step'] > 0:
                validate_groups(prior.task_groups, prior.reference_time, prior.machine_list)
            counts['verified_update_inputs'] += 1
        if row['phase'] == 'scan':
            hashes[(row['pattern'], row['size'])].add(digest)
        output = Experiments.from_json((directory / 'output.json').read_text())
        if row['outcome'] == 'VALID':
            validate_groups(output.task_groups, output.reference_time, output.machine_list)
            counts['verified_valid_outputs'] += 1
            if row['phase'] == 'io':
                saved_json = Experiments.from_json((directory / 'saved/experiments.json').read_text())
                saved_pickle = Experiments.from_pickle(
                    dumped=(directory / 'saved/experiments.pkl').read_bytes())
                digest = fingerprint(snapshot(output))
                assert digest == fingerprint(snapshot(saved_json)) == fingerprint(snapshot(saved_pickle))
                assert all((directory / 'drawing' / f'gantt_chart.{extension}').stat().st_size > 0
                           for extension in ('png', 'pdf', 'svg'))
                counts['verified_io_roundtrips'] += 1
            if row['phase'] != 'operation':
                before = {(g.experiment_uuid, g.task_group_id): g for g in manager.task_groups}
                penalty = 0
                for group in output.task_groups:
                    original = before[(group.experiment_uuid, group.task_group_id)]
                    assert len(group.tasks) == len(original.tasks)
                    if original.is_not_started():
                        start = group.tasks[0].scheduled_time - group.tasks[0].interval
                        penalty += original.penalty_type.calculate_penalty(start, original.optimal_start_time)
                    else:
                        for old, new in zip(original.tasks, group.tasks):
                            assert old.scheduled_time == new.scheduled_time
                            if old.allocated_machine_id is not None:
                                assert old.allocated_machine_id == new.allocated_machine_id
                assert penalty == row['movable_penalty'] == row['reports'][-1]['penalty']
                counts['verified_static_quality_and_fixed_work'] += 1
            else:
                assert len(row['updates']) == config['operation_steps']
                assert len(row['reports']) == sum(len(u['calls']) for u in row['updates'])
                assert all(u['result']['status'] == 'ok' for u in row['updates'])
                expected_additions = len(config['operation_add_at_steps']) * config['operation_add_count']
                assert sum(len(u['added_experiments']) for u in row['updates']) == expected_additions
                assert len(output.experiments) == row['size'] + expected_additions
        else:
            assert (directory / 'error.txt').is_file()
            counts['preserved_failures'] += 1
    assert all(len(values) == 1 for values in hashes.values()), hashes
    assert {r['pattern'] for r in rows if r['phase'] == 'operation'} == set(config['patterns'])
    assert sum(r['phase'] == 'operation' for r in rows) == len(config['patterns']) * len(config['search_limits_seconds'])
    report = dict(counts, trial_count=len(rows), paired_static_conditions=len(hashes),
                  outcomes=dict(Counter(r['outcome'] for r in rows)),
                  phases=dict(Counter(r['phase'] for r in rows)))
    (root / 'artifact_verification.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('directory', type=Path)
    verify(parser.parse_args().directory)
