"""Confirm representative conditions on final source without pooling old samples."""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from gems_python.common.scheduling import validate_groups
from gems_python.multi_machine_problem_interval_task.transition_manager import Experiments
from scripts.measure_scheduler_capacity import environment, fingerprint, make_case, snapshot


def confirm(study, output):
    output.mkdir(parents=True, exist_ok=False)
    config = json.loads((study / 'config.json').read_text())
    env = environment()
    (output / 'environment.json').write_text(json.dumps(env, indent=2))
    (output / 'config.json').write_text(json.dumps(config, indent=2))
    (output / 'results.jsonl').touch()
    primary_env = json.loads((study / 'environment.json').read_text())
    deadline = datetime.fromisoformat(primary_env['started_at_utc']).timestamp() + config['total_runtime_seconds']
    original = [json.loads(line) for line in (study / 'results.jsonl').read_text().splitlines()]
    scans = defaultdict(list)
    for row in original:
        if row['phase'] == 'scan':
            scans[(row['pattern'], row['size'], row['search_limit_seconds'])].append(row)
    chosen = []
    for pattern in config['patterns']:
        one_second = max(size for (p, size, limit), rows in scans.items()
                         if p == pattern and limit == 1 and all(r['outcome'] == 'VALID' for r in rows))
        for phase, size, limit in [('scan', one_second, 1), ('scan', max(config['sizes']), 60),
                                   ('operation', one_second, 1)]:
            chosen.append(next(r for r in original if (r['pattern'], r['phase'], r['size'],
                               r['search_limit_seconds'], r['repetition']) == (pattern, phase, size, limit, 0)))
    verified = []
    for source in chosen:
        # Original confirmation keeps the exact paired input. A smaller
        # operation after UNKNOWN is a separately labelled supplemental probe.
        supplemental = source.get('confirmation_kind') == 'operation_lower_size'
        name = (f"lower-operation-{source['pattern']}-{source['size']}-1"
                if supplemental else Path(source['artifact_directory']).name)
        directory = output / 'trials' / name
        directory.mkdir(parents=True)
        if supplemental:
            value = snapshot(make_case(source['pattern'], source['size'], directory / 'workspace', config))
            request = dict(pattern=source['pattern'], size=source['size'], phase='operation',
                           repetition=0, search_limit_seconds=1.0, input_hash=fingerprint(value),
                           input=value, config=config)
        else:
            request = json.loads((study / source['artifact_directory'] / 'request.json').read_text())
        path = directory / 'request.json'
        path.write_text(json.dumps(request, indent=2))
        limit = request['search_limit_seconds']
        safeguard = limit * (config['operation_steps'] * 3 + 1) + 120 if request['phase'] == 'operation' else limit + 120
        remaining = deadline - datetime.now(timezone.utc).timestamp()
        if remaining < safeguard:
            raise RuntimeError('Confirmation would exceed the approved combined six-hour allowance')
        started = time.perf_counter()
        with (directory / 'worker.log').open('w') as log:
            subprocess.run([sys.executable, 'scripts/measure_scheduler_capacity.py', '--worker', str(path)],
                           stdout=log, stderr=subprocess.STDOUT, check=True, timeout=safeguard)
        row = json.loads((directory / 'result.json').read_text())
        row['worker_process_seconds'] = time.perf_counter() - started
        row['artifact_directory'] = str(directory.relative_to(output))
        row['confirmation_kind'] = 'operation_lower_size' if supplemental else 'original_condition'
        before = Experiments.from_dict(request['input'])
        assert fingerprint(snapshot(before)) == row['input_hash'] == row['actual_input_hash']
        assert all(c['solver_time_limit_seconds'] == limit for c in row['reports'])
        if row['outcome'] == 'VALID':
            after = Experiments.from_json((directory / 'output.json').read_text())
            validate_groups(after.task_groups, after.reference_time, after.machine_list)
            if row['phase'] == 'operation':
                assert len(row['updates']) == config['operation_steps']
                assert all(u['result']['status'] == 'ok' for u in row['updates'])
                assert sum(len(u['added_experiments']) for u in row['updates']) == (
                    len(config['operation_add_at_steps']) * config['operation_add_count'])
        with (output / 'results.jsonl').open('a') as stream:
            stream.write(json.dumps(row, separators=(',', ':')) + '\n')
        verified.append(dict(pattern=row['pattern'], phase=row['phase'], size=row['size'],
                             search_limit_seconds=limit, outcome=row['outcome'],
                             input_hash=row['input_hash'], total_seconds=row['total_seconds'],
                             confirmation_kind=row['confirmation_kind'],
                             solver_calls=len(row['reports']), updates=len(row.get('updates', [])),
                             successful_updates=sum(u['result']['status'] == 'ok' for u in row.get('updates', []))))
        print(f"{name}: {row['outcome']} calls={len(row['reports'])} seconds={row['total_seconds']:.3f}", flush=True)
        if row['phase'] == 'operation' and (
            row['outcome'] == 'NO_SOLUTION' or (not supplemental and source['outcome'] == 'NO_SOLUTION')
        ) and row['size'] > min(config['sizes']):
            chosen.append(dict(pattern=row['pattern'], size=row['size'] // 2,
                               confirmation_kind='operation_lower_size'))
    (output / 'confirmation.json').write_text(json.dumps(dict(
        source_commit=env['commit'], primary_commit=primary_env['commit'],
        results=verified, note='Independent final-source confirmation; not pooled with primary repetitions.'
    ), indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('study', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    confirm(args.study, args.output)
