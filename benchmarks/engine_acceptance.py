#!/usr/bin/env python3
"""Judge a completed custom-engine case against a matched stock case and execution receipts."""
import argparse
import json
import math
import re
from pathlib import Path


def read(path):
    return json.loads(path.read_text())


def request_identity(value):
    return {k: v for k, v in value.items() if k not in ('session_id', 'reap_sessions')}


def runtime_identity(case, start_ms):
    """Require explicit endpoint PID and mapped-library hashes; never infer from timing."""
    try:
        value = read(case / 'runtime-identity.json')
        revision = value['engine_revision']
        libraries = value['runtime_libraries']
        required = {'libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'libggml-cuda.so'}
        valid = (value.get('evidence_source') == 'proc_maps_and_sha256' and
                 type(value.get('pid')) is int and value['pid'] > 0 and
                 re.fullmatch(r'[0-9a-f]{40}', revision) is not None and
                 0 <= start_ms / 1000 - value['captured_unix'] <= 300 and
                 isinstance(libraries, list) and len(libraries) == len(required))
        names = set()
        for item in libraries:
            names.add(item['name'])
            valid = valid and re.fullmatch(r'[0-9a-f]{64}', item['sha256']) is not None
            valid = valid and Path(item['path']).is_absolute()
            valid = valid and ('merlin-engines-' + revision[:7]) in Path(item['path']).parts
        return value if valid and names == required else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def judge(case, baseline, telemetry, minimum_improvement=0.0):
    if not math.isfinite(minimum_improvement) or not 0 <= minimum_improvement < 1:
        raise ValueError('Improvement threshold must be finite and in [0,1)')
    actual, reference = read(case / 'measurement.json'), read(baseline / 'measurement.json')
    start = read(case / 'attempt.json')['started_unix'] * 1000
    end = start + actual['wall_seconds'] * 1000
    before, stock_before = read(case / 'before.json'), read(baseline / 'before.json')
    correctness = read(case / 'verdict.json').get('retrieval_passed') is True
    stock_correct = read(baseline / 'verdict.json').get('retrieval_passed') is True
    controls = ('alias', 'class', 'context_size', 'slots', 'batch_size')
    matched = all(k in before and before[k] == stock_before.get(k) for k in controls)
    matched = matched and before.get('kv_cells_resident') == stock_before.get('kv_cells_resident') == 0
    matched = matched and request_identity(read(case / 'request.json')) == request_identity(read(baseline / 'request.json'))
    matched = matched and actual.get('usage') == reference.get('usage') and stock_correct
    ttft, stock_ttft = actual.get('ttft_seconds'), reference.get('ttft_seconds')
    times_valid = all(isinstance(x, (int, float)) and math.isfinite(x) and x > 0 for x in (ttft, stock_ttft))
    benefit = matched and times_valid and ttft < stock_ttft * (1 - minimum_improvement)
    runtime = runtime_identity(case, start)
    profile_pids, proofs = set(), set()
    with telemetry.open() as stream:
        for number, line in enumerate(stream):
            if number >= 1_000_000 or len(line) > 65536:
                raise ValueError('Telemetry exceeds bounded acceptance input')
            row = json.loads(line)
            if runtime is None or row.get('mode') != 'inference' or row.get('pid') != runtime['pid']:
                continue
            if (row.get('event') == 'merlin_profile_status' and row.get('profile_match') is True and
                    row.get('engine_revision') == runtime['engine_revision']):
                profile_pids.add(row.get('pid'))
            if row.get('gpu_completion') is not True or row.get('completed_custom_operations', 0) <= 0:
                continue
            submitted = row.get('submitted_unix_ms')
            if not isinstance(submitted, (int, float)) or not start <= submitted <= end:
                continue
            if row.get('event') == 'merlin_kernel_completion' and row.get('kernel') not in (None, '', 'prism'):
                identity = (row.get('pid'), row.get('device'), 'kernel', row.get('completion_id'))
            elif row.get('event') == 'merlin_graph_completion':
                identity = (row.get('pid'), row.get('device'), 'graph', row.get('monitor_id'),
                            row.get('generation'), row.get('submission'))
            else:
                continue
            if None not in identity:
                proofs.add(identity)
    completions = sum(identity[0] in profile_pids for identity in proofs)
    passed = correctness and completions > 0 and benefit
    return {'retrieval_passed': correctness, 'matched_baseline': matched,
            'runtime_identity_verified': runtime is not None,
            'missing_runtime_proof': None if runtime else 'Need fresh endpoint PID, engine revision and five mapped-library hashes in runtime-identity.json',
            'distinct_sampled_custom_completions': completions,
            'custom_execution_proven': completions > 0,
            'matched_ttft_improvement_fraction': 1 - ttft / stock_ttft if matched and times_valid else None,
            'matched_performance_benefit': bool(benefit), 'experiment_passed': bool(passed),
            'classification': 'custom_case_improved' if passed else
                'fallback_correctness_only' if correctness and completions == 0 else 'experiment_not_demonstrated',
            'engine_viability_proven': False,
            'scope': 'One matched case; full logits, batching, context range and stability acceptance remain separate.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--case', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--telemetry', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--minimum-improvement', type=float, default=0.0)
    args = parser.parse_args()
    result = judge(args.case, args.baseline, args.telemetry, args.minimum_improvement)
    with args.output.open('x') as output:
        json.dump(result, output, indent=2)
        output.write('\n')
    print(json.dumps(result))
    return 0 if result['experiment_passed'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
