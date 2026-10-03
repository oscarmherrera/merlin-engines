#!/usr/bin/env python3
"""Calibrate representative shapes before endpoint startup, with persistent logs."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import subprocess
import time

from calibration_shapes import backend_shapes, pq2_shapes, requested_workload


def verify_receipts(receipt, manifest, requested):
    if not receipt.is_file():
        raise ValueError('No current calibration executions were recorded')
    observed = set()
    for line in receipt.read_text().splitlines():
        record = json.loads(line)
        if (record.get('schema') != 2 or record.get('profile_saved') is not True or
                record.get('reference_valid') is not True or record.get('records', 0) < 1 or
                any(record.get(field) != manifest[field] for field in
                    ('engine_revision', 'runtime_revision', 'cutlass_revision'))):
            raise ValueError('Calibration receipt does not establish a valid current profile')
        shape = (record['m'], record['n'], record['k'], record['sequence_batch'],
                 record['tokens_in_flight'], record['phase'])
        if shape not in requested:
            raise ValueError('Unexpected calibration shape')
        observed.add(shape)
    if not observed:
        raise ValueError('Zero calibration shapes executed')
    return observed


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runtime', type=Path, required=True)
    parser.add_argument('--log-dir', type=Path, required=True)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ubatch-size', type=int, default=2048)
    args, _ = parser.parse_known_args()
    if not 1 <= args.ubatch_size <= 65536:
        parser.error('ubatch-size must be 1..65536')
    profile_value = os.environ.get('MERLIN_ENGINE_PROFILE')
    if not profile_value:
        parser.error('MERLIN_ENGINE_PROFILE is required')
    profile = Path(profile_value)
    profile.parent.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    shapes_path = args.log_dir / 'calibration-shapes.txt'
    with args.model.open('rb') as stream:
        matrices = pq2_shapes(stream)
    shapes = list(backend_shapes(matrices, args.ubatch_size))
    if not shapes:
        raise SystemExit('No PQ2 shapes available for calibration')
    shapes_path.write_text('\n'.join(shapes) + '\n')
    receipt = args.log_dir / 'calibration-receipt.jsonl'
    if receipt.exists():
        raise SystemExit('Calibration receipt must be a new file')
    environment = dict(os.environ,
                       MERLIN_ENGINE_CALIBRATE='1',
                       MERLIN_CALIBRATION_RECEIPT=str(receipt),
                       MERLIN_KERNEL_LOG=str(args.log_dir / 'calibration-kernels.jsonl'),
                       LD_LIBRARY_PATH=str(args.runtime))
    # Operation diagnostics synchronize per node and would contaminate medians.
    environment.pop('MERLIN_CUDA_PROFILE', None)
    print(f'merlin-engine: startup calibration {len(shapes)} graphs; log {args.log_dir / "calibration.log"}', flush=True)
    started = time.monotonic()
    with Path(str(profile) + '.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        with (args.log_dir / 'calibration.log').open('x') as output:
            result = subprocess.run(
                [str(args.runtime / 'test-backend-ops'), 'perf', '-b', 'CUDA0',
                 '-o', 'MUL_MAT', '--test-file', str(shapes_path)],
                env=environment, stdout=output, stderr=subprocess.STDOUT, check=False)
    summary = {'graphs_requested': len(shapes), 'exit_code': result.returncode,
               'elapsed_seconds': time.monotonic() - started,
               'profile_present': profile.is_file() and profile.stat().st_size > 0}
    requested = {requested_workload(line) for line in shapes}
    try:
        manifest = json.loads((args.runtime / 'manifest.json').read_text())
        observed = verify_receipts(receipt, manifest, requested)
        summary['verified_calibrated_shapes'] = len(observed)
        summary['unprofiled_reference_shapes'] = sorted(requested - observed)
    except (ValueError, KeyError, OSError) as error:
        summary['validation_error'] = str(error)
    (args.log_dir / 'calibration-result.json').write_text(json.dumps(summary, indent=2) + '\n')
    if result.returncode or not summary['profile_present'] or 'validation_error' in summary:
        raise SystemExit('Startup calibration failed; see calibration.log and calibration-result.json')
    print(f'merlin-engine: startup calibration finished in {summary["elapsed_seconds"]:.3f}s', flush=True)


if __name__ == '__main__':
    main()
