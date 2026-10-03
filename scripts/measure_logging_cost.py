#!/usr/bin/env python3
"""Host-only production logger A/B; this does not measure CUDA or endpoint overhead."""
import json
import os
from pathlib import Path
import statistics
import subprocess
import tempfile


def main():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as temporary:
        directory = Path(temporary)
        binary = directory / 'logging-cost'
        subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-O2', '-Wall', '-Wextra', '-Werror',
            '-I', str(root / 'engine/telemetry'), str(root / 'engine/tests/logging_cost.cpp'), '-o', str(binary)], check=True)
        results = []
        for index in range(6):
            enabled = index % 2 == 0
            output = directory / f'{index}.jsonl'
            env = dict(os.environ, MERLIN_KERNEL_LOG=str(output), MERLIN_ENGINE_LOG_DISABLE='0' if enabled else '1')
            run = subprocess.run([str(binary)], env=env, check=True, capture_output=True, text=True)
            result = json.loads(run.stdout)
            result['records'] = len(output.read_text().splitlines()) if output.exists() else 0
            results.append(result)
        on = statistics.median(r['host_wall_seconds'] for r in results if r['logging_enabled'])
        off = statistics.median(r['host_wall_seconds'] for r in results if not r['logging_enabled'])
        print(json.dumps({'scope': 'Host-only; 100000 sampled dispatch calls plus one completion per128 calls',
            'runs': results, 'median_on_seconds': on, 'median_off_seconds': off,
            'additional_ns_per_dispatch': (on - off) * 1e9 / 100000,
            'gpu_or_endpoint_overhead_measured': False}, indent=2))


if __name__ == '__main__':
    main()
