#!/usr/bin/env python3
"""Summarize diagnostic graph timings without calling them throughput."""
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import sys


def summarize(path):
    totals = defaultdict(lambda: {'calls': 0, 'work_ms': 0.0})
    seen = set()
    for line in path.read_text().splitlines():
        graph = json.loads(line)
        if graph['schema'] != 1 or graph['cuda_graphs'] is not False:
            raise ValueError('Unsupported timing mode')
        identity = (graph['device'], graph['graph'])
        if identity in seen or not 0 <= graph['graph'] < 128:
            raise ValueError('Duplicate or invalid graph')
        seen.add(identity)
        end = -1
        for op in graph['operations']:
            first, last = op['first'], op['last']
            if not end < first <= last < graph['nodes'] or len(op['ops']) != last - first + 1:
                raise ValueError('Overlapping or incomplete fused range')
            end = last
            if not math.isfinite(op['ms']) or op['ms'] < 0:
                raise ValueError('Invalid elapsed time')
            key = (graph['device'], tuple(op['ops']), json.dumps(
                [op['src0'], op['src1'], op['dst']], sort_keys=True))
            totals[key]['calls'] += 1
            totals[key]['work_ms'] += op['ms']
    if not seen or not totals:
        raise ValueError('No measured operations')
    rows = [{'device': key[0], 'ops': key[1], 'shapes': json.loads(key[2]), **value}
            for key, value in totals.items()]
    return {'graphs': len(seen), 'capture_limit_reached': len(seen) == 128,
            'interpretation': 'Diagnostic work time; streams may overlap. Not wall time or kernel-only time.',
            'operations': sorted(rows, key=lambda r: r['work_ms'], reverse=True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('profile', type=Path)
    args = parser.parse_args()
    try:
        report = summarize(args.profile)
    except (ValueError, KeyError, TypeError, OSError):
        print('Invalid or incomplete profile; no report produced.', file=sys.stderr)
        return 1
    print(json.dumps(report, indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
