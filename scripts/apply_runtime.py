#!/usr/bin/env python3
"""Apply the resident experiment to the exact pinned Prism source."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
from apply_decode import apply as apply_decode
from apply_prefill import apply as apply_prefill
from configure_runtime import apply as configure_runtime
from apply_calibration import apply as apply_calibration
from apply_workload import apply as apply_workload
from apply_graph import apply as apply_graph


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    lock = json.loads((root / 'runtime.lock.json').read_text())
    revision = subprocess.check_output(['git', '-C', str(args.source), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != lock['runtime_revision']:
        raise SystemExit('Source revision differs from runtime.lock.json')
    if subprocess.check_output(['git', '-C', str(args.source), 'status', '--porcelain']):
        raise SystemExit('Source must be clean before applying the engine overlay')
    path = args.source / 'ggml/src/ggml-cuda/ggml-cuda.cu'
    text = path.read_text()
    changes = [
        ('#include <vector>\n', '#include <vector>\n#include "ggml-cuda/merlin-profile.cuh"\n'),
        ('    bool graph_evaluated_or_captured = false;\n',
         '    merlin_cuda_profile merlin_profile(cgraph, cuda_ctx->device);\n'
         '    bool graph_evaluated_or_captured = false;\n'),
        ('                // The normalized pre-attention residual is consumed only by a\n',
         '                merlin_cuda_profile::scope merlin_scope(merlin_profile, i,\n'
         '                    cuda_ctx->curr_stream_no, cuda_ctx->stream());\n\n'
         '                // The normalized pre-attention residual is consumed only by a\n'),
        ('    if (graph->is_enabled()) {\n',
         '    if (graph->is_enabled() && !merlin_cuda_profile_enabled() && !merlin_dispatch_calibrating()) {\n'),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise SystemExit('Pinned source integration anchor changed')
        text = text.replace(before, after)
    path.write_text(text)
    shutil.copyfile(root / 'engine/telemetry/merlin-profile.cuh', path.parent / 'merlin-profile.cuh')
    shutil.copyfile(root / 'engine/telemetry/merlin-kernel-log.cuh', path.parent / 'merlin-kernel-log.cuh')
    for header in (root / 'engine/dispatch').glob('merlin-dispatch*'):
        shutil.copyfile(header, path.parent / header.name)
    apply_decode(args.source, root)
    apply_prefill(args.source, root)
    apply_calibration(args.source)
    apply_workload(args.source, root)
    apply_graph(args.source, root)
    configure_runtime(args.source, root, lock)
    print('Applied Merlin packed decode, tiled prefill and runtime logging.', flush=True)


if __name__ == '__main__':
    main()
