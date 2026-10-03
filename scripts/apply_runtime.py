#!/usr/bin/env python3
"""Apply the resident experiment to the exact pinned Prism source."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess


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
         '    if (graph->is_enabled() && !merlin_cuda_profile_enabled()) {\n'),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise SystemExit('Pinned source integration anchor changed')
        text = text.replace(before, after)
    path.write_text(text)
    shutil.copyfile(root / 'runtime/merlin-profile.cuh', path.parent / 'merlin-profile.cuh')
    print('Applied graph profiler; ordinary inference dispatch is unchanged.')


if __name__ == '__main__':
    main()
