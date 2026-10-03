#!/usr/bin/env python3
"""Verify workload overlays and shared-library transport against the pinned source headers."""
import argparse
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

from apply_workload import apply


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as directory:
        stage = Path(directory)
        for relative in ('ggml/src/ggml-backend.cpp', 'src/llama-context.cpp', 'tests/test-backend-ops.cpp'):
            target = stage / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(args.source / relative, target)
        (stage / 'ggml/include').mkdir()
        apply(stage, root)
        # The real producer and both test entry points must carry a scope; no source-cache edits.
        llama = (stage / 'src/llama-context.cpp').read_text()
        assert llama.index('const merlin_workload_scope merlin_workload(') < llama.index(
            'const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);')
        assert (stage / 'tests/test-backend-ops.cpp').read_text().count(
            'const merlin_workload_scope merlin_scope(merlin_workload);') == 2
        compiler = [os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror', '-pthread',
                    '-I', str(root / 'engine/core'), '-I', str(args.source / 'ggml/include'),
                    '-I', str(args.source / 'include'), '-I', str(args.source / 'src')]
        shared = ['-dynamiclib'] if sys.platform == 'darwin' else ['-shared', '-fPIC']
        suffix = '.dylib' if sys.platform == 'darwin' else '.so'
        central = stage / ('libworkload' + suffix)
        implementation = stage / 'workload.cpp'
        implementation.write_text('#include "merlin-workload-impl.h"\n')
        subprocess.run(compiler + shared + [str(implementation), '-o', str(central)], check=True)
        consumer = stage / ('consumer' + suffix)
        subprocess.run(compiler + shared + [str(root / 'engine/tests/workload_consumer.cpp'), str(central),
                                          '-o', str(consumer)], check=True)
        executable = stage / 'workload-transport'
        subprocess.run(compiler + [str(root / 'engine/tests/workload_transport.cpp'), str(central),
                                   '-o', str(executable)] + ([] if sys.platform == 'darwin' else ['-ldl']), check=True)
        subprocess.run([str(executable), str(consumer)], check=True)


if __name__ == '__main__':
    main()
