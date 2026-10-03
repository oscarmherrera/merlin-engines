#!/usr/bin/env python3
"""Build and stage the RTX8000 diagnostic runtime; never load or install it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


def run(*args, **kwargs):
    subprocess.run([str(a) for a in args], check=True, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--host-cxx', default='/usr/bin/g++-13')
    parser.add_argument('--host-cc', default='/usr/bin/gcc-13')
    parser.add_argument('--jobs', type=int, default=8)
    parser.add_argument('--source-cache', type=Path, help='Existing local Git checkout of the pinned runtime')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error('jobs must be 1..32')
    root = Path(__file__).resolve().parents[1]
    if subprocess.check_output(['git', '-C', str(root), 'status', '--porcelain']):
        parser.error('Build from a clean, committed engine repository')
    engine_revision = subprocess.check_output(['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    lock = json.loads((root / 'runtime.lock.json').read_text())
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    cuda = args.cuda.resolve()
    nvcc = cuda / 'bin/nvcc'
    compiler = subprocess.check_output([str(nvcc), '--version'], text=True)
    if 'release 12.' not in compiler:
        parser.error('The resident CUDA build requires CUDA 12.x (later P40 support)')
    source, build, bundle = output / 'source', output / 'build', output / 'runtime'
    repository = str(args.source_cache.resolve()) if args.source_cache else lock['runtime_repository']
    run('git', 'clone', '--depth', '1', '--branch', lock['runtime_tag'], repository, source)
    run(sys.executable, root / 'scripts/apply_runtime.py', source)
    environment = dict(os.environ, CUDACXX=str(nvcc), CUDAHOSTCXX=args.host_cxx)
    environment['PATH'] = str(cuda / 'bin') + os.pathsep + environment.get('PATH', '')
    run('cmake', '-S', source, '-B', build,
        '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF',
        '-DGGML_CUDA=ON', '-DGGML_CUDA_FA_ALL_QUANTS=ON', '-DCMAKE_CUDA_ARCHITECTURES=75',
        f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCUDAToolkit_ROOT={cuda}',
        f'-DCMAKE_CUDA_HOST_COMPILER={args.host_cxx}', f'-DCMAKE_CXX_COMPILER={args.host_cxx}',
        f'-DCMAKE_C_COMPILER={args.host_cc}',
        '-DCMAKE_INSTALL_RPATH=$ORIGIN', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
        '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=ON', env=environment)
    run('cmake', '--build', build, '--parallel', args.jobs,
        '--target', 'llama-server', 'llama-bench', 'test-backend-ops', env=environment)
    shutil.copytree(build / 'bin', bundle, symlinks=True)
    # Stage CUDA user-space dependencies alongside $ORIGIN-linked GGML libraries.
    # The device driver is provided by the eventual target host.
    for pattern in ('libcudart.so*', 'libcublas.so*', 'libcublasLt.so*'):
        matches = list((cuda / 'lib').glob(pattern))
        if not matches:
            raise RuntimeError('Missing CUDA runtime dependency: ' + pattern)
        for path in matches:
            shutil.copy2(path, bundle / path.name, follow_symlinks=False)
    shutil.copyfile(source / 'LICENSE', bundle / 'LICENSE.llama.cpp')
    manifest = {'engine_revision': engine_revision, 'runtime_revision': lock['runtime_revision'],
                'endpoint_binding': lock['endpoint_binding'], 'cuda_architectures': ['75'],
                'nvcc': compiler.strip(), 'host_cxx': subprocess.check_output(
                    [args.host_cxx, '--version'], text=True).splitlines()[0],
                'gpu_validation': 'NOT RUN', 'files': {}}
    correction = cuda / 'merlin-glibc-compat.json'
    if correction.exists():
        manifest['cuda_header_correction'] = json.loads(correction.read_text())
    with (cuda / 'include/crt/math_functions.h').open('rb') as f:
        manifest['cuda_math_header_sha256'] = hashlib.file_digest(f, 'sha256').hexdigest()
    for path in sorted(bundle.iterdir()):
        if path.is_symlink():
            manifest['files'][path.name] = {'symlink': os.readlink(path)}
        elif path.is_file():
            with path.open('rb') as f:
                manifest['files'][path.name] = {'sha256': hashlib.file_digest(f, 'sha256').hexdigest(),
                                               'bytes': path.stat().st_size}
    (bundle / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Staged runtime:', bundle, flush=True)
    print('GPU correctness, profiling, and endpoint validation have NOT run.', flush=True)


if __name__ == '__main__':
    main()
