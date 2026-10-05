#!/usr/bin/env python3
"""Build and stage the RTX8000 custom runtime; never load or install it."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import shlex
from cutlass_dependency import prepare as prepare_cutlass


def run(*args, **kwargs):
    started = time.monotonic()
    print(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
          'START', shlex.join(str(a) for a in args), flush=True)
    subprocess.run([str(a) for a in args], check=True, **kwargs)
    print('DONE', str(args[0]), 'elapsed_seconds=%.3f' % (time.monotonic() - started), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cuda', type=Path, required=True)
    parser.add_argument('--host-cxx', default='/usr/bin/g++-13')
    parser.add_argument('--host-cc', default='/usr/bin/gcc-13')
    parser.add_argument('--jobs', type=int, default=6)
    parser.add_argument('--cuda-arch', choices=('75', '86'), default='75')
    parser.add_argument('--source-cache', type=Path, help='Existing local Git checkout of the pinned runtime')
    parser.add_argument('--scratch', type=Path, help='New directory for temporary source and objects')
    parser.add_argument('--reuse-scratch', action='store_true',
                        help='Reuse objects for the SAME pinned upstream source; source must be restored clean first')
    args = parser.parse_args()
    if not 1 <= args.jobs <= 32:
        parser.error('jobs must be 1..32')
    if args.reuse_scratch and not args.scratch:
        parser.error('--reuse-scratch requires --scratch')
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
    scratch = args.scratch.resolve() if args.scratch else output
    if args.scratch and not args.reuse_scratch:
        scratch.mkdir(parents=True, exist_ok=False)
    source, build, bundle = scratch / 'source', scratch / 'build', output / 'runtime'
    repository = str(args.source_cache.resolve()) if args.source_cache else lock['runtime_repository']
    if args.reuse_scratch:
        cache = (build / 'CMakeCache.txt').read_text()
        expected = 'CMAKE_HOME_DIRECTORY:INTERNAL=' + str(source)
        if expected not in cache.splitlines():
            parser.error('Existing objects do not belong to this source directory')
        print('Reusing object directory for the unchanged pinned upstream revision.', flush=True)
    else:
        run('git', 'clone', '--depth', '1', '--branch', lock['runtime_tag'], repository, source)
    cutlass = prepare_cutlass(lock, scratch / 'cutlass', run)
    run(sys.executable, root / 'scripts/apply_runtime.py', source)
    environment = dict(os.environ, CUDACXX=str(nvcc), CUDAHOSTCXX=args.host_cxx)
    environment['PATH'] = str(cuda / 'bin') + os.pathsep + environment.get('PATH', '')
    environment['LD_LIBRARY_PATH'] = os.pathsep.join(filter(None, (
        str(cuda / 'lib'), environment.get('LD_LIBRARY_PATH'))))
    backend_objects = build / 'ggml/src/ggml-cuda/CMakeFiles/ggml-cuda.dir'
    flags_path = backend_objects / 'flags.make'
    includes_path = backend_objects / 'includes_CUDA.rsp'
    def flags_value(path):
        return '\n'.join(line for line in path.read_text().splitlines()
                         if line.strip() and not line.lstrip().startswith('#'))
    cached_flags = None
    if args.reuse_scratch and flags_path.exists() and includes_path.exists():
        objects = list(backend_objects.rglob('*.o'))
        if objects:
            cached_flags = (flags_value(flags_path), includes_path.read_bytes(),
                            min(item.stat().st_mtime_ns for item in objects) - 1000000000)
    run('cmake', '-S', source, '-B', build,
        '-DCMAKE_BUILD_TYPE=Release', '-DBUILD_SHARED_LIBS=ON', '-DGGML_NATIVE=OFF',
        '-DGGML_CUDA=ON', '-DGGML_CUDA_FA_ALL_QUANTS=OFF', f'-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}',
        f'-DGGML_CUDA_CUTLASS_DIR={cutlass}',
        f'-DCMAKE_CUDA_COMPILER={nvcc}', f'-DCUDAToolkit_ROOT={cuda}',
        f'-DCMAKE_CUDA_HOST_COMPILER={args.host_cxx}', f'-DCMAKE_CXX_COMPILER={args.host_cxx}',
        f'-DCMAKE_C_COMPILER={args.host_cc}',
        '-DCMAKE_INSTALL_RPATH=$ORIGIN', '-DCMAKE_BUILD_WITH_INSTALL_RPATH=ON',
        '-DLLAMA_CURL=OFF', '-DLLAMA_BUILD_TESTS=ON', '-DLLAMA_BUILD_EXAMPLES=OFF',
        '-DLLAMA_BUILD_SERVER=OFF', '-DLLAMA_BUILD_UI=OFF', env=environment)
    # CMake's source-option comments can touch the shared flags file without
    # changing any global compiler argument. Overlay sources are freshly written;
    # keep unrelated objects only when the flags AND include response are identical.
    if cached_flags and (flags_value(flags_path), includes_path.read_bytes()) == cached_flags[:2]:
        os.utime(flags_path, ns=(flags_path.stat().st_atime_ns, cached_flags[2]))
        print('CUDA backend global flags/includes unchanged; preserved cached-object dependencies.', flush=True)
    run('cmake', '--build', build, '--parallel', args.jobs,
        '--target', 'llama', 'llama-bench', 'test-backend-ops', env=environment)
    shutil.copytree(build / 'bin', bundle, symlinks=True)
    # Stage CUDA user-space dependencies alongside $ORIGIN-linked GGML libraries.
    # The device driver is provided by the eventual target host.
    for pattern in ('libcudart.so*', 'libcublas.so*', 'libcublasLt.so*'):
        matches = list((cuda / 'lib').glob(pattern))
        if not matches:
            raise RuntimeError('Missing CUDA runtime dependency: ' + pattern)
        for path in matches:
            target = bundle / path.name
            if path.is_symlink():
                target.symlink_to(os.readlink(path))
            else:
                os.link(path, target)
    shutil.copyfile(source / 'LICENSE', bundle / 'LICENSE.llama.cpp')
    shutil.copyfile(cutlass / 'LICENSE.txt', bundle / 'LICENSE.cutlass')
    for name in ('calibrate_runtime.py', 'calibration_shapes.py'):
        shutil.copyfile(root / 'scripts' / name, bundle / name)
    manifest = {'engine_revision': engine_revision, 'runtime_revision': lock['runtime_revision'],
                'cutlass_revision': lock['cutlass_revision'], 'cutlass_tag': lock['cutlass_tag'],
                'endpoint_binding': lock['endpoint_binding'], 'cuda_architectures': [args.cuda_arch],
                'reused_objects_same_upstream': args.reuse_scratch,
                'build_jobs': args.jobs,
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
