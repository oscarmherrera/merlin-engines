"""Fetch only the pinned CUTLASS headers; do not build its kernel catalogue."""
import subprocess
from pathlib import Path


def prepare(lock: dict, destination: Path, run) -> Path:
    if not destination.exists():
        run('git', 'clone', '--depth', '1', '--branch', lock['cutlass_tag'],
            lock['cutlass_repository'], destination)
    revision = subprocess.check_output(
        ['git', '-C', str(destination), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != lock['cutlass_revision']:
        raise RuntimeError('CUTLASS checkout differs from runtime.lock.json')
    if subprocess.check_output(['git', '-C', str(destination), 'status', '--porcelain']):
        raise RuntimeError('CUTLASS checkout has local changes')
    if not (destination / 'include/cutlass/cutlass.h').is_file():
        raise RuntimeError('CUTLASS checkout is missing its headers')
    return destination.resolve()
