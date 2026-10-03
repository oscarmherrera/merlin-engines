#!/usr/bin/env python3
"""Apply the pinned upstream build.md CUDA/new-glibc declaration correction."""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cuda', type=Path)
    args = parser.parse_args()
    header = args.cuda / 'include/crt/math_functions.h'
    original = header.read_bytes()
    text = original.decode()
    for function in ('rsqrt', 'sinpi', 'cospi'):
        for kind, suffix, spaces in [('double', '', 17), ('float', 'f', 18)]:
            before = (f'extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ {kind}'
                      + ' ' * spaces + f'{function}{suffix}({kind} x);')
            if text.count(before) != 1:
                raise SystemExit('Unexpected or already corrected CUDA header; left unchanged')
            text = text.replace(before, before[:-1] + ' noexcept (true);')
    backup = header.with_suffix('.h.pre-merlin-glibc')
    with backup.open('xb') as f:
        f.write(original)
    header.write_text(text)
    manifest = {'correction': 'six math declarations: noexcept(true)',
                'reference': 'Prism adfffbe docs/build.md:253-285',
                'before_sha256': hashlib.sha256(original).hexdigest(),
                'after_sha256': hashlib.sha256(header.read_bytes()).hexdigest()}
    (args.cuda / 'merlin-glibc-compat.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
