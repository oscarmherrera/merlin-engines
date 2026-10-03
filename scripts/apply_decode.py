"""Attach packed PQ2 decode to the pinned Prism MMVQ production entry point."""
from pathlib import Path
import shutil


def apply(source: Path, root: Path):
    path = source / 'ggml/src/ggml-cuda/mmvq.cu'
    text = path.read_text()
    changes = [
        ('#include "vecdotq.cuh"\n',
         '#include "vecdotq.cuh"\n#include "merlin-decode.cuh"\n'),
        ('    // If src0 is a temporary compute buffer, clear any potential padding.\n',
         '    if (merlin_cuda_decode(ctx, src0, src1, ids, dst, fusion, fusion_local)) {\n'
         '        return;\n'
         '    }\n\n'
         '    // If src0 is a temporary compute buffer, clear any potential padding.\n'),
    ]
    if '#include "merlin-decode.cuh"' in text:
        raise ValueError('Decode overlay is already applied')
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Pinned decode integration anchor changed')
        text = text.replace(before, after)
    shutil.copyfile(root / 'runtime/merlin-decode.cuh', path.parent / 'merlin-decode.cuh')
    path.write_text(text)
