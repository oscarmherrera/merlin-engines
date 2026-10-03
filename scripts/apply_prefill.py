"""Integrate the packed PQ2 prefill kernel into pinned Prism's production dispatcher."""
from pathlib import Path
import shutil


def apply(source: Path, root: Path) -> None:
    path = source / 'ggml/src/ggml-cuda/ggml-cuda.cu'
    text = path.read_text()
    anchors = [
        ('#include "ggml-cuda/mmq.cuh"\n',
         '#include "ggml-cuda/mmq.cuh"\n#include "ggml-cuda/merlin-prefill.cuh"\n'),
        ('    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.\n',
         '    if (try_merlin_prefill(ctx, src0, src1, dst)) {\n'
         '        return;\n'
         '    }\n\n'
         '    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.\n'),
    ]
    for before, after in anchors:
        if text.count(before) != 1:
            raise RuntimeError('Pinned source prefill integration anchor changed')
        text = text.replace(before, after)
    test_path = source / 'tests/test-backend-ops.cpp'
    tests = test_path.read_text()
    anchor = '    // PTQ1_0 / PQ2_0 integer-dot mat-vec: Bonsai-2 shapes, odd row counts (row tail), batches and multi-column B\n'
    if tests.count(anchor) != 1:
        raise RuntimeError('Pinned source prefill test anchor changed')
    tests = tests.replace(anchor,
        '    // Resident prefill: exercise real backend graph dispatch and both tile tails.\n'
        '    for (int64_t n : {16, 17, 32, 65}) {\n'
        '        for (int64_t k : {128, 5120, 17408}) {\n'
        '            test_cases.emplace_back(new test_mul_mat(GGML_TYPE_PQ2_0, GGML_TYPE_F32, 67, n, k, {1, 1}, {1, 1}));\n'
        '        }\n'
        '    }\n\n' + anchor)
    path.write_text(text)
    test_path.write_text(tests)
    shutil.copyfile(root / 'runtime/merlin-prefill.cuh', path.parent / 'merlin-prefill.cuh')
