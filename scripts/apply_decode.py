"""Attach packed PQ2 decode to the pinned Prism MMVQ production entry point."""
from pathlib import Path
import shutil


def apply(source: Path, root: Path):
    path = source / 'ggml/src/ggml-cuda/mmvq.cu'
    text = path.read_text()
    quantize = """    {
        const int64_t s11 = src1->nb[1] / ts_src1;
        const int64_t s12 = src1->nb[2] / ts_src1;
        const int64_t s13 = src1->nb[3] / ts_src1;
        quantize_row_q8_1_cuda(src1_d, nullptr, src1_q8_1.get(), src0->type, ne10, s11, s12, s13, ne10_padded, ne11, ne12, ne13, stream);
    }
"""
    reference = """    mul_mat_vec_q_switch_type(
        src0->data, src0->type, src1_q8_1.get(), ids_d, fusion_local, dst_d, ne00,
        ne01,              ncols_dst,     s01, stride_col_y,     stride_col_dst,
        ne02, nchannels_y, nchannels_dst, s02, stride_channel_y, stride_channel_dst,
        ne03,              ne3,           s03, s13,              s3,               ids_stride, stream);
"""
    changes = [
        ('#include "vecdotq.cuh"\n', '#include "vecdotq.cuh"\n#include "merlin-decode.cuh"\n'),
        (quantize, quantize.replace('    {\n', '    const auto merlin_prepare_q8 = [&]() {\n', 1).replace('    }\n', '    };\n')),
        (reference, '    const auto merlin_reference = [&]() {\n'
         '        merlin_prepare_q8();\n' + ''.join('    ' + line for line in reference.splitlines(True)) + '    };\n'
         '    if (merlin_cuda_decode(ctx, src0, src1, ids, dst, fusion, fusion_local,\n'
         '            reinterpret_cast<const block_q8_1 *>(src1_q8_1.get()), ne10_padded / QK8_1,\n'
         '            merlin_prepare_q8, merlin_reference)) { return; }\n'
         '    merlin_reference();\n'),
    ]
    if '#include "merlin-decode.cuh"' in text:
        raise ValueError('Decode overlay is already applied')
    for before, after in changes:
        if text.count(before) != 1:
            raise ValueError('Pinned decode integration anchor changed')
        text = text.replace(before, after)
    shutil.copyfile(root / 'engine/backends/cuda/decode_ternary/merlin-decode.cuh', path.parent / 'merlin-decode.cuh')
    path.write_text(text)
