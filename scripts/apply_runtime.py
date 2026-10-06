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
from apply_mmq import apply as apply_mmq


def apply_attention(source):
    path = source / 'ggml/src/ggml-cuda/fattn.cu'
    text = path.read_text()
    before = ('    if (use_gqa_opt && gqa_ratio > 4) {\n'
              '        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 8>(ctx, dst);\n')
    after = ('    // GQA 6: two heads per tile cut 225K Q8 attention by 26% on SM75 and 20% on SM86.\n'
             '    if ((cc == GGML_CUDA_CC_TURING || cc == 860) && use_gqa_opt && gqa_ratio == 6 && DKQ == 256 && DV == 256 && Q->ne[1] > 8) {\n'
             '        ggml_cuda_flash_attn_ext_mma_f16_switch_ncols1<DKQ, DV, 2>(ctx, dst);\n'
             '        return;\n'
             '    }\n\n' + before)
    if text.count(before) != 1:
        raise SystemExit('Pinned attention dispatch anchor changed')
    path.write_text(text.replace(before, after))


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
    shutil.copyfile(root / 'engine/telemetry/merlin-engine-log.h', path.parent / 'merlin-engine-log.h')
    for header in (root / 'engine/dispatch').glob('merlin-dispatch*'):
        shutil.copyfile(header, path.parent / header.name)
    apply_decode(args.source, root)
    apply_prefill(args.source, root)
    apply_mmq(args.source)
    apply_attention(args.source)
    subprocess.run(['patch', '--batch', '--forward', '--fuzz=0', '-p1', '-d', str(args.source),
                    '-i', str(root / 'engine/backends/cuda/attention/grouped-q8-vector.patch')], check=True)
    apply_calibration(args.source)
    apply_workload(args.source, root)
    apply_graph(args.source, root)
    configure_runtime(args.source, root, lock)
    print('Applied Merlin packed decode, tiled prefill, SM86 MMQ, SM75/SM86 attention and runtime logging.', flush=True)


if __name__ == '__main__':
    main()
