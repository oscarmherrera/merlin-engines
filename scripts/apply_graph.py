"""Attach completion/drift monitoring to pinned Prism's real CUDA graph lifecycle."""
from pathlib import Path
import shutil


def apply(source: Path, root: Path) -> None:
    path = source / 'ggml/src/ggml-cuda/ggml-cuda.cu'
    text = path.read_text()
    changes = [
        ('#include "ggml-cuda/mmq.cuh"\n',
         '#include "ggml-cuda/mmq.cuh"\n#include "ggml-cuda/merlin-graph-monitor.cuh"\n'),
        ('    delete cuda_ctx;\n',
         '#ifdef USE_CUDA_GRAPH\n'
         '    merlin_graph::release(*cuda_ctx);\n'
         '#endif\n'
         '    delete cuda_ctx;\n'),
        ('        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));\n',
         '        merlin_graph::launch(*cuda_ctx, graph->instance);\n'),
        ('    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);\n\n'
         '    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);\n',
         '    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);\n\n'
         '    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);\n'
         '    auto * merlin_monitor = (!merlin_dispatch_calibrating() && !merlin_cuda_profile_enabled()) ?\n'
         '        merlin_graph::acquire(*cuda_ctx, graph_key, graph) : nullptr;\n'
         '    const bool merlin_changed = merlin_monitor &&\n'
         '        merlin_graph::prepare(*merlin_monitor, *cuda_ctx, cgraph);\n'),
        ('    if (graph->is_enabled() && !merlin_cuda_profile_enabled() && !merlin_dispatch_calibrating()) {\n',
         '    if (merlin_monitor && graph->is_enabled() && !merlin_cuda_profile_enabled() && !merlin_dispatch_calibrating()) {\n'),
        ('            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph);\n',
         '            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph) || merlin_changed;\n'),
        ('    if (use_cuda_graph && cuda_graph_update_required) {\n        // Start CUDA graph capture\n',
         '#ifdef USE_CUDA_GRAPH\n'
         '    merlin_graph::capture_scope merlin_capture(merlin_monitor, use_cuda_graph && cuda_graph_update_required);\n'
         '#endif\n\n'
         '    if (use_cuda_graph && cuda_graph_update_required) {\n        // Start CUDA graph capture\n'),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise RuntimeError('Pinned CUDA graph monitoring integration anchor changed: ' + before[:80])
        text = text.replace(before, after)
    path.write_text(text)
    shutil.copyfile(root / 'engine/telemetry/merlin-graph-monitor.cuh', path.parent / 'merlin-graph-monitor.cuh')
    shutil.copyfile(root / 'engine/dispatch/merlin-graph-policy.h', path.parent / 'merlin-graph-policy.h')
