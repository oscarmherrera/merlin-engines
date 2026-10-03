"""Install endpoint-owned CUTLASS scratch before any inference or CUDA capture."""
from pathlib import Path
import shutil


def apply(source: Path, root: Path) -> None:
    path = source / 'ggml/src/ggml-cuda/ggml-cuda.cu'
    text = path.read_text()
    if "merlin_prefill_initialize(*ctx," in text:
        raise RuntimeError("Workspace overlay already applied")
    changes = [
        ('    ggml_backend_t cuda_backend = new ggml_backend {\n',
         '    merlin_prefill_initialize(*ctx, merlin_prefill_profile_eligible(ctx->device));\n\n'
         '    ggml_backend_t cuda_backend = new ggml_backend {\n'),
        ('    delete cuda_ctx;\n', '    merlin_prefill_release(*cuda_ctx);\n    delete cuda_ctx;\n'),
        ('    bool use_cuda_graph             = false;\n',
         '    merlin_prefill_prepare_graph(*cuda_ctx, cgraph);\n\n'
         '    bool use_cuda_graph             = false;\n'),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise RuntimeError('Pinned source workspace integration anchor changed')
        text = text.replace(before, after)
    path.write_text(text)
    for name in ('merlin-prefill-workspace.cuh', 'merlin-prefill-workspace-impl.cuh'):
        shutil.copyfile(root / 'engine/backends/cuda/prefill_cutlass' / name, path.parent / name)
