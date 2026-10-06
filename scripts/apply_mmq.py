"""Enable Bonsai's existing async Q8 MMQ buffer for SM86 PQ2 prefill."""
from pathlib import Path


def apply(source: Path) -> None:
    path = source / 'ggml/src/ggml-cuda/mmq.cuh'
    text = path.read_text()
    changes = (
        ('#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK\n'
         '    constexpr bool async_buffer_y =\n'
         '        type == GGML_TYPE_Q1_0 || type == GGML_TYPE_Q2_0 || type == GGML_TYPE_PQ2_0;',
         '#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK || __CUDA_ARCH__ == 860)\n'
         '    constexpr bool async_buffer_y =\n'
         '        type == GGML_TYPE_PQ2_0 || (__CUDA_ARCH__ == GGML_CUDA_CC_DGX_SPARK &&\n'
         '            (type == GGML_TYPE_Q1_0 || type == GGML_TYPE_Q2_0));'),
        ('const bool async_buffer_y = cc == GGML_CUDA_CC_DGX_SPARK &&\n'
         '        (config.type == GGML_TYPE_Q1_0 || config.type == GGML_TYPE_Q2_0 || config.type == GGML_TYPE_PQ2_0);',
         'const bool async_buffer_y = (cc == GGML_CUDA_CC_DGX_SPARK &&\n'
         '        (config.type == GGML_TYPE_Q1_0 || config.type == GGML_TYPE_Q2_0 || config.type == GGML_TYPE_PQ2_0)) ||\n'
         '        (cc == 860 && config.type == GGML_TYPE_PQ2_0);'),
    )
    for before, after in changes:
        if text.count(before) != 1:
            raise SystemExit('Pinned MMQ async anchor changed')
        text = text.replace(before, after)
    path.write_text(text)
