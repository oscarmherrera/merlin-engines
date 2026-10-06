"""Apply measured PQ2 MMQ configurations for SM61 and SM86."""
from pathlib import Path


def apply(source: Path) -> None:
    path = source / 'ggml/src/ggml-cuda/mmq.cuh'
    text = path.read_text()
    changes = (
        ('static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc) {\n',
         'static __host__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(const ggml_type type, const int J, const bool fallback, const int cc) {\n'
         '    if (cc == 610 && type == GGML_TYPE_PQ2_0 && J == 64) {\n'
         '        return ggml_cuda_mmq_config(type, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);\n'
         '    }\n'),
        ('static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback) {\n',
         'static constexpr __device__ ggml_cuda_mmq_config ggml_cuda_mmq_get_config(ggml_type type, int J, bool fallback) {\n'
         '#if __CUDA_ARCH__ == 610\n'
         '    if (type == GGML_TYPE_PQ2_0 && J == 64) {\n'
         '        return ggml_cuda_mmq_config(type, 256, 2, 128, 64, GGML_CUDA_MMQ_SRAM_LAYOUT_Q8_0, MMQ_ITER_K, false, fallback);\n'
         '    }\n'
         '#endif\n'),
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
            raise SystemExit('Pinned MMQ configuration anchor changed')
        text = text.replace(before, after)
    path.write_text(text)
