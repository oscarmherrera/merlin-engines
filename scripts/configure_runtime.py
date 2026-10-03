"""Attach the pinned header dependency and runtime calibration identity."""
import json
from pathlib import Path
import subprocess


def apply(source: Path, root: Path, lock: dict) -> None:
    cuda = source / 'ggml/src/ggml-cuda'
    path = cuda / 'CMakeLists.txt'
    text = path.read_text()
    anchor = '    add_compile_definitions(GGML_CUDA_PEER_MAX_BATCH_SIZE=${GGML_CUDA_PEER_MAX_BATCH_SIZE})\n'
    if text.count(anchor) != 1:
        raise RuntimeError('Pinned CUDA CMake integration anchor changed')
    addition = '''    if (NOT EXISTS "${GGML_CUDA_CUTLASS_DIR}/include/cutlass/cutlass.h")
        message(FATAL_ERROR "Merlin requires the CUTLASS headers pinned in runtime.lock.json")
    endif()
    # Scope flags to our prefill translation unit; preserve unrelated cached objects.
    set_property(SOURCE ggml-cuda.cu mmvq.cu APPEND PROPERTY INCLUDE_DIRECTORIES
        "${GGML_CUDA_CUTLASS_DIR}/include")
    set_property(SOURCE ggml-cuda.cu mmvq.cu APPEND PROPERTY COMPILE_OPTIONS
        --expt-relaxed-constexpr)

'''
    path.write_text(text.replace(anchor, addition + anchor))
    revision = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    (cuda / 'merlin-build.h').write_text(
        '#pragma once\n' +
        '#define MERLIN_ENGINE_BUILD_ID ' + json.dumps(revision) + '\n' +
        '#define MERLIN_PRISM_REVISION ' + json.dumps(lock['runtime_revision']) + '\n' +
        '#define MERLIN_CUTLASS_REVISION ' + json.dumps(lock['cutlass_revision']) + '\n')
