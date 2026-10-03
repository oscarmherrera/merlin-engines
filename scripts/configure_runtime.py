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
    sources_anchor = '    ggml_add_backend_library(ggml-cuda\n'
    if text.count(sources_anchor) != 1:
        raise RuntimeError('Pinned CUDA source list anchor changed')
    text = text.replace(sources_anchor,
        '    list(REMOVE_ITEM GGML_SOURCES_CUDA "${CMAKE_CURRENT_SOURCE_DIR}/merlin-prefill.cu")\n\n' + sources_anchor)
    addition = '''    if (NOT EXISTS "${GGML_CUDA_CUTLASS_DIR}/include/cutlass/cutlass.h")
        message(FATAL_ERROR "Merlin requires the CUTLASS headers pinned in runtime.lock.json")
    endif()
    # A separate object target keeps CUTLASS flags out of the backend's shared flags.make.
    add_library(merlin-cutlass OBJECT merlin-prefill.cu)
    set_target_properties(merlin-cutlass PROPERTIES POSITION_INDEPENDENT_CODE ON
        CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
    target_include_directories(merlin-cutlass PRIVATE
        $<TARGET_PROPERTY:ggml-cuda,INCLUDE_DIRECTORIES>
        "${GGML_CUDA_CUTLASS_DIR}/include")
    target_compile_definitions(merlin-cutlass PRIVATE
        $<TARGET_PROPERTY:ggml-cuda,COMPILE_DEFINITIONS>)
    target_compile_options(merlin-cutlass PRIVATE
        $<TARGET_PROPERTY:ggml-cuda,COMPILE_OPTIONS>
        --expt-relaxed-constexpr)
    target_sources(ggml-cuda PRIVATE $<TARGET_OBJECTS:merlin-cutlass>)

'''
    path.write_text(text.replace(anchor, addition + anchor))
    revision = subprocess.check_output(
        ['git', '-C', str(root), 'rev-parse', 'HEAD'], text=True).strip()
    (cuda / 'merlin-build.h').write_text(
        '#pragma once\n' +
        '#define MERLIN_ENGINE_BUILD_ID ' + json.dumps(revision) + '\n' +
        '#define MERLIN_PRISM_REVISION ' + json.dumps(lock['runtime_revision']) + '\n' +
        '#define MERLIN_CUTLASS_REVISION ' + json.dumps(lock['cutlass_revision']) + '\n')
