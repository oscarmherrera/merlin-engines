"""Carry actual llama microbatch metadata to synchronous backend graph submissions."""
from pathlib import Path
import shutil


def replace_once(text, before, after):
    if text.count(before) != 1:
        raise RuntimeError('Pinned workload integration anchor changed')
    return text.replace(before, after)


def apply(source: Path, root: Path):
    backend = source / 'ggml/src/ggml-backend.cpp'
    llama = source / 'src/llama-context.cpp'
    tests = source / 'tests/test-backend-ops.cpp'
    backend_text = replace_once(backend.read_text(), '#include "ggml-backend-impl.h"\n',
                               '#include "ggml-backend-impl.h"\n#include "merlin-workload-impl.h"\n')
    llama_text = replace_once(llama.read_text(), '#include "llama-context.h"\n',
                             '#include "llama-context.h"\n#include "merlin-workload-llama.h"\n')
    llama_text = replace_once(llama_text,
        '    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);\n',
        '    const merlin_workload_scope merlin_workload(merlin_workload_from_ubatch(ubatch, gtype == LLM_GRAPH_TYPE_ENCODER));\n'
        '    const auto status = graph_compute(res->get_gf(), ubatch.n_tokens > 1);\n')
    text = replace_once(tests.read_text(), '#include "ggml-backend.h"\n',
                        '#include "ggml-backend.h"\n#include "merlin-workload.h"\n')
    text = replace_once(text, 'struct test_case {\n',
        'struct test_case {\n'
        '    ggml_merlin_workload merlin_workload{0, 0, GGML_MERLIN_PHASE_UNKNOWN, GGML_MERLIN_SOURCE_UNKNOWN};\n')
    text = replace_once(text,
        '    bool eval_perf(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {\n',
        '    bool eval_perf(ggml_backend_t backend, const char * op_names_filter, printer * output_printer) {\n'
        '        const merlin_workload_scope merlin_scope(merlin_workload);\n')
    text = replace_once(text,
        '                       printer *      output_printer) {\n'
        '        mode = MODE_TEST;\n',
        '                       printer *      output_printer) {\n'
        '        const merlin_workload_scope merlin_scope(merlin_workload);\n'
        '        mode = MODE_TEST;\n')
    anchor = '        test_cases.emplace_back(new test_generic_op(op, type, ne, op_params, sources, std::move(name)));\n'
    text = replace_once(text, anchor, anchor +
        '        iss >> std::ws;\n'
        '        if (!iss.eof()) {\n'
        '            auto & workload = test_cases.back()->merlin_workload;\n'
        '            if (!(iss >> workload.sequence_batch >> workload.tokens_in_flight >> workload.phase)) {\n'
        '                throw std::runtime_error("Invalid explicit workload metadata");\n'
        '            }\n'
        '            workload.source = GGML_MERLIN_SOURCE_CALIBRATION;\n'
        '            iss >> std::ws;\n'
        '            if (!merlin_workload_valid(workload) || !iss.eof()) {\n'
        '                throw std::runtime_error("Invalid explicit workload metadata");\n'
        '            }\n'
        '        }\n')
    backend.write_text(backend_text)
    llama.write_text(llama_text)
    tests.write_text(text)
    for name, directory in [('merlin-workload.h', 'ggml/include'),
                            ('merlin-workload-impl.h', 'ggml/src'),
                            ('merlin-workload-llama.h', 'src')]:
        shutil.copyfile(root / 'engine/core' / name, source / directory / name)
