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
    text = replace_once(text,
        'static std::vector<std::unique_ptr<test_case>> make_test_cases_from_file(const char * path) {',
        '#include "merlin-calibration-case.h"\n\n'
        'static std::vector<std::unique_ptr<test_case>> make_test_cases_from_file(const char * path) {')
    anchor = '        test_cases.emplace_back(new test_generic_op(op, type, ne, op_params, sources, std::move(name)));\n'
    replacement = r'''        ggml_merlin_workload workload{0, 0, GGML_MERLIN_PHASE_UNKNOWN, GGML_MERLIN_SOURCE_UNKNOWN};
        int64_t fusion = 0;
        int fixture = 0;
        iss >> std::ws;
        if (!iss.eof()) {
            if (!(iss >> workload.sequence_batch >> workload.tokens_in_flight >> workload.phase >> fusion >> fixture)) {
                throw std::runtime_error("Invalid explicit workload metadata");
            }
            workload.source = GGML_MERLIN_SOURCE_CALIBRATION;
            iss >> std::ws;
            const int64_t glu = (fusion >> 8) - 1;
            const int64_t expected = (fusion & 1) ? (fusion & ~int64_t(7)) | 1 | (fusion & 2 ? 6 : 0) : 2;
            const bool valid_fusion = fusion == 0 || (fusion > 0 && ne[1] == 1 && fusion == expected &&
                (!(fusion & 1) || glu == GGML_GLU_OP_SWIGLU || glu == GGML_GLU_OP_GEGLU || glu == GGML_GLU_OP_SWIGLU_OAI));
            if (!merlin_workload_valid(workload) || !iss.eof() || !valid_fusion ||
                    (fixture != 0 && fixture != 1) || op != GGML_OP_MUL_MAT || type != GGML_TYPE_F32 ||
                    sources.size() != 2 || sources[0].type != GGML_TYPE_PQ2_0 || sources[1].type != GGML_TYPE_F32) {
                throw std::runtime_error("Invalid explicit workload metadata");
            }
        }
        if (merlin_workload_valid(workload)) {
            test_cases.emplace_back(new test_merlin_calibration(op, type, ne, op_params, sources, std::move(name), fusion, fixture));
        } else {
            test_cases.emplace_back(new test_generic_op(op, type, ne, op_params, sources, std::move(name)));
        }
        test_cases.back()->merlin_workload = workload;
'''
    text = replace_once(text, anchor, replacement)
    backend.write_text(backend_text)
    llama.write_text(llama_text)
    tests.write_text(text)
    for name, directory in [('merlin-workload.h', 'ggml/include'),
                            ('merlin-workload-impl.h', 'ggml/src'),
                            ('merlin-workload-llama.h', 'src')]:
        shutil.copyfile(root / 'engine/core' / name, source / directory / name)

    shutil.copyfile(root / 'engine/tests/merlin-calibration-case.h', source / 'tests/merlin-calibration-case.h')
