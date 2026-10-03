"""Run the real backend graph once; the registry owns calibration repetitions."""
from pathlib import Path


def apply(source: Path) -> None:
    path = source / 'tests/test-backend-ops.cpp'
    text = path.read_text()
    changes = [
        ('        // determine number of runs\n',
         '        const char * merlin_calibrate = std::getenv("MERLIN_ENGINE_CALIBRATE");\n'
         '        if (merlin_calibrate && std::strcmp(merlin_calibrate, "1") == 0) {\n'
         '            ggml_backend_synchronize(backend);\n'
         '            printf("merlin-engine: calibration graph completed %s %s\\n",\n'
         '                   current_op_name.c_str(), vars().c_str());\n'
         '            return true;\n'
         '        }\n\n'
         '        // determine number of runs\n'),
        ('        for (auto & test : test_cases) {\n'
         '            test->eval_perf(backend, op_names_filter, output_printer);\n'
         '        }\n'
         '        return true;\n',
         '        bool passed = true;\n'
         '        for (auto & test : test_cases) {\n'
         '            passed = test->eval_perf(backend, op_names_filter, output_printer) && passed;\n'
         '        }\n'
         '        return passed;\n'),
    ]
    for before, after in changes:
        if text.count(before) != 1:
            raise RuntimeError('Pinned calibration harness anchor changed')
        text = text.replace(before, after)
    path.write_text(text)
