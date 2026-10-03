#include "merlin-workload.h"

extern "C" ggml_merlin_workload read_workload_from_other_library() {
    return ggml_merlin_workload_get();
}
