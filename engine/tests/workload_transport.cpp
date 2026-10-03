#include "merlin-workload-llama.h"

#include <cassert>
#include <dlfcn.h>
#include <thread>

int main(int argc, char ** argv) {
    assert(argc == 2);
    void * library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    assert(library);
    auto read_other = reinterpret_cast<ggml_merlin_workload (*)()>(dlsym(library, "read_workload_from_other_library"));
    assert(read_other);
    assert(!merlin_workload_valid(read_other()));

    llama_seq_id first[] = {17}, second[] = {29};
    llama_seq_id * ids[] = {first, first, first, second};
    int32_t counts[] = {1, 1, 1, 1};
    llama_ubatch batch{};
    batch.n_tokens = 4;
    batch.n_seqs_unq = 2;
    batch.seq_id = ids;
    batch.n_seq_id = counts;
    {
        const merlin_workload_scope scope(merlin_workload_from_ubatch(batch, false));
        const auto observed = read_other();
        assert(observed.sequence_batch == 2 && observed.tokens_in_flight == 4);
        assert(observed.phase == GGML_MERLIN_PHASE_MIXED && observed.source == GGML_MERLIN_SOURCE_UBATCH);
        std::thread independent([&] {
            assert(!merlin_workload_valid(read_other()));
            const merlin_workload_scope isolated({1, 128, GGML_MERLIN_PHASE_MULTI_TOKEN, GGML_MERLIN_SOURCE_CALIBRATION});
            assert(read_other().sequence_batch == 1 && read_other().tokens_in_flight == 128);
        });
        independent.join();
        assert(read_other().phase == GGML_MERLIN_PHASE_MIXED);
        {
            const merlin_workload_scope encoder(merlin_workload_from_ubatch(batch, true));
            assert(read_other().phase == GGML_MERLIN_PHASE_ENCODER);
        }
        assert(read_other().phase == GGML_MERLIN_PHASE_MIXED);
    }
    assert(!merlin_workload_valid(read_other()));

    // Four matrix columns can represent one prompt sequence or four decode sequences.
    ids[3] = first;
    batch.n_seqs_unq = 1;
    {
        const merlin_workload_scope scope(merlin_workload_from_ubatch(batch, false));
        assert(read_other().sequence_batch == 1 && read_other().tokens_in_flight == 4);
        assert(read_other().phase == GGML_MERLIN_PHASE_MULTI_TOKEN);
    }
    llama_seq_id third[] = {35}, fourth[] = {49};
    ids[0] = first; ids[1] = second; ids[2] = third; ids[3] = fourth;
    batch.n_seqs_unq = 4;
    {
        const merlin_workload_scope scope(merlin_workload_from_ubatch(batch, false));
        assert(read_other().sequence_batch == 4 && read_other().tokens_in_flight == 4);
        assert(read_other().phase == GGML_MERLIN_PHASE_SINGLE_TOKEN);
    }
    // A shared token may belong to multiple sequences; sequence count may exceed token count.
    llama_seq_id coupled[] = {17, 29};
    ids[0] = coupled; counts[0] = 2;
    batch.n_tokens = 1; batch.n_seqs_unq = 2;
    {
        const merlin_workload_scope scope(merlin_workload_from_ubatch(batch, false));
        assert(read_other().sequence_batch == 2 && read_other().tokens_in_flight == 1);
        assert(read_other().phase == GGML_MERLIN_PHASE_SINGLE_TOKEN);
    }
    batch.n_seqs_unq = 3;
    {
        const merlin_workload_scope invalid(merlin_workload_from_ubatch(batch, false));
        assert(!merlin_workload_valid(read_other()));
    }
    dlclose(library);
    std::puts("workload transport: shared-library storage, threads, scopes, real ubatch metadata PASS");
}
