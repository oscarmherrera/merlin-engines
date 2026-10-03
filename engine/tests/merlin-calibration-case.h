#pragma once

// Included after the pinned backend test classes; graphs use their actual builders.
struct test_merlin_calibration : test_generic_op {
    int64_t fusion;
    bool packed_fixture;

    test_merlin_calibration(ggml_op op, ggml_type type, std::array<int64_t, 4> ne,
            std::array<int32_t, GGML_MAX_OP_PARAMS / sizeof(int32_t)> params,
            std::vector<input_tensor> sources, std::string name, int64_t fusion, bool fixture)
        : test_generic_op(op, type, ne, params, std::move(sources), std::move(name)),
          fusion(fusion), packed_fixture(fixture) {}

    std::string op_desc(ggml_tensor *) override { return "MUL_MAT"; }
    bool run_whole_graph() override { return fusion != 0; }

    ggml_tensor * build_graph(ggml_context * ctx) override {
        if (!fusion) { return test_generic_op::build_graph(ctx); }
        const bool gate = (fusion & 1) != 0;
        const bool bias = (fusion & 2) != 0;
        const auto glu = gate ? static_cast<ggml_glu_op>((fusion >> 8) - 1) : GGML_GLU_OP_SWIGLU;
        test_mul_mat_vec_fusion builder(GGML_TYPE_PQ2_0, glu, ne[1], ne[0], sources[0].ne[0],
            false, 1, 1, false, bias, gate, false, {1, 1});
        builder.mode = mode;
        builder.sentinels = sentinels;
        ggml_tensor * out = builder.build_graph(ctx);
        sentinels = std::move(builder.sentinels);
        return out;
    }

    void initialize_tensors(ggml_context * ctx) override {
        if (fusion) { test_case::initialize_tensors(ctx); }
        else { test_generic_op::initialize_tensors(ctx); }
        if (!packed_fixture) { return; }
        unsigned operand = 0, float_operand = 0;
        for (ggml_tensor * t = ggml_get_first_tensor(ctx); t; t = ggml_get_next_tensor(ctx, t)) {
            if (t->op != GGML_OP_NONE || t->view_src ||
                    std::find(sentinels.begin(), sentinels.end(), t) != sentinels.end()) { continue; }
            if (t->type == GGML_TYPE_PQ2_0) {
                GGML_ASSERT(ggml_type_size(t->type) == 34 && ggml_blck_size(t->type) == 128);
                std::vector<uint8_t> bytes(ggml_nbytes(t));
                for (size_t b = 0; b < bytes.size() / 34; ++b) {
                    const ggml_fp16_t scale = ggml_fp32_to_fp16(b % 2 ? 0.25f : 0.125f);
                    std::memcpy(bytes.data() + b * 34, &scale, sizeof(scale));
                    for (unsigned q = 0; q < 32; ++q) {
                        unsigned packed = 0;
                        for (unsigned i = 0; i < 4; ++i) {
                            packed |= ((i + q + b + operand) % 4) << (2 * i);
                        }
                        bytes[b * 34 + 2 + q] = uint8_t(packed);
                    }
                }
                ggml_backend_tensor_set(t, bytes.data(), 0, bytes.size());
                ++operand;
            } else if (t->type == GGML_TYPE_F32) {
                // Binary fractions and a fixed unsigned generator reproduce activation/bias inputs.
                std::vector<float> values(ggml_nelements(t));
                uint32_t state = 0x9e3779b9u + ++float_operand;
                for (float & value : values) {
                    state = state * 1664525u + 1013904223u;
                    value = (int32_t(state >> 21) - 1024) / 1024.0f;
                }
                ggml_backend_tensor_set(t, values.data(), 0, values.size() * sizeof(float));
            }
        }
    }
};
