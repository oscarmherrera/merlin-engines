#!/usr/bin/env python3
"""Fusion 2 (applied after apply_fwht_signed_vulkan.py): the subgroup FWHT shader also writes the q8_1 x4
blocks of its output into the backend's shared quantized-activation buffer and records the reuse memo, so
the consuming vector matmul skips its own quantize dispatch. Same arithmetic as quantize_q8_1.comp:
amax over 32, d = amax/127, q = round(v * (1/d)), s = sum(q) * d."""
import sys
from pathlib import Path

root = Path(sys.argv[1])

def edit(rel, before, after):
    p = root / rel
    t = p.read_text()
    n = t.count(before)
    assert n == 1, f"{rel}: anchor count {n}: {before[:70]!r}"
    p.write_text(t.replace(before, after))
    print("edited", rel, "->", after.strip().splitlines()[0][:70])

def edit_n(rel, before, after, expected):
    p = root / rel
    t = p.read_text()
    n = t.count(before)
    assert n == expected, f"{rel}: anchor count {n} != {expected}: {before[:70]!r}"
    p.write_text(t.replace(before, after))
    print("edited", rel, f"x{n} ->", after.strip().splitlines()[0][:70])

sh = "ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp"
edit(sh,
     "#ifndef FWHT_SHMEM\n#extension GL_KHR_shader_subgroup_basic : enable\n#extension GL_KHR_shader_subgroup_shuffle : enable\n#endif\n",
     "#ifndef FWHT_SHMEM\n#extension GL_KHR_shader_subgroup_basic : enable\n#extension GL_KHR_shader_subgroup_shuffle : enable\n#endif\n"
     "#ifdef FWHT_Q8\n#extension GL_KHR_shader_subgroup_clustered : require\n#include \"types.glsl\"\n#endif\n")
edit(sh,
     "#ifdef FWHT_SIGNS\nlayout(binding = 2, std430) readonly buffer S { float data_s[]; };\n#endif\n",
     "#ifdef FWHT_SIGNS\nlayout(binding = 2, std430) readonly buffer S { float data_s[]; };\n#endif\n"
     "#ifdef FWHT_Q8\n#ifdef FWHT_SIGNS\nlayout(binding = 3, std430) writeonly buffer Q { block_q8_1_x4 data_q[]; };\n#else\n"
     "layout(binding = 2, std430) writeonly buffer Q { block_q8_1_x4 data_q[]; };\n#endif\n#endif\n")
# epilogue: after the last butterfly stage, before the f32 store (subgroup variant only)
edit(sh,
     "#ifdef FWHT_SHMEM\n        if (row < n_rows) {\n#endif\n            [[unroll]]\n            for (uint i = 0; i < EL_W; ++i) {\n                data_d[dst_offset + row_offset + i * BLOCK_SIZE + tid] = reg[i];\n",
     "#if defined(FWHT_Q8) && !defined(FWHT_SHMEM)\n"
     "        // q8_1 x4 epilogue: a 32-element block is 32 consecutive lanes at one i; identical to quantize_q8_1.comp\n"
     "        [[unroll]]\n        for (uint i = 0; i < EL_W; ++i) {\n"
     "            const float v = reg[i];\n"
     "            const float amax = subgroupClusteredMax(abs(v), 32);\n"
     "            const float d = amax / 127.0;\n"
     "            const float d_inv = d != 0.0 ? 1.0 / d : 0.0;\n"
     "            const float q = round(v * d_inv);\n"
     "            const float s = subgroupClusteredAdd(q, 32);\n"
     "            const float q1 = subgroupShuffle(q, tid + 1);\n"
     "            const float q2 = subgroupShuffle(q, tid + 2);\n"
     "            const float q3 = subgroupShuffle(q, tid + 3);\n"
     "            const uint e = row_offset + i * BLOCK_SIZE + tid;\n"
     "            const uint ib = e >> 5;\n"
     "            if ((tid & 3) == 0) {\n"
     "                data_q[ib >> 2].qs[(ib & 3) * 8 + ((e & 31) >> 2)] = pack32(i8vec4(round(vec4(q, q1, q2, q3))));\n"
     "            }\n"
     "            if ((tid & 31) == 0) {\n"
     "                data_q[ib >> 2].ds[ib & 3] = f16vec2(vec2(d, s * d));\n"
     "            }\n"
     "        }\n"
     "#endif\n"
     "#ifdef FWHT_SHMEM\n        if (row < n_rows) {\n#endif\n            [[unroll]]\n            for (uint i = 0; i < EL_W; ++i) {\n                data_d[dst_offset + row_offset + i * BLOCK_SIZE + tid] = reg[i];\n")

gen = "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"
edit(gen,
     '    string_to_spv("fwht_signed_shmem_f16", "fwht.comp", {{"FWHT_SIGNS", "1"}, {"FWHT_F16", "1"}, {"FWHT_SHMEM", "1"}});\n',
     '    string_to_spv("fwht_signed_shmem_f16", "fwht.comp", {{"FWHT_SIGNS", "1"}, {"FWHT_F16", "1"}, {"FWHT_SHMEM", "1"}});\n'
     '    string_to_spv("fwht_q8_f32", "fwht.comp", {{"FWHT_Q8", "1"}});\n'
     '    string_to_spv("fwht_signed_q8_f32", "fwht.comp", {{"FWHT_SIGNS", "1"}, {"FWHT_Q8", "1"}});\n')

vk = "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
edit(vk,
     "    vk_pipeline pipeline_fwht_signed_f16[GGML_VK_FWHT_NUM_SIZES];\n",
     "    vk_pipeline pipeline_fwht_signed_f16[GGML_VK_FWHT_NUM_SIZES];\n"
     "    // subgroup-variant transforms that also emit the q8_1 x4 blocks of their output into prealloc_y\n"
     "    vk_pipeline pipeline_fwht_q8_f32[GGML_VK_FWHT_NUM_SIZES];\n"
     "    vk_pipeline pipeline_fwht_signed_q8_f32[GGML_VK_FWHT_NUM_SIZES];\n")
edit(vk,
     '                    ggml_vk_create_pipeline(device, device->pipeline_fwht_signed_f32[idx], "fwht_signed_f32", fwht_signed_f32_len, fwht_signed_f32_data, "main", 3, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n',
     '                    ggml_vk_create_pipeline(device, device->pipeline_fwht_signed_f32[idx], "fwht_signed_f32", fwht_signed_f32_len, fwht_signed_f32_data, "main", 3, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n'
     '                    if (device->subgroup_clustered && device->subgroup_size % 32 == 0) {\n'
     '                        ggml_vk_create_pipeline(device, device->pipeline_fwht_q8_f32[idx], "fwht_q8_f32", fwht_q8_f32_len, fwht_q8_f32_data, "main", 3, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n'
     '                        ggml_vk_create_pipeline(device, device->pipeline_fwht_signed_q8_f32[idx], "fwht_signed_q8_f32", fwht_signed_q8_f32_len, fwht_signed_q8_f32_data, "main", 4, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n'
     '                    }\n')
# ggml_vk_fwht: optional q8_1 output
edit(vk,
     "static void ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src, ggml_tensor * dst,\n"
     "                         const ggml_tensor * signs = nullptr, uint32_t width = 0) {\n",
     "static void ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src, ggml_tensor * dst,\n"
     "                         const ggml_tensor * signs = nullptr, uint32_t width = 0, bool quantize_out = false) {\n")
edit(vk,
     "    vk_pipeline pipeline = signs ? (src->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_signed_f16[idx] : ctx->device->pipeline_fwht_signed_f32[idx])\n"
     "                                 : (src->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_f16[idx]        : ctx->device->pipeline_fwht_f32[idx]);\n",
     "    vk_pipeline pipeline = signs ? (src->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_signed_f16[idx] : ctx->device->pipeline_fwht_signed_f32[idx])\n"
     "                                 : (src->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_f16[idx]        : ctx->device->pipeline_fwht_f32[idx]);\n"
     "    // q8_1 epilogue only on the f32 subgroup variants; the consumer validates the memo before trusting it\n"
     "    quantize_out = quantize_out && src->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32 && ctx->device->integer_dot_product &&\n"
     "                   (signs ? ctx->device->pipeline_fwht_signed_q8_f32[idx] : ctx->device->pipeline_fwht_q8_f32[idx]) != nullptr;\n"
     "    if (quantize_out) {\n"
     "        pipeline = signs ? ctx->device->pipeline_fwht_signed_q8_f32[idx] : ctx->device->pipeline_fwht_q8_f32[idx];\n"
     "    }\n")
edit(vk,
     "    if (signs) {\n"
     "        pc.signs_offset = get_misalign_bytes(ctx, signs) / ggml_type_size(signs->type);\n"
     "        const vk_subbuffer signs_buf = ggml_vk_tensor_subbuffer(ctx, signs, true);\n"
     "        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf, signs_buf }, pc, { workgroups_x, 1, 1 });\n"
     "        return;\n"
     "    }\n"
     "    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });\n",
     "    vk_subbuffer q8_buf;\n"
     "    if (quantize_out) {\n"
     "        const uint64_t y_ne = (uint64_t) n_rows * N;\n"
     "        const uint64_t y_sz = ggml_vk_align_size(y_ne, 128) * ggml_type_size(GGML_TYPE_Q8_1) / ggml_blck_size(GGML_TYPE_Q8_1);\n"
     "        if (ctx->prealloc_size_y < y_sz) {\n"
     "            ctx->prealloc_size_y = y_sz;\n"
     "            ggml_vk_preallocate_buffers(ctx, subctx);\n"
     "        }\n"
     "        if (ctx->prealloc_y_need_sync) {\n"
     "            ggml_vk_sync_buffers(ctx, subctx);\n"
     "        }\n"
     "        q8_buf = { ctx->prealloc_y, 0, ctx->prealloc_y->size };\n"
     "    }\n"
     "    if (signs) {\n"
     "        pc.signs_offset = get_misalign_bytes(ctx, signs) / ggml_type_size(signs->type);\n"
     "        const vk_subbuffer signs_buf = ggml_vk_tensor_subbuffer(ctx, signs, true);\n"
     "        if (quantize_out) {\n"
     "            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf, signs_buf, q8_buf }, pc, { workgroups_x, 1, 1 });\n"
     "        } else {\n"
     "            ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf, signs_buf }, pc, { workgroups_x, 1, 1 });\n"
     "        }\n"
     "    } else if (quantize_out) {\n"
     "        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf, q8_buf }, pc, { workgroups_x, 1, 1 });\n"
     "    } else {\n"
     "        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });\n"
     "    }\n"
     "    if (quantize_out) {\n"
     "        ggml_vk_sync_buffers(ctx, subctx);\n"
     "        ctx->prealloc_y_last_pipeline_used = ggml_vk_get_quantize_pipeline(ctx, GGML_TYPE_Q8_1).get();\n"
     "        ctx->prealloc_y_last_tensor_used = dst;\n"
     "        ctx->prealloc_y_last_decode_vector_staging = false;\n"
     "    }\n")
# consumers: fused and unfused transform calls request the epilogue
edit(vk,
     "    ggml_vk_fwht(ctx, subctx, mul->src[0], mm, mul->src[1], (uint32_t) mm->src[0]->ne[0]);\n",
     "    ggml_vk_fwht(ctx, subctx, mul->src[0], mm, mul->src[1], (uint32_t) mm->src[0]->ne[0], true);\n")
edit(vk,
     "        ggml_vk_fwht(ctx, subctx, src1, dst);\n",
     "        ggml_vk_fwht(ctx, subctx, src1, dst, nullptr, 0, true);\n")
# memo (vector and prefill matmul paths, identical blocks): a whole-tensor view of the transform output counts as the same activation
edit_n(vk,
     "    if (quantize_y) {\n"
     "        if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||\n"
     "            ctx->prealloc_y_last_tensor_used != src1 ||\n"
     "            ctx->prealloc_y_last_decode_vector_staging) {\n"
     "            if (ctx->prealloc_y_need_sync) {\n"
     "                ggml_vk_sync_buffers(ctx, subctx);\n"
     "            }\n"
     "            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);\n",
     "    if (quantize_y) {\n"
     "        const bool y_is_whole_view_of_last = src1->view_src != nullptr && src1->view_src == ctx->prealloc_y_last_tensor_used &&\n"
     "                                             src1->view_offs == 0 && ggml_nbytes(src1) == ggml_nbytes(src1->view_src);\n"
     "        if (ctx->prealloc_y_last_pipeline_used != to_q8_1.get() ||\n"
     "            (ctx->prealloc_y_last_tensor_used != src1 && !y_is_whole_view_of_last) ||\n"
     "            ctx->prealloc_y_last_decode_vector_staging) {\n"
     "            if (ctx->prealloc_y_need_sync) {\n"
     "                ggml_vk_sync_buffers(ctx, subctx);\n"
     "            }\n"
     "            ggml_vk_quantize_q8_1(ctx, subctx, ggml_vk_subbuffer(ctx, d_Qy, qy_buf_offset), ggml_vk_subbuffer(ctx, d_Y, 0), y_ne);\n", 2)
print("all edits applied")
