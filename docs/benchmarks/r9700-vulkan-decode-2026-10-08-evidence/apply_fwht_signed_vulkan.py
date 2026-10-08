#!/usr/bin/env python3
"""Vulkan port of the CUDA sign-fused Hadamard transform (MUL + RESHAPE + FWHT-hint MUL_MAT -> one dispatch).
Exact-anchor edits on the pinned Prism source; every anchor must match exactly once."""
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

# --- shader -------------------------------------------------------------------
sh = "ggml/src/ggml-vulkan/vulkan-shaders/fwht.comp"
edit(sh,
     "    float scale;\n};\n",
     "    float scale;\n    uint n_blk;\n    uint signs_offset;\n};\n")
edit(sh,
     "layout(binding = 1, std430) writeonly buffer D { float data_d[]; };\n",
     "layout(binding = 1, std430) writeonly buffer D { float data_d[]; };\n"
     "#ifdef FWHT_SIGNS\n"
     "layout(binding = 2, std430) readonly buffer S { float data_s[]; };\n"
     "#endif\n")
edit(sh,
     "            reg[i] = row < n_rows ? float(data_a[src_offset + row_offset + i * BLOCK_SIZE + tid]) * scale : 0.0;\n",
     "            reg[i] = row < n_rows ? float(data_a[src_offset + row_offset + i * BLOCK_SIZE + tid]) * scale : 0.0;\n"
     "#ifdef FWHT_SIGNS\n"
     "            reg[i] *= data_s[signs_offset + (row % n_blk) * N + i * BLOCK_SIZE + tid];\n"
     "#endif\n")

# --- shader generator ----------------------------------------------------------
gen = "ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp"
edit(gen,
     '    string_to_spv("fwht_shmem_f16", "fwht.comp", {{"FWHT_F16", "1"}, {"FWHT_SHMEM", "1"}});\n',
     '    string_to_spv("fwht_shmem_f16", "fwht.comp", {{"FWHT_F16", "1"}, {"FWHT_SHMEM", "1"}});\n'
     '    string_to_spv("fwht_signed_f32", "fwht.comp", {{"FWHT_SIGNS", "1"}});\n'
     '    string_to_spv("fwht_signed_shmem_f32", "fwht.comp", {{"FWHT_SIGNS", "1"}, {"FWHT_SHMEM", "1"}});\n'
     '    string_to_spv("fwht_signed_f16", "fwht.comp", {{"FWHT_SIGNS", "1"}, {"FWHT_F16", "1"}});\n'
     '    string_to_spv("fwht_signed_shmem_f16", "fwht.comp", {{"FWHT_SIGNS", "1"}, {"FWHT_F16", "1"}, {"FWHT_SHMEM", "1"}});\n')

# --- backend -------------------------------------------------------------------
vk = "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
edit(vk,
     "struct vk_op_fwht_push_constants {\n    uint32_t n_rows;\n    uint32_t src_offset;\n    uint32_t dst_offset;\n    float scale;\n};\n",
     "struct vk_op_fwht_push_constants {\n    uint32_t n_rows;\n    uint32_t src_offset;\n    uint32_t dst_offset;\n    float scale;\n    uint32_t n_blk;\n    uint32_t signs_offset;\n};\n")
edit(vk,
     "    vk_pipeline pipeline_fwht_f16[GGML_VK_FWHT_NUM_SIZES];\n",
     "    vk_pipeline pipeline_fwht_f16[GGML_VK_FWHT_NUM_SIZES];\n"
     "    // sign-fused variants: binding 2 carries the Hadamard sign vector (CUDA fwht_signed port)\n"
     "    vk_pipeline pipeline_fwht_signed_f32[GGML_VK_FWHT_NUM_SIZES];\n"
     "    vk_pipeline pipeline_fwht_signed_f16[GGML_VK_FWHT_NUM_SIZES];\n")
# pipeline creation: twin each of the four creates with a 3-binding signed pipeline
edit(vk,
     '                    ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n',
     '                    ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_f32", fwht_f32_len, fwht_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n'
     '                    ggml_vk_create_pipeline(device, device->pipeline_fwht_signed_f32[idx], "fwht_signed_f32", fwht_signed_f32_len, fwht_signed_f32_data, "main", 3, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n')
edit(vk,
     '                        ggml_vk_create_pipeline(device, device->pipeline_fwht_f16[idx], "fwht_f16", fwht_f16_len, fwht_f16_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n',
     '                        ggml_vk_create_pipeline(device, device->pipeline_fwht_f16[idx], "fwht_f16", fwht_f16_len, fwht_f16_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n'
     '                        ggml_vk_create_pipeline(device, device->pipeline_fwht_signed_f16[idx], "fwht_signed_f16", fwht_signed_f16_len, fwht_signed_f16_data, "main", 3, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { device->subgroup_size, n, GGML_VK_FWHT_ROWS }, 1, true, true, device->subgroup_size);\n')
edit(vk,
     '                    ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n, rows }, 1);\n',
     '                    ggml_vk_create_pipeline(device, device->pipeline_fwht_f32[idx], "fwht_shmem_f32", fwht_shmem_f32_len, fwht_shmem_f32_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n, rows }, 1);\n'
     '                    ggml_vk_create_pipeline(device, device->pipeline_fwht_signed_f32[idx], "fwht_signed_shmem_f32", fwht_signed_shmem_f32_len, fwht_signed_shmem_f32_data, "main", 3, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n, rows }, 1);\n')
edit(vk,
     '                        ggml_vk_create_pipeline(device, device->pipeline_fwht_f16[idx], "fwht_shmem_f16", fwht_shmem_f16_len, fwht_shmem_f16_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n, rows }, 1);\n',
     '                        ggml_vk_create_pipeline(device, device->pipeline_fwht_f16[idx], "fwht_shmem_f16", fwht_shmem_f16_len, fwht_shmem_f16_data, "main", 2, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n, rows }, 1);\n'
     '                        ggml_vk_create_pipeline(device, device->pipeline_fwht_signed_f16[idx], "fwht_signed_shmem_f16", fwht_signed_shmem_f16_len, fwht_signed_shmem_f16_data, "main", 3, sizeof(vk_op_fwht_push_constants), {1, 1, 1}, { block_size, n, rows }, 1);\n')
# the transform dispatch: optional signs operand and explicit width
edit(vk,
     "static void ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src, ggml_tensor * dst) {\n"
     "    const int idx = ggml_vk_fwht_pipeline_idx(src->ne[0]);\n"
     "    vk_pipeline pipeline = src->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_f16[idx] : ctx->device->pipeline_fwht_f32[idx];\n",
     "static void ggml_vk_fwht(ggml_backend_vk_context * ctx, vk_context& subctx, const ggml_tensor * src, ggml_tensor * dst,\n"
     "                         const ggml_tensor * signs = nullptr, uint32_t width = 0) {\n"
     "    // width: transform size N; src rows are width-sized. Fused callers pass the unreshaped activation\n"
     "    // and the Hadamard width explicitly; signs (f32, n_blk*width) is multiplied in during the load.\n"
     "    const uint32_t N = width ? width : (uint32_t) src->ne[0];\n"
     "    const int idx = ggml_vk_fwht_pipeline_idx(N);\n"
     "    vk_pipeline pipeline = signs ? (src->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_signed_f16[idx] : ctx->device->pipeline_fwht_signed_f32[idx])\n"
     "                                 : (src->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_f16[idx]        : ctx->device->pipeline_fwht_f32[idx]);\n")
edit(vk,
     "    GGML_ASSERT(rows_per_workgroup > 0);\n    const uint32_t n_rows = (uint32_t)ggml_nrows(src);\n",
     "    GGML_ASSERT(rows_per_workgroup > 0);\n    const uint32_t n_rows = (uint32_t)(ggml_nelements(src) / N);\n")
edit(vk,
     "    vk_op_fwht_push_constants pc = {\n        n_rows,\n        0,\n        0,\n        1.0f / std::sqrt((float)src->ne[0]),\n    };\n"
     "    init_pushconst_tensor_offsets(ctx, pc, src, nullptr, nullptr, nullptr, dst);\n\n"
     "    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });\n",
     "    vk_op_fwht_push_constants pc = {\n        n_rows,\n        0,\n        0,\n        1.0f / std::sqrt((float)N),\n"
     "        signs ? (uint32_t)(signs->ne[0] / N) : 1u,\n        0,\n    };\n"
     "    init_pushconst_tensor_offsets(ctx, pc, src, nullptr, nullptr, nullptr, dst);\n\n"
     "    if (signs) {\n"
     "        pc.signs_offset = get_misalign_bytes(ctx, signs) / ggml_type_size(signs->type);\n"
     "        const vk_subbuffer signs_buf = ggml_vk_tensor_subbuffer(ctx, signs, true);\n"
     "        ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf, signs_buf }, pc, { workgroups_x, 1, 1 });\n"
     "        return;\n"
     "    }\n"
     "    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline, { src_buf, dst_buf }, pc, { workgroups_x, 1, 1 });\n")
# fusion predicate + fused dispatch, placed before ggml_vk_can_fuse
edit(vk,
     "static bool ggml_vk_can_fuse(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {\n",
     "// Hadamard sign flip + reshape + FWHT-hint matmul: multiply the sign vector during the transform's\n"
     "// load instead of a separate pass (port of the CUDA backend's fwht_signed fusion).\n"
     "static bool ggml_vk_can_fuse_fwht_signed(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx) {\n"
     "    if (!ggml_can_fuse_subgraph(cgraph, node_idx, { GGML_OP_MUL, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, { node_idx + 2 })) {\n"
     "        return false;\n"
     "    }\n"
     "    const ggml_tensor * mul     = cgraph->nodes[node_idx];\n"
     "    const ggml_tensor * reshape = cgraph->nodes[node_idx + 1];\n"
     "    const ggml_tensor * mm      = cgraph->nodes[node_idx + 2];\n"
     "    const ggml_tensor * x       = mul->src[0];\n"
     "    const ggml_tensor * signs   = mul->src[1];\n"
     "    if (ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD || mm->src[1] != reshape || reshape->src[0] != mul) {\n"
     "        return false;\n"
     "    }\n"
     "    if (signs->ne[1] != 1 || signs->ne[2] != 1 || signs->ne[3] != 1 || signs->type != GGML_TYPE_F32) {\n"
     "        return false;\n"
     "    }\n"
     "    if ((x->type != GGML_TYPE_F32 && x->type != GGML_TYPE_F16) || mul->type != x->type || mm->type != GGML_TYPE_F32) {\n"
     "        return false;\n"
     "    }\n"
     "    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(signs) || !ggml_is_contiguous(mm)) {\n"
     "        return false;\n"
     "    }\n"
     "    const int64_t n = mm->src[0]->ne[0];\n"
     "    if (signs->ne[0] != x->ne[0] || signs->ne[0] % n != 0) {\n"
     "        return false;\n"
     "    }\n"
     "    const int idx = ggml_vk_fwht_pipeline_idx(n);\n"
     "    if (idx < 0) {\n"
     "        return false;\n"
     "    }\n"
     "    const vk_pipeline & p = x->type == GGML_TYPE_F16 ? ctx->device->pipeline_fwht_signed_f16[idx] : ctx->device->pipeline_fwht_signed_f32[idx];\n"
     "    return p != nullptr;\n"
     "}\n\n"
     "static void ggml_vk_fwht_signed_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, int node_idx) {\n"
     "    const ggml_tensor * mul = cgraph->nodes[node_idx];\n"
     "    ggml_tensor *       mm  = cgraph->nodes[node_idx + 2];\n"
     "    ggml_vk_fwht(ctx, subctx, mul->src[0], mm, mul->src[1], (uint32_t) mm->src[0]->ne[0]);\n"
     "}\n\n"
     "static bool ggml_vk_can_fuse(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {\n")
# registration, ahead of the other MUL-rooted fusion
edit(vk,
     "            } else if (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {\n",
     "            } else if (ggml_vk_can_fuse_fwht_signed(ctx, cgraph, i)) {\n"
     "                ctx->num_additional_fused_ops = 2;\n"
     "                fusion_string = \"FWHT_SIGNED\";\n"
     "                op_srcs_fused_elementwise[0] = false;\n"
     "                op_srcs_fused_elementwise[1] = false;\n"
     "                op_srcs_fused_elementwise[2] = false;\n"
     "            } else if (ggml_vk_can_fuse_snake(ctx, cgraph, i)) {\n")
# dispatch
edit(vk,
     "    case GGML_OP_MUL:\n        if (ctx->num_additional_fused_ops) {\n            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, node_idx);\n",
     "    case GGML_OP_MUL:\n"
     "        if (ctx->num_additional_fused_ops == 2 && cgraph->nodes[node_idx + 2]->op == GGML_OP_MUL_MAT) {\n"
     "            ggml_vk_fwht_signed_dispatch_fused(ctx, compute_ctx, cgraph, node_idx);\n"
     "        } else if (ctx->num_additional_fused_ops) {\n"
     "            ggml_vk_snake_dispatch_fused(ctx, compute_ctx, cgraph, node_idx);\n")
# forward declaration: the graph builder is defined before the fused dispatch helper
edit(vk,
     "static bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int node_idx, ggml_tensor *node_begin, int node_idx_begin, bool last_node, bool almost_ready, bool submit){\n",
     "static void ggml_vk_fwht_signed_dispatch_fused(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_cgraph * cgraph, int node_idx);\n\n"
     "static bool ggml_vk_build_graph(ggml_backend_vk_context * ctx, ggml_cgraph * cgraph, int node_idx, ggml_tensor *node_begin, int node_idx_begin, bool last_node, bool almost_ready, bool submit){\n")
print("all edits applied")
