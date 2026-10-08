#!/usr/bin/env python3
"""Fusion 3: Vulkan parity for the fused gated-delta-net op (applied after apply_fwht_signed_vulkan.py).
 - raw gates: beta = sigmoid(beta), g = a[h] * softplus(g + dt_bias[h]) inside the kernel (src[7], src[8]);
 - rows-mode state read (src[6]): each sequence's state read straight from the 2D cache view;
 - fused state write-back: GDN + views + CPY(cache) -> the kernel writes the K==1 state into the cache view;
 - the model builder's capability lists accept the Vulkan backend for raw gates and rows mode.
Reference semantics: ggml-cpu ops.cpp and ggml-cuda gated_delta_net.cu (same formulas)."""
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

# ---------------------------------------------------------------- shader
sh = "ggml/src/ggml-vulkan/vulkan-shaders/gated_delta_net.comp"
edit(sh,
     "    float scale;\n    uint K;\n};\n",
     "    float scale;\n    uint K;\n"
     "    uint raw;                // beta/g arrive pre-activation; apply sigmoid / a*softplus(g+dt_bias)\n"
     "    uint rows_mode;          // state read at cache row data_rows[seq] (row length state_row_size floats)\n"
     "    uint state_row_size;\n"
     "    uint state_out_mode;     // 1: write the final state (K == 1) into StateOut rows instead of the dst tail\n"
     "    uint state_out_row_size;\n"
     "};\n")
edit(sh,
     "layout(binding = 6)           buffer DstBuf   { FLOAT_TYPE data_dst[];   };\n",
     "layout(binding = 6)           buffer DstBuf   { FLOAT_TYPE data_dst[];   };\n"
     "layout(binding = 7) readonly  buffer RowsBuf  { int        data_rows[];  };\n"
     "layout(binding = 8) readonly  buffer DtBuf    { FLOAT_TYPE data_dt[];    };\n"
     "layout(binding = 9) readonly  buffer ABuf     { FLOAT_TYPE data_a[];     };\n"
     "layout(binding = 10)          buffer SOutBuf  { FLOAT_TYPE data_sout[];  };\n")
edit(sh,
     "    const uint state_in_base       = (seq_id * H + head_id) * state_size;\n",
     "    const uint state_in_base       = rows_mode != 0u ? uint(data_rows[seq_id]) * state_row_size + head_id * state_size\n"
     "                                                     : (seq_id * H + head_id) * state_size;\n")
edit(sh,
     "        const FLOAT_TYPE beta_val = FLOAT_TYPE(data_beta[gb_off]);\n",
     "        FLOAT_TYPE beta_val = FLOAT_TYPE(data_beta[gb_off]);\n"
     "        if (raw != 0u) {\n"
     "            beta_val = FLOAT_TYPE(1.0) / (FLOAT_TYPE(1.0) + exp(-beta_val));\n"
     "        }\n")
edit(sh,
     "            const FLOAT_TYPE g_val = exp(FLOAT_TYPE(data_g[gb_off]));\n",
     "            FLOAT_TYPE g0 = FLOAT_TYPE(data_g[gb_off]);\n"
     "            if (raw != 0u) {\n"
     "                const FLOAT_TYPE x = g0 + FLOAT_TYPE(data_dt[head_id]);\n"
     "                g0 = FLOAT_TYPE(data_a[head_id]) * ((x > FLOAT_TYPE(20.0)) ? x : log(FLOAT_TYPE(1.0) + exp(x)));\n"
     "            }\n"
     "            const FLOAT_TYPE g_val = exp(g0);\n")
edit(sh,
     "    if (K == 1u) {\n        [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {\n            data_dst[s_off + state_out_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];\n        }\n    }\n",
     "    if (K == 1u) {\n"
     "        if (state_out_mode != 0u) {\n"
     "            const uint sout_base = seq_id * state_out_row_size + head_id * state_size;\n"
     "            [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {\n"
     "                data_sout[sout_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];\n"
     "            }\n"
     "        } else {\n"
     "            [[unroll]] for (uint r = 0; r < ROWS_PER_LANE; r++) {\n"
     "                data_dst[s_off + state_out_base + col * S_V + r * LANES_PER_COLUMN + lane] = s_shard[r];\n"
     "            }\n"
     "        }\n"
     "    }\n")

# ---------------------------------------------------------------- backend host
vk = "ggml/src/ggml-vulkan/ggml-vulkan.cpp"
edit(vk,
     "    float scale;\n    uint32_t K;\n};\n\nstruct vk_op_ssm_scan_push_constants {\n",
     "    float scale;\n    uint32_t K;\n    uint32_t raw;\n    uint32_t rows_mode;\n    uint32_t state_row_size;\n"
     "    uint32_t state_out_mode;\n    uint32_t state_out_row_size;\n};\n\nstruct vk_op_ssm_scan_push_constants {\n")
edit(vk,
     '                    gdn_names[si][kda], gdn_len, gdn_data, "main", 7, sizeof(vk_op_gated_delta_net_push_constants),',
     '                    gdn_names[si][kda], gdn_len, gdn_data, "main", 11, sizeof(vk_op_gated_delta_net_push_constants),')
edit(vk,
     "                // rows-indexed state read (src[6]) not implemented on Vulkan yet\n"
     "                if (op->src[6] != nullptr) {\n"
     "                    return false;\n"
     "                }\n",
     "                // rows-indexed state read (src[6]): I32 row indices into the 2D cache view\n"
     "                if (op->src[6] != nullptr && op->src[6]->type != GGML_TYPE_I32) {\n"
     "                    return false;\n"
     "                }\n")
edit(vk,
     "                // raw gates (ggml_gated_delta_net_set_raw_gates): beta and g arrive\n"
     "                // pre-activation and need beta = sigmoid(beta) and\n"
     "                // g = a * softplus(g + dt_bias), with dt_bias in src[7] and a in src[8].\n"
     "                // gated_delta_net.comp has neither those bindings nor that math - it\n"
     "                // applies exp(g) unconditionally - so the shader silently returns wrong\n"
     "                // results for this case. Decline it and let it fall back to the CPU.\n"
     "                if (ggml_get_op_params_i32(op, 1) != 0) {\n"
     "                    return false;\n"
     "                }\n",
     "                // raw gates (ggml_gated_delta_net_set_raw_gates): the shader applies sigmoid(beta) and\n"
     "                // a * softplus(g + dt_bias) itself; scalar gate only, dt_bias in src[7] and a in src[8]\n"
     "                if (ggml_get_op_params_i32(op, 1) != 0) {\n"
     "                    if (op->src[7] == nullptr || op->src[8] == nullptr ||\n"
     "                        op->src[7]->type != GGML_TYPE_F32 || op->src[8]->type != GGML_TYPE_F32 ||\n"
     "                        op->src[3]->ne[0] != 1) {\n"
     "                        return false;\n"
     "                    }\n"
     "                }\n")
edit(vk,
     "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst) {\n",
     "// cache_out: when the following CPY of the final state into the recurrent cache was fused away, the\n"
     "// [D, n_seqs] cache view the kernel writes that state into instead of the dst tail (K == 1 only).\n"
     "static void ggml_vk_gated_delta_net(ggml_backend_vk_context * ctx, vk_context& subctx, ggml_tensor * dst, const ggml_tensor * cache_out = nullptr) {\n")
edit(vk,
     "    const vk_op_gated_delta_net_push_constants pc = {\n"
     "        H, n_tokens, n_seqs, s_off,\n"
     "        sq1, sq2, sq3,\n"
     "        sv1, sv2, sv3,\n"
     "        sb1, sb2, sb3,\n"
     "        neq1, rq3,\n"
     "        scale,\n"
     "        K\n"
     "    };\n"
     "\n"
     "    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,\n"
     "        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf},\n"
     "        pc, { H, n_seqs, S_v });\n",
     "    const bool raw  = ggml_get_op_params_i32(dst, 1) != 0;\n"
     "    const bool rows = dst->src[6] != nullptr;\n"
     "    GGML_ASSERT(!cache_out || K == 1);\n"
     "    // unused bindings are given a valid buffer (dst) the shader never reads in that mode\n"
     "    const vk_subbuffer rows_buf = rows      ? ggml_vk_tensor_subbuffer(ctx, dst->src[6]) : dst_buf;\n"
     "    const vk_subbuffer dt_buf   = raw       ? ggml_vk_tensor_subbuffer(ctx, dst->src[7]) : dst_buf;\n"
     "    const vk_subbuffer a_buf    = raw       ? ggml_vk_tensor_subbuffer(ctx, dst->src[8]) : dst_buf;\n"
     "    const vk_subbuffer sout_buf = cache_out ? ggml_vk_tensor_subbuffer(ctx, cache_out)   : dst_buf;\n"
     "\n"
     "    const vk_op_gated_delta_net_push_constants pc = {\n"
     "        H, n_tokens, n_seqs, s_off,\n"
     "        sq1, sq2, sq3,\n"
     "        sv1, sv2, sv3,\n"
     "        sb1, sb2, sb3,\n"
     "        neq1, rq3,\n"
     "        scale,\n"
     "        K,\n"
     "        raw ? 1u : 0u,\n"
     "        rows ? 1u : 0u,\n"
     "        rows ? (uint32_t)(dst->src[5]->nb[1] / sizeof(float)) : 0u,\n"
     "        cache_out ? 1u : 0u,\n"
     "        cache_out ? (uint32_t)(cache_out->nb[1] / sizeof(float)) : 0u\n"
     "    };\n"
     "\n"
     "    ggml_vk_dispatch_pipeline(ctx, subctx, pipeline,\n"
     "        {src_buf[0], src_buf[1], src_buf[2], src_buf[3], src_buf[4], src_buf[5], dst_buf, rows_buf, dt_buf, a_buf, sout_buf},\n"
     "        pc, { H, n_seqs, S_v });\n")
# fusion predicate (placed before ggml_vk_can_fuse): GDN + views + CPY(cache) -> kernel writes the cache
edit(vk,
     "static bool ggml_vk_can_fuse(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {\n",
     "// gated_delta_net followed (through views) by the CPY that stores its final state into the\n"
     "// recurrent cache: the kernel writes the cache view directly and the CPY is skipped (K == 1).\n"
     "// Returns the number of nodes after the GDN that the fusion covers, 0 when it does not apply.\n"
     "static int ggml_vk_can_fuse_gdn_cache(const struct ggml_cgraph * cgraph, int node_idx) {\n"
     "    const ggml_tensor * gdn = cgraph->nodes[node_idx];\n"
     "    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 || (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {\n"
     "        return 0;\n"
     "    }\n"
     "    if (ggml_get_op_params_i32(gdn, 0) != 1) {\n"
     "        return 0;\n"
     "    }\n"
     "    const ggml_tensor * src_v = gdn->src[2];\n"
     "    const int64_t S_v = src_v->ne[0];\n"
     "    const int64_t H = src_v->ne[1];\n"
     "    const int64_t n_tokens = src_v->ne[2];\n"
     "    const int64_t n_seqs = src_v->ne[3];\n"
     "    const int64_t D = S_v * S_v * H;\n"
     "    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);\n"
     "    const ggml_tensor * cpy = nullptr;\n"
     "    int skip = 0;\n"
     "    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {\n"
     "        const ggml_tensor * n = cgraph->nodes[j];\n"
     "        if (ggml_is_empty(n) || n->op == GGML_OP_RESHAPE || n->op == GGML_OP_TRANSPOSE ||\n"
     "            n->op == GGML_OP_VIEW || n->op == GGML_OP_PERMUTE || n->op == GGML_OP_NONE) {\n"
     "            continue;\n"
     "        }\n"
     "        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {\n"
     "            return 0;\n"
     "        }\n"
     "        cpy = n;\n"
     "        skip = j - node_idx;\n"
     "    }\n"
     "    if (cpy == nullptr) {\n"
     "        return 0;\n"
     "    }\n"
     "    const ggml_tensor * src = cpy->src[0];\n"
     "    const ggml_tensor * dst = cpy->src[1];\n"
     "    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off || !ggml_is_contiguous(src)) {\n"
     "        return 0;\n"
     "    }\n"
     "    if (dst->op != GGML_OP_VIEW || dst->type != GGML_TYPE_F32 || dst->ne[0] != D || dst->ne[1] != n_seqs || dst->ne[2] != 1 || dst->ne[3] != 1 ||\n"
     "        dst->nb[0] != sizeof(float) || dst->nb[1] % sizeof(float) != 0 || dst->nb[1] < (size_t) D * sizeof(float)) {\n"
     "        return 0;\n"
     "    }\n"
     "    return skip <= 12 ? skip : 0; // op_srcs_fused_elementwise holds 13 nodes\n"
     "}\n\n"
     "static bool ggml_vk_can_fuse(const ggml_backend_vk_context * ctx, const struct ggml_cgraph * cgraph, int node_idx, std::initializer_list<enum ggml_op> ops) {\n")
edit(vk,
     "        const char *fusion_string {};\n",
     "        const char *fusion_string {};\n        int gdn_cache_skip = 0;\n")
edit(vk,
     "            } else if (ggml_vk_can_fuse_fwht_signed(ctx, cgraph, i)) {\n",
     "            } else if ((gdn_cache_skip = ggml_vk_can_fuse_gdn_cache(cgraph, i)) > 0) {\n"
     "                ctx->num_additional_fused_ops = gdn_cache_skip;\n"
     "                fusion_string = \"GDN_CACHE\";\n"
     "                ctx->fused_ops_write_mask |= 1; // the kernel still writes the attention scores into the gdn output\n"
     "                for (int f = 0; f <= gdn_cache_skip; ++f) {\n"
     "                    op_srcs_fused_elementwise[f] = false;\n"
     "                }\n"
     "            } else if (ggml_vk_can_fuse_fwht_signed(ctx, cgraph, i)) {\n")
edit(vk,
     "    case GGML_OP_GATED_DELTA_NET:\n        ggml_vk_gated_delta_net(ctx, compute_ctx, node);\n",
     "    case GGML_OP_GATED_DELTA_NET:\n"
     "        if (ctx->num_additional_fused_ops > 0) {\n"
     "            ggml_vk_gated_delta_net(ctx, compute_ctx, node, cgraph->nodes[node_idx + ctx->num_additional_fused_ops]->src[1]);\n"
     "        } else {\n"
     "            ggml_vk_gated_delta_net(ctx, compute_ctx, node);\n"
     "        }\n")


# overlap check: a CPY never reads src[1] -- it is the destination the fused kernel writes (cpy's result is a
# view of it), so it must not count as an input overlapping the output; and a disabled fusion must not keep
# its profiler label.
edit(vk,
     "                            if (!src || src->op == GGML_OP_NONE) {\n"
     "                                continue;\n"
     "                            }\n"
     "                            if (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {\n",
     "                            if (!src || src->op == GGML_OP_NONE) {\n"
     "                                continue;\n"
     "                            }\n"
     "                            if (s == 1 && cgraph->nodes[i + k]->op == GGML_OP_CPY) {\n"
     "                                continue; // cpy's src[1] is its destination, never read\n"
     "                            }\n"
     "                            if (ggml_vk_tensors_overlap(src, dst, op_srcs_fused_elementwise[k])) {\n")
edit(vk,
     "            if (need_disable) {\n"
     "                ctx->num_additional_fused_ops = 0;\n"
     "                ctx->fused_ops_write_mask = 1;\n",
     "            if (need_disable) {\n"
     "                fusion_string = {};\n"
     "                ctx->num_additional_fused_ops = 0;\n"
     "                ctx->fused_ops_write_mask = 1;\n")

# ---------------------------------------------------------------- model builder capability lists
q = "src/models/qwen35.cpp"
edit(q,
     '        if (is_gpu && strcmp(reg_name, "MTL") != 0) {\n            gdn_state_rows_dev_ok = false;\n        }\n',
     '        if (is_gpu && strcmp(reg_name, "MTL") != 0 && strcmp(reg_name, "Vulkan") != 0) {\n            gdn_state_rows_dev_ok = false;\n        }\n')
edit(q,
     '        if (strcmp(reg_name, "MTL") != 0 && strcmp(reg_name, "CUDA") != 0 &&\n'
     '            strcmp(reg_name, "ROCm") != 0 && strcmp(reg_name, "MUSA") != 0 && strcmp(reg_name, "CPU") != 0) {\n'
     '            gdn_raw_gates_dev_ok = false;\n        }\n',
     '        if (strcmp(reg_name, "MTL") != 0 && strcmp(reg_name, "CUDA") != 0 &&\n'
     '            strcmp(reg_name, "ROCm") != 0 && strcmp(reg_name, "MUSA") != 0 && strcmp(reg_name, "CPU") != 0 &&\n'
     '            strcmp(reg_name, "Vulkan") != 0) {\n'
     '            gdn_raw_gates_dev_ok = false;\n        }\n')
print("all edits applied")
