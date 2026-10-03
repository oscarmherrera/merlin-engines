"""CPU checks of actual scratch lifecycle, production overlay, and telemetry JSON."""
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from apply_workspace import apply

ROOT = Path(__file__).resolve().parents[1]
PREFILL = ROOT / 'engine/backends/cuda/prefill_cutlass'

STUB = r'''
#pragma once
#include <cassert>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <map>
#define GGML_CUDA_MAX_STREAMS 8
#define GGML_CUDA_CC_TURING 750
#define GGML_OP_MUL_MAT 1
#define GGML_ASSERT(x) assert(x)
#define GGML_UNUSED(x) (void)(x)
#define CUDA_CHECK(x) assert((x)==0)
struct ggml_tensor { int op; ggml_tensor * src[2]{}; };
struct ggml_cgraph { ggml_tensor * node=nullptr; };
inline int ggml_graph_n_nodes(ggml_cgraph * graph) { return graph->node ? 1 : 0; }
inline ggml_tensor * ggml_graph_node(ggml_cgraph * graph, int) { return graph->node; }
struct event { std::map<const ggml_tensor *,int> stream_mapping; };
struct stream_context_type { std::map<const ggml_tensor *,event> concurrent_events; };
struct ggml_backend_cuda_context {
 int device=0, curr_stream_no=0;
 int streams[2][8]{};
 stream_context_type context;
 stream_context_type & stream_context() { return context; }
};
struct info_type { struct { int cc; } devices[2]{{750},{610}}; };
inline info_type ggml_cuda_info() { return {}; }
inline void ggml_cuda_set_device(int) {}
inline int allocations=0, frees=0, syncs=0;
inline int cudaMalloc(void ** p, size_t n) { assert(n==128u*1024*1024); *p=malloc(1); ++allocations; return 0; }
inline int cudaFree(void * p) { free(p); ++frees; return 0; }
inline int cudaStreamSynchronize(int) { ++syncs; return 0; }
inline bool merlin_prefill_eligible(ggml_backend_cuda_context &, const ggml_tensor *, const ggml_tensor *, ggml_tensor *) { return true; }
'''

class WorkspaceTests(unittest.TestCase):
    def test_production_owner_lifecycle(self):
        body = r'''
#include "merlin-prefill-workspace-impl.cuh"
int main() {
 ggml_backend_cuda_context ctx, other; ggml_cgraph graph;
 merlin_prefill_initialize(ctx,false); assert(allocations==0 && !merlin_prefill_scratch(ctx));
 other.device=1; merlin_prefill_initialize(other,true); assert(allocations==0);
 merlin_prefill_initialize(ctx,true); char * primary=merlin_prefill_scratch(ctx);
 merlin_prefill_initialize(ctx,true); assert(allocations==1 && primary);
 ggml_tensor input{}, weight{}, node{GGML_OP_MUL_MAT,{&weight,&input}};
 graph.node=&node;
 for (int s=1;s<8;++s) {
   ctx.context.concurrent_events[&node].stream_mapping[&node]=s;
   merlin_prefill_prepare_graph(ctx,&graph);
 }
 assert(allocations==8 && merlin_prefill_reserved_bytes(ctx)==8*merlin_prefill_slab_bytes);
 merlin_prefill_prepare_graph(ctx,&graph); assert(allocations==8);
 ctx.curr_stream_no=7; assert(merlin_prefill_scratch(ctx)!=primary);
 for(int s=0;s<8;++s) ctx.streams[0][s]=s+1;
 merlin_prefill_release(ctx); assert(frees==8 && syncs==8 && !merlin_prefill_scratch(ctx));
 merlin_prefill_release(ctx); assert(frees==8);
 ctx.curr_stream_no=0; merlin_prefill_initialize(ctx,true); assert(allocations==9);
 merlin_prefill_release(ctx); assert(frees==9);
}
'''
        with tempfile.TemporaryDirectory() as directory:
            temp=Path(directory)
            (temp/'common.cuh').write_text(STUB)
            # Copy production headers so their common.cuh resolves to the CUDA API test double.
            for name in ('merlin-prefill-workspace.cuh','merlin-prefill-workspace-impl.cuh'):
                shutil.copyfile(PREFILL/name,temp/name)
            (temp/'test.cpp').write_text(body)
            subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror',str(temp/'test.cpp'),'-o',str(temp/'test')],check=True)
            subprocess.run([str(temp/'test')],check=True,capture_output=True)

    def test_overlay_rejects_wrong_anchor_without_write(self):
        with tempfile.TemporaryDirectory() as directory:
            temp=Path(directory); path=temp/'ggml/src/ggml-cuda/ggml-cuda.cu'
            path.parent.mkdir(parents=True); path.write_text('unrecognized source\n')
            with self.assertRaises(RuntimeError): apply(temp,ROOT)
            self.assertEqual(path.read_text(),'unrecognized source\n')

    def test_overlay_orders_owner_lifecycle(self):
        with tempfile.TemporaryDirectory() as directory:
            temp=Path(directory); path=temp/'ggml/src/ggml-cuda/ggml-cuda.cu'
            path.parent.mkdir(parents=True)
            path.write_text('    ggml_backend_t cuda_backend = new ggml_backend {\n    bool use_cuda_graph             = false;\n    delete cuda_ctx;\n')
            apply(temp,ROOT); text=path.read_text()
            self.assertLess(text.index('merlin_prefill_initialize'),text.index('new ggml_backend'))
            self.assertLess(text.index('merlin_prefill_prepare_graph'),text.index('bool use_cuda_graph'))
            self.assertLess(text.index('merlin_prefill_release'),text.index('delete cuda_ctx'))
            with self.assertRaises(RuntimeError): apply(temp,ROOT)

    def test_production_telemetry_values_and_unknowns(self):
        with tempfile.TemporaryDirectory() as directory:
            temp=Path(directory)
            (temp/'test.cpp').write_text(r'''
#include "merlin-kernel-log.cuh"
int main() {
 merlin_kernel_log("multi_token_per_sequence","cutlass",0,32,16,128,2304,false,8960,{1600,2048},134217728,2,16,"explicit_calibration_workload");
 merlin_kernel_log("single_token_per_sequence","prism",0,32,1,128,SIZE_MAX,true,SIZE_MAX);
}
''')
            subprocess.run(['c++','-std=c++17','-Wall','-Wextra','-Werror','-I',str(ROOT/'engine/telemetry'),str(temp/'test.cpp'),'-o',str(temp/'test')],check=True)
            result=subprocess.run([str(temp/'test')],check=True,capture_output=True,text=True)
            first,second=map(json.loads,result.stderr.splitlines())
            self.assertEqual(first['logical_tensor_read_bytes_estimate'],1600)
            self.assertEqual(first['logical_tensor_write_bytes_estimate'],2048)
            self.assertEqual(first['workspace_reserved_bytes'],134217728)
            self.assertEqual(first['sequence_batch'],2)
            self.assertEqual(first['tokens_in_flight'],16)
            self.assertFalse(first['physical_dram_traffic_measured'])
            self.assertEqual(first['phase_source'],'explicit_calibration_workload')
            for name in ('workspace_bytes','workspace_reserved_bytes','shared_bytes_per_block','logical_tensor_read_bytes_estimate'):
                self.assertIsNone(second[name])

if __name__ == '__main__': unittest.main()
