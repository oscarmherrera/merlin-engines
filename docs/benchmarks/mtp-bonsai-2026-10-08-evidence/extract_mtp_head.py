#!/usr/bin/env python3
"""Write a head-only MTP GGUF from a full GGUF that embeds the NextN block (the shape of ggml-org's
mtp-Qwen3.8-Flash-Next-Q4_0.gguf: output, token_embd, blk.<n_layer>.* incl. nextn.*). All metadata is copied.
usage: extract_mtp_head.py <src.gguf> <dst.gguf>"""
import sys
import numpy as np
from gguf import GGUFReader, GGUFWriter
src, dst = sys.argv[1], sys.argv[2]
r = GGUFReader(src)
arch = r.fields["general.architecture"].parts[-1].tobytes().decode()
block_count = int(r.fields[f"{arch}.block_count"].parts[-1][0])
nextn = int(r.fields[f"{arch}.nextn_predict_layers"].parts[-1][0]) if f"{arch}.nextn_predict_layers" in r.fields else 0
assert nextn >= 1, "source has no nextn_predict_layers"
head_blocks = [f"blk.{i}." for i in range(block_count - nextn, block_count)]
print("arch", arch, "block_count", block_count, "nextn", nextn, "head blocks", head_blocks)
w = GGUFWriter(dst, arch)
# copy fields the way gguf-py's gguf_new_metadata does
from gguf import GGUFValueType
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vt = f.types[0]
    if vt == GGUFValueType.ARRAY:
        et = f.types[1]
        if et == GGUFValueType.STRING:
            val = [f.parts[i].tobytes().decode("utf-8") for i in f.data]
        else:
            val = [f.parts[i][0].item() for i in f.data]
        w.add_array(name, val)
    elif vt == GGUFValueType.STRING:
        w.add_string(name, f.parts[-1].tobytes().decode("utf-8"))
    else:
        w.add_key_value(name, f.parts[-1][0].item(), vt)
keep = []
for t in r.tensors:
    if t.name in ("output.weight", "token_embd.weight", "output_norm.weight") or any(t.name.startswith(p) for p in head_blocks):
        keep.append(t)
total = 0
for t in keep:
    # byte shape as gguf-py's own gguf_new_metadata does; the writer derives the quantized logical shape
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
    total += t.data.nbytes
print("tensors kept", len(keep), "bytes", total)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data)
w.close()
print("wrote", dst)
