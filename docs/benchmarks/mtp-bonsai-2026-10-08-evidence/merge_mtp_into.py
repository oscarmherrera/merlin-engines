#!/usr/bin/env python3
"""Append a NextN (MTP) block from a head-only GGUF to a target GGUF so the fork's llama-server can run
--spec-type draft-mtp (its MTP path reads the head from the main model). The target keeps its own
token_embd / output / output_norm; only blk.<n>.* of the head is added; block_count += nextn.
usage: merge_mtp_into.py <target.gguf> <head.gguf> <out.gguf>"""
import sys
from gguf import GGUFReader, GGUFWriter, GGUFValueType
target, head, out = sys.argv[1:4]
r = GGUFReader(target); h = GGUFReader(head)
arch = r.fields["general.architecture"].parts[-1].tobytes().decode()
harch = h.fields["general.architecture"].parts[-1].tobytes().decode()
assert arch == harch, (arch, harch)
n_layer = int(r.fields[f"{arch}.block_count"].parts[-1][0])
nextn = int(h.fields[f"{arch}.nextn_predict_layers"].parts[-1][0])
h_blocks = int(h.fields[f"{arch}.block_count"].parts[-1][0])
assert h_blocks == n_layer + nextn, (h_blocks, n_layer, nextn)
assert f"{arch}.nextn_predict_layers" not in r.fields
prefixes = [f"blk.{i}." for i in range(n_layer, n_layer + nextn)]
print("arch", arch, "target blocks", n_layer, "adding", prefixes)
w = GGUFWriter(out, arch)
for name, f in r.fields.items():
    if name.startswith("GGUF.") or name == "general.architecture":
        continue
    vt = f.types[0]
    if name == f"{arch}.block_count":
        w.add_uint32(name, n_layer + nextn); continue
    if vt == GGUFValueType.ARRAY:
        et = f.types[1]
        val = [f.parts[i].tobytes().decode("utf-8") for i in f.data] if et == GGUFValueType.STRING else [f.parts[i][0].item() for i in f.data]
        w.add_array(name, val)
    elif vt == GGUFValueType.STRING:
        w.add_string(name, f.parts[-1].tobytes().decode("utf-8"))
    else:
        w.add_key_value(name, f.parts[-1][0].item(), vt)
w.add_uint32(f"{arch}.nextn_predict_layers", nextn)
keep = list(r.tensors) + [t for t in h.tensors if any(t.name.startswith(p) for p in prefixes)]
names = [t.name for t in keep]; assert len(names) == len(set(names))
for t in keep:
    w.add_tensor_info(t.name, t.data.shape, t.data.dtype, t.data.nbytes, raw_dtype=t.tensor_type)
w.write_header_to_file(); w.write_kv_data_to_file(); w.write_ti_data_to_file()
for t in keep:
    w.write_tensor_data(t.data)
w.close()
print("tensors", len(keep), "wrote", out)
