#!/usr/bin/env python3
"""Acceptance of an MTP head on the fork's llama-server: four prompts (warehouse filler at two depths, prose,
code), temperature 0, 512 tokens; draft_n / draft_n_accepted and predicted_per_second from the server's own
timings. Refuses to send while a Merlin run is in flight. usage: mtp-accept.py <label> <base_url> <out.jsonl>"""
import json, sys, time, urllib.request
label, base, out_path = sys.argv[1:4]
FILLER = "\n".join(
    f"Record Q{i:05d}: warehouse unit {i % 137} holds {i * 13 % 997} parts; route {i * 7 % 61}; "
    f"inspection state reviewed; supplier group {chr(65 + i % 26)}."
    for i in range(60000))
MARKER = "\nTarget M-73419: owner Leena; code COPPER-8362.\n"
TASK = ("\nFirst state the owner and code of target M-73419 on one line. Then write a detailed "
        "inventory audit of at least 600 words using the supplied records.")
def warehouse(chars):
    body = FILLER[:chars]
    return "Read the following warehouse records.\n" + body[:len(body)//2] + MARKER + body[len(body)//2:] + TASK
PROMPTS = [
    ("warehouse-8k", warehouse(33000)),
    ("warehouse-30k", warehouse(92000)),  # ~3.1 chars/token on these records
    ("prose", "Explain, for an engineer new to inference systems, how speculative decoding with a multi-token prediction head works, why the output distribution is unchanged, and what determines the achieved speedup. About 600 words, no headings."),
    ("code", "Write a Go package that parses the header of a GGUF file: magic, version, tensor count, metadata key-value pairs (all value types including arrays and strings) and tensor infos (name, dims, type, offset). Return typed structs, handle truncated files with errors, and include a table-driven test. Output only code."),
]
def call(url, payload=None, timeout=3600):
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)
import os
done = set()
if os.path.exists(out_path):
    done = {json.loads(l)["kind"] for l in open(out_path) if l.strip()}
with open(out_path, "a") as out:
    for kind, prompt in PROMPTS:
        if kind in done:
            continue
        total = call("http://10.200.55.20:8080/health", timeout=10)["running_runs"]["total"]
        if total != 0:
            sys.exit(f"running_runs.total is {total}: a run is in flight, not sending")
        t0 = time.monotonic()
        r = call(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": prompt}], "temperature": 0,
                 "seed": 12345, "max_tokens": 512, "cache_prompt": False, "chat_template_kwargs": {"enable_thinking": False}})
        wall = time.monotonic() - t0
        msg = r["choices"][0]["message"]; content = msg.get("content") or ""
        t = r.get("timings", {})
        row = {"label": label, "kind": kind, "wall_s": round(wall, 1), "prompt_n": t.get("prompt_n"), "predicted_n": t.get("predicted_n"),
               "prompt_tps": round(t.get("prompt_per_second") or 0, 1), "decode_tps": round(t.get("predicted_per_second") or 0, 2),
               "draft_n": t.get("draft_n"), "draft_n_accepted": t.get("draft_n_accepted"),
               "acceptance": (round(t["draft_n_accepted"] / t["draft_n"], 3) if t.get("draft_n") else None),
               "finish": r["choices"][0].get("finish_reason"), "lookup": ("COPPER-8362" in content and "Leena" in content) if kind.startswith("warehouse") else None}
        out.write(json.dumps(dict(row, content=content)) + "\n"); out.flush()
        print(json.dumps(row), flush=True)
