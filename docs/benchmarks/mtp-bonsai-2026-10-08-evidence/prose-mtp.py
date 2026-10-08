#!/usr/bin/env python3
"""Two natural prompts (prose, code) against a direct merlin-endpoint; the repetitive warehouse filler of
measure-runtime.py is easy to draft, these are the honest acceptance read. Rates from `Completion finished`."""
import json, sys, time, urllib.request, uuid
base, label, log = sys.argv[1], sys.argv[2], sys.argv[3]
PROMPTS = {
    "prose": "Explain, for an engineer new to inference systems, how speculative decoding with a multi-token prediction head works, why the output distribution is unchanged, and what determines the achieved speedup. About 600 words, no headings.",
    "code": "Write a Go package that parses the header of a GGUF file: magic, version, tensor count, metadata key-value pairs (all value types including arrays and strings) and tensor infos (name, dims, type, offset). Return typed structs, handle truncated files with errors, and include a table-driven test. Output only code.",
}
started = time.time()
for kind, text in PROMPTS.items():
    session = f"{label}-{kind}-{uuid.uuid4().hex[:8]}"
    req = urllib.request.Request(base + "/v1/chat/completions", data=json.dumps({
        "session_id": session, "messages": [{"role": "user", "content": text}], "temperature": 0, "seed": 12345,
        "max_tokens": 512, "native_prefix_reuse": False, "reap_sessions": [session]}).encode(),
        headers={"Content-Type": "application/json", "X-Merlin-Disable-Thinking": "true"})
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=3600) as r:
        body = json.load(r)
    wall = time.monotonic() - t0
    content = body["choices"][0]["message"].get("content") or ""
    open(f"/home/oscar/merlin-r9700-direct-20261008/mtp-{label}-{kind}.txt", "w").write(content)
    print(json.dumps({"session": session, "kind": kind, "wall_s": round(wall, 1), "usage": body.get("usage"), "finish": body["choices"][0].get("finish_reason")}))
time.sleep(2)
for line in open(log, errors="replace"):
    if "Completion finished" in line and "{" in line:
        try: d = json.loads(line[line.index("{"):])
        except ValueError: continue
        if str(d.get("session_id", "")).startswith(label + "-"):
            keys = [k for k in ("session_id", "tokens_in", "tokens_out", "first_token_ms", "last_token_ms", "queue_ms", "decode_tokens_per_sec", "mtp_drafted", "mtp_accepted", "finish_reason") if k in d]
            out = {k: d[k] for k in keys}
            if all(k in d for k in ("first_token_ms", "last_token_ms", "tokens_out")) and d["last_token_ms"] > d["first_token_ms"]:
                out["decode_caller_tps"] = round((d["tokens_out"] - 1) / ((d["last_token_ms"] - d["first_token_ms"]) / 1000.0), 2)
            print(out)
