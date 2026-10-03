#!/usr/bin/env python3
"""Prepare, or explicitly execute, one cold endpoint measurement."""
import argparse
import hashlib
import http.client
import json
from pathlib import Path
import socket
import sys
import threading
import time
from urllib.parse import urlsplit
from urllib.request import urlopen
import uuid


def write_json(path, value):
    with path.open("x") as f:
        json.dump(value, f, indent=2)
        f.write("\n")


def read_json(url):
    with urlopen(url, timeout=10) as response:
        return json.load(response)


def snapshot(state):
    keys = ("alias", "endpoint_build", "class", "context_size", "slots",
            "batch_size", "slots_busy", "generations_in_flight", "queue_depth",
            "kv_cells_resident", "served", "failed", "tokens_in", "tokens_out")
    return {k: state[k] for k in keys if k in state}


def prepare(output):
    output.mkdir(parents=True, exist_ok=False)
    session = "resident-baseline-" + uuid.uuid4().hex
    lines = [f"Record {i:04d}: component sample-{i:04d}; status archived; region north."
             for i in range(1024)]
    lines.insert(512, "Record TARGET: the verification code is CEDAR-731.")
    text = ('Read the records and return only a JSON object with exactly one key, '
            '"code", whose string value is the verification code from Record TARGET.\n'
            + "\n".join(lines))
    request = {
        "messages": [{"role": "system", "content": "Answer using the supplied records."},
                     {"role": "user", "content": text}],
        "max_tokens": 128, "temperature": 0, "seed": 12345,
        "stream": True, "native_prefix_reuse": False,
        "session_id": session, "reap_sessions": [session],
    }
    write_json(output / "request.json", request)
    write_json(output / "expectation.json", {"answer": {"code": "CEDAR-731"}})
    print(f"Prepared {len(text)} characters; token count is unmeasured. No network request sent.")


def stream_once(endpoint, body, output, seconds):
    parsed = urlsplit(endpoint)
    if parsed.scheme not in ("http", "https") or parsed.username or parsed.password:
        raise ValueError("Endpoint must be an HTTP(S) URL without credentials")
    cls = http.client.HTTPSConnection if parsed.scheme == "https" else http.client.HTTPConnection
    conn = cls(parsed.hostname, parsed.port, timeout=seconds)
    started = time.monotonic()
    conn.connect()
    sock = conn.sock

    def cancel():
        try:
            sock.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass

    timer = threading.Timer(max(0.001, seconds - (time.monotonic() - started)), cancel)
    timer.daemon = True
    timer.start()
    result = {"ttft_seconds": None, "keepalive_frames": 0, "usage": None,
              "finish_reason": None, "done": False, "content": "", "reasoning": ""}
    try:
        conn.request("POST", parsed.path.rstrip("/") + "/v1/chat/completions", body,
                     {"Content-Type": "application/json", "X-Merlin-Disable-Thinking": "true"})
        response = conn.getresponse()
        result["http_status"] = response.status
        if response.status != 200:
            raise RuntimeError("Endpoint rejected the request")
        with (output / "response.sse").open("xb") as raw:
            while True:
                line = response.readline()
                if not line:
                    break
                raw.write(line)
                raw.flush()
                if time.monotonic() - started >= seconds:
                    raise TimeoutError("Request wall bound exceeded")
                if line.startswith(b":"):
                    result["keepalive_frames"] += 1
                if not line.startswith(b"data:"):
                    continue
                payload = line[5:].strip()
                if payload == b"[DONE]":
                    result["done"] = True
                    break
                frame = json.loads(payload)
                if "error" in frame:
                    raise RuntimeError("Endpoint emitted an error frame")
                if frame.get("usage"):
                    result["usage"] = frame["usage"]
                for choice in frame.get("choices", []):
                    delta = choice.get("delta", {})
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    if (content or reasoning) and result["ttft_seconds"] is None:
                        result["ttft_seconds"] = time.monotonic() - started
                    result["content"] += content
                    result["reasoning"] += reasoning
                    if choice.get("finish_reason"):
                        result["finish_reason"] = choice["finish_reason"]
        if not result["done"] or not result["usage"] or not result["finish_reason"]:
            raise RuntimeError("Incomplete stream: DONE, usage and finish reason are required")
        if not result["content"] or result["ttft_seconds"] is None:
            raise RuntimeError("No answer tokens received")
        return result
    finally:
        timer.cancel()
        conn.close()
        result["wall_seconds"] = time.monotonic() - started
        write_json(output / "measurement.json", result)


def execute(args):
    output = args.output
    if not (output / "request.json").is_file() or (output / "attempt.json").exists():
        raise ValueError("Prepare a new output directory first; an attempt cannot be overwritten")
    lock = json.loads((Path(__file__).resolve().parents[1] / "runtime.lock.json").read_text())
    health = read_json(args.robot.rstrip("/") + "/health")
    runs = health.get("running_runs", {}).get("total")
    if type(runs) is not int or runs != 0:
        raise RuntimeError("Fleet running_runs.total is nonzero or unknown; no inference sent")
    state = read_json(args.endpoint.rstrip("/") + "/state")
    for key in ("slots_busy", "generations_in_flight", "queue_depth"):
        if type(state.get(key)) is not int or state[key] != 0:
            raise RuntimeError("Endpoint is busy or its idleness is unknown; no inference sent")
    expected = {"model_digest": lock["endpoint_model_digest"], "kv_type": lock["kv_type"]}
    if any(state.get("class", {}).get(k) != v for k, v in expected.items()):
        raise RuntimeError("Model or KV type differs from the pinned pilot")
    if state.get("endpoint_build", {}).get("binding") != lock["endpoint_binding"]:
        raise RuntimeError("Endpoint binding differs from the pinned pilot")
    if any(state.get(k) != lock[k] for k in ("context_size", "slots")):
        raise RuntimeError("Endpoint window or slots differ from the pinned pilot")
    body = (output / "request.json").read_bytes()
    request = json.loads(body)
    if (request.get("native_prefix_reuse") is not False or request.get("stream") is not True
            or request.get("max_tokens") != 128
            or request.get("reap_sessions") != [request.get("session_id")]
            or not str(request.get("session_id", "")).startswith("resident-baseline-")):
        raise ValueError("Request must retain the prepared cold, bounded, own-session controls")
    write_json(output / "before.json", snapshot(state))
    write_json(output / "attempt.json", {"started_unix": time.time(), "running_runs": runs,
               "request_sha256": hashlib.sha256(body).hexdigest(), "client_bound_seconds": args.seconds})
    try:
        result = stream_once(args.endpoint, body, output, args.seconds)
        expected_answer = json.loads((output / "expectation.json").read_text())["answer"]
        passed = json.loads(result["content"]) == expected_answer and result["finish_reason"] == "stop"
        write_json(output / "verdict.json", {"retrieval_passed": passed})
        if not passed:
            raise RuntimeError("Retrieval answer or finish reason failed")
        print(json.dumps({k: result[k] for k in ("usage", "ttft_seconds", "wall_seconds", "finish_reason")}))
    finally:
        try:
            write_json(output / "after.json", snapshot(read_json(args.endpoint.rstrip("/") + "/state")))
        except Exception:
            write_json(output / "after-unavailable.json", {"state": "unknown"})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--robot", default="http://10.200.55.20:8080")
    parser.add_argument("--endpoint", default="http://10.200.55.43:8081")
    parser.add_argument("--seconds", type=int, default=300, choices=range(1, 301), metavar="1..300")
    args = parser.parse_args()
    try:
        execute(args) if args.execute else prepare(args.output)
    except Exception as exc:
        # Do not echo transport exception text: it can contain URLs or remote payloads.
        print(f"Baseline stopped ({type(exc).__name__}); inspect saved results.", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
