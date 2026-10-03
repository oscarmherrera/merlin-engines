"""Exercise the real CLI and HTTP/SSE path without a GPU."""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
import unittest


ROOT = Path(__file__).resolve().parents[1]
CLI = ROOT / "benchmarks/single_request.py"
LOCK = json.loads((ROOT / "runtime.lock.json").read_text())


class BaselineCLI(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.output = Path(self.temp.name) / "result"
        self.health = {"running_runs": {"total": 0}}
        self.state = {
            "class": {"model_digest": LOCK["endpoint_model_digest"], "kv_type": LOCK["kv_type"]},
            "endpoint_build": {"binding": LOCK["endpoint_binding"]},
            "context_size": LOCK["context_size"], "slots": LOCK["slots"],
            "slots_busy": 0, "generations_in_flight": 0, "queue_depth": 0,
        }
        self.mode = "success"
        self.posts = 0
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                data = owner.health if self.path == "/health" else owner.state
                body = json.dumps(data).encode()
                self.send_response(200)
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                owner.posts += 1
                owner.request = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                owner.path = self.path
                if owner.mode == "http_error":
                    self.send_response(503)
                    self.end_headers()
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.end_headers()
                if owner.mode == "timeout":
                    time.sleep(2)
                    return
                self.wfile.write(b": prefill\n\n")
                self.wfile.flush()
                time.sleep(0.03)
                answer = '{"code":"CEDAR-731"}' if owner.mode != "wrong" else '{"code":"WRONG"}'
                frames = [{"choices": [{"delta": {"content": answer}}]}]
                if owner.mode != "incomplete":
                    frames += [{"choices": [{"delta": {}, "finish_reason": "stop"}]},
                               {"choices": [], "usage": {"prompt_tokens": 19000,
                                "completion_tokens": 9, "total_tokens": 19009}}]
                for frame in frames:
                    self.wfile.write(b"data: " + json.dumps(frame).encode() + b"\n\n")
                self.wfile.write(b"data: [DONE]\n\n")

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.url = f"http://127.0.0.1:{self.server.server_port}"
        self.assertEqual(self.run_cli().returncode, 0)

    def tearDown(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()
        self.temp.cleanup()

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(CLI), "--output", str(self.output),
                               "--robot", self.url, "--endpoint", self.url, *args],
                              capture_output=True, text=True, timeout=10)

    def test_prepare_is_offline_and_preserves_existing_request(self):
        data = (self.output / "request.json").read_bytes()
        self.assertEqual(self.posts, 0)
        self.assertNotEqual(self.run_cli().returncode, 0)
        self.assertEqual((self.output / "request.json").read_bytes(), data)

    def test_success_uses_real_stream_and_own_session_cleanup(self):
        run = self.run_cli("--execute")
        self.assertEqual(run.returncode, 0, run.stderr)
        self.assertEqual(self.posts, 1)
        self.assertEqual(self.path, "/v1/chat/completions")
        self.assertEqual(self.request["reap_sessions"], [self.request["session_id"]])
        self.assertFalse(self.request["native_prefix_reuse"])
        result = json.loads((self.output / "measurement.json").read_text())
        self.assertGreaterEqual(result["ttft_seconds"], 0.03)
        self.assertEqual(result["keepalive_frames"], 1)
        self.assertEqual(result["usage"]["prompt_tokens"], 19000)
        self.assertTrue(json.loads((self.output / "verdict.json").read_text())["retrieval_passed"])
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertEqual(self.posts, 1)

    def test_running_run_refuses_before_inference(self):
        self.health["running_runs"]["total"] = 1
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertEqual(self.posts, 0)

    def test_unknown_running_runs_refuses(self):
        self.health = {"active_sessions": 0}
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertEqual(self.posts, 0)

    def test_busy_endpoint_refuses(self):
        self.state["generations_in_flight"] = 1
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertEqual(self.posts, 0)

    def test_wrong_model_refuses(self):
        self.state["class"]["model_digest"] = "different"
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertEqual(self.posts, 0)

    def test_changed_window_refuses(self):
        self.state["context_size"] = 131072
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertEqual(self.posts, 0)

    def test_incomplete_stream_cannot_pass(self):
        self.mode = "incomplete"
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertFalse((self.output / "verdict.json").exists())
        self.assertTrue((self.output / "after.json").exists())

    def test_http_error_cannot_pass(self):
        self.mode = "http_error"
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertEqual(json.loads((self.output / "measurement.json").read_text())["http_status"], 503)

    def test_wrong_answer_cannot_pass(self):
        self.mode = "wrong"
        self.assertNotEqual(self.run_cli("--execute").returncode, 0)
        self.assertFalse(json.loads((self.output / "verdict.json").read_text())["retrieval_passed"])

    def test_wall_bound_interrupts_silent_stream(self):
        self.mode = "timeout"
        started = time.monotonic()
        self.assertNotEqual(self.run_cli("--execute", "--seconds", "1").returncode, 0)
        self.assertLess(time.monotonic() - started, 1.8)
        self.assertTrue((self.output / "after.json").exists())


if __name__ == "__main__":
    unittest.main()
