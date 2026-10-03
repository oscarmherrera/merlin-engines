"""Exercise the graph drift policy used by real CUDA completion monitoring."""
import os
import json
from pathlib import Path
import subprocess
import tempfile
import unittest


class GraphPolicyTest(unittest.TestCase):
    def test_healthy_drift_and_new_workload(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'graph-policy'
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(root / 'engine/dispatch'), str(root / 'engine/tests/graph_policy.cpp'),
                            '-o', str(output)], check=True)
            subprocess.run([str(output)], check=True)

    def test_production_monitor_lifecycle_and_persisted_invalidation(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'graph-monitor'
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(root / 'engine/tests/graph-stubs'),
                            '-I', str(root / 'engine/dispatch'), '-I', str(root / 'engine/telemetry'),
                            str(root / 'engine/tests/graph_monitor.cpp'), '-o', str(output)], check=True)
            telemetry = Path(temporary) / 'engine.jsonl'
            result = subprocess.run([str(output), str(Path(temporary) / 'profile.tsv'), str(telemetry)],
                                    check=True, capture_output=True, text=True)
            self.assertEqual(result.stderr, '')
            rows = [json.loads(line) for line in telemetry.read_text().splitlines()]
            self.assertTrue(any(row.get('event') == 'merlin_test_lifecycle' for row in rows))
            self.assertTrue(any(row.get('event') == 'merlin_kernel_dispatch' for row in rows))
            self.assertTrue(any(row.get('event') == 'merlin_kernel_completion' for row in rows))
            completions = [row for row in rows if row.get('event') == 'merlin_graph_completion']
            self.assertTrue(completions)
            self.assertTrue(all(row['completed_custom_operations'] == 1 for row in completions))
            self.assertTrue(all(row['mode'] == 'inference' and row['submitted_unix_ms'] > 0 for row in completions))
            self.assertTrue(any('reason=graph_cohort_latency_drift' in row.get('message', '') for row in rows))
            self.assertTrue(any('per_kernel_cause_known=false profile_saved=1' in row.get('message', '') for row in rows))



if __name__ == '__main__':
    unittest.main()
