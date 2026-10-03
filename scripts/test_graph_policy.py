"""Exercise the graph drift policy used by real CUDA completion monitoring."""
import os
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
            result = subprocess.run([str(output), str(Path(temporary) / 'profile.tsv')],
                                    check=True, capture_output=True, text=True)
            self.assertIn('reason=graph_cohort_latency_drift', result.stderr)
            self.assertIn('per_kernel_cause_known=false profile_saved=1', result.stderr)


if __name__ == '__main__':
    unittest.main()
