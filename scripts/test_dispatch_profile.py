"""Build and run the same portable selection/profile code consumed by CUDA dispatch."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest


class DispatchProfileTest(unittest.TestCase):
    def test_production_profile_and_selection(self):
        root = Path(__file__).resolve().parents[1]
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / 'dispatch-profile-test'
            subprocess.run([os.environ.get('CXX', 'c++'), '-std=c++17', '-Wall', '-Wextra', '-Werror',
                            '-I', str(root / 'engine/dispatch'), str(root / 'engine/tests/dispatch_profile.cpp'),
                            '-o', str(output)], check=True)
            subprocess.run([str(output), str(Path(temporary) / 'profile.tsv')], check=True)


if __name__ == '__main__':
    unittest.main()
