"""Drive the report CLI with graph records shaped like the CUDA producer."""
import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


CLI = Path(__file__).with_name('profile_report.py')
SHAPE = {'type': 'f32', 'ne': [128, 8, 1, 1]}
GRAPH = {'schema': 1, 'graph': 0, 'device': 0, 'nodes': 4, 'cuda_graphs': False,
         'operations': [
             {'first': 0, 'last': 1, 'stream': 0, 'ms': 2.0,
              'ops': ['RMS_NORM', 'MUL'], 'src0': SHAPE, 'src1': None, 'dst': SHAPE},
             {'first': 3, 'last': 3, 'stream': 1, 'ms': 3.0,
              'ops': ['MUL_MAT'], 'src0': SHAPE, 'src1': SHAPE, 'dst': SHAPE}]}


class ProfileCLI(unittest.TestCase):
    def run_report(self, records, tail=''):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'profile.jsonl'
            path.write_text(''.join(json.dumps(r) + '\n' for r in records) + tail)
            return subprocess.run([sys.executable, str(CLI), str(path)],
                                  text=True, capture_output=True, timeout=10)

    def test_fused_and_unfused_work_is_counted_once_per_shape(self):
        second = copy.deepcopy(GRAPH)
        second['graph'] = 1
        result = self.run_report([GRAPH, second])
        self.assertEqual(result.returncode, 0, result.stderr)
        report = json.loads(result.stdout)
        self.assertEqual(report['graphs'], 2)
        self.assertEqual([r['work_ms'] for r in report['operations']], [6.0, 4.0])
        self.assertEqual(report['operations'][1]['ops'], ['RMS_NORM', 'MUL'])
        self.assertEqual(report['operations'][1]['calls'], 2)
        self.assertIn('Not wall time', report['interpretation'])

    def test_invalid_capture_cannot_produce_a_report(self):
        cases = []
        for mutation in ['overlap', 'missing_fused_op', 'negative', 'nan', 'cuda_graphs']:
            graph = copy.deepcopy(GRAPH)
            if mutation == 'overlap': graph['operations'][1]['first'] = 1
            if mutation == 'missing_fused_op': graph['operations'][0]['ops'] = ['RMS_NORM']
            if mutation == 'negative': graph['operations'][0]['ms'] = -1
            if mutation == 'nan': graph['operations'][0]['ms'] = float('nan')
            if mutation == 'cuda_graphs': graph['cuda_graphs'] = True
            cases.append([graph])
        cases += [[], [GRAPH, GRAPH]]
        for records in cases:
            with self.subTest(records=records):
                result = self.run_report(records)
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(result.stdout, '')
        result = self.run_report([GRAPH], '{"schema":')
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(result.stdout, '')


if __name__ == '__main__':
    unittest.main()
