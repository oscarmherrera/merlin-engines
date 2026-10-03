import json
from pathlib import Path
import tempfile
import unittest

from calibrate_runtime import verify_receipts


class CalibrationReceiptTest(unittest.TestCase):
    def test_only_current_executed_shapes_can_pass(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'receipt.jsonl'
            manifest = dict(engine_revision='new', runtime_revision='prism', cutlass_revision='cutlass')
            requested = {(5120, 2048, 17408, 1, 2048, 1)}
            with self.assertRaisesRegex(ValueError, 'No current'):
                verify_receipts(path, manifest, requested)
            path.write_text('')
            with self.assertRaisesRegex(ValueError, 'Zero'):
                verify_receipts(path, manifest, requested)
            record = dict(manifest, schema=2, m=5120, n=2048, k=17408,
                          sequence_batch=1, tokens_in_flight=2048, phase=1,
                          profile_saved=True, reference_valid=True, records=3)
            path.write_text(json.dumps(record) + '\n')
            self.assertEqual(verify_receipts(path, manifest, requested), requested)
            for bad in (dict(record, engine_revision='old'), dict(record, profile_saved=False),
                        dict(record, reference_valid=False), dict(record, n=16),
                        dict(record, sequence_batch=2048), dict(record, phase=0)):
                path.write_text(json.dumps(bad) + '\n')
                with self.assertRaises(ValueError):
                    verify_receipts(path, manifest, requested)


if __name__ == '__main__':
    unittest.main()
