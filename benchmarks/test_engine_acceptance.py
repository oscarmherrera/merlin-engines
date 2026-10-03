"""Positive and negative controls for custom-engine experiment acceptance."""
import json
from pathlib import Path
import tempfile
import unittest
from engine_acceptance import judge


class EngineAcceptance(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.case, self.baseline = self.root / 'custom', self.root / 'stock'
        self.telemetry = self.root / 'events.jsonl'
        for path, elapsed in ((self.case, 8), (self.baseline, 10)):
            path.mkdir()
            self.write(path, 'measurement.json', {'ttft_seconds': elapsed, 'wall_seconds': 12,
                'usage': {'prompt_tokens': 1000, 'completion_tokens': 10}})
            self.write(path, 'before.json', {'alias': 'rtx8000', 'class': {'model_digest': 'same'},
                'context_size': 262144, 'slots': 4, 'batch_size': 4096, 'kv_cells_resident': 0})
            self.write(path, 'request.json', {'messages': ['same'], 'session_id': str(path),
                'reap_sessions': [str(path)], 'temperature': 0})
            self.write(path, 'verdict.json', {'retrieval_passed': True})
        self.write(self.case, 'attempt.json', {'started_unix': 1000})
        self.runtime = {'pid': 20, 'engine_revision': 'a' * 40, 'captured_unix': 999,
            'evidence_source': 'proc_maps_and_sha256', 'runtime_libraries': [
                {'name': name, 'path': '/opt/merlin/lib/merlin-engines-aaaaaaa/' + name + '.0', 'sha256': 'b' * 64}
                for name in ('libllama.so', 'libggml.so', 'libggml-base.so', 'libggml-cpu.so', 'libggml-cuda.so')]}
        self.write(self.case, 'runtime-identity.json', self.runtime)
        self.profile = {'engine_revision': 'a' * 40, 'event': 'merlin_profile_status', 'profile_match': True, 'pid': 20, 'mode': 'inference'}
        self.completed = {'event': 'merlin_graph_completion', 'mode': 'inference', 'pid': 20, 'device': 0,
            'submitted_unix_ms': 1001000, 'gpu_completion': True, 'completed_custom_operations': 5,
            'monitor_id': 1, 'generation': 2, 'submission': 3}

    def write(self, directory, name, value):
        (directory / name).write_text(json.dumps(value))

    def verdict(self, rows):
        self.telemetry.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        return judge(self.case, self.baseline, self.telemetry)

    def test_completed_custom_and_matched_improvement_passes_case_only(self):
        result = self.verdict([self.profile, self.completed, self.completed])
        self.assertTrue(result['experiment_passed'])
        self.assertEqual(result['distinct_sampled_custom_completions'], 1)
        self.assertFalse(result['engine_viability_proven'])

    def test_reference_correctness_does_not_pass_engine_experiment(self):
        result = self.verdict([self.profile, dict(self.completed, completed_custom_operations=0)])
        self.assertFalse(result['experiment_passed'])
        self.assertEqual(result['classification'], 'fallback_correctness_only')

    def test_calibration_host_submission_wrong_process_and_other_request_are_not_execution(self):
        for change in ({'mode': 'calibration'}, {'event': 'merlin_kernel_dispatch', 'gpu_completion': False},
                       {'pid': 21}, {'submitted_unix_ms': 999000}, {'monitor_id': None}):
            with self.subTest(change=change):
                self.assertFalse(self.verdict([self.profile, dict(self.completed, **change)])['experiment_passed'])

    def test_slower_or_unmatched_case_fails_despite_custom_completion(self):
        original = json.loads((self.case / 'measurement.json').read_text())
        self.write(self.case, 'measurement.json', dict(original, ttft_seconds=11))
        self.assertFalse(self.verdict([self.profile, self.completed])['experiment_passed'])
        self.write(self.case, 'measurement.json', original)
        self.write(self.case, 'request.json', {'messages': ['different']})
        self.assertFalse(self.verdict([self.profile, self.completed])['experiment_passed'])

    def test_missing_wrong_runtime_identity_and_unrelated_process_fail(self):
        self.assertFalse(self.verdict([dict(self.profile, pid=21), dict(self.completed, pid=21)])['experiment_passed'])
        self.assertFalse(self.verdict([dict(self.profile, engine_revision='c' * 40), self.completed])['experiment_passed'])
        (self.case / 'runtime-identity.json').unlink()
        result = self.verdict([self.profile, self.completed])
        self.assertFalse(result['experiment_passed'])
        self.assertIsNotNone(result['missing_runtime_proof'])

    def test_actual_uncaptured_custom_receipt_is_accepted(self):
        receipt = dict(self.completed, event='merlin_kernel_completion', completion_id=7, kernel='pq2_q8_1_addsub_warp')
        self.assertTrue(self.verdict([self.profile, receipt])['experiment_passed'])

    def test_bad_answer_or_unmatched_profile_fails(self):
        self.assertFalse(self.verdict([dict(self.profile, profile_match=False), self.completed])['experiment_passed'])
        self.write(self.case, 'verdict.json', {'retrieval_passed': False})
        self.assertFalse(self.verdict([self.profile, self.completed])['experiment_passed'])


if __name__ == '__main__':
    unittest.main()
