from pathlib import Path
import subprocess
import tempfile
import unittest

from cutlass_dependency import prepare


class CutlassDependencyTest(unittest.TestCase):
    def test_rejects_wrong_revision_or_dirty_headers(self):
        with tempfile.TemporaryDirectory() as name:
            root = Path(name)
            subprocess.run(['git', 'init', '-q', str(root)], check=True)
            header = root / 'include/cutlass/cutlass.h'
            header.parent.mkdir(parents=True)
            header.write_text('// dependency\n')
            subprocess.run(['git', '-C', name, 'add', '.'], check=True)
            subprocess.run(['git', '-C', name, '-c', 'user.name=Test', '-c', 'user.email=test@example.invalid',
                            '-c', 'commit.gpgsign=false', 'commit', '-qm', 'fixture'], check=True)
            revision = subprocess.check_output(['git', '-C', name, 'rev-parse', 'HEAD'], text=True).strip()
            lock = {'cutlass_revision': revision}
            def no_fetch(*args):
                self.fail('Existing dependency must not be downloaded again')
            self.assertEqual(prepare(lock, root, no_fetch), root.resolve())
            with self.assertRaisesRegex(RuntimeError, 'differs'):
                prepare(dict(lock, cutlass_revision='0' * 40), root, no_fetch)
            header.write_text('// modified dependency\n')
            with self.assertRaisesRegex(RuntimeError, 'local changes'):
                prepare(lock, root, no_fetch)


if __name__ == '__main__':
    unittest.main()
