import io
import struct
import unittest

from calibration_shapes import MAX_OUTPUT_BYTES, backend_shapes, pq2_shapes, requested_workload


def u32(value):
    return struct.pack('<I', value)


def u64(value):
    return struct.pack('<Q', value)


def string(value):
    value = value.encode()
    return u64(len(value)) + value


def tensor(name, dims, kind=142):
    return string(name) + u32(len(dims)) + b''.join(u64(n) for n in dims) + u32(kind) + u64(0)


class HeaderOnly(io.BytesIO):
    def __init__(self, header):
        super().__init__(header + b'WEIGHTS MUST NOT BE READ')
        self.header_end = len(header)

    def read(self, n=-1):
        if n < 0 or self.tell() + n > self.header_end:
            raise AssertionError('Touched model weight data')
        return super().read(n)


class CalibrationShapesTest(unittest.TestCase):
    def test_reads_real_shapes_without_loading_weights(self):
        metadata = (string('tokens') + u32(9) + u32(8) + u64(2) + string('a') + string('b') +
                    string('version') + u32(4) + u32(1))
        records = (tensor('up.weight', [5120, 17408]) + tensor('gate.weight', [5120, 17408]) +
                   tensor('down.weight', [17408, 5120]) + tensor('norm.weight', [5120], 0))
        source = HeaderOnly(b'GGUF' + u32(3) + u64(4) + u64(2) + metadata + records)
        self.assertEqual(pq2_shapes(source), [(5120, 17408), (17408, 5120)])
        self.assertEqual(source.tell(), source.header_end)

    def test_truncated_and_malformed_headers_rejected(self):
        cases = [b'GGUF', b'GGUF' + u32(1),
                 b'GGUF' + u32(3) + u64(0) + u64(1) + u64(1000000),
                 b'GGUF' + u32(3) + u64(1) + u64(0) + tensor('bad', [129, 32])]
        for data in cases:
            with self.subTest(data=data), self.assertRaises(ValueError):
                pq2_shapes(io.BytesIO(data))

    def test_batch_matrix_and_packed_strides(self):
        lines = list(backend_shapes([(17408, 5120)], 2048))
        self.assertEqual(len(lines), 32)
        self.assertEqual(sorted({int(line.split()[3]) for line in lines}), [1, 2, 4, 8, 16, 128, 512, 2048])
        self.assertIn(
                         '29 0 17408 2048 1 1 0 2 142 5120 17408 1 1 34 1360 23674880 23674880 '
                         '0 5120 2048 1 1 4 20480 41943040 41943040 pq2_m17408_n2048_k5120 1 2048 1', lines)
        keys = {requested_workload(line) for line in lines}
        self.assertIn((17408, 4, 5120, 4, 4, 0), keys)
        self.assertIn((17408, 4, 5120, 1, 4, 1), keys)
        self.assertIn((17408, 4, 5120, 2, 4, 1), keys)
        self.assertIn((17408, 4, 5120, 2, 4, 2), keys)

    def test_workspace_bound_and_small_ubatch(self):
        self.assertEqual(len(list(backend_shapes([(67, 128)], 4))), 7)
        for line in backend_shapes([(200000, 5120)], 2048):
            fields = line.split()
            self.assertLessEqual(int(fields[2]) * int(fields[3]) * 4, MAX_OUTPUT_BYTES)


if __name__ == '__main__':
    unittest.main()
