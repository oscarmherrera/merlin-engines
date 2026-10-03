#!/usr/bin/env python3
"""Read only GGUF metadata and emit representative PQ2 backend calibration graphs."""
import argparse
from pathlib import Path
import struct


PQ2_TYPE = 142  # pinned Prism ggml_type; 128 values per 34-byte packed block
MUL_MAT_OP = 29
MAX_OUTPUT_BYTES = 256 * 1024 * 1024
SCALAR_BYTES = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1,
                10: 8, 11: 8, 12: 8}


def read_exact(stream, length):
    value = stream.read(length)
    if len(value) != length:
        raise ValueError('Truncated GGUF header')
    return value


def number(stream, fmt):
    return struct.unpack('<' + fmt, read_exact(stream, struct.calcsize(fmt)))[0]


def skip(stream, length):
    # Metadata lengths are untrusted; reject seeking past EOF without reading weights.
    start = stream.tell()
    end = stream.seek(0, 2)
    if length < 0 or start + length > end:
        raise ValueError('GGUF metadata length exceeds file')
    stream.seek(start + length)


def skip_value(stream, kind, depth=0):
    if depth > 8:
        raise ValueError('GGUF metadata nesting too deep')
    if kind in SCALAR_BYTES:
        skip(stream, SCALAR_BYTES[kind])
    elif kind == 8:
        skip(stream, number(stream, 'Q'))
    elif kind == 9:
        element, count = number(stream, 'I'), number(stream, 'Q')
        if element in SCALAR_BYTES:
            skip(stream, SCALAR_BYTES[element] * count)
        else:
            if count > 10000000:
                raise ValueError('GGUF metadata array too large')
            for _ in range(count):
                skip_value(stream, element, depth + 1)
    else:
        raise ValueError('Unsupported GGUF metadata type')


def pq2_shapes(stream, projections=None, fusion_shapes=None):
    if read_exact(stream, 4) != b'GGUF' or number(stream, 'I') not in (2, 3):
        raise ValueError('Expected GGUF v2 or v3')
    tensors, metadata = number(stream, 'Q'), number(stream, 'Q')
    if max(tensors, metadata) > 1000000:
        raise ValueError('Unreasonable GGUF header count')
    for _ in range(metadata):
        skip(stream, number(stream, 'Q'))
        skip_value(stream, number(stream, 'I'))
    shapes = set()
    for _ in range(tensors):
        name = read_exact(stream, number(stream, 'Q')).decode('utf-8')
        rank = number(stream, 'I')
        if not 1 <= rank <= 4:
            raise ValueError('Invalid tensor rank')
        dims = [number(stream, 'Q') for _ in range(rank)]
        kind = number(stream, 'I')
        number(stream, 'Q')  # aligned data offset; never visit tensor data
        if kind == PQ2_TYPE and rank == 2:
            k, m = dims
            if k <= 0 or m <= 0 or k % 128:
                raise ValueError('Unsupported PQ2 matrix shape')
            shapes.add((m, k))
            if projections is not None and name in ('output.weight', 'token_embd.weight'):
                projections.add((m, k))
            if fusion_shapes is not None and name.endswith(('ffn_up.weight', 'ffn_gate.weight',
                    'ffn_down.weight', 'attn_output.weight', 'ssm_out.weight')):
                fusion_shapes.add((m, k))
    return sorted(shapes)


# Bias only, then gate GLU with/without both biases; matches the CUDA fusion signature.
FUSIONS = (2, 513, 519, 769, 775, 1025, 1031)


def shape_row(m, n, k, sequences, tokens, phase, fusion=0, fixture=0):
    row_bytes = k // 128 * 34
    return (f'{MUL_MAT_OP} 0 {m} {n} 1 1 0 2 '
            f'{PQ2_TYPE} {k} {m} 1 1 34 {row_bytes} {row_bytes*m} {row_bytes*m} '
            f'0 {k} {n} 1 1 4 {k*4} {k*n*4} {k*n*4} '
            f'pq2_m{m}_n{n}_k{k}_f{fusion}_c{fixture} {sequences} {tokens} {phase} {fusion} {fixture}')


def backend_shapes(matrices, ubatch, projections=(), fusion_shapes=None):
    matrices = list(matrices)
    projections = set(projections)
    fusion_shapes = set(matrices) - projections if fusion_shapes is None else set(fusion_shapes)
    if not matrices:
        return
    tokens = sorted({n for n in (1, 2, 4, 8, 16, 128, 512, ubatch) if n <= ubatch})
    seen = set()
    rows = []

    def add(m, n, k, sequences, total_tokens, phase, fusion=0, fixture=0):
        key = (m, n, k, sequences, total_tokens, phase, fusion)
        if key not in seen and m * n * 4 <= MAX_OUTPUT_BYTES:
            seen.add(key)
            rows.append(shape_row(*key, fixture))

    # All-code controls have unique small dimensions and run through the same CUDA dispatch.
    for n in (1, 2, 4, 8):
        if n <= ubatch:
            add(67, n, 256, n, n, 0, fixture=1)
    for fusion in FUSIONS:
        add(67, 1, 256, 1, 1, 0, fusion, fixture=1)
    for m, k in matrices:
        for n in tokens:
            for sequences, total_tokens, phase in representative_workloads(n):
                add(m, n, k, sequences, total_tokens, phase)
        if (m, k) in fusion_shapes:
            for fusion in FUSIONS:
                add(m, 1, k, 1, 1, 0, fusion)
    # Logits may select one output per sequence from a larger prompt microbatch.
    projection_tokens = sorted(set(tokens) | ({66} if ubatch >= 66 else set()))
    for m, k in projections:
        for n in (1, 2, 4, 8):
            for total_tokens in projection_tokens:
                if n <= ubatch and total_tokens >= 2 * n:
                    add(m, n, k, n, total_tokens, 1)
    yield from rows


def representative_workloads(tokens):
    # Deliberately specified workloads; the backend parser never infers these from tensor shape.
    if tokens in (1, 2, 4, 8):
        yield tokens, tokens, 0  # one token for every independent decode sequence
    for sequences in (1, 2, 4, 8):
        if tokens >= 2 * sequences and tokens % sequences == 0:
            yield sequences, tokens, 1  # equal multi-token prompts
    if tokens >= 3:
        yield 2, tokens, 2  # one sequence has one token; the other has the remaining tokens


def requested_workload(line):
    parts = line.split()
    return (int(parts[2]), int(parts[3]), int(parts[9]),
            int(parts[-5]), int(parts[-4]), int(parts[-3]), int(parts[-2]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', type=Path, required=True)
    parser.add_argument('--ubatch', type=int, default=2048)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.ubatch <= 65536:
        parser.error('ubatch must be 1..65536')
    projections, fusion_shapes = set(), set()
    with args.model.open('rb') as stream:
        matrices = pq2_shapes(stream, projections, fusion_shapes)
    rows = list(backend_shapes(matrices, args.ubatch, projections, fusion_shapes))
    if not rows:
        raise SystemExit('No compatible PQ2 calibration shapes found')
    args.output.write_text('\n'.join(rows) + '\n')
    print(f'merlin-engine: calibration has {len(matrices)} matrix shapes, {len(rows)} graphs', flush=True)


if __name__ == '__main__':
    main()
