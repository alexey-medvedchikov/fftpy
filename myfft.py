#!/usr/bin/env python3.5

import math
import cmath
import functools
import operator


def generate_signal(frames, precision):
    return [
        int(precision * math.sin(i * math.pi / 10))
        for i in range(0, frames)
    ]


def print_signal(sig):
    for e in sig:
        print('{:.2f}'.format(e), end=' ')
    print()


def bit_inverse(n, pow2):
    k = 0
    for i in range(pow2-1, -1, -1):
        k += (n & 0b1) << i
        n >>= 1
    return k


def binary_inverse_perm(sig, pow2):
    return [
        sig[bit_inverse(i, pow2)]
        for i in range(len(sig))
    ]

__cached_w = {}


def w(k, n):
    r = __cached_w.get((k, n))
    if r is None:
        r = cmath.exp(-1j * 2 * math.pi * k / n)
        __cached_w[(k, n)] = r
    return r


__cached_rw = {}


def rw(k, n):
    r = __cached_w.get((k, n))
    if r is None:
        r = cmath.exp(1j * 2 * math.pi * k / n)
        __cached_w[(k, n)] = r
    return r


def fft(sig, pow2):
    sig = binary_inverse_perm(sig, pow2)
    for stage in range(0, pow2):
        base_step = 2 ** (stage + 1)
        step = 2 ** stage
        for base_idx in range(0, len(sig), base_step):
            for idx in range(0, base_step // 2):
                idx_a = base_idx + idx
                idx_b = idx_a + step
                a, b = sig[idx_a], sig[idx_b]
                sig[idx_a] = a + w(idx, base_step) * b
                sig[idx_b] = a - w(idx, base_step) * b
    return sig


def rfft(sig, pow2):
    sig = binary_inverse_perm(sig, pow2)
    for stage in range(0, pow2):
        base_step = 2 ** (stage + 1)
        step = 2 ** stage
        for base_idx in range(0, len(sig), base_step):
            for idx in range(0, base_step // 2):
                idx_a = base_idx + idx
                idx_b = idx_a + step
                a, b = sig[idx_a], sig[idx_b]
                sig[idx_a] = a + rw(idx, base_step) * b
                sig[idx_b] = a - rw(idx, base_step) * b
    return sig


def fft_naive(fft_k):
    return [
        functools.reduce(operator.add, (
                fft_k[k] * cmath.exp(-1j * 2 * math.pi * n * k / len(fft_k))
                for k in range(0, len(fft_k))
        ), 0) / len(fft_k)
        for n in range(0, len(fft_k))
    ]


def rfft_naive(fft_k):
    return [
        functools.reduce(operator.add, (
                fft_k[k] * cmath.exp(1j * 2 * math.pi * n * k / len(fft_k))
                for k in range(0, len(fft_k))
        ), 0) / len(fft_k)
        for n in range(0, len(fft_k))
    ]


def main():
    pow2 = 4
    width = 2 ** pow2
    s = generate_signal(width, 20)
    s1 = fft(s, pow2)
    s2 = rfft_naive(s1)
    print_signal(s)
    print_signal(s2)

if __name__ == '__main__':
    main()
