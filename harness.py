"""Shared timing and verification helpers for the performance lab."""
from dataclasses import dataclass
import multiprocessing as mp
import time
from typing import Callable


@dataclass
class BenchStats:
    min_ms: float
    median_ms: float
    iqr_ms: float


def fnv1a_hash(value: str) -> int:
    h = 14695981039346656037
    for byte in value.encode("utf-8"):
        h = ((h ^ byte) * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return h


def run_benchmark(fn: Callable[[], None], warmup: int = 2, runs: int = 7) -> BenchStats:
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(runs):
        start = time.perf_counter_ns()
        fn()
        samples.append((time.perf_counter_ns() - start) / 1e6)
    samples.sort()
    return BenchStats(samples[0], samples[len(samples) // 2],
                      samples[(3 * len(samples)) // 4] - samples[len(samples) // 4])


def emit_verify_tag(phase: str, seed: int, m1: float, m2: float) -> str:
    hw = mp.cpu_count()
    sig = fnv1a_hash(f"{phase}|{seed}|{hw}|{m1:.4f}|{m2:.4f}")
    return f"VERIFY_HASH[{phase}]: {sig:016X} (P_log={hw})"
