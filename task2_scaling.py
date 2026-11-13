"""Phase 2: strong and weak scaling using a persistent process pool."""
import math
import multiprocessing as mp
import time
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103200
N_STRONG = 1_200_000
N0 = 200_000


def stage_a(n: int) -> float:
    acc = 0.0
    for i in range(n):
        x = i * 1e-6 + 0.1
        for _ in range(10):
            acc += math.sin(x) * math.sqrt(abs(x) + 1.0)
    return acc


def stage_b(bounds: tuple[int, int]) -> float:
    lo, hi = bounds
    acc = 0.0
    for i in range(lo, hi):
        x = i * 1e-6 + 0.1
        for _ in range(30):
            acc += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0) * 1e-3
    return acc


def pipeline(pool, n: int, p: int) -> None:
    stage_a(max(1, int(n * 0.05)))
    q, r = divmod(n, p)
    bounds = []
    lo = 0
    for i in range(p):
        hi = lo + q + (i < r)
        bounds.append((lo, hi))
        lo = hi
    pool.map(stage_b, bounds)


def time_stage_a(n: int) -> float:
    start = time.perf_counter_ns()
    stage_a(n)
    return (time.perf_counter_ns() - start) / 1e6


def main() -> None:
    maxp = mp.cpu_count()
    strong = {}
    inflection = None
    for p in range(1, maxp + 1):
        with mp.Pool(p) as pool:
            stats = run_benchmark(lambda p=p: pipeline(pool, N_STRONG, p), warmup=0, runs=3)
            strong[p] = stats.median_ms
            print(f"strong p={p}: {stats.median_ms:.3f} ms speedup={strong[1]/stats.median_ms:.3f} efficiency={strong[1]/stats.median_ms/p:.3f}")
    a0 = time_stage_a(max(1, int(N_STRONG * .05)))
    s = a0 / strong[1]
    print(f"Amdahl serial fraction={s:.5f}; cap={1/s:.3f}")
    weak = {}
    gustafson = {}
    for p in range(1, maxp + 1):
        with mp.Pool(p) as pool:
            stats = run_benchmark(lambda p=p: pipeline(pool, p * N0, p), warmup=0, runs=3)
            weak[p] = stats.median_ms
            alpha = time_stage_a(max(1, int(p * N0 * .05))) / weak[p]
            scaled_speedup = p - alpha * (p - 1)
            gustafson[p] = scaled_speedup
            print(f"weak p={p}: {stats.median_ms:.3f} ms efficiency={weak[1]/stats.median_ms:.3f} Gustafson={scaled_speedup:.3f}")
    with mp.Pool(maxp + 1) as pool:
        inflection = run_benchmark(lambda: pipeline(pool, N_STRONG, maxp + 1), warmup=0, runs=3).median_ms
    s8 = strong[maxp] and strong[1] / strong[maxp]
    s9 = strong[1] / inflection
    print(f"oversubscription p={maxp+1}: {inflection:.3f} ms, speedup={s9:.3f}; marginal speedup p={maxp}->{maxp+1}: {s9-s8:+.3f}")
    print(emit_verify_tag("PHASE2", STUDENT_ID,
                          strong[maxp] and strong[1]/strong[maxp],
                          gustafson[maxp]))


if __name__ == "__main__":
    main()
