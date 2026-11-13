"""Phase 3: memory/compute ceilings and three GEMM loop organizations."""
import multiprocessing as mp
from multiprocessing import shared_memory
import ctypes
import time
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103200
N = 256
PPHYS = 8


def copy_worker(args):
    name_a, name_b, lo, hi = args
    a = shared_memory.SharedMemory(name=name_a)
    b = shared_memory.SharedMemory(name=name_b)
    av, bv = a.buf, b.buf
    av[lo:hi] = bv[lo:hi]
    del av, bv
    a.close(); b.close()


def fma_worker(rounds):
    x = [float(i + 1) / 17 for i in range(16)]
    y = [float(i + 3) / 19 for i in range(16)]
    for _ in range(rounds):
        x[0] = x[0] * y[0] + 1.0; x[1] = x[1] * y[1] + 1.0
        x[2] = x[2] * y[2] + 1.0; x[3] = x[3] * y[3] + 1.0
        x[4] = x[4] * y[4] + 1.0; x[5] = x[5] * y[5] + 1.0
        x[6] = x[6] * y[6] + 1.0; x[7] = x[7] * y[7] + 1.0
        x[8] = x[8] * y[8] + 1.0; x[9] = x[9] * y[9] + 1.0
        x[10] = x[10] * y[10] + 1.0; x[11] = x[11] * y[11] + 1.0
        x[12] = x[12] * y[12] + 1.0; x[13] = x[13] * y[13] + 1.0
        x[14] = x[14] * y[14] + 1.0; x[15] = x[15] * y[15] + 1.0
    return sum(x)


def roofline():
    size = 64 * 1024 * 1024
    a = shared_memory.SharedMemory(create=True, size=size)
    b = shared_memory.SharedMemory(create=True, size=size)
    try:
        b.buf[:] = b"x" * size
        q, r = divmod(size, PPHYS)
        chunks = []
        lo = 0
        for i in range(PPHYS):
            hi = lo + q + (i < r)
            chunks.append((a.name, b.name, lo, hi))
            lo = hi
        with mp.Pool(PPHYS) as pool:
            copy = run_benchmark(lambda: pool.map(copy_worker, chunks), warmup=0, runs=3)
            bandwidth = 2 * size / (copy.median_ms * 1e6)
            rounds = 60_000_000 // 32
            compute = run_benchmark(lambda: pool.map(fma_worker, [rounds] * PPHYS), warmup=0, runs=3)
            gflops = 60_000_000 * PPHYS / (compute.median_ms * 1e6)
        print(f"DRAM copy: {copy.median_ms:.3f} ms; {bandwidth:.3f} GB/s")
        print(f"Scalar compute: {compute.median_ms:.3f} ms; {gflops:.3f} GFLOP/s; ridge={gflops/bandwidth:.3f} FLOP/B")
    finally:
        a.close(); a.unlink(); b.close(); b.unlink()


def gemm_naive(a, b, n):
    c = [0.0] * (n*n)
    for i in range(n):
        for j in range(n):
            x = 0.0
            for k in range(n):
                x += a[i*n+k] * b[k*n+j]
            c[i*n+j] = x
    return c


def gemm_ikj(a, b, n):
    c = [0.0] * (n*n)
    for i in range(n):
        for k in range(n):
            aik = a[i*n+k]
            for j in range(n):
                c[i*n+j] += aik * b[k*n+j]
    return c


def gemm_blocked(a, b, n, bs):
    c = [0.0] * (n*n)
    for ii in range(0, n, bs):
        for kk in range(0, n, bs):
            for jj in range(0, n, bs):
                for i in range(ii, min(ii+bs, n)):
                    for k in range(kk, min(kk+bs, n)):
                        aik = a[i*n+k]
                        row = i*n
                        for j in range(jj, min(jj+bs, n)):
                            c[row+j] += aik * b[k*n+j]
    return c


def gemm_rows(args):
    a, b, n, first, last, variant, bs = args
    c = [0.0] * ((last - first) * n)
    if variant == "blocked":
        for ii in range(first, last, bs):
            for kk in range(0, n, bs):
                for jj in range(0, n, bs):
                    for i in range(ii, min(ii+bs, last)):
                        row = (i - first) * n
                        for k in range(kk, min(kk+bs, n)):
                            aik = a[i*n+k]
                            for j in range(jj, min(jj+bs, n)):
                                c[row+j] += aik * b[k*n+j]
        return first, c
    for i in range(first, last):
        row = (i - first) * n
        if variant == "naive":
            for j in range(n):
                total = 0.0
                for k in range(n):
                    total += a[i*n+k] * b[k*n+j]
                c[row+j] = total
        elif variant == "ikj":
            for k in range(n):
                aik = a[i*n+k]
                for j in range(n):
                    c[row+j] += aik * b[k*n+j]
    return first, c


def parallel_gemm(pool, a, b, n, variant, bs=None):
    q, r = divmod(n, PPHYS)
    chunks = []
    first = 0
    for worker in range(PPHYS):
        last = first + q + (worker < r)
        chunks.append((a, b, n, first, last, variant, bs))
        first = last
    output = [0.0] * (n*n)
    for first, part in pool.map(gemm_rows, chunks):
        output[first*n:(first+len(part)//n)*n] = part
    return output


def main():
    roofline()
    n = N
    a = [((i % 31) - 15) / 17 for i in range(n*n)]
    b = [((i % 29) - 14) / 13 for i in range(n*n)]
    stats = {}
    outputs = {}
    best = None
    with mp.Pool(PPHYS) as pool:
        for name, variant in (("naive", "naive"), ("ikj", "ikj")):
            fn = lambda variant=variant: parallel_gemm(pool, a, b, n, variant)
            bench = run_benchmark(fn, warmup=0, runs=3)
            stats[name] = bench.median_ms
            outputs[name] = fn()
            print(f"{name}: {bench.median_ms:.3f} ms, {2*n**3/(bench.median_ms*1e6):.4f} GFLOP/s")
        for bs in (8,16,32,64,96,128):
            fn = lambda bs=bs: parallel_gemm(pool, a, b, n, "blocked", bs)
            bench = run_benchmark(fn, warmup=0, runs=3)
            result = fn()
            err = max(abs(x-y) for x,y in zip(result, outputs["ikj"]))
            rate = 2*n**3/(bench.median_ms*1e6)
            print(f"BS={bs}: {bench.median_ms:.3f} ms, {rate:.4f} GFLOP/s, max_error={err:.3g}, Wtile={3*bs*bs*8} B, I≈{bs/12:.3f}")
            if best is None or bench.median_ms < best[0]:
                best = (bench.median_ms, rate, bs)
    print(f"best tile BS={best[2]}; tile footprint={3*best[2]**2*8} B vs L1d=65536 B and L2=4194304 B")
    print(emit_verify_tag("PHASE3", STUDENT_ID,
                          2*n**3/(stats["naive"]*1e6), best[1]))


if __name__ == "__main__":
    main()
