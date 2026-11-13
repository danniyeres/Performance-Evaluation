"""Phase 1: shared-cache-line stride sweep and process scaling."""
import ctypes
import multiprocessing as mp
from multiprocessing import shared_memory
from harness import run_benchmark, emit_verify_tag

STUDENT_ID = 230103200
M = 16_000_000
MASK = (1 << 64) - 1


def fs_worker(shm_name: str, slot_idx: int, iters: int) -> None:
    shm = shared_memory.SharedMemory(name=shm_name)
    arr = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
    acc = arr[slot_idx]
    for i in range(iters):
        acc = (acc * 6364136223846793005 + (i | 1)) & MASK
        arr[slot_idx] = acc
    del arr
    shm.close()


def trial(p: int, stride: int, total: int = M) -> None:
    shm = shared_memory.SharedMemory(create=True, size=4096)
    workers = []
    try:
        slots = (ctypes.c_uint64 * 512).from_buffer(shm.buf)
        for i in range(p):
            slots[i * stride] = i + 1
        del slots
        base, rem = divmod(total, p)
        for i in range(p):
            proc = mp.Process(target=fs_worker,
                              args=(shm.name, i * stride, base + (i < rem)))
            workers.append(proc)
        for proc in workers:
            proc.start()
        for proc in workers:
            proc.join()
            if proc.exitcode:
                raise RuntimeError(f"worker exited with status {proc.exitcode}")
    finally:
        shm.close()
        shm.unlink()


def main() -> None:
    pphys = 8
    print("stride sweep: stride, median_ms, IQR_ms, Mops/s")
    sweep = {}
    for stride in (1, 2, 4, 8, 16, 32):
        stats = run_benchmark(lambda: trial(pphys, stride), warmup=0, runs=3)
        sweep[stride] = stats.median_ms
        print(f"{stride}, {stats.median_ms:.3f}, {stats.iqr_ms:.3f}, {M/stats.median_ms/1000:.3f}")
    print("scaling: p, stride, median_ms, speedup, efficiency, karp_flatt")
    for stride in (1, 16):
        t1 = None
        for p in (1, 2, 4, 8):
            stats = run_benchmark(lambda p=p, stride=stride: trial(p, stride), warmup=0, runs=3)
            t1 = stats.median_ms if p == 1 else t1
            speedup = t1 / stats.median_ms
            e = (1/speedup - 1/p) / (1 - 1/p) if p > 1 else 0.0
            print(f"{p}, {stride}, {stats.median_ms:.3f}, {speedup:.3f}, {speedup/p:.3f}, {e:.5f}")
    print(emit_verify_tag("PHASE1", STUDENT_ID, sweep[1], sweep[16]))


if __name__ == "__main__":
    main()
