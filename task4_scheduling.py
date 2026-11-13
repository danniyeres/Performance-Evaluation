"""Phase 4: static and dynamic scheduling for a skewed workload."""
import math
import multiprocessing as mp
import time
from harness import emit_verify_tag

STUDENT_ID = 230103200
N = 8000


def work(i: int) -> float:
    acc = 0.0
    for j in range(1 + int((i / 40) ** 2)):
        x = (i + j) * 1e-4
        acc += math.sin(x) * math.cos(x) + math.sqrt(abs(x) + 1.0)
    return acc


def worker_static(items, timings, slot):
    start = time.perf_counter()
    for i in items:
        work(i)
    timings[slot] = time.perf_counter() - start


def worker_dynamic(counter, lock, timings, slot, chunk):
    start = time.perf_counter()
    while True:
        with lock:
            lo = counter.value
            counter.value += chunk
        if lo >= N:
            break
        for i in range(lo, min(N, lo + chunk)):
            work(i)
    timings[slot] = time.perf_counter() - start


def run_static(p, cyclic=False):
    timings = mp.Array('d', p)
    procs = []
    for rank in range(p):
        items = range(rank, N, p) if cyclic else range(rank*N//p, (rank+1)*N//p)
        procs.append(mp.Process(target=worker_static, args=(items, timings, rank)))
    start=time.perf_counter()
    for x in procs: x.start()
    for x in procs: x.join()
    wall=time.perf_counter()-start
    values=list(timings)
    imbalance=max(values)/(sum(values)/p)
    return wall, imbalance


def run_dynamic(p, chunk):
    counter=mp.Value('i', 0)
    lock=mp.Lock()
    timings=mp.Array('d', p)
    procs=[mp.Process(target=worker_dynamic,args=(counter,lock,timings,i,chunk)) for i in range(p)]
    start=time.perf_counter()
    for x in procs: x.start()
    for x in procs: x.join()
    wall=time.perf_counter()-start
    values=list(timings)
    return wall, max(values)/(sum(values)/p)


def main():
    p=mp.cpu_count()
    block, lb=run_static(p)
    cyc, lc=run_static(p, True)
    print(f"block: {block:.3f}s lambda={lb:.3f}; cyclic: {cyc:.3f}s lambda={lc:.3f}")
    results={}
    for c in (1,8,64,512):
        results[c]=run_dynamic(p,c)
        print(f"dynamic C={c}: {results[c][0]:.3f}s lambda={results[c][1]:.3f}")
    best=min(results,key=lambda c:results[c][0])
    print(f"selected C*={best}")
    print(emit_verify_tag("PHASE4", STUDENT_ID, block*1000, results[best][0]*1000))


if __name__ == "__main__": main()
