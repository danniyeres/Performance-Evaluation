"""Phase 0: lightweight runtime telemetry."""
import multiprocessing as mp
import time
from harness import emit_verify_tag

STUDENT_ID = 230103200

if __name__ == "__main__":
    n = 1_000_000
    start = time.perf_counter_ns()
    for _ in range(n):
        time.perf_counter_ns()
    latency = (time.perf_counter_ns() - start) / n
    print(f"mp.cpu_count()={mp.cpu_count()}")
    print(f"perf_counter_ns average={latency:.2f} ns ({n:,} calls)")
    print(emit_verify_tag("PHASE0", STUDENT_ID, latency, latency))
