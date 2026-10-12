"""
Simple standalone GPU test for Qiskit Aer (no project code needed).

1. Shows versions, the GPU (nvidia-smi) and runs a 2-qubit Bell circuit on the GPU.
2. Runs the same QAOA-like circuit (ring graph, RZZ + RX layers, fixed angles) on CPU and on
   GPU with the statevector method, for growing qubit counts, and prints the time of each and
   the speedup. It also checks that CPU and GPU give the same answer (within shot noise).

Requirements: qiskit + qiskit-aer-gpu-cu11 (see requirements-gpu.txt), NVIDIA driver working
(`nvidia-smi` must show the GPU table).

Usage:
    python gpu_test.py                         # 20, 24, 26, 28, 30 qubits
    python gpu_test.py --qubits 20,25 --depth 2
    python gpu_test.py --gpu-only              # skip the CPU runs (faster)

Memory: statevector needs 16 bytes * 2^n -> 30 qubits = 16 GB (CPU RAM and GPU VRAM).
"""
import argparse
import math
import subprocess
import sys
import time

import numpy as np
from qiskit import QuantumCircuit, __version__ as qiskit_version
from qiskit_aer import AerSimulator, __version__ as aer_version


def header(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def check_gpu():
    header("1) ENVIRONMENT AND GPU CHECK")
    print(f"Python {sys.version.split()[0]} | qiskit {qiskit_version} | qiskit-aer {aer_version}")
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total,memory.used",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=10)
        print("nvidia-smi:", (out.stdout or out.stderr).strip())
    except FileNotFoundError:
        print("nvidia-smi: not found (NVIDIA driver not installed?)")

    qc = QuantumCircuit(2)
    qc.h(0)
    qc.cx(0, 1)
    qc.measure_all()
    try:
        t0 = time.perf_counter()
        counts = AerSimulator(method="statevector", device="GPU").run(qc, shots=1000, seed_simulator=1).result().get_counts()
        print(f"Bell test on GPU: {counts}  ({time.perf_counter() - t0:.2f} s, includes GPU start-up)")
    except Exception as e:
        print(f"\nGPU NOT USABLE: {type(e).__name__}: {e}")
        print("Hints:\n"
              "  - 'Driver/library version mismatch' in nvidia-smi -> the driver was updated: reboot the machine\n"
              "  - 'No CUDA device available!' -> nvidia-smi must work first\n"
              "  - Simulator without GPU support -> pip uninstall -y qiskit-aer && pip install qiskit-aer-gpu-cu11==0.17.2")
        sys.exit(1)
    ok = set(counts) <= {"00", "11"}
    print("Bell result:", "OK (only 00 and 11)" if ok else "UNEXPECTED")
    if not ok:
        sys.exit(1)


def qaoa_like_circuit(n, depth, seed=1234):
    """H on all qubits, then `depth` layers of RZZ on a ring + RX mixer, fixed random angles."""
    rng = np.random.default_rng(seed + n + 100 * depth)
    qc = QuantumCircuit(n)
    qc.h(range(n))
    for _ in range(depth):
        gamma, beta = rng.uniform(0, np.pi, 2)
        for i in range(n):
            qc.rzz(2 * gamma, i, (i + 1) % n)
        qc.rx(2 * beta, range(n))
    qc.measure_all()
    return qc


def run(qc, device, shots):
    sim = AerSimulator(method="statevector", device=device)
    t0 = time.perf_counter()
    result = sim.run(qc, shots=shots, seed_simulator=7).result()
    elapsed = time.perf_counter() - t0
    counts = result.get_counts()
    if sum(counts.values()) == 0:
        raise MemoryError("simulator returned 0 shots (state does not fit in memory)")
    return elapsed, counts


def hamming_stats(counts):
    """Mean and variance of the number of 1s per sample (a simple check that two runs agree)."""
    w = np.array([b.count("1") for b in counts for _ in range(counts[b])], dtype=float)
    return w.mean(), w.var(), len(w)


def benchmark(qubits, depth, shots, gpu_only):
    header(f"2) CPU vs GPU  (statevector, depth {depth}, {shots} shots)")
    print(f"{'qubits':>6} {'memory':>8} {'CPU s':>8} {'GPU s':>8} {'speedup':>8} {'same answer?':>14}")
    for n in qubits:
        qc = qaoa_like_circuit(n, depth)
        mem = 16 * 2 ** n / 1024 ** 3
        try:
            t_gpu, c_gpu = run(qc, "GPU", shots)
        except Exception as e:
            print(f"{n:6d} {mem:7.2f}G {'':>8} {'FAIL':>8}   GPU error: {str(e)[:60]}")
            continue
        if gpu_only:
            print(f"{n:6d} {mem:7.2f}G {'-':>8} {t_gpu:8.2f} {'-':>8} {'-':>14}", flush=True)
            continue
        try:
            t_cpu, c_cpu = run(qc, "CPU", shots)
        except Exception as e:
            print(f"{n:6d} {mem:7.2f}G {'FAIL':>8} {t_gpu:8.2f}   CPU error: {str(e)[:60]}")
            continue
        m1, v1, k1 = hamming_stats(c_cpu)
        m2, v2, k2 = hamming_stats(c_gpu)
        z = abs(m1 - m2) / math.sqrt(v1 / k1 + v2 / k2 + 1e-12)
        same = "yes" if z < 4 else f"NO (z={z:.1f})"
        print(f"{n:6d} {mem:7.2f}G {t_cpu:8.2f} {t_gpu:8.2f} {t_cpu / t_gpu:7.1f}x {same:>14}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Simple Qiskit Aer GPU test")
    ap.add_argument("--qubits", default="20,24,26,28,30", help="comma-separated qubit counts")
    ap.add_argument("--depth", type=int, default=3, help="number of QAOA-like layers (default 3)")
    ap.add_argument("--shots", type=int, default=1024)
    ap.add_argument("--gpu-only", action="store_true", help="skip the CPU runs")
    args = ap.parse_args()

    check_gpu()
    benchmark([int(x) for x in args.qubits.split(",")], args.depth, args.shots, args.gpu_only)
    print("\nDone. 'same answer? yes' = CPU and GPU agree within shot noise.")
