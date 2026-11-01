"""
Runs QAOA circuits on IBM Quantum hardware (Z-basis sampling only).

Adapted from the advisor's angles_estimation_ibm.py (IBMQEMTomography.measure_counts):
the X/Y/Z tomography is not needed for the MIS, so only the Z basis is measured, which
uses a third of the QPU time. Additions: the transpiled-circuit statistics (to check a
circuit before spending QPU time) and the job information (job id, QPU seconds, wall time).

Choosing the circuit: the transpiler is randomized, so the same circuit can come out with more or
fewer two-qubit gates (we saw 413 to 493 for the 30-pole QAOA). transpile_best() transpiles it
--tries times with different seeds and keeps the one with the highest estimated success
probability (ESP = product of (1 - error) of every gate and measurement, from the backend's
calibration). ESP prefers fewer two-qubit gates and better qubits.

Credentials: run the advisor's ibm_setup.py ONCE on your own computer (outside this
repository, never commit the API key). After that QiskitRuntimeService() finds the key.

Dry runs need no account: IBMRunner(fake="FakeFez") transpiles for a fake copy of a
156-qubit IBM device and reports depth / two-qubit gates. Fake backends are only used
for transpiling here, never to execute (no artificial noise).

Error mitigation (mitigation="none" | "dd" | "twirl" | "dd+twirl"): IBM's own techniques, applied by
the runtime on the real hardware (no artificial noise is added anywhere):
  * dd: dynamical decoupling. Idle qubits get pulse pairs (XpXm) that cancel the slow phase noise
    while they wait for the two-qubit gates of other qubits.
  * twirl: Pauli twirling of the two-qubit gates and of the measurement. Each shot runs a randomized
    but equivalent version of the circuit, which turns coherent errors (that add up gate after gate)
    into random noise; measurement twirling also removes the 0/1 bias of the readout.
Neither changes the counts format, so the analysis is the same. Twirling uses a little more QPU time.

Requires: pip install qiskit-ibm-runtime
"""
import time

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager


def two_qubit_gate_count(circuit):
    return sum(1 for inst in circuit.data if inst.operation.num_qubits == 2 and inst.operation.name != "barrier")


def circuit_stats(isa_circuit):
    """Size of a transpiled (ISA) circuit: what the hardware will actually run."""
    ops = dict(isa_circuit.count_ops())
    return {
        "physical_qubits_used": len({isa_circuit.find_bit(q).index
                                     for inst in isa_circuit.data for q in inst.qubits}),
        "depth": isa_circuit.depth(),
        "depth_2q": isa_circuit.depth(lambda inst: inst.operation.num_qubits == 2),
        "two_qubit_gates": two_qubit_gate_count(isa_circuit),
        "ops": {k: int(v) for k, v in ops.items()},
    }


def estimated_success_probability(isa_circuit, target):
    """ESP: product of (1 - error) over every gate and measurement of a transpiled circuit, using the
    backend calibration. A rough predictor of how much of the ideal result survives the noise."""
    log_esp = 0.0
    for inst in isa_circuit.data:
        name = inst.operation.name
        if name in ("barrier", "delay") or name not in target.operation_names:
            continue
        qargs = tuple(isa_circuit.find_bit(q).index for q in inst.qubits)
        props = target[name].get(qargs) if hasattr(target[name], "get") else None
        err = getattr(props, "error", None) if props is not None else None
        if err:
            log_esp += np.log1p(-min(err, 0.999999))
    return float(np.exp(log_esp))


MITIGATIONS = ("none", "dd", "twirl", "dd+twirl")


def sampler_options(mitigation="none"):
    """SamplerV2 options for one of MITIGATIONS."""
    if mitigation not in MITIGATIONS:
        raise ValueError(f"mitigation must be one of {MITIGATIONS}, got {mitigation!r}")
    options = {}
    if "dd" in mitigation:
        options["dynamical_decoupling"] = {"enable": True, "sequence_type": "XpXm"}
    if "twirl" in mitigation:
        options["twirling"] = {"enable_gates": True, "enable_measure": True,
                               "num_randomizations": "auto", "shots_per_randomization": "auto"}
    return options


def fake_backend(name):
    """A fake IBM backend by class name, e.g. FakeFez or FakeMarrakesh (156 qubits)."""
    from qiskit_ibm_runtime import fake_provider
    try:
        return getattr(fake_provider, name)()
    except AttributeError:
        raise SystemExit(f"Unknown fake backend {name!r}; try FakeFez, FakeMarrakesh or FakeTorino")


class IBMRunner:
    """One backend, pass manager and sampler, reused for every circuit."""

    def __init__(self, backend_name=None, *, min_qubits=1, optimization_level=3, fake=None, service=None,
                 mitigation="none"):
        self.mitigation = mitigation
        self.options = sampler_options(mitigation)  # validates the name before any connection
        if fake:
            self.backend = fake_backend(fake)
            self.is_fake = True
        else:
            from qiskit_ibm_runtime import QiskitRuntimeService
            service = service or QiskitRuntimeService()
            if backend_name:
                self.backend = service.backend(backend_name)
            else:
                self.backend = service.least_busy(operational=True, simulator=False, min_num_qubits=min_qubits)
            self.is_fake = False
        if self.backend.num_qubits < min_qubits:
            raise SystemExit(f"{self.backend.name} has {self.backend.num_qubits} qubits, the circuit needs {min_qubits}")
        self.pass_manager = generate_preset_pass_manager(backend=self.backend, optimization_level=optimization_level)
        self.optimization_level = optimization_level
        self._sampler = None

    @property
    def name(self):
        return self.backend.name

    def transpile(self, circuit):
        """Adds the Z measurement of every qubit and transpiles for the backend."""
        circuit = circuit.remove_final_measurements(inplace=False)
        if circuit.num_clbits:
            raise ValueError("The circuit must not contain other classical registers")
        circuit.measure_all()
        t0 = time.time()
        isa = self.pass_manager.run(circuit)
        return isa, time.time() - t0

    def transpile_best(self, circuit, tries=10):
        """Transpiles with `tries` different transpiler seeds and keeps the circuit with the highest
        ESP. Returns the circuit, its stats (with esp, seed and the range over all tries) and the time."""
        circuit = circuit.remove_final_measurements(inplace=False)
        if circuit.num_clbits:
            raise ValueError("The circuit must not contain other classical registers")
        circuit.measure_all()
        t0 = time.time()
        best, tried = None, []
        for seed in range(tries):
            pm = generate_preset_pass_manager(backend=self.backend, optimization_level=self.optimization_level,
                                              seed_transpiler=seed)
            isa = pm.run(circuit)
            esp = estimated_success_probability(isa, self.backend.target)
            gates = two_qubit_gate_count(isa)
            tried.append((esp, gates))
            if best is None or esp > best[0]:
                best = (esp, seed, isa)
        esp, seed, isa = best
        stats = circuit_stats(isa) | {
            "esp": esp, "transpiler_seed": seed, "tries": tries,
            "two_qubit_gates_range": [min(g for _, g in tried), max(g for _, g in tried)],
            "esp_range": [min(e for e, _ in tried), max(e for e, _ in tried)],
        }
        return isa, stats, time.time() - t0

    def sampler(self):
        """SamplerV2 for this backend with the chosen mitigation options (created once)."""
        if self._sampler is None:
            from qiskit_ibm_runtime import SamplerV2 as Sampler
            self._sampler = Sampler(mode=self.backend, options=self.options)
        return self._sampler

    def run_many(self, isa_circuits, shots):
        """Samples several transpiled circuits in ONE job (one queue wait, less QPU overhead).
        Returns a list of counts and the job information."""
        if self.is_fake:
            raise RuntimeError("Fake backends are only for dry runs: pass --backend or no --fake to use the QPU")
        t0 = time.time()
        job = self.sampler().run(list(isa_circuits), shots=shots)
        print(f"  job {job.job_id()} ({len(isa_circuits)} circuits) submitted to {self.name}; waiting in the queue...",
              flush=True)
        result = job.result()
        counts = [r.data.meas.get_counts() for r in result]
        try:
            qpu_seconds = job.usage()
        except Exception:
            qpu_seconds = None
        info = {"job_id": job.job_id(), "backend": self.name, "shots": shots, "circuits": len(isa_circuits),
                "mitigation": self.mitigation, "qpu_seconds": qpu_seconds, "wall_seconds": time.time() - t0}
        return counts, info

    def run(self, isa_circuit, shots):
        """Samples an already transpiled circuit on the QPU. Returns counts (Qiskit bit order,
        highest qubit on the left, same as Aer) and the job information."""
        if self.is_fake:
            raise RuntimeError("Fake backends are only for dry runs: pass --backend or no --fake to use the QPU")
        t0 = time.time()
        job = self.sampler().run([isa_circuit], shots=shots)
        print(f"  job {job.job_id()} submitted to {self.name}; waiting in the queue...", flush=True)
        result = job.result()
        counts = result[0].data.meas.get_counts()
        try:
            qpu_seconds = job.usage()
        except Exception:  # usage is not available on every plan / version
            qpu_seconds = None
        info = {"job_id": job.job_id(), "backend": self.name, "shots": shots, "mitigation": self.mitigation,
                "qpu_seconds": qpu_seconds, "wall_seconds": time.time() - t0}
        return counts, info


def fetch_job(job_id, service=None):
    """Retrieves a finished job by its id (e.g. after the terminal was closed or the connection dropped).
    Returns the counts of each circuit, the job information and the transpiled circuits that were
    submitted (None where they cannot be read back)."""
    from qiskit import QuantumCircuit
    from qiskit_ibm_runtime import QiskitRuntimeService
    service = service or QiskitRuntimeService()
    job = service.job(job_id)
    status = str(job.status())
    if status != "DONE":
        raise SystemExit(f"Job {job_id} is {status}; wait until it is DONE (see quantum.cloud.ibm.com/workloads)")
    result = job.result()
    counts = [r.data.meas.get_counts() for r in result]
    circuits = []
    try:
        for pub in job.inputs.get("pubs", []):
            c = getattr(pub, "circuit", None)
            if c is None and isinstance(pub, (list, tuple)) and pub:
                c = pub[0]
            circuits.append(c if isinstance(c, QuantumCircuit) else None)
    except Exception:
        circuits = []
    if len(circuits) != len(counts):
        circuits = [None] * len(counts)
    try:
        qpu_seconds = job.usage()
    except Exception:
        qpu_seconds = None
    backend = job.backend()
    info = {"job_id": job_id, "backend": getattr(backend, "name", str(backend)),
            "shots": sum(counts[0].values()) if counts else None, "circuits": len(counts),
            "qpu_seconds": qpu_seconds, "retrieved_by_id": True}
    return counts, info, circuits
