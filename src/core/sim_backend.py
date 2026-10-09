"""
Aer simulator settings shared by every experiment (method, device, precision).

The defaults in config.py (matrix_product_state, CPU, double) produce exactly the same
backend options as before this module existed, so old results stay comparable and the
result folders keep their old names. Any other combination gets a folder suffix, so
SKIP_COMPLETED_RUNS never mixes runs from different simulators.
"""

LEGACY = ("matrix_product_state", "CPU", "double")
GPU_METHODS = ("statevector", "density_matrix", "unitary")  # Aer methods with a GPU implementation


def _resolve(method=None, device=None, precision=None):
    import config
    method = method or config.SIMULATION_METHOD
    device = (device or getattr(config, "SIMULATION_DEVICE", "CPU")).upper()
    precision = (precision or getattr(config, "SIMULATION_PRECISION", "double")).lower()
    if device not in ("CPU", "GPU"):
        raise ValueError(f"SIMULATION_DEVICE must be 'CPU' or 'GPU', got {device!r}")
    if precision not in ("double", "single"):
        raise ValueError(f"SIMULATION_PRECISION must be 'double' or 'single', got {precision!r}")
    if device == "GPU" and method not in GPU_METHODS:
        raise ValueError(f"Aer has no GPU implementation of {method!r}: use SIMULATION_METHOD = 'statevector'")
    if precision == "single" and method != "statevector":
        raise ValueError("SIMULATION_PRECISION = 'single' is only supported here with 'statevector'")
    return method, device, precision


def backend_options(method=None, device=None, precision=None, custatevec=None):
    """Aer backend_options dict. Only non-default keys are added, so the legacy setup is unchanged."""
    import config
    method, device, precision = _resolve(method, device, precision)
    opts = {"method": method}
    if device == "GPU":
        opts["device"] = "GPU"
        if custatevec is None:
            custatevec = getattr(config, "SIMULATION_CUSTATEVEC", False)
        if custatevec:
            opts["cuStateVec_enable"] = True
    if precision == "single":
        opts["precision"] = "single"
    return opts


def results_suffix(method=None, device=None, precision=None):
    """'' for the legacy setup, otherwise e.g. '_statevector_GPU' or '_statevector_GPU_single'."""
    method, device, precision = _resolve(method, device, precision)
    if (method, device, precision) == LEGACY:
        return ""
    return f"_{method}_{device}" + ("_single" if precision == "single" else "")


def describe(method=None, device=None, precision=None):
    method, device, precision = _resolve(method, device, precision)
    return f"{method} | {device} | {precision}"


def check_gpu():
    """Run a 2-qubit circuit on the GPU. Returns None if it works, otherwise the error text.
    (AerSimulator().available_devices() lists 'GPU' even when no GPU is usable.)"""
    try:
        from qiskit import QuantumCircuit
        from qiskit_aer import AerSimulator
        qc = QuantumCircuit(2)
        qc.h(0)
        qc.cx(0, 1)
        qc.measure_all()
        AerSimulator(method="statevector", device="GPU").run(qc, shots=10).result()
        return None
    except Exception as e:  # ImportError, "No CUDA device available!", driver errors...
        return f"{type(e).__name__}: {e}"


def require_gpu_if_selected():
    """Stop early with a clear message when config asks for the GPU but it is not usable."""
    import config
    if getattr(config, "SIMULATION_DEVICE", "CPU").upper() != "GPU":
        return
    err = check_gpu()
    if err:
        raise SystemExit(
            f"SIMULATION_DEVICE = 'GPU' but the GPU is not usable:\n  {err}\n"
            "Check `nvidia-smi` and that the qaoa-gpu env (requirements-gpu.txt) is active."
        )
