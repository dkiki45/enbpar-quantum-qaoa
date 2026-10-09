import pytest
from core.sim_backend import backend_options, results_suffix


def test_legacy_setup_is_unchanged():
    # MPS + CPU + double: same options and same folder names as before
    assert backend_options("matrix_product_state", "CPU", "double") == {"method": "matrix_product_state"}
    assert results_suffix("matrix_product_state", "CPU", "double") == ""


def test_gpu_options_and_suffix():
    assert backend_options("statevector", "GPU", "double", custatevec=False) == {"method": "statevector", "device": "GPU"}
    assert backend_options("statevector", "GPU", "single", custatevec=True) == {
        "method": "statevector", "device": "GPU", "cuStateVec_enable": True, "precision": "single"}
    assert results_suffix("statevector", "GPU", "double") == "_statevector_GPU"
    assert results_suffix("statevector", "CPU", "double") == "_statevector_CPU"
    assert results_suffix("statevector", "GPU", "single") == "_statevector_GPU_single"


def test_invalid_combinations():
    with pytest.raises(ValueError):
        backend_options("matrix_product_state", "GPU", "double")  # Aer has no GPU MPS
    with pytest.raises(ValueError):
        backend_options("matrix_product_state", "CPU", "single")
    with pytest.raises(ValueError):
        backend_options("statevector", "TPU", "double")
