import importlib.util

import pytest


@pytest.fixture(autouse=True)
def _default_quant_backend():
    """Tests that install kernel requests must not leak them into the next test."""
    yield
    if importlib.util.find_spec("torch") is None:
        # Metal venv (macOS): no torch, so no quant backend state to reset.
        return
    from freetoken.layers.quantization import QuantBackend, set_quant_backend

    set_quant_backend(QuantBackend())
