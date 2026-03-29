"""
ONNX export and validation tests for the RoMa model.

Test ordering is intentional:
  1. test_pytorch_*  - establish a PyTorch reference output BEFORE any ONNX export
  2. test_onnx_*     - export to ONNX, then verify outputs are numerically consistent
                       with the reference

Run with:
    pytest tests/test_onnx.py -v

Notes
-----
* Tested with upsample_preds=False, symmetric=False (single-pass coarse forward).
* The export uses a deterministic GP solver (torch.linalg.solve instead of
  Cholesky+cholesky_solve) so that the ONNX graph is portable.
* Batch size is fixed to 1 because the native torch local-correlation falls back
  to a Python loop over the batch dimension, which is unrolled at trace time.
"""

import os
import math
import tempfile

import numpy as np
import pytest
import torch

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _set_precision():
    """RoMa requires highest float32 matmul precision."""
    torch.set_float32_matmul_precision("highest")


def _get_pytorch_output(matcher, im_A: torch.Tensor, im_B: torch.Tensor):
    """
    Run the patched (ONNX-equivalent) PyTorch forward pass and return
    (flow, certainty) as numpy arrays so they can be compared with ONNX outputs.
    """
    from romatch.onnx_export import RoMaONNXWrapper, onnx_patches

    with onnx_patches(matcher):
        wrapper = RoMaONNXWrapper(matcher).to(im_A.device).eval()
        with torch.no_grad():
            flow, certainty = wrapper(im_A, im_B)

    return flow.cpu().numpy(), certainty.cpu().numpy()


def _get_onnx_output(onnx_path: str, im_A_np: np.ndarray, im_B_np: np.ndarray):
    """Run inference using ONNX Runtime."""
    import onnxruntime as ort

    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    flow, certainty = session.run(None, {"im_A": im_A_np, "im_B": im_B_np})
    return flow, certainty


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def roma_model():
    _set_precision()
    from romatch import roma_outdoor

    device = torch.device("cpu")
    model = roma_outdoor(
        device=device,
        upsample_preds=False,
        symmetric=False,
        use_custom_corr=False,
    )
    model.eval()
    return model


@pytest.fixture(scope="session")
def fixed_inputs(roma_model):
    """A deterministic pair of (im_A, im_B) tensors at coarse resolution."""
    torch.manual_seed(0)
    h, w = roma_model.h_resized, roma_model.w_resized
    im_A = torch.randn(1, 3, h, w)
    im_B = torch.randn(1, 3, h, w)
    return im_A, im_B


@pytest.fixture(scope="session")
def pytorch_reference(roma_model, fixed_inputs):
    """PyTorch (flow, certainty) reference outputs for fixed_inputs."""
    im_A, im_B = fixed_inputs
    flow, certainty = _get_pytorch_output(roma_model, im_A, im_B)
    return flow, certainty


@pytest.fixture(scope="session")
def onnx_model_path(tmp_path_factory, roma_model):
    """Export the model to ONNX once per test session and return the path."""
    from romatch.onnx_export import export_roma_to_onnx

    out_dir = tmp_path_factory.mktemp("onnx")
    path = str(out_dir / "roma_outdoor.onnx")
    export_roma_to_onnx(roma_model, path)
    return path


# ---------------------------------------------------------------------------
# 1. PyTorch baseline tests  (run BEFORE ONNX export)
# ---------------------------------------------------------------------------

class TestPyTorchBaseline:
    """Validate the PyTorch model before any ONNX export."""

    def test_output_shapes(self, roma_model, fixed_inputs):
        im_A, im_B = fixed_inputs
        h, w = roma_model.h_resized, roma_model.w_resized

        flow, certainty = _get_pytorch_output(roma_model, im_A, im_B)

        assert flow.shape == (1, 2, h, w), (
            f"Expected flow shape (1, 2, {h}, {w}), got {flow.shape}"
        )
        assert certainty.shape == (1, 1, h, w), (
            f"Expected certainty shape (1, 1, {h}, {w}), got {certainty.shape}"
        )

    def test_flow_range(self, roma_model, fixed_inputs):
        """Flow values must lie in [-1, 1] (normalized coordinates)."""
        im_A, im_B = fixed_inputs
        flow, _ = _get_pytorch_output(roma_model, im_A, im_B)

        assert np.all(np.isfinite(flow)), "Flow contains NaN or Inf"
        # A small fraction of out-of-range values is acceptable (clamped later),
        # but the vast majority should be within range.
        in_range_frac = np.mean(np.abs(flow) <= 1.0)
        assert in_range_frac > 0.9, (
            f"Too many flow values outside [-1, 1]: only {in_range_frac:.1%} in range"
        )

    def test_certainty_is_finite(self, roma_model, fixed_inputs):
        im_A, im_B = fixed_inputs
        _, certainty = _get_pytorch_output(roma_model, im_A, im_B)
        assert np.all(np.isfinite(certainty)), "Certainty contains NaN or Inf"

    def test_deterministic(self, roma_model, fixed_inputs):
        """Same inputs must produce identical outputs on repeated calls."""
        im_A, im_B = fixed_inputs
        flow1, cert1 = _get_pytorch_output(roma_model, im_A, im_B)
        flow2, cert2 = _get_pytorch_output(roma_model, im_A, im_B)

        np.testing.assert_array_equal(flow1, flow2, err_msg="PyTorch flow is not deterministic")
        np.testing.assert_array_equal(cert1, cert2, err_msg="PyTorch certainty is not deterministic")


# ---------------------------------------------------------------------------
# 2. ONNX export and validation tests
# ---------------------------------------------------------------------------

class TestONNXExport:
    """Verify that the exported ONNX model produces the same output as PyTorch."""

    def test_export_creates_file(self, onnx_model_path):
        assert os.path.isfile(onnx_model_path), "ONNX export did not create a file"
        assert os.path.getsize(onnx_model_path) > 0, "Exported ONNX file is empty"

    def test_onnx_model_is_valid(self, onnx_model_path):
        """The exported ONNX graph must pass onnx.checker validation."""
        import onnx
        model_proto = onnx.load(onnx_model_path)
        onnx.checker.check_model(model_proto)

    def test_onnx_output_shapes(self, onnx_model_path, fixed_inputs, roma_model):
        im_A_np = fixed_inputs[0].numpy()
        im_B_np = fixed_inputs[1].numpy()
        h, w = roma_model.h_resized, roma_model.w_resized

        flow, certainty = _get_onnx_output(onnx_model_path, im_A_np, im_B_np)

        assert flow.shape == (1, 2, h, w), (
            f"ONNX flow shape mismatch: expected (1, 2, {h}, {w}), got {flow.shape}"
        )
        assert certainty.shape == (1, 1, h, w), (
            f"ONNX certainty shape mismatch: expected (1, 1, {h}, {w}), got {certainty.shape}"
        )

    def test_onnx_matches_pytorch_flow(
        self, onnx_model_path, fixed_inputs, pytorch_reference
    ):
        """ONNX flow must be numerically close to the PyTorch reference."""
        im_A_np = fixed_inputs[0].numpy()
        im_B_np = fixed_inputs[1].numpy()
        pt_flow, _ = pytorch_reference

        onnx_flow, _ = _get_onnx_output(onnx_model_path, im_A_np, im_B_np)

        np.testing.assert_allclose(
            onnx_flow,
            pt_flow,
            rtol=1e-3,
            atol=1e-4,
            err_msg="ONNX flow output deviates from PyTorch reference",
        )

    def test_onnx_matches_pytorch_certainty(
        self, onnx_model_path, fixed_inputs, pytorch_reference
    ):
        """ONNX certainty must be numerically close to the PyTorch reference."""
        im_A_np = fixed_inputs[0].numpy()
        im_B_np = fixed_inputs[1].numpy()
        _, pt_cert = pytorch_reference

        _, onnx_cert = _get_onnx_output(onnx_model_path, im_A_np, im_B_np)

        np.testing.assert_allclose(
            onnx_cert,
            pt_cert,
            rtol=1e-3,
            atol=1e-4,
            err_msg="ONNX certainty output deviates from PyTorch reference",
        )

    def test_onnx_deterministic(self, onnx_model_path, fixed_inputs):
        """The ONNX model must produce identical outputs on repeated calls."""
        im_A_np = fixed_inputs[0].numpy()
        im_B_np = fixed_inputs[1].numpy()

        flow1, cert1 = _get_onnx_output(onnx_model_path, im_A_np, im_B_np)
        flow2, cert2 = _get_onnx_output(onnx_model_path, im_A_np, im_B_np)

        np.testing.assert_array_equal(flow1, flow2, err_msg="ONNX flow is not deterministic")
        np.testing.assert_array_equal(cert1, cert2, err_msg="ONNX certainty is not deterministic")
