#!/usr/bin/env python3
"""
Visual comparison of PyTorch vs ONNX RoMa outputs.

Generates side-by-side 2×3 panel images and a timing JSON for both backends.

Panel layout
------------
  Top-left     : Image A (original)
  Top-center   : Image B (original)
  Top-right    : Image B warped into A's frame
  Bottom-left  : Confidence map  (bright = high, inferno colourmap)
  Bottom-center: Alpha blend     (certainty × warped_B + (1−certainty) × im_A)
  Bottom-right : Dense correspondences  (HSV colour-wheel flow encoding)

Usage
-----
    python tests/visual_comparison.py \\
        [--im_A  assets/sacre_coeur_A.jpg] \\
        [--im_B  assets/sacre_coeur_B.jpg] \\
        [--onnx  roma_outdoor.onnx]        \\
        [--out_dir docs/visual]            \\
        [--export]   # re-export ONNX even if the file already exists

Output files
------------
    <out_dir>/pytorch_panel.png
    <out_dir>/onnx_panel.png
    <out_dir>/timing.json
"""

import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")  # headless — no display needed
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

# ImageNet statistics — identical to RoMa training preprocessing
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Image helpers
# ---------------------------------------------------------------------------

def load_image(path: str, h: int, w: int) -> torch.Tensor:
    """Load and ImageNet-normalise an image → (1, 3, H, W) float32 tensor."""
    arr = np.array(Image.open(path).convert("RGB").resize((w, h)), dtype=np.float32)
    arr = (arr / 255.0 - _MEAN) / _STD
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def denormalize(t: torch.Tensor) -> np.ndarray:
    """(1, 3, H, W) ImageNet-normalised tensor → (H, W, 3) uint8 array."""
    arr = t[0].permute(1, 2, 0).cpu().float().numpy()
    arr = np.clip(arr * _STD + _MEAN, 0.0, 1.0)
    return (arr * 255).astype(np.uint8)


def flow_to_rgb(flow: np.ndarray) -> np.ndarray:
    """
    Colour-encode a (2, H, W) flow field as an HSV colour-wheel image.

    Hue   = flow direction (angle in [0, 2π] → [0, 1])
    Value = normalised flow magnitude
    Saturation = 1.0
    """
    u, v   = flow[0], flow[1]
    mag    = np.sqrt(u ** 2 + v ** 2)
    mag_n  = mag / (mag.max() + 1e-8)
    hue    = (np.arctan2(v, u) + math.pi) / (2.0 * math.pi)
    hsv    = np.stack([hue, np.ones_like(hue), mag_n], axis=-1).astype(np.float32)
    return (mcolors.hsv_to_rgb(hsv) * 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# Inference helpers
# ---------------------------------------------------------------------------

def run_pytorch(
    matcher,
    im_A: torch.Tensor,
    im_B: torch.Tensor,
):
    """
    Run the ONNX-patched PyTorch forward pass.

    Returns
    -------
    flow       : (2, H, W) numpy — correspondence field in [-1, 1]
    certainty  : (1, H, W) numpy — logits (apply sigmoid → probabilities)
    elapsed_ms : float — wall-clock time of a single forward pass
    """
    from romatch.onnx_export import RoMaONNXWrapper, onnx_patches

    with onnx_patches(matcher):
        wrapper = RoMaONNXWrapper(matcher).to(im_A.device).eval()
        # warmup
        with torch.no_grad():
            wrapper(im_A, im_B)
        # timed run
        if im_A.device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.no_grad():
            flow, certainty = wrapper(im_A, im_B)
        if im_A.device.type == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t0) * 1000

    return flow[0].cpu().numpy(), certainty[0].cpu().numpy(), elapsed_ms


def run_onnx(
    path: str,
    im_A_np: np.ndarray,
    im_B_np: np.ndarray,
    n_warmup: int = 1,
    n_runs: int = 3,
):
    """
    Run ONNX Runtime inference.

    Returns
    -------
    flow       : (2, H, W) numpy
    certainty  : (1, H, W) numpy
    median_ms  : float — median over n_runs forward passes
    provider   : str   — active execution provider
    """
    import onnxruntime as ort

    sess = ort.InferenceSession(
        path,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    feed = {"im_A": im_A_np, "im_B": im_B_np}

    for _ in range(n_warmup):
        sess.run(None, feed)

    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        flow, certainty = sess.run(None, feed)
        times.append((time.perf_counter() - t0) * 1000)

    provider = sess.get_providers()[0]
    return flow[0], certainty[0], float(np.median(times)), provider


# ---------------------------------------------------------------------------
# Visualisation helpers
# ---------------------------------------------------------------------------

def build_visuals(im_A_t, im_B_t, flow, certainty):
    """
    Produce the six sub-images for a panel.

    Parameters
    ----------
    im_A_t, im_B_t : (1, 3, H, W) float32 tensors (ImageNet-normalised, CPU)
    flow           : (2, H, W) numpy — correspondence field in [-1, 1]
    certainty      : (1, H, W) numpy — logits

    Returns
    -------
    Six (H, W, 3) uint8 arrays:
        im_A_rgb, im_B_rgb, warped_rgb, conf_rgb, blend_rgb, flow_rgb
    """
    cert_prob = torch.sigmoid(torch.from_numpy(certainty)).unsqueeze(0)  # (1, 1, H, W)

    grid     = torch.from_numpy(flow).unsqueeze(0).permute(0, 2, 3, 1)  # (1, H, W, 2)
    warped_B = F.grid_sample(im_B_t, grid, mode="bilinear", align_corners=False)

    blend    = cert_prob * warped_B + (1.0 - cert_prob) * im_A_t

    conf_np  = cert_prob[0, 0].numpy()
    conf_rgb = (plt.get_cmap("inferno")(conf_np)[:, :, :3] * 255).astype(np.uint8)

    return (
        denormalize(im_A_t),
        denormalize(im_B_t),
        denormalize(warped_B),
        conf_rgb,
        denormalize(blend),
        flow_to_rgb(flow),
    )


def make_panel(im_A_rgb, im_B_rgb, warped_rgb, conf_rgb, blend_rgb, flow_rgb, title):
    """Compose six sub-images into a labelled 2×3 matplotlib figure."""
    labels = [
        "Image A",
        "Image B",
        "Image B warped → A",
        "Confidence  (bright = high)",
        "Alpha blend",
        "Dense correspondences",
    ]
    images = [im_A_rgb, im_B_rgb, warped_rgb, conf_rgb, blend_rgb, flow_rgb]

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    for ax, img, lbl in zip(axes.flat, images, labels):
        ax.imshow(img)
        ax.set_title(lbl, fontsize=10)
        ax.axis("off")
    plt.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(
        description="Generate PyTorch vs ONNX visual comparison panels for RoMa."
    )
    ap.add_argument("--im_A",    default="assets/sacre_coeur_A.jpg",
                    help="Path to image A (default: assets/sacre_coeur_A.jpg)")
    ap.add_argument("--im_B",    default="assets/sacre_coeur_B.jpg",
                    help="Path to image B (default: assets/sacre_coeur_B.jpg)")
    ap.add_argument("--onnx",    default="roma_outdoor.onnx",
                    help="Path to the ONNX model file (default: roma_outdoor.onnx)")
    ap.add_argument("--out_dir", default="docs/visual",
                    help="Directory for output images and timing JSON (default: docs/visual)")
    ap.add_argument("--export",  action="store_true",
                    help="Re-export the ONNX model even if the file already exists")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    # ── Load RoMa model ───────────────────────────────────────────────────────
    torch.set_float32_matmul_precision("highest")
    from romatch import roma_outdoor
    from romatch.onnx_export import export_roma_to_onnx

    device  = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matcher = roma_outdoor(
        device=device,
        upsample_preds=False,
        symmetric=False,
        use_custom_corr=False,
    )
    matcher.eval()
    h, w = matcher.h_resized, matcher.w_resized
    print(f"Model loaded on {device}  (coarse resolution {h}×{w})")

    # ── Preprocess images ─────────────────────────────────────────────────────
    im_A   = load_image(args.im_A, h, w).to(device)
    im_B   = load_image(args.im_B, h, w).to(device)
    im_A_c = im_A.cpu()
    im_B_c = im_B.cpu()

    # ── PyTorch forward ───────────────────────────────────────────────────────
    print("\nRunning PyTorch forward …")
    pt_flow, pt_cert, pt_ms = run_pytorch(matcher, im_A, im_B)
    print(f"  Elapsed: {pt_ms:.1f} ms")

    pt_fig  = make_panel(
        *build_visuals(im_A_c, im_B_c, pt_flow, pt_cert),
        title=f"PyTorch  ({device})  —  {pt_ms:.1f} ms",
    )
    pt_path = os.path.join(args.out_dir, "pytorch_panel.png")
    pt_fig.savefig(pt_path, dpi=120, bbox_inches="tight")
    plt.close(pt_fig)
    print(f"  Saved  → {pt_path}")

    # ── ONNX export ───────────────────────────────────────────────────────────
    export_ms = None
    if not os.path.exists(args.onnx) or args.export:
        print(f"\nExporting ONNX model → {args.onnx} …")
        t0 = time.perf_counter()
        export_roma_to_onnx(matcher, args.onnx)
        export_ms = (time.perf_counter() - t0) * 1000
        print(f"  Export done in {export_ms / 1000:.1f} s")
    else:
        print(f"\nUsing existing ONNX model at {args.onnx}")

    # ── ONNX inference ────────────────────────────────────────────────────────
    print("\nRunning ONNX forward …")
    onnx_flow, onnx_cert, onnx_ms, provider = run_onnx(
        args.onnx,
        im_A.cpu().numpy(),
        im_B.cpu().numpy(),
    )
    print(f"  Elapsed: {onnx_ms:.1f} ms  (provider: {provider})")

    onnx_fig  = make_panel(
        *build_visuals(im_A_c, im_B_c, onnx_flow, onnx_cert),
        title=f"ONNX  ({provider})  —  {onnx_ms:.1f} ms",
    )
    onnx_path = os.path.join(args.out_dir, "onnx_panel.png")
    onnx_fig.savefig(onnx_path, dpi=120, bbox_inches="tight")
    plt.close(onnx_fig)
    print(f"  Saved  → {onnx_path}")

    # ── Numerical difference ──────────────────────────────────────────────────
    flow_diff = np.abs(pt_flow - onnx_flow)
    cert_diff = np.abs(pt_cert - onnx_cert)
    print(f"\nNumerical difference (PyTorch vs ONNX):")
    print(f"  flow      : max |Δ| = {flow_diff.max():.2e}   mean |Δ| = {flow_diff.mean():.2e}")
    print(f"  certainty : max |Δ| = {cert_diff.max():.2e}   mean |Δ| = {cert_diff.mean():.2e}")

    # ── Save timing JSON ──────────────────────────────────────────────────────
    timing = {
        "device": str(device),
        "pytorch_ms": round(pt_ms, 2),
        "onnx_ms": round(onnx_ms, 2),
        "onnx_provider": provider,
        "export_ms": round(export_ms, 2) if export_ms is not None else None,
        "flow_max_diff": float(flow_diff.max()),
        "flow_mean_diff": float(flow_diff.mean()),
        "cert_max_diff": float(cert_diff.max()),
        "cert_mean_diff": float(cert_diff.mean()),
    }
    t_path = os.path.join(args.out_dir, "timing.json")
    with open(t_path, "w") as fh:
        json.dump(timing, fh, indent=2)
    print(f"\nTiming JSON → {t_path}")


if __name__ == "__main__":
    main()
