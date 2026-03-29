<p align="center">
  <h1 align="center"> <ins>RoMa</ins> 🏛️:<br> Robust Dense Feature Matching <br> ⭐CVPR 2024⭐</h1>
  <p align="center">
    <a href="https://scholar.google.com/citations?user=Ul-vMR0AAAAJ">Johan Edstedt</a>
    ·
    <a href="https://scholar.google.com/citations?user=HS2WuHkAAAAJ">Qiyu Sun</a>
    ·
    <a href="https://scholar.google.com/citations?user=FUE3Wd0AAAAJ">Georg Bökman</a>
    ·
    <a href="https://scholar.google.com/citations?user=6WRQpCQAAAAJ">Mårten Wadenbäck</a>
    ·
    <a href="https://scholar.google.com/citations?user=lkWfR08AAAAJ">Michael Felsberg</a>
  </p>
  <h2 align="center"><p>
    <a href="https://arxiv.org/abs/2305.15404" align="center">Paper</a> |
    <a href="https://parskatt.github.io/RoMa" align="center">Project Page</a>
  </p></h2>
  <div align="center"></div>
</p>
<br/>
<p align="center">
    <img src="https://github.com/Parskatt/RoMa/assets/22053118/15d8fea7-aa6d-479f-8a93-350d950d006b" alt="example" width=80%>
    <br>
    <em>RoMa is the robust dense feature matcher capable of estimating pixel-dense warps and reliable certainties for almost any image pair.</em>
</p>

---

## Contents

1. [ONNX Export](#onnx-export)
2. [Setup / Install](#setupinstall)
3. [Demo / How to Use](#demo--how-to-use)
4. [Settings](#settings)
5. [Tiny RoMa](#tiny-roma)
6. [Reproducing Results](#reproducing-results)
7. [Reproducibility Notes](#reproducibility-notes)
8. [License / Acknowledgement / BibTeX](#license)

---

## ONNX Export

This fork adds ONNX export for the coarse single-pass forward pass, enabling
deployment with ONNX Runtime on CPU and GPU without a PyTorch dependency.

### Requirements

```bash
pip install onnx onnxruntime          # CPU inference
pip install onnxruntime-gpu           # GPU inference (Linux / Windows)
pip install matplotlib pillow         # visual comparison script
```

### Export to ONNX

```python
import torch
from romatch import roma_outdoor
from romatch.onnx_export import export_roma_to_onnx

torch.set_float32_matmul_precision("highest")

matcher = roma_outdoor(
    device=torch.device("cpu"),   # or "cuda" — the .onnx file is device-agnostic
    upsample_preds=False,         # only the coarse single-pass forward is exported
    symmetric=False,
    use_custom_corr=False,        # the custom CUDA kernel cannot be JIT-traced
)
matcher.eval()

export_roma_to_onnx(matcher, "roma_outdoor.onnx")
```

The exported file is roughly **400 MB** (VGG19-BN + ViT-L/14 weights combined).

> **What `export_roma_to_onnx` does internally**
>
> 1. Patches the GP forward pass to use a 200-iteration Conjugate Gradient solver
>    instead of `torch.linalg.cholesky` + `torch.cholesky_solve` (neither has an
>    ONNX operator).
> 2. Patches DINOv2's `interpolate_pos_encoding` to use an explicit integer `size`
>    instead of a `scale_factor` derived from `x.shape` (which the JIT tracer turns
>    into a symbolic tensor that cannot be emitted as an ONNX constant).
> 3. Calls `torch.onnx.export` with `opset_version=16` (minimum for `GridSample`).
> 4. Post-processes the raw `.onnx` file to eliminate spurious `float64` constants
>    emitted by the JIT tracer (Python float literals become ONNX `DOUBLE` constants;
>    ONNX Runtime has no `float64` kernel for `GridSample` or `Conv` on CPU).

### Run ONNX Inference

```python
import numpy as np
import onnxruntime as ort
from PIL import Image

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

def load_image(path, h=560, w=560):
    arr = np.array(Image.open(path).convert("RGB").resize((w, h)), dtype=np.float32)
    arr = (arr / 255.0 - MEAN) / STD
    return arr.transpose(2, 0, 1)[None]   # (1, 3, H, W)

sess = ort.InferenceSession(
    "roma_outdoor.onnx",
    providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
)

im_A = load_image("assets/sacre_coeur_A.jpg")
im_B = load_image("assets/sacre_coeur_B.jpg")

flow, certainty = sess.run(None, {"im_A": im_A, "im_B": im_B})
# flow      : (1, 2, H, W)  float32  — correspondence field in [-1, 1]
# certainty : (1, 1, H, W)  float32  — logits; apply sigmoid() → probabilities in [0, 1]
```

#### Output tensors

| Output | Shape | Range | Meaning |
|--------|-------|-------|---------|
| `flow` | `(1, 2, H, W)` | `[−1, 1]` | For each pixel in A, the `(x, y)` coordinate of the matching pixel in B — normalised grid coordinates suitable as input to `F.grid_sample` |
| `certainty` | `(1, 1, H, W)` | `(−∞, +∞)` | Log-odds of a valid correspondence; apply `sigmoid` to obtain match probability in `[0, 1]` |

#### Warp image B into A's frame

```python
import torch
import torch.nn.functional as F

flow_t   = torch.from_numpy(flow)                   # (1, 2, H, W)
grid     = flow_t.permute(0, 2, 3, 1)               # (1, H, W, 2) — x, y order
im_B_t   = torch.from_numpy(im_B)                   # (1, 3, H, W)
warped_B = F.grid_sample(im_B_t, grid,
                         mode="bilinear",
                         align_corners=False)        # (1, 3, H, W)
```

### Visual Comparison

Generate side-by-side panels (PyTorch vs ONNX) with:

```bash
python tests/visual_comparison.py \
    --im_A   assets/sacre_coeur_A.jpg \
    --im_B   assets/sacre_coeur_B.jpg \
    --onnx   roma_outdoor.onnx \
    --out_dir docs/visual
```

Add `--export` to force re-export even if `roma_outdoor.onnx` already exists.

Each panel is a 2 × 3 grid:

| Position | Content |
|----------|---------|
| Top-left | **Image A** — original input |
| Top-center | **Image B** — original input |
| Top-right | **Image B warped → A** — `grid_sample(im_B, flow)` |
| Bottom-left | **Confidence** — `sigmoid(certainty)` mapped to the *inferno* colourmap; bright = high confidence |
| Bottom-center | **Alpha blend** — `certainty × warped_B + (1−certainty) × im_A` |
| Bottom-right | **Dense correspondences** — HSV colour-wheel encoding (hue = direction, value = magnitude) |

**PyTorch**

![PyTorch panel](docs/visual/pytorch_panel.png)

**ONNX**

![ONNX panel](docs/visual/onnx_panel.png)

### Timing

Numbers measured on **macOS CPU (Apple M-series)** and **Linux GPU (NVIDIA RTX 3090)**.
Run the comparison script on your hardware and inspect `docs/visual/timing.json` for
exact figures.

| Backend | Device | Inference (ms) |
|---------|--------|----------------|
| PyTorch | macOS CPU | **11 316** |
| ONNX Runtime | CPUExecutionProvider (macOS) | **15 109** |
| PyTorch | NVIDIA RTX 3090 | ~400 |
| ONNX Runtime | CUDAExecutionProvider | ~350 |

> On macOS, PyTorch benefits from Apple's Accelerate/BLAS while ONNX Runtime uses a
> generic MLAS backend. On Linux CPU the ordering flips and ONNX Runtime is ~10–15 %
> faster.

#### Numerical fidelity (PyTorch vs ONNX, sacre_coeur images)

| Metric | Flow | Certainty (logits) |
|--------|------|--------------------|
| Max absolute difference | 1.58 × 10⁻⁴ | 1.35 × 10⁻² |
| Mean absolute difference | 6.07 × 10⁻⁷ | 6.81 × 10⁻⁵ |

The flow mean error is sub-pixel. The larger certainty max difference is confined
to a small number of pixels with extreme logit values; mean error is well within
`atol=1e-4` (verified by `tests/test_onnx.py`).

### ONNX Limitations

| Limitation | Detail |
|------------|--------|
| Batch size fixed to 1 | The local-correlation Python loop is unrolled by the JIT tracer at the batch size seen during export |
| No upsampling path | `upsample_preds=True` is not exported; use the full PyTorch model for full-resolution output |
| No symmetric mode | Only A→B flow is produced; use the PyTorch model for bidirectional warps |
| GP solver approximation | Cholesky + `cholesky_solve` have no ONNX operator; replaced with 200-iteration CG (condition number ~3700, converges to < 10⁻⁵ flow error) |

### Tests

```bash
pip install onnx onnxruntime pytest
pytest tests/test_onnx.py -v
```

10 tests: PyTorch baseline (shapes, flow range, finiteness, determinism) + ONNX
validation (file validity, shapes, numerical match to PyTorch, determinism).
Passes on CPU (macOS, ~3 min 40 s) and GPU (Linux/RTX 3090, ~1 min 45 s).

---

## Setup/Install

In your python environment (tested on Linux python 3.12), run:
```bash
uv pip install -e .
```
or
```bash
uv sync
```
You can also install `romatch` directly as a package from PyPI by
```bash
uv pip install romatch
```
or
```bash
uv add romatch
```

### Fused local correlation kernel

```bash
uv sync --extra fused-local-corr
```
or
```bash
uv pip install romatch[fused-local-corr]
```

---

## Demo / How to Use

Two demos are provided in the [demos folder](demo).

```python
from romatch import roma_outdoor
roma_model = roma_outdoor(device=device)
# Match
warp, certainty = roma_model.match(imA_path, imB_path, device=device)
# Sample matches for estimation
matches, certainty = roma_model.sample(warp, certainty)
# Convert to pixel coordinates (RoMa produces matches in [-1,1]x[-1,1])
kptsA, kptsB = roma_model.to_pixel_coordinates(matches, H_A, W_A, H_B, W_B)
# Find a fundamental matrix (or anything else of interest)
F, mask = cv2.findFundamentalMat(
    kptsA.cpu().numpy(), kptsB.cpu().numpy(), ransacReprojThreshold=0.2, method=cv2.USAC_MAGSAC, confidence=0.999999, maxIters=10000
)
```

**New**: You can also match arbitrary keypoints with RoMa. See [match_keypoints](romatch/models/matcher.py) in `RegressionMatcher`.

---

## Settings

### Resolution

By default RoMa uses an initial resolution of (560, 560) which is then upsampled
to (864, 864). You can change this at construction (see `roma_outdoor` kwargs), or
later by setting `roma_model.w_resized`, `roma_model.h_resized`, and
`roma_model.upsample_res`.

### Sampling

`roma_model.sample_thresh` controls the threshold used when sampling matches. A
lower or higher threshold may improve results depending on the scene.

---

## Tiny RoMa

If RoMa is too heavy, try Tiny RoMa built on top of XFeat:

```python
from romatch import tiny_roma_v1_outdoor
tiny_roma_model = tiny_roma_v1_outdoor(device=device)
```

**Mega1500:**

|  | AUC@5 | AUC@10 | AUC@20 |
|----------|----------|----------|----------|
| XFeat    | 46.4    | 58.9    | 69.2    |
| XFeat*   | 51.9    | 67.2    | 78.9    |
| Tiny RoMa v1 | 56.4 | 69.5 | 79.5   |

**Mega-8-Scenes:**

|  | AUC@5 | AUC@10 | AUC@20 |
|----------|----------|----------|----------|
| XFeat*   | 50.1    | 64.4    | 75.2    |
| Tiny RoMa v1 | 57.7 | 70.5 | 79.6   |

**IMC22:**

|  | mAA@10 |
|----------|----------|
| XFeat    | 42.1    |
| Tiny RoMa v1 | 42.2 |

---

## Reproducing Results

The experiments are in the [experiments folder](experiments).

### Training

1. Follow the instructions at https://github.com/Parskatt/DKM for downloading and
   preprocessing datasets.
2. Run the relevant experiment:
```bash
torchrun --nproc_per_node=4 --nnodes=1 --rdzv_backend=c10d experiments/roma_outdoor.py
```

### Testing

```bash
python experiments/roma_outdoor.py --only_test --benchmark mega-1500
```

---

## Reproducibility Notes

1. The `scale_factor` in the `match` method is now relative to the original training
   resolution of `560` (previously based on the set coarse resolution).
2. Newer PyTorch versions may produce slightly different results.
3. Both RANSAC and chosen correspondences are stochastic in `Mega1500`.
4. Matrix inverse in GP has been replaced with Cholesky decomposition.

Differences > 0.5 in benchmark numbers likely indicate a problem.

---

## License

All code except DINOv2 is MIT licensed.
DINOv2 is Apache 2.0 — see [DINOv2 license](https://github.com/facebookresearch/dinov2/blob/main/LICENSE).

## Acknowledgement

Codebase builds on [DKM](https://github.com/Parskatt/DKM).

## BibTeX

```bibtex
@inproceedings{edstedt2024roma,
  title={{RoMa: Robust Dense Feature Matching}},
  author={Edstedt, Johan and Sun, Qiyu and Bökman, Georg and Wadenbäck, Mårten and Felsberg, Michael},
  booktitle={IEEE Conference on Computer Vision and Pattern Recognition},
  year={2024}
}
```
