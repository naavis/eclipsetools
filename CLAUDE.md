# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

Conda is the supported environment manager; pinned versions live in `environment.yml`.

```bash
conda env create -f environment.yml
conda activate eclipsetools
```

Code is formatted with `black` (also pinned in `environment.yml`).

## Running

The CLI entry point is `main.py`, a `click` group composed of four subgroups registered in `main.py`:

- `align` — register images against a reference (corona-based or moon-based).
- `stack` — combine pre-aligned images (`average` for equal exposures, `hdr` for mixed exposures).
- `filter` — adaptive unsharp masking (single or multi-scale).
- `utils` — helpers: `find-moon`, `create-moon-mask`, `log-stretch`, `color-calibrate`.

Use `python main.py <group> <command> --help` to discover options. Many commands accept `--n-jobs` (joblib parallelism, default `-1`) and `--moon-min-radius` / `--moon-max-radius` (Hough circle detection bounds in pixels).

## Tests

```bash
pytest -n auto                       # full suite, parallelised via pytest-xdist
pytest tests/alignment/test_transform.py::test_name   # single test
```

`tests/alignment/test_transform.py` uses `pytest_generate_tests` to parametrise from a fixed seed (`TEST_SEED`), and depends on the binary asset `tests/images/eclipse_5ms.CR3`.

## Architecture

The pipeline is **align → stack → filter**, with each stage operating on float32/float64 images normalised to `[0, 1]`.

### Image I/O (`eclipsetools/common/`)

- `image_reader.open_image` is the single entry point for reading. It transparently handles RAW files via `rawpy` (with `output_color=raw`, gamma 1.0, no auto-bright) and TIFFs via OpenCV (BGR→RGB swap, dtype-aware normalisation). All downstream code assumes its outputs.
- `image_writer.save_tiff` writes zlib-compressed TIFFs and can embed an sRGB ICC profile for display-referred output (`embed_srgb=True` for filtered/stretched outputs; off for linear intermediates).
- `circle_finder.find_circle` is the moon detector used everywhere — Hough circles followed by a least-squares refinement. Returns `DetectedCircle(center=(y, x), radius)`. Note the **(y, x)** convention used throughout.

### Alignment (`eclipsetools/alignment/` + `commands/align.py`)

Two strategies share a transform-matrix builder (`_get_transform_matrix` in `commands/align.py`):

1. **`align corona`** — full similarity transform (scale, rotation, translation) recovered from coronal structure. `transform.find_transform` uses log-polar FFTs + phase correlation for scale/rotation, then a second phase correlation for translation. Images are preprocessed via `preprocessing/workflows.py` (radial high-pass + annulus mask around the moon) before correlation; preprocessing is what makes phase correlation tractable on eclipse images.
2. **`align moon`** — translation-only, derived from the offset between detected moon centres.

Mask sizing supports two modes (`MaskMode.AUTO_PER_IMAGE` vs `MaskMode.MAX`): per-image auto-sizing or a single max radius computed across the batch via `find_max_mask_inner_radius` so all images use the same mask.

### Stacking (`eclipsetools/stacking/` + `commands/stack.py`)

`stack hdr` is the non-trivial path:

1. Sort images by brightness (`sorting.sort_images_by_brightness`, sampled mean).
2. Pairwise linear-fit consecutive (by brightness) images on non-moon pixels (`linear_fit.fit_eclipse_image_pair`, RANSAC-backed). The reference image must be one of the inputs.
3. Solve a global linear fit so every image is in the reference's units (`solve_global_linear_fits`).
4. Composite with a hat-shaped weight function (`weighting.weight_function_hat`) that down-weights near-saturated and near-zero pixels, and zeros out moon pixels using `get_binary_moon_mask`. Unfilled pixels (entirely masked/saturated) fall back to the reference image.

### Filtering (`eclipsetools/filtering/` + `commands/filter.py`)

`unsharp_mask` and `multi_unsharp_mask` work in CIELAB L\*, doing all filtering on the lightness channel then converting back. The non-obvious piece is `_convolve_with_infill`: pixels just outside the moon mask are inpainted (`filtering/inpainting.py`, numba) before convolution to prevent ringing at the moon limb. When `sigma_radial == sigma_tangent` it uses OpenCV's GaussianBlur; otherwise it uses `partial_convolution` with a spatially varying anisotropic kernel oriented relative to the moon centre. `multi_unsharp_mask` stacks several such filters (Druckmüller-style ACHF).

### Performance notes

- Parallelism is `joblib.Parallel` with `prefer="threads"`
- Hot kernel work in `filtering/` uses numba (`numba_progress.ProgressBar` for progress reporting).
- `phase_correlation.py` and `preprocessing/masking.py` use `@functools.cache` for window/Gaussian-weight lookup — keep these functions' arguments hashable.
- Phase correlation uses `rfft2`/`irfft2` (`phase_correlation.py`). Note it allocates fresh FFT arrays (`fft1`, `fft2`, `cross_power_spectrum`, the `irfft2` result) on every call — there is no buffer reuse, so each call has a large transient footprint that is multiplied by `n_jobs` when alignment runs in parallel threads.
