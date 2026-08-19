"""
eda.py
======
Exploratory data analysis and all report figures that do not involve the
U-Net (those live in evaluate.py).

Every function saves to config.FIGURE_DIR and returns the path, so a notebook
cell reads `show(eda.sample_grid())` and the report's figure list is exactly
the contents of outputs/figures/.

Plotting conventions: no seaborn, no explicit colours except where a specific
comparison needs them, one chart per figure - matplotlib defaults are fine and
the marks are for what the figures *show*, not for styling.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")          # headless-safe; notebooks override this
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from . import config, data


def _save(fig, name: str) -> Path:
    path = config.FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Figure 1: sample images across density regimes
# --------------------------------------------------------------------------
def sample_grid(n_per_density: int = 2, name: str = "fig01_sample_grid.png") -> Path:
    """Grid of example images, one row per density regime.

    Sampling by regime rather than at random is deliberate: the regimes are the
    axis along which every later result varies, so the reader should see them
    at the outset.
    """
    md = data.load_metadata()
    regimes = ["sparse", "normal", "dense", "clustered"]

    fig, axes = plt.subplots(
        len(regimes), n_per_density * 2, figsize=(3 * n_per_density * 2, 3 * len(regimes))
    )
    for r, regime in enumerate(regimes):
        ids = md[(md.density == regime) & (md.split == "train")].image_id.tolist()[:n_per_density]
        for c, image_id in enumerate(ids):
            s = data.load_sample(image_id, "train")
            ax_img = axes[r, 2 * c]
            ax_msk = axes[r, 2 * c + 1]
            ax_img.imshow(s.gray, cmap="gray", vmin=0, vmax=1)
            ax_img.set_title(f"{image_id}\n{regime}, n={md.loc[image_id, 'n_objects']}",
                             fontsize=9)
            ax_msk.imshow(s.mask, cmap="gray")
            ax_msk.set_title("ground-truth mask", fontsize=9)
            for a in (ax_img, ax_msk):
                a.axis("off")
    fig.suptitle("Sample images and ground-truth masks by density regime", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 2: intensity histograms
# --------------------------------------------------------------------------
def intensity_histograms(n_images: int = 6,
                         name: str = "fig02_intensity_hist.png") -> Path:
    """Per-image and pooled intensity histograms on a log count axis.

    The log axis matters: nuclei occupy ~8% of pixels on average, so on a
    linear axis the foreground mode is invisible next to the background spike -
    and the bimodality that justifies using Otsu at all cannot be seen.
    """
    md = data.load_metadata()
    ids = md[md.split == "train"].image_id.tolist()[:n_images]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    for image_id in ids:
        s = data.load_sample(image_id, "train")
        axes[0].hist(s.gray.ravel(), bins=64, histtype="step", label=image_id)
    axes[0].set_yscale("log")
    axes[0].set_xlabel("normalised intensity")
    axes[0].set_ylabel("pixel count (log)")
    axes[0].set_title(f"Per-image intensity histograms (n={n_images} train images)")
    axes[0].legend(fontsize=7)

    pooled = np.concatenate(
        [data.load_sample(i, "train").gray.ravel() for i in md[md.split == "train"].image_id[:20]]
    )
    axes[1].hist(pooled, bins=128, color="grey")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("normalised intensity")
    axes[1].set_ylabel("pixel count (log)")
    axes[1].set_title("Pooled histogram, 20 training images")

    from skimage.filters import threshold_otsu
    thr = threshold_otsu(pooled)
    axes[1].axvline(thr, ls="--", color="red", label=f"global Otsu = {thr:.3f}")
    axes[1].legend()

    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 3: grayscale conversion comparison
# --------------------------------------------------------------------------
def grayscale_comparison(image_id: str = "train_001",
                         name: str = "fig03_grayscale_methods.png") -> tuple[Path, pd.DataFrame]:
    """Compare blue-channel / luminance / max grayscale conversions.

    Justifies a preprocessing choice with a measurement rather than an
    assertion: we report foreground-background separation (Otsu threshold and
    the resulting Dice against ground truth) for each conversion.
    """
    from skimage.filters import threshold_otsu

    from . import classical

    s = data.load_sample(image_id, "train", gray_method="blue")
    rgb = s.rgb
    methods = ["blue", "luminance", "max"]

    rows = []
    fig, axes = plt.subplots(1, len(methods) + 1, figsize=(4 * (len(methods) + 1), 4))
    axes[0].imshow(rgb)
    axes[0].set_title(f"original RGB\n{image_id}")
    axes[0].axis("off")

    for i, m in enumerate(methods, start=1):
        g = data.to_grayscale(rgb, m)
        seg = classical.otsu_segment(g)
        inter = (seg.mask & s.mask).sum()
        dice = 2 * inter / (seg.mask.sum() + s.mask.sum() + 1e-8)
        rows.append(
            dict(method=m, otsu_threshold=round(float(threshold_otsu(g)), 4),
                 fg_mean=round(float(g[s.mask].mean()), 4),
                 bg_mean=round(float(g[~s.mask].mean()), 4),
                 separation=round(float(g[s.mask].mean() - g[~s.mask].mean()), 4),
                 otsu_dice=round(float(dice), 4))
        )
        axes[i].imshow(g, cmap="gray", vmin=0, vmax=1)
        axes[i].set_title(f"{m}\nOtsu Dice = {dice:.3f}")
        axes[i].axis("off")

    fig.suptitle("Grayscale conversion choice and its effect on segmentation")
    fig.tight_layout()
    path = _save(fig, name)
    return path, pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Figure 4: dataset composition
# --------------------------------------------------------------------------
def dataset_composition(name: str = "fig04_composition.png") -> Path:
    """Object-count distribution by split and density regime."""
    md = data.load_metadata()
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    order = ["sparse", "normal", "dense", "clustered"]
    counts = md.groupby(["split", "density"]).size().unstack(fill_value=0)[order]
    counts.plot(kind="bar", stacked=True, ax=axes[0])
    axes[0].set_title("Images per split and density regime")
    axes[0].set_ylabel("images")
    axes[0].tick_params(axis="x", rotation=0)

    axes[1].hist(md.n_objects, bins=20, color="grey", edgecolor="k")
    axes[1].set_xlabel("objects per image (ground truth)")
    axes[1].set_ylabel("images")
    axes[1].set_title(f"Object count, all {len(md)} images")

    for regime in order:
        sub = md[md.density == regime]
        axes[2].scatter(sub.n_objects, sub.area_fraction, label=regime, alpha=0.7)
    axes[2].set_xlabel("objects per image")
    axes[2].set_ylabel("foreground area fraction")
    axes[2].set_title("Count vs. coverage")
    axes[2].legend(fontsize=8)

    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 5: classical segmentation walkthrough
# --------------------------------------------------------------------------
def otsu_walkthrough(image_id: str = "val_000", split: str = "val",
                     name: str = "fig05_otsu_steps.png") -> Path:
    """Show each stage of the classical pipeline on one image."""
    from skimage import morphology
    from skimage.filters import threshold_otsu

    from . import classical

    s = data.load_sample(image_id, split)
    thr = threshold_otsu(s.gray)
    raw = s.gray > thr
    opened = classical._opening(raw, config.MORPH_DISK_RADIUS)
    cleaned = classical._remove_small_objects(opened, config.MIN_OBJECT_AREA)
    seg = classical.otsu_segment(s.gray)

    panels = [
        (s.gray, f"grayscale ({config.GRAYSCALE_METHOD})", "gray"),
        (raw, f"Otsu threshold = {thr:.3f}", "gray"),
        (opened, "after opening", "gray"),
        (cleaned, "after small-object removal", "gray"),
        (seg.label_map, f"labelled: {seg.n_objects} objects", "nipy_spectral"),
        (s.mask, "ground truth", "gray"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(3 * len(panels), 3.2))
    for ax, (img, title, cmap) in zip(axes, panels):
        ax.imshow(img, cmap=cmap)
        ax.set_title(title, fontsize=9)
        ax.axis("off")
    fig.suptitle(f"Classical pipeline stages - {image_id}")
    fig.tight_layout()
    return _save(fig, name)


# --------------------------------------------------------------------------
# Figure 6: where Otsu's object counting fails
# --------------------------------------------------------------------------
def counting_error_figure(df: pd.DataFrame,
                          name: str = "fig06_count_error.png") -> Path:
    """Predicted vs. true object count, coloured by density regime.

    `df` must have columns: true_n, pred_n, density.
    """
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for regime, sub in df.groupby("density"):
        ax.scatter(sub.true_n, sub.pred_n, label=regime, alpha=0.75)
    lim = max(df.true_n.max(), df.pred_n.max()) * 1.05
    ax.plot([0, lim], [0, lim], "k--", lw=1, label="perfect")
    ax.set_xlabel("true object count")
    ax.set_ylabel("detected object count")
    ax.set_title("Connected-component counting vs. ground truth")
    ax.legend(fontsize=8)
    fig.tight_layout()
    return _save(fig, name)


def show(path: Path) -> None:
    """Display a saved figure inside a notebook."""
    from IPython.display import Image as IPImage, display

    display(IPImage(filename=str(path)))
