"""
data.py
=======
Loading and preprocessing for the synthetic stained-nuclei dataset.

The dataset ships as
    nuclei_dataset/{train,val,test}/{images,masks,labels}/*.png
    nuclei_dataset/test_corrupted/images/*.png
    nuclei_dataset/metadata.csv

`images` are 256x256 RGB, `masks` are binary (0/255), `labels` are 16-bit
instance maps.  Everything downstream in this project consumes the output of
`load_split()`, so the rest of the code never touches the raw folder layout.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Literal, Sequence

import numpy as np
import pandas as pd
from PIL import Image

from . import config

GrayMethod = Literal["blue", "luminance", "max"]


# --------------------------------------------------------------------------
# Core record type
# --------------------------------------------------------------------------
@dataclass
class Sample:
    """One image and (where available) its ground truth.

    Attributes
    ----------
    image_id : str          e.g. "train_000"
    split    : str          train / val / test / test_corrupted
    gray     : np.ndarray   float32, shape (256, 256), range [0, 1]
    rgb      : np.ndarray   uint8,   shape (256, 256, 3) - kept for the VLM step
    mask     : np.ndarray   bool,    shape (256, 256) or None
    labels   : np.ndarray   uint16 instance map or None
    """

    image_id: str
    split: str
    gray: np.ndarray
    rgb: np.ndarray
    mask: np.ndarray | None = None
    labels: np.ndarray | None = None

    @property
    def path_png(self) -> Path:
        return config.DATA_ROOT / self.split / "images" / f"{self.image_id}.png"


# --------------------------------------------------------------------------
# Preprocessing primitives
# --------------------------------------------------------------------------
def to_grayscale(rgb: np.ndarray, method: GrayMethod = None) -> np.ndarray:
    """Collapse an RGB image to a single channel in [0, 1].

    Three options are offered because the choice is *not* neutral for this
    modality.  The nuclei are DAPI-like, i.e. essentially pure blue on a dark
    field, and the standard luminance weighting gives blue a coefficient of
    only 0.114 - it throws away ~89% of the signal amplitude before we have
    even started.  `blue` keeps the stain channel intact.  The EDA notebook
    quantifies the difference; see also report Section 2.

    Parameters
    ----------
    rgb : (H, W, 3) uint8 array, or (H, W) if already single channel.
    method : one of "blue", "luminance", "max".  Defaults to
        config.GRAYSCALE_METHOD.
    """
    method = method or config.GRAYSCALE_METHOD
    arr = np.asarray(rgb)

    if arr.ndim == 2:                       # already grayscale
        gray = arr.astype(np.float32)
    elif method == "blue":
        gray = arr[..., 2].astype(np.float32)
    elif method == "luminance":
        gray = (
            0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        ).astype(np.float32)
    elif method == "max":
        gray = arr.max(axis=-1).astype(np.float32)
    else:
        raise ValueError(f"unknown grayscale method: {method!r}")

    # Normalise to [0, 1].  We divide by 255 rather than by the per-image max
    # so that absolute brightness remains comparable across images - the
    # low-contrast corruption in the robustness experiment must stay visible.
    return gray / 255.0


def resize(arr: np.ndarray, size: tuple[int, int] = None, *, is_mask: bool = False):
    """Resize to `size` (default 256x256).

    For this dataset every image is already 256x256, so this is usually a
    no-op - but it is applied unconditionally so that the pipeline still holds
    if a differently-sized image is dropped in.  Masks use nearest-neighbour
    interpolation to avoid inventing intermediate label values.
    """
    size = size or config.IMAGE_SIZE
    if arr.shape[:2] == tuple(size):
        return arr

    resample = Image.NEAREST if is_mask else Image.BILINEAR
    if arr.dtype == np.float32 or arr.dtype == np.float64:
        im = Image.fromarray((arr * 255).astype(np.uint8))
        out = np.asarray(im.resize(size[::-1], resample)).astype(np.float32) / 255.0
    else:
        im = Image.fromarray(arr)
        out = np.asarray(im.resize(size[::-1], resample))
    return out


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_sample(image_id: str, split: str, *, gray_method: GrayMethod = None) -> Sample:
    """Load a single image (+ mask/labels if the split has them)."""
    img_path = config.DATA_ROOT / split / "images" / f"{image_id}.png"
    if not img_path.exists():
        raise FileNotFoundError(img_path)

    rgb = np.asarray(Image.open(img_path).convert("RGB"))
    rgb = resize(rgb)
    gray = to_grayscale(rgb, gray_method)

    mask = labels = None
    mask_path = config.DATA_ROOT / split / "masks" / f"{image_id}.png"
    if mask_path.exists():
        mask = resize(np.asarray(Image.open(mask_path)), is_mask=True) > 127

    label_path = config.DATA_ROOT / split / "labels" / f"{image_id}.png"
    if label_path.exists():
        labels = resize(np.asarray(Image.open(label_path)), is_mask=True)

    return Sample(image_id=image_id, split=split, gray=gray, rgb=rgb,
                  mask=mask, labels=labels)


def list_ids(split: str) -> list[str]:
    """Sorted image ids present in a split."""
    d = config.DATA_ROOT / split / "images"
    return sorted(p.stem for p in d.glob("*.png"))


def load_split(split: str, *, limit: int | None = None,
               gray_method: GrayMethod = None) -> list[Sample]:
    """Load an entire split into memory.

    The dataset is tiny (112 images at 256x256 ~ 7 MB as float32), so eager
    loading keeps the rest of the code simple.  `limit` is handy for smoke
    tests in Colab.
    """
    ids = list_ids(split)
    if limit is not None:
        ids = ids[:limit]
    return [load_sample(i, split, gray_method=gray_method) for i in ids]


def iter_split(split: str) -> Iterator[Sample]:
    """Lazy variant of `load_split` for memory-constrained runtimes."""
    for image_id in list_ids(split):
        yield load_sample(image_id, split)


# --------------------------------------------------------------------------
# Ground-truth metadata
# --------------------------------------------------------------------------
def load_metadata() -> pd.DataFrame:
    """Per-image ground truth shipped with the dataset.

    Columns: image_id, split, density, n_objects, mean_intensity,
             area_fraction, seed

    This is the answer key.  We never feed it to a model; we use it only to
    *score* what the pipeline produced, which is what lets the report make
    evidence-based claims about LLM accuracy rather than vibes.
    """
    df = pd.read_csv(config.METADATA_CSV)
    return df.set_index("image_id", drop=False)


def ground_truth_for(image_id: str) -> dict:
    """Convenience lookup returning a plain dict (or {} for corrupted images)."""
    md = load_metadata()
    base = image_id.split("_blur")[0].split("_lowcontrast")[0]
    if base in md.index:
        return md.loc[base].to_dict()
    return {}


# --------------------------------------------------------------------------
# Corrupted variants (robustness extension)
# --------------------------------------------------------------------------
def list_corrupted() -> list[str]:
    """Ids in test_corrupted, e.g. ['test_000_blur', 'test_000_lowcontrast']."""
    return list_ids("test_corrupted")


def load_corrupted(image_id: str) -> Sample:
    return load_sample(image_id, "test_corrupted")


def summarise_dataset() -> pd.DataFrame:
    """Small table used in the EDA section of the report."""
    md = load_metadata()
    rows = []
    for split in ("train", "val", "test"):
        sub = md[md.split == split]
        rows.append(
            dict(
                split=split,
                n_images=len(sub),
                n_objects_mean=round(sub.n_objects.mean(), 1),
                n_objects_min=int(sub.n_objects.min()),
                n_objects_max=int(sub.n_objects.max()),
                area_fraction_mean=round(sub.area_fraction.mean(), 4),
                densities=", ".join(
                    f"{k}:{v}" for k, v in sub.density.value_counts().sort_index().items()
                ),
            )
        )
    return pd.DataFrame(rows)
