"""
evaluate.py
===========
Scoring and figures for Task 3, plus the head-to-head against Otsu that
question 2 of the report asks about.

The central design decision here: we evaluate at **two levels**.

  * Pixel level  - Dice and IoU. What the brief asks for, and what almost all
    segmentation papers report.
  * Object level - counting error and count-derived density class. What the
    downstream JSON record actually depends on.

Reporting only the first would hide the most interesting result in this
assignment, which is that the two levels disagree sharply on this dataset.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from . import classical, config, data
from .train import get_device, predict_mask


def _save(fig, name: str) -> Path:
    path = config.FIGURE_DIR / name
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------
# Per-image scoring
# --------------------------------------------------------------------------
def dice_np(pred: np.ndarray, target: np.ndarray) -> float:
    """Dice on numpy boolean arrays (single image)."""
    pred, target = pred.astype(bool), target.astype(bool)
    denom = pred.sum() + target.sum()
    if denom == 0:
        return 1.0
    return float(2 * (pred & target).sum() / denom)


def iou_np(pred: np.ndarray, target: np.ndarray) -> float:
    pred, target = pred.astype(bool), target.astype(bool)
    union = (pred | target).sum()
    if union == 0:
        return 1.0
    return float((pred & target).sum() / union)


def per_image_scores(model, split: str = "val", device=None,
                     threshold: float = None) -> pd.DataFrame:
    """Score every image in a split with both U-Net and Otsu.

    Columns: image_id, density, true_n, unet_dice, unet_iou, otsu_dice,
             otsu_iou, unet_n, otsu_n, and the two signed count errors.
    """
    device = device or get_device()
    md = data.load_metadata()
    rows = []

    for image_id in data.list_ids(split):
        s = data.load_sample(image_id, split)
        gt = md.loc[image_id]

        unet_mask, prob = predict_mask(model, s.gray, device=device,
                                       threshold=threshold)
        unet_res = classical.mask_to_result(unet_mask, source="unet")
        otsu_res = classical.otsu_segment(s.gray)

        rows.append(dict(
            image_id=image_id,
            density=gt.density,
            true_n=int(gt.n_objects),
            unet_dice=round(dice_np(unet_res.mask, s.mask), 4),
            unet_iou=round(iou_np(unet_res.mask, s.mask), 4),
            otsu_dice=round(dice_np(otsu_res.mask, s.mask), 4),
            otsu_iou=round(iou_np(otsu_res.mask, s.mask), 4),
            unet_n=unet_res.n_objects,
            otsu_n=otsu_res.n_objects,
            unet_count_err=unet_res.n_objects - int(gt.n_objects),
            otsu_count_err=otsu_res.n_objects - int(gt.n_objects),
            mean_prob_fg=round(float(prob[s.mask].mean()), 4) if s.mask.any() else np.nan,
            mean_prob_bg=round(float(prob[~s.mask].mean()), 4),
        ))

    df = pd.DataFrame(rows)
    df.to_csv(config.RESULT_DIR / f"per_image_scores_{split}.csv", index=False)
    return df


def summarise_scores(df: pd.DataFrame) -> pd.DataFrame:
    """Headline table: metrics overall and stratified by density regime."""
    def block(sub: pd.DataFrame, label: str) -> dict:
        return dict(
            group=label,
            n_images=len(sub),
            unet_dice=round(sub.unet_dice.mean(), 4),
            unet_iou=round(sub.unet_iou.mean(), 4),
            otsu_dice=round(sub.otsu_dice.mean(), 4),
            otsu_iou=round(sub.otsu_iou.mean(), 4),
            unet_count_mae=round(sub.unet_count_err.abs().mean(), 2),
            otsu_count_mae=round(sub.otsu_count_err.abs().mean(), 2),
        )

    rows = [block(df, "ALL")]
    for regime in ["sparse", "normal", "dense", "clustered"]:
        sub = df[df.density == regime]
        if len(sub):
            rows.append(block(sub, regime))
    out = pd.DataFrame(rows)
    out.to_csv(config.RESULT_DIR / "metric_summary.csv", index=False)
    return out


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def prediction_panels(model, image_ids: list[str], split: str = "val",
                      device=None, name: str = "fig07_unet_panels.png") -> Path:
    """Input / ground truth / U-Net probability / U-Net mask / Otsu mask.

    Including the raw probability map, not just the thresholded mask, is worth
    the column: it shows *where the model is unsure*, which is how we identify
    that failures concentrate on nucleus boundaries and touching pairs.
    """
    device = device or get_device()
    md = data.load_metadata()
    n = len(image_ids)
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.2 * n))
    if n == 1:
        axes = axes[np.newaxis, :]

    for r, image_id in enumerate(image_ids):
        s = data.load_sample(image_id, split)
        mask, prob = predict_mask(model, s.gray, device=device)
        otsu = classical.otsu_segment(s.gray)
        d_u = dice_np(mask, s.mask)
        d_o = dice_np(otsu.mask, s.mask)

        panels = [
            (s.gray, f"{image_id} ({md.loc[image_id,'density']}, n={md.loc[image_id,'n_objects']})", "gray", None),
            (s.mask, "ground truth", "gray", None),
            (prob, "U-Net probability", "viridis", (0, 1)),
            (mask, f"U-Net mask, Dice {d_u:.3f}", "gray", None),
            (otsu.mask, f"Otsu mask, Dice {d_o:.3f}", "gray", None),
        ]
        for c, (img, title, cmap, lim) in enumerate(panels):
            ax = axes[r, c]
            if lim:
                ax.imshow(img, cmap=cmap, vmin=lim[0], vmax=lim[1])
            else:
                ax.imshow(img, cmap=cmap)
            ax.set_title(title, fontsize=9)
            ax.axis("off")

    fig.suptitle("U-Net predictions vs. ground truth and the Otsu baseline", fontsize=12)
    fig.tight_layout()
    return _save(fig, name)


def training_curves(histories: dict[str, pd.DataFrame],
                    name: str = "fig08_training_curves.png") -> Path:
    """Loss and validation Dice per epoch, one line per loss function."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.2))

    for tag, h in histories.items():
        axes[0].plot(h.epoch, h.train_loss, label=f"{tag} train")
        axes[0].plot(h.epoch, h.val_loss, ls="--", label=f"{tag} val")
        axes[1].plot(h.epoch, h.val_dice, label=tag)
        axes[2].plot(h.epoch, h.val_iou, label=tag)

    axes[0].set_title("Loss")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss value")
    axes[0].legend(fontsize=7)
    # Loss values are not comparable across different loss functions - only the
    # shape of each curve is. Said in the caption too.
    axes[1].set_title("Validation Dice")
    axes[1].set_xlabel("epoch")
    axes[1].legend(fontsize=8)
    axes[2].set_title("Validation IoU")
    axes[2].set_xlabel("epoch")
    axes[2].legend(fontsize=8)

    fig.suptitle("Training dynamics (loss magnitudes are not comparable across losses)")
    fig.tight_layout()
    return _save(fig, name)


def unet_vs_otsu_figure(df: pd.DataFrame,
                        name: str = "fig09_unet_vs_otsu.png") -> Path:
    """Paired comparison of the two segmenters at pixel and object level."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))

    axes[0].scatter(df.otsu_dice, df.unet_dice, c="tab:blue")
    lo = min(df.otsu_dice.min(), df.unet_dice.min()) - 0.01
    axes[0].plot([lo, 1], [lo, 1], "k--", lw=1)
    axes[0].set_xlabel("Otsu Dice")
    axes[0].set_ylabel("U-Net Dice")
    axes[0].set_title("Pixel overlap, per image\n(above the line = U-Net better)")

    idx = np.arange(len(df))
    w = 0.4
    axes[1].bar(idx - w / 2, df.otsu_count_err.abs(), w, label="Otsu")
    axes[1].bar(idx + w / 2, df.unet_count_err.abs(), w, label="U-Net")
    axes[1].set_xticks(idx)
    axes[1].set_xticklabels(df.image_id, rotation=90, fontsize=6)
    axes[1].set_ylabel("|count error|")
    axes[1].set_title("Object-count error per image")
    axes[1].legend()

    grouped = df.groupby("density")[["otsu_dice", "unet_dice"]].mean()
    grouped.plot(kind="bar", ax=axes[2], rot=0)
    axes[2].set_ylim(min(0.8, grouped.values.min() - 0.02), 1.0)
    axes[2].set_ylabel("mean Dice")
    axes[2].set_title("Dice by density regime")

    fig.tight_layout()
    return _save(fig, name)


def error_map_figure(model, image_id: str, split: str = "val", device=None,
                     name: str = "fig10_error_map.png") -> Path:
    """False positives and false negatives in colour, to localise the errors.

    Answers "where does the model make its mistakes" spatially rather than in
    the abstract: if the red/blue pixels form thin rings around every object,
    the error is boundary localisation; if they fill whole objects, it is
    detection.
    """
    device = device or get_device()
    s = data.load_sample(image_id, split)
    mask, prob = predict_mask(model, s.gray, device=device)

    overlay = np.zeros((*s.gray.shape, 3))
    overlay[..., 0] = mask & ~s.mask          # red   = false positive
    overlay[..., 2] = ~mask & s.mask          # blue  = false negative
    overlay[..., 1] = mask & s.mask           # green = true positive

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    axes[0].imshow(s.gray, cmap="gray")
    axes[0].set_title(f"{image_id} input")
    axes[1].imshow(prob, cmap="viridis", vmin=0, vmax=1)
    axes[1].set_title("predicted probability")
    axes[2].imshow(overlay)
    axes[2].set_title("green TP / red FP / blue FN")
    for a in axes:
        a.axis("off")
    fig.tight_layout()
    return _save(fig, name)


def pick_contrast_examples(df: pd.DataFrame) -> dict:
    """Find the images where each method most outperforms the other.

    Used to answer question 2 with concrete examples instead of averages.
    """
    d = df.copy()
    d["dice_gap"] = d.unet_dice - d.otsu_dice
    d["count_gap"] = d.otsu_count_err.abs() - d.unet_count_err.abs()
    return dict(
        unet_best_pixel=d.loc[d.dice_gap.idxmax()].to_dict(),
        otsu_best_pixel=d.loc[d.dice_gap.idxmin()].to_dict(),
        unet_best_count=d.loc[d.count_gap.idxmax()].to_dict(),
        otsu_best_count=d.loc[d.count_gap.idxmin()].to_dict(),
    )
