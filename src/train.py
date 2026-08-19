"""
train.py
========
Task 3: training the U-Net, and the loss ablation used for the extension.

Everything here is deterministic given `config.SEED` (barring cuDNN
non-determinism), and every run writes:
    outputs/models/unet_<loss>.pt          weights
    outputs/results/history_<loss>.csv     per-epoch metrics
so that the curves in the report can be regenerated without retraining.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset

from . import config, data
from .unet import LOSS_REGISTRY, UNet, dice_coefficient, iou_score


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------
class NucleiDataset(Dataset):
    """Wraps a split as (image, mask) float tensors of shape (1, H, W).

    Augmentation is limited to 90-degree rotations and flips - the dihedral
    group of the square.  These are the only transforms that are certainly
    label-preserving here: nuclei have no canonical orientation, and unlike
    (say) elastic deformation or brightness jitter, they cannot create
    physically impossible images or shift the intensity statistics that the
    downstream quality flag depends on.
    """

    def __init__(self, split: str, augment: bool = False,
                 gray_method: str = None, samples: list = None):
        self.samples = samples if samples is not None else data.load_split(
            split, gray_method=gray_method
        )
        self.augment = augment
        self.split = split

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        img = s.gray.astype(np.float32)
        msk = s.mask.astype(np.float32)

        if self.augment:
            k = np.random.randint(4)                 # 0/90/180/270 rotation
            img, msk = np.rot90(img, k), np.rot90(msk, k)
            if np.random.rand() < 0.5:               # horizontal flip
                img, msk = np.fliplr(img), np.fliplr(msk)
            if np.random.rand() < 0.5:               # vertical flip
                img, msk = np.flipud(img), np.flipud(msk)
            img, msk = np.ascontiguousarray(img), np.ascontiguousarray(msk)

        return (
            torch.from_numpy(img).unsqueeze(0),
            torch.from_numpy(msk).unsqueeze(0),
            s.image_id,
        )


def make_loaders(batch_size: int = None, augment: bool = True,
                 num_workers: int = 0) -> tuple[DataLoader, DataLoader]:
    """Train/val DataLoaders.

    The dataset ships with a fixed train/val/test split, which we honour
    exactly - resplitting would make our numbers incomparable with everyone
    else's on the same assignment, and the test split is never touched until
    Task 4.
    """
    batch_size = batch_size or config.BATCH_SIZE
    train_ds = NucleiDataset("train", augment=augment)
    val_ds = NucleiDataset("val", augment=False)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                   num_workers=num_workers, drop_last=False),
        DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                   num_workers=num_workers),
    )


# --------------------------------------------------------------------------
# Train / evaluate
# --------------------------------------------------------------------------
def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader,
             loss_fn, device: torch.device,
             threshold: float = None) -> dict:
    """Mean loss, Dice and IoU over a loader."""
    threshold = config.VAL_THRESHOLD if threshold is None else threshold
    model.eval()
    losses, dices, ious, n = 0.0, 0.0, 0.0, 0
    for x, y, _ in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x)
        b = x.shape[0]
        losses += float(loss_fn(logits, y)) * b
        dices += float(dice_coefficient(logits, y, threshold)) * b
        ious += float(iou_score(logits, y, threshold)) * b
        n += b
    return dict(loss=losses / n, dice=dices / n, iou=ious / n)


def train_model(
    loss_name: str = "bce_dice",
    *,
    epochs: int = None,
    lr: float = None,
    base: int = None,
    augment: bool = True,
    batch_size: int = None,
    device: torch.device = None,
    verbose: bool = True,
    tag: str = None,
) -> tuple[torch.nn.Module, pd.DataFrame]:
    """Train one U-Net and return (best model, per-epoch history).

    Model selection is by best **validation Dice**, not by final epoch: with 80
    training images the validation curve is noisy, and taking the last epoch
    would report a number that depends on where we happened to stop.
    """
    config.set_seed()
    epochs = epochs or config.EPOCHS
    lr = lr or config.LEARNING_RATE
    base = base or config.UNET_BASE_CHANNELS
    device = device or get_device()
    tag = tag or loss_name

    train_loader, val_loader = make_loaders(batch_size=batch_size, augment=augment)
    model = UNet(base=base).to(device)
    loss_fn = LOSS_REGISTRY[loss_name]
    optimiser = torch.optim.Adam(model.parameters(), lr=lr)
    # Cosine decay: no schedule tuning to justify, and it removes the
    # "did you just stop at a lucky epoch" question from the results.
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=epochs)

    if verbose:
        print(f"[{tag}] device={device} params={model.count_parameters():,} "
              f"train={len(train_loader.dataset)} val={len(val_loader.dataset)}")

    history, best_dice, best_state = [], -1.0, None
    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, n = time.time(), 0.0, 0
        for x, y, _ in train_loader:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            logits = model(x)
            loss = loss_fn(logits, y)
            loss.backward()
            optimiser.step()
            running += float(loss) * x.shape[0]
            n += x.shape[0]
        scheduler.step()

        val = evaluate(model, val_loader, loss_fn, device)
        row = dict(epoch=epoch, train_loss=running / n, val_loss=val["loss"],
                   val_dice=val["dice"], val_iou=val["iou"],
                   lr=optimiser.param_groups[0]["lr"],
                   seconds=round(time.time() - t0, 1))
        history.append(row)

        if val["dice"] > best_dice:
            best_dice = val["dice"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        if verbose:
            print(f"  epoch {epoch:3d}/{epochs}  train {row['train_loss']:.4f}  "
                  f"val {row['val_loss']:.4f}  dice {row['val_dice']:.4f}  "
                  f"iou {row['val_iou']:.4f}  ({row['seconds']}s)")

    if best_state is not None:
        model.load_state_dict(best_state)

    hist = pd.DataFrame(history)
    hist.to_csv(config.RESULT_DIR / f"history_{tag}.csv", index=False)
    torch.save(
        {"state_dict": model.state_dict(), "base": base,
         "loss": loss_name, "best_val_dice": best_dice},
        config.MODEL_DIR / f"unet_{tag}.pt",
    )
    if verbose:
        print(f"[{tag}] best val Dice = {best_dice:.4f}")
    return model, hist


def load_model(tag: str = "bce_dice", device: torch.device = None) -> torch.nn.Module:
    """Reload a trained checkpoint (so the report can be rebuilt without retraining)."""
    device = device or get_device()
    ckpt = torch.load(config.MODEL_DIR / f"unet_{tag}.pt", map_location=device,
                      weights_only=False)
    model = UNet(base=ckpt["base"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


@torch.no_grad()
def predict_mask(model: torch.nn.Module, gray: np.ndarray,
                 device: torch.device = None,
                 threshold: float = None) -> tuple[np.ndarray, np.ndarray]:
    """Run the U-Net on a single grayscale image.

    Returns (binary_mask, probability_map).
    """
    device = device or get_device()
    threshold = config.VAL_THRESHOLD if threshold is None else threshold
    model.eval()
    x = torch.from_numpy(gray.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(device)
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()
    return prob > threshold, prob


# --------------------------------------------------------------------------
# Loss ablation (extension)
# --------------------------------------------------------------------------
def loss_ablation(losses: tuple[str, ...] = ("bce", "dice", "bce_dice"),
                  **kwargs) -> pd.DataFrame:
    """Train one model per loss under identical conditions and compare.

    Identical seed, identical augmentation, identical schedule - the loss is
    the only thing that varies, which is what makes this an ablation rather
    than three unrelated runs.
    """
    rows = []
    for loss_name in losses:
        model, hist = train_model(loss_name=loss_name, tag=loss_name, **kwargs)
        best = hist.loc[hist.val_dice.idxmax()]
        rows.append(dict(
            loss=loss_name,
            best_epoch=int(best.epoch),
            best_val_dice=round(float(best.val_dice), 4),
            val_iou_at_best=round(float(best.val_iou), 4),
            final_train_loss=round(float(hist.train_loss.iloc[-1]), 4),
            total_minutes=round(hist.seconds.sum() / 60, 1),
        ))
    df = pd.DataFrame(rows)
    df.to_csv(config.RESULT_DIR / "loss_ablation.csv", index=False)
    return df
