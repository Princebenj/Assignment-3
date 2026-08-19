"""
config.py
=========
Single place for every path, constant and hyper-parameter used in the
assignment.  Nothing else in the project hard-codes a path, so the whole
pipeline can be re-pointed at a new dataset by editing this file only.

Why a config module at all?  The marking rubric asks that "another user can
install the dependencies, understand it, and re-run it as submitted".  Keeping
the mutable state in one file is the cheapest way to make that true.
"""
from __future__ import annotations

import os
import random
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
# PROJECT_ROOT is the directory that contains `src/`.  Using a path relative to
# this file (rather than the current working directory) means the code behaves
# identically whether it is run from a notebook, a shell, or Colab.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# The dataset is cloned/unzipped next to the project by `setup.sh`.
# Override with the BIOMED_DATA_ROOT environment variable if you keep it
# elsewhere (Colab users mounting Drive will want this).
DATA_ROOT = Path(
    os.environ.get("BIOMED_DATA_ROOT", PROJECT_ROOT / "data" / "nuclei_dataset")
)

OUTPUT_ROOT = Path(os.environ.get("BIOMED_OUTPUT_ROOT", PROJECT_ROOT / "outputs"))
FIGURE_DIR = OUTPUT_ROOT / "figures"
RESULT_DIR = OUTPUT_ROOT / "results"
LLM_LOG_DIR = OUTPUT_ROOT / "llm_logs"   # every raw LLM response is archived here
MODEL_DIR = OUTPUT_ROOT / "models"

for _d in (FIGURE_DIR, RESULT_DIR, LLM_LOG_DIR, MODEL_DIR):
    _d.mkdir(parents=True, exist_ok=True)

METADATA_CSV = DATA_ROOT / "metadata.csv"

# --------------------------------------------------------------------------
# Image / preprocessing constants
# --------------------------------------------------------------------------
IMAGE_SIZE = (256, 256)          # target size required by the brief

# How to collapse RGB -> single channel.  See src/data.py for the comparison.
#   "blue"      : take the blue channel only  (DAPI-like stain lives here)
#   "luminance" : ITU-R BT.601 weighting, the textbook default
#   "max"       : per-pixel channel maximum
GRAYSCALE_METHOD = "blue"

# --------------------------------------------------------------------------
# Classical segmentation (Task 2)
# --------------------------------------------------------------------------
MORPH_DISK_RADIUS = 2      # radius of the structuring element for opening
MIN_OBJECT_AREA = 20       # px; discard specks smaller than this after Otsu
CLEAR_BORDER = False       # nuclei touching the frame edge are still real nuclei

# --------------------------------------------------------------------------
# U-Net (Task 3)
# --------------------------------------------------------------------------
UNET_BASE_CHANNELS = 16    # matches the Lab 4 network exactly
BATCH_SIZE = 4
EPOCHS = 25
LEARNING_RATE = 1e-3
VAL_THRESHOLD = 0.5        # probability -> binary mask cut-off

# --------------------------------------------------------------------------
# LLM (Tasks 1, 2, 4)
# --------------------------------------------------------------------------
def _normalise_host(value: str) -> str:
    """Ensure the Ollama host carries an http:// scheme.

    The Ollama CLI and the labs set OLLAMA_HOST as a bare "127.0.0.1:11434".
    The `ollama` Python client tolerates that, but `requests` does not - it
    raises MissingSchema - so a bare value silently broke availability checks
    while ordinary calls kept working. Normalising here fixes both paths.
    """
    value = value.strip()
    if not value.startswith(("http://", "https://")):
        value = "http://" + value
    return value.rstrip("/")


OLLAMA_HOST = _normalise_host(os.environ.get("OLLAMA_HOST", "127.0.0.1:11434"))
# Model names follow the labs: llava:7b / moondream / llama3.2-vision for
# images, llama3.2 / qwen2.5:3b / phi3:mini for text.
#
# Why llava:7b rather than llama3.2-vision: the latter downloads correctly but
# current Ollama builds refuse to load it, failing with
#     unknown model architecture: 'mllama'
# and pinning Ollama back to 0.6.8 then fails at the registry with HTTP 412.
# The revised assignment brief permits a comparable model, so llava:7b is the
# default here. Override with the VISION_MODEL environment variable if a build
# that supports mllama becomes available.
VISION_MODEL = os.environ.get("VISION_MODEL", "llava:7b")
TEXT_MODEL = os.environ.get("TEXT_MODEL", "llama3.2")
# Alternatives used by the model-comparison extension.
ALT_VISION_MODELS = ("moondream", "llama3.2-vision")
ALT_TEXT_MODELS = ("qwen2.5:3b", "phi3:mini")

# Two temperatures, used for two different jobs:
#   PRODUCTION - 0.0, greedy decoding. Lab 2 showed this makes the LLM step
#     reproducible, which is a precondition for an auditable pipeline, so every
#     record-generating call in Tasks 2 and 4 uses it.
#   DEMO - Ollama's default (~0.8). Task 1 has to *demonstrate* that repeated
#     runs differ, and at temperature 0 they mostly do not; the variability
#     experiment therefore runs at the default a user would get out of the box.
LLM_TEMPERATURE = 0.0
LLM_TEMPERATURE_DEMO = 0.8
LLM_TIMEOUT_S = 600        # a 11B vision model on a cold cache is slow
N_REPEAT_RUNS = 3          # repeats used for the determinism demonstration

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
SEED = 42


def set_seed(seed: int = SEED) -> None:
    """Seed python, numpy and (if installed) torch.

    Note this makes *our* code reproducible.  It does not make the LLM
    reproducible - that is the point of Task 1's repeated-run experiment.
    """
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    except ImportError:      # torch is optional for Tasks 1-2
        pass


@dataclass
class DensityBands:
    """Thresholds mapping an object count to a human-readable density class.

    These are derived from the dataset's own metadata (object counts run 5-85,
    median 29) rather than plucked from the air, and they are applied
    *deterministically in Python*.  The LLM is asked to echo the class, never
    to invent it - which is what makes the JSON record auditable.
    """

    sparse_max: int = 15
    normal_max: int = 35
    dense_max: int = 60
    # anything above dense_max is "very_dense"
    names: tuple = field(
        default_factory=lambda: ("sparse", "normal", "dense", "very_dense")
    )

    def classify(self, n_objects: int) -> str:
        if n_objects <= self.sparse_max:
            return self.names[0]
        if n_objects <= self.normal_max:
            return self.names[1]
        if n_objects <= self.dense_max:
            return self.names[2]
        return self.names[3]


DENSITY_BANDS = DensityBands()
