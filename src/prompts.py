"""
prompts.py
==========
Every prompt used anywhere in the pipeline lives here, versioned and named, so
that the exact strings quoted in the report are provably the strings the code
sent.  (Prompts embedded inline in notebook cells drift away from the report
within about two edits.)

Design rationale - the four levers used in the optimised prompts
----------------------------------------------------------------
1. ROLE ANCHORING.  The model is told it is a *descriptive* component inside a
   larger measurement pipeline, explicitly not a diagnostic tool.  Vision-
   language models default to a clinical-report register when shown anything
   that looks medical, because that is what dominates their fine-tuning data;
   without an anchor, llama3.2-vision volunteers pathology it cannot possibly
   see in a synthetic image.
2. SCHEMA FORCING.  The response format is stated as a literal JSON skeleton
   with enumerated allowed values, and Ollama's `format` parameter is used to
   constrain decoding as well.  Belt and braces: the API constraint guarantees
   parseable JSON, the in-prompt schema guarantees the *right* JSON.
3. LICENSED UNCERTAINTY.  "uncertain" is named as a first-class allowed value
   for every field.  If a model is given no way to say "I don't know", the
   likeliest completion is a confident guess - refusal has to be made cheaper
   than fabrication.
4. SCOPE FENCING.  An explicit list of things not to output (patient details,
   diagnosis, staging, treatment, invented measurements) plus an instruction to
   emit no prose outside the JSON object.
"""
from __future__ import annotations

# ==========================================================================
# TASK 1 - direct visual description by a multimodal model
# ==========================================================================

# --- Baseline for comparison. Deliberately bad: open-ended, invites diagnosis,
# --- no structure, no escape hatch. This is the control condition.
NAIVE_VISION_PROMPT = """What is this medical image? Describe what you see and tell me what condition the patient has."""


VISION_SYSTEM = """You are an image-description module inside an automated research pipeline for biomedical microscopy. You describe visual appearance only. You are not a diagnostic system, you have no patient context, and nothing you output will be used for clinical decisions.

Rules you must follow:
- Describe only what is directly visible in the pixels.
- Never state or imply a diagnosis, disease, prognosis, or clinical recommendation.
- Never invent counts, sizes, or measurements. Quantities are measured elsewhere in the pipeline by dedicated software, not by you.
- If a field cannot be determined from the image, output the exact string "uncertain". Saying "uncertain" is always preferred to guessing and is never penalised.
- Output one JSON object and nothing else: no preamble, no explanation, no markdown code fences."""


OPTIMISED_VISION_PROMPT = """Describe this grayscale biomedical microscopy image.

Return exactly this JSON object:

{
  "modality": "<what imaging technique the appearance is consistent with, or \\"uncertain\\">",
  "tissue_type": "<what biological material is visible, or \\"uncertain\\">",
  "notable_features": ["<up to 5 short visual observations>"],
  "image_quality": "<one of: good | acceptable | poor | uncertain>"
}

Field guidance:
- "modality": judge from visual appearance only (e.g. bright objects on a dark field is consistent with fluorescence microscopy). If several are plausible, choose "uncertain".
- "tissue_type": describe the visible structures (e.g. "cell nuclei"), not an organ or a patient. If not determinable, "uncertain".
- "notable_features": qualitative appearance only - shape, distribution, clumping, brightness, artefacts. Do NOT include numbers or counts.
- "image_quality": judge focus, contrast and noise only.

Output the JSON object only."""


# A middle rung used in the prompt-ablation table: structured output is
# requested, but "uncertain" is never offered as an option.  Isolating this one
# variable shows how much of the improvement comes from licensing uncertainty
# rather than from asking for JSON.
STRUCTURED_NO_UNCERTAINTY_PROMPT = """Describe this grayscale biomedical microscopy image.

Return exactly this JSON object:

{
  "modality": "<imaging technique>",
  "tissue_type": "<biological material visible>",
  "notable_features": ["<up to 5 short visual observations>"],
  "image_quality": "<one of: good | acceptable | poor>"
}

Output the JSON object only."""


# JSON schema passed to Ollama's structured-output `format` parameter and used
# by our own validator.  Keys match the four required by the brief.
VISION_SCHEMA = {
    "type": "object",
    "properties": {
        "modality": {"type": "string"},
        "tissue_type": {"type": "string"},
        "notable_features": {"type": "array", "items": {"type": "string"}},
        "image_quality": {
            "type": "string",
            "enum": ["good", "acceptable", "poor", "uncertain"],
        },
    },
    "required": ["modality", "tissue_type", "notable_features", "image_quality"],
}


# ==========================================================================
# TASK 2 - numbers-first interpretation (the model never sees the image)
# ==========================================================================
NUMBERS_SYSTEM = """You are a reporting module inside an automated microscopy analysis pipeline.

You have NOT seen the image. You are given only measurements that were computed by validated image-processing software (scikit-image). Your job is to render those measurements into readable language and a structured record.

Rules you must follow:
- Use only the numbers supplied to you. Never introduce a value that is not in the input, and never round away meaning.
- Never describe visual appearance you were not told about (colour, staining, artefacts, anatomy). You cannot see it.
- Never state or imply a diagnosis or clinical interpretation.
- If the measurements do not support a field, output "uncertain".
- Output must be one JSON object and nothing else."""


OPTIMISED_NUMBERS_PROMPT = """Below are quantitative measurements extracted from one microscopy image by automated segmentation.

--- MEASUREMENTS ---
{measurements}
--- END MEASUREMENTS ---

Classification rules you must apply exactly as written:
- density_class: "sparse" if n_objects <= 15; "normal" if 16-35; "dense" if 36-60; "very_dense" if > 60.
- shape_regularity: "regular" if mean solidity >= 0.95 and mean circularity >= 0.80; "mostly_regular" if mean solidity >= 0.88; otherwise "irregular".
- quality_flag: "good" if contrast ratio >= 0.45; "acceptable" if 0.25-0.45; "poor_low_contrast" if < 0.25; "unusable" if no objects were detected.

Return exactly this JSON object:

{{
  "n_objects": <integer, copied from the measurements>,
  "density_class": "<sparse | normal | dense | very_dense>",
  "shape_regularity": "<regular | mostly_regular | irregular | uncertain>",
  "quality_flag": "<good | acceptable | poor_low_contrast | unusable | uncertain>",
  "description": "<one paragraph, maximum 80 words, describing what these measurements indicate. Reference specific values. Do not speculate about anything not measured.>"
}}

Output the JSON object only."""


NAIVE_NUMBERS_PROMPT = """Here are some measurements from a microscopy image:

{measurements}

What can you tell me about this sample?"""


NUMBERS_SCHEMA = {
    "type": "object",
    "properties": {
        "n_objects": {"type": "integer"},
        "density_class": {
            "type": "string",
            "enum": ["sparse", "normal", "dense", "very_dense", "uncertain"],
        },
        "shape_regularity": {
            "type": "string",
            "enum": ["regular", "mostly_regular", "irregular", "uncertain"],
        },
        "quality_flag": {
            "type": "string",
            "enum": ["good", "acceptable", "poor_low_contrast",
                     "questionable_blurred_or_merged", "unusable", "uncertain"],
        },
        "description": {"type": "string"},
    },
    "required": ["n_objects", "density_class", "shape_regularity",
                 "quality_flag", "description"],
}


# ==========================================================================
# TASK 4 - hybrid pipeline record + narrative
# ==========================================================================
HYBRID_SYSTEM = NUMBERS_SYSTEM


OPTIMISED_HYBRID_PROMPT = """An automated pipeline segmented one microscopy image with a trained U-Net and measured every detected object. The verified results are below.

--- VERIFIED PIPELINE OUTPUT ---
{measurements}
--- END ---

These values are the source of truth. They have already been computed and validated; your task is presentation, not calculation.

Return exactly this JSON object:

{{
  "image_id": "{image_id}",
  "n_objects": {n_objects},
  "mean_area": {mean_area},
  "density_class": "{density_class}",
  "quality_flag": "{quality_flag}",
  "narrative": "<one paragraph, maximum 70 words, in plain professional English, summarising the field for a lab notebook. State the object count, the typical object size, the density regime and the image quality. Add no numbers beyond those given. Add no clinical interpretation.>"
}}

The first five fields must be copied verbatim from the values above. Only "narrative" is yours to write. Output the JSON object only."""


HYBRID_SCHEMA = {
    "type": "object",
    "properties": {
        "image_id": {"type": "string"},
        "n_objects": {"type": "integer"},
        "mean_area": {"type": "number"},
        "density_class": {"type": "string"},
        "quality_flag": {"type": "string"},
        "narrative": {"type": "string"},
    },
    "required": ["image_id", "n_objects", "mean_area", "density_class",
                 "quality_flag", "narrative"],
}


# Registry so notebooks and the report can enumerate prompts programmatically.
PROMPT_REGISTRY = {
    "vision_naive": NAIVE_VISION_PROMPT,
    "vision_structured_no_uncertainty": STRUCTURED_NO_UNCERTAINTY_PROMPT,
    "vision_optimised": OPTIMISED_VISION_PROMPT,
    "numbers_naive": NAIVE_NUMBERS_PROMPT,
    "numbers_optimised": OPTIMISED_NUMBERS_PROMPT,
    "hybrid_optimised": OPTIMISED_HYBRID_PROMPT,
}
