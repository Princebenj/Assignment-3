"""Build notebooks/assignment.ipynb from a declarative cell list.

Keeping the notebook in a build script (rather than hand-editing JSON) means
the notebook can be regenerated after any change to src/, and the two never
drift apart.  Run:  python build_notebook.py
"""
import json
from pathlib import Path

C = []          # (kind, source)


def md(s):
    C.append(("markdown", s.strip("\n")))


def code(s):
    C.append(("code", s.strip("\n")))


# ===========================================================================
md("""
# Assignment 3 — Hybrid Biomedical Image-Analysis Pipeline
### Modality: fluorescence microscopy of stained cell nuclei

**Pipeline:** raw image → segmentation → quantitative region features →
structured JSON record → narrative.

This notebook is a thin driver. All logic lives in `src/`, which keeps each
task readable and means every function is unit-testable and reusable:

| module | responsibility |
|---|---|
| `config.py` | paths, constants, seeds, density bands |
| `data.py` | loading, grayscale conversion, resizing, ground-truth metadata |
| `eda.py` | Task 1 exploratory figures |
| `prompts.py` | every prompt, versioned in one place |
| `llm.py` | Ollama client, JSON recovery, schema validation, logging |
| `classical.py` | Task 2: Otsu, morphology, `regionprops_table`, summaries |
| `unet.py` | Task 3: the Lab 4 U-Net, losses, Dice/IoU |
| `train.py` | training loop, loss ablation |
| `evaluate.py` | metrics, panels, curves, U-Net vs Otsu |
| `pipeline.py` | Task 4: end-to-end run, quality gate, auditing |

> **Educational use only.** None of these models are validated for clinical
> use. Hallucinations in a medical context can cause harm.
""")

md("## 0. Setup\n\nRun these cells once on a fresh Colab runtime with a **T4 GPU** "
   "(*Runtime → Change runtime type → T4 GPU*).")

code("""
# +-- Setup 1 of 4: check the GPU --
# The U-Net trains in ~2 min on a T4 and ~30 min on CPU; llama3.2-vision is
# effectively unusable without a GPU.
import subprocess
r = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                    '--format=csv,noheader'], capture_output=True, text=True)
print('GPU:', r.stdout.strip() if r.returncode == 0 else 'NONE - switch runtime to T4')
""")

code("""
# +-- Setup 2 of 4: get the project code and the dataset --
# WHAT: make sure src/ is importable and download the nuclei dataset.
# WHY:  every later cell imports from src and reads data/nuclei_dataset.
# HOW:  subprocess (not ! magics) so the same cell also runs in plain Jupyter.
import os, subprocess, sys

subprocess.run([sys.executable, '-m', 'pip', '-q', 'install',
                'ollama', 'scikit-image'], check=False)

if not os.path.isdir('src'):
    raise SystemExit('src/ not found. Upload the project folder next to '
                     'this notebook, or clone your repo first.')

if not os.path.isdir('data/nuclei_dataset'):
    subprocess.run(['git', 'clone', '-q', '--depth', '1',
                    'https://github.com/Nickolay-K/Assingnment-3-dataset.git',
                    '_ds'], check=True)
    os.makedirs('data', exist_ok=True)
    subprocess.run(['unzip', '-q', '_ds/nuclei_dataset.zip', '-d', 'data'],
                   check=True)
    subprocess.run(['rm', '-rf', '_ds'], check=False)

print('dataset present:', os.path.isdir('data/nuclei_dataset'))
""")

code("""
# +-- Setup 3 of 4: install and start Ollama --
# Colab has no systemd, so the server must be launched manually and polled
# until its HTTP API answers (same pattern as Lab 5).
#
# CRITICAL: install pciutils and lshw FIRST. Ollama detects the GPU by shelling
# out to lspci/lshw, and Colab images do not ship them. Without them the
# installer prints "Unable to detect NVIDIA/AMD GPU" and every model then runs
# on CPU - which llama3.2-vision (11B) cannot survive in Colab's RAM.
import subprocess, time, requests, os

!apt-get -qq update && apt-get -qq install -y pciutils lshw
!curl -fsSL https://ollama.com/install.sh | sh
os.environ['OLLAMA_HOST'] = '127.0.0.1:11434'

def ollama_up():
    try:
        requests.get('http://127.0.0.1:11434/api/tags', timeout=2)
        return True
    except Exception:
        return False

if not ollama_up():
    _log = open('ollama_serve.log', 'w')
    subprocess.Popen(['ollama', 'serve'], stdout=_log, stderr=_log)
    for _ in range(60):
        if ollama_up():
            break
        time.sleep(1)
print('Ollama reachable:', ollama_up())

# Confirm the server found the GPU. If this prints no CUDA line, stop and
# re-run this cell - running the vision model on CPU will fail.
!grep -iE -m3 'cuda|gpu|inference compute' ollama_serve.log || echo '(no GPU line yet)'
""")

code("""
# +-- Setup 4 of 4: pull the models --
# llama3.2-vision is NOT used: it downloads but current Ollama builds refuse to
# load it ("unknown model architecture: 'mllama'"), and pinning Ollama to 0.6.8
# then fails at the registry with HTTP 412. The revised brief permits a
# comparable model, so llava:7b is used instead.
!ollama pull llava:7b       # vision, Task 1        (~4.7 GB)
!ollama pull llama3.2       # text, Tasks 2 and 4   (~2.0 GB)
!ollama pull moondream      # second vision model, Extension B (~1.7 GB)
!ollama list
""")

code("""
# +-- Imports --
import warnings; warnings.filterwarnings('ignore')
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
%matplotlib inline

from src import config, data, eda, classical, llm, prompts, pipeline
config.set_seed()

print('Data root :', config.DATA_ROOT)
print('Outputs   :', config.OUTPUT_ROOT)
print(data.summarise_dataset().to_string(index=False))
""")

# ===========================================================================
md("""
---
## Task 1 — Data preparation, EDA, and multimodal LLM description

### 1.1 Preprocessing

Two decisions worth stating, because neither is neutral:

1. **Grayscale conversion.** The nuclei are DAPI-like, essentially pure blue on
   a dark field. The standard luminance weighting gives blue a coefficient of
   0.114, discarding most of the signal amplitude before analysis begins. We
   compare three conversions and pick on measured separability, not convention.
2. **Resizing.** Every image is already 256×256, so the resize is a no-op here.
   It is still applied unconditionally so the pipeline holds for other inputs.
""")

code("""
# Preprocessing is applied inside data.load_sample(); this shows what it does.
s = data.load_sample('train_001', 'train')
print('RGB in :', s.rgb.shape, s.rgb.dtype)
print('Gray out:', s.gray.shape, s.gray.dtype, f'range [{s.gray.min():.2f}, {s.gray.max():.2f}]')
print('Mask    :', s.mask.shape, s.mask.dtype, f'{s.mask.sum()} foreground px')
""")

code("""
# Figure 3: does the grayscale choice actually matter? Measure, do not assume.
path, gray_table = eda.grayscale_comparison()
display(gray_table)
eda.show(path)
""")

md("### 1.2 Exploratory data analysis")

code("""
eda.show(eda.sample_grid())          # Fig 1: images + masks by density regime
""")

code("""
eda.show(eda.intensity_histograms()) # Fig 2: intensity distributions + Otsu
""")

code("""
eda.show(eda.dataset_composition())  # Fig 4: splits, counts, coverage
""")

md("""
### 1.3 Multimodal LLM description

Three prompts are compared on the same image, isolating one variable at a time:

| prompt | structured? | uncertainty allowed? |
|---|---|---|
| `vision_naive` | no | no |
| `vision_structured_no_uncertainty` | yes | no |
| `vision_optimised` | yes | **yes** |

The third is the one used downstream. Every response is logged to
`outputs/llm_logs/` as JSONL, so the report can quote real output and the run
survives a Colab disconnect.
""")

code("""
client = llm.OllamaClient()
print('Ollama available:', client.available())
print('Models:', client.list_models())

# The representative image sent to the VLM.
vlm_sample = data.load_sample('test_002', 'test')
plt.figure(figsize=(4, 4)); plt.imshow(vlm_sample.gray, cmap='gray')
plt.title(f'{vlm_sample.image_id} (sent to the VLM)'); plt.axis('off'); plt.show()
""")

code("""
# --- Naive prompt: the control condition ---
print(prompts.NAIVE_VISION_PROMPT)
print('=' * 70)
naive = client.structured(
    config.VISION_MODEL, prompts.NAIVE_VISION_PROMPT,
    prompt_name='vision_naive', schema=None, use_format=False,
    image=vlm_sample.gray, image_id=vlm_sample.image_id,
)
print(naive.raw_text if naive.raw_text else f'CALL FAILED: {naive.error}')
""")

code("""
# --- Optimised prompt: role anchor + schema + licensed uncertainty ---
optimised = client.structured(
    config.VISION_MODEL, prompts.OPTIMISED_VISION_PROMPT,
    prompt_name='vision_optimised', schema=prompts.VISION_SCHEMA,
    system=prompts.VISION_SYSTEM,
    image=vlm_sample.gray, image_id=vlm_sample.image_id,
)
print('Schema-valid:', optimised.valid, optimised.validation_errors)
if optimised.error:
    print('CALL FAILED:', optimised.error)   # e.g. model not loaded / OOM
print('RAW:', optimised.raw_text[:400])
print(json.dumps(optimised.parsed, indent=2))
""")

code("""
# --- Middle rung: structured, but with no way to say "uncertain" ---
no_unc = client.structured(
    config.VISION_MODEL, prompts.STRUCTURED_NO_UNCERTAINTY_PROMPT,
    prompt_name='vision_structured_no_uncertainty', schema=prompts.VISION_SCHEMA,
    image=vlm_sample.gray, image_id=vlm_sample.image_id,
)
print(json.dumps(no_unc.parsed, indent=2))
""")

md("""
#### Run-to-run variability

The brief asks us to show that repeated runs are not identical. Lab 2 showed
`temperature=0` makes decoding greedy and therefore reproducible, so the
demonstration has to run at the **default** temperature a user would get out of
the box. We then confirm the opposite at `temperature=0`, which is why every
record-producing call later in the pipeline uses greedy decoding.
""")

code("""
# Default temperature: expect the wording, and possibly the fields, to vary.
runs_default = client.repeat(
    n=3, model=config.VISION_MODEL, prompt=prompts.OPTIMISED_VISION_PROMPT,
    prompt_name='vision_repeat_default', schema=prompts.VISION_SCHEMA,
    system=prompts.VISION_SYSTEM, image=vlm_sample.gray,
    image_id=vlm_sample.image_id, temperature=config.LLM_TEMPERATURE_DEMO,
)
for i, r in enumerate(runs_default):
    print(f'--- run {i + 1} ---')
    print(json.dumps(r.parsed, indent=2) if r.parsed else r.raw_text)

agree = llm.response_agreement(
    runs_default, ['modality', 'tissue_type', 'notable_features', 'image_quality'])
print('\\nAgreement across runs:')
display(pd.Series(agree))
""")

code("""
# temperature=0: the same call should now be reproducible.
runs_greedy = client.repeat(
    n=3, model=config.VISION_MODEL, prompt=prompts.OPTIMISED_VISION_PROMPT,
    prompt_name='vision_repeat_greedy', schema=prompts.VISION_SCHEMA,
    system=prompts.VISION_SYSTEM, image=vlm_sample.gray,
    image_id=vlm_sample.image_id, temperature=0.0,
)
print('Raw text identical across 3 greedy runs:',
      llm.response_agreement(runs_greedy, ['modality'])['raw_text_identical'])
""")

code("""
# --- Prompt reliability table: validity across prompts and repeats ---
# One number that summarises the whole prompt-engineering section.
rows = []
for name, prompt, schema, system in [
    ('naive', prompts.NAIVE_VISION_PROMPT, None, None),
    ('structured_no_uncertainty', prompts.STRUCTURED_NO_UNCERTAINTY_PROMPT,
     prompts.VISION_SCHEMA, None),
    ('optimised', prompts.OPTIMISED_VISION_PROMPT, prompts.VISION_SCHEMA,
     prompts.VISION_SYSTEM),
]:
    reps = client.repeat(
        n=3, model=config.VISION_MODEL, prompt=prompt,
        prompt_name=f'ablation_{name}', schema=schema, system=system,
        use_format=(schema is not None), image=vlm_sample.gray,
        image_id=vlm_sample.image_id, temperature=config.LLM_TEMPERATURE_DEMO,
    )
    parsed_ok = sum(r.parsed is not None for r in reps) / len(reps)
    rows.append(dict(prompt=name,
                     parsed_rate=parsed_ok,
                     schema_valid_rate=llm.validity_rate(reps) if schema else None,
                     mean_latency_s=round(np.mean([r.latency_s for r in reps]), 1),
                     said_uncertain=any(
                         'uncertain' in json.dumps(r.parsed or {}).lower() for r in reps)))
prompt_table = pd.DataFrame(rows)
prompt_table.to_csv(config.RESULT_DIR / 'prompt_ablation.csv', index=False)
display(prompt_table)
""")

# ===========================================================================
md("""
---
## Task 2 — Classical features and numbers-first LLM interpretation

Otsu → morphological cleanup → connected components → `regionprops_table` →
a numeric brief → LLM. **The model never sees the image**; that constraint is
enforced structurally, not just requested in the prompt.
""")

code("""
eda.show(eda.otsu_walkthrough())     # Fig 5: every stage of the classical arm
""")

code("""
# Per-object feature table for one image.
s = data.load_sample('val_003', 'val')
result = classical.run_classical(s.gray, 'val_003')
print(f"Detected {result['record']['n_objects']} objects "
      f"(ground truth: {data.ground_truth_for('val_003')['n_objects']})")
display(result['table'].head(8).round(3))
""")

code("""
# The numbers-only brief that will be handed to the LLM.
print(result['text_summary'])
""")

code("""
# Numbers-first interpretation. Note: no image argument is passed at all.
numbers_resp = client.structured(
    config.TEXT_MODEL,
    prompts.OPTIMISED_NUMBERS_PROMPT.format(measurements=result['text_summary']),
    prompt_name='numbers_optimised', schema=prompts.NUMBERS_SCHEMA,
    system=prompts.NUMBERS_SYSTEM, image_id='val_003',
)
print('Schema-valid:', numbers_resp.valid)
print(json.dumps(numbers_resp.parsed, indent=2))
""")

code("""
# Does the LLM's classification match the deterministic Python rules?
# Any disagreement is the LLM failing to apply rules it was given verbatim.
py_record = result['record']
print(f"{'field':<20}{'Python':<18}{'LLM':<18}match")
for k in ['n_objects', 'density_class', 'shape_regularity', 'quality_flag']:
    a, b = py_record.get(k), (numbers_resp.parsed or {}).get(k)
    print(f'{k:<20}{str(a):<18}{str(b):<18}{a == b}')
""")

code("""
# Otsu counting accuracy across every image - the evidence for the report's
# claim that pixel metrics and object metrics disagree on this dataset.
rows = []
for split in ['train', 'val', 'test']:
    for iid in data.list_ids(split):
        sm = data.load_sample(iid, split)
        seg = classical.otsu_segment(sm.gray)
        gt = data.ground_truth_for(iid)
        inter = (seg.mask & sm.mask).sum()
        rows.append(dict(image_id=iid, split=split, density=gt['density'],
                         true_n=gt['n_objects'], pred_n=seg.n_objects,
                         dice=2 * inter / (seg.mask.sum() + sm.mask.sum())))
otsu_df = pd.DataFrame(rows)
otsu_df['count_err'] = otsu_df.pred_n - otsu_df.true_n
otsu_df.to_csv(config.RESULT_DIR / 'otsu_baseline_all.csv', index=False)

display(otsu_df.groupby('density').agg(
    n=('image_id', 'size'), mean_dice=('dice', 'mean'),
    mean_count_err=('count_err', 'mean'),
    count_mae=('count_err', lambda x: x.abs().mean())).round(3))
eda.show(eda.counting_error_figure(otsu_df))
""")

# ===========================================================================
md("""
---
## Task 3 — U-Net segmentation

The architecture is the Lab 4 network reproduced layer-for-layer
(`UNet(in_ch=1, out_ch=1, base=16)`), run at 256×256 rather than the labs' 128×128
— it is fully convolutional, so no change is needed, and the nuclei are small
enough (~10–20 px) that downsampling would hurt.
""")

code("""
import torch
from src import unet, train, evaluate

model_blank = unet.UNet()
print(f'Parameters: {model_blank.count_parameters():,}')
print('Device:', train.get_device())
""")

code("""
# Train the primary model (BCE+Dice). ~2 min on a T4.
model, history = train.train_model(loss_name='bce_dice', epochs=config.EPOCHS)
display(history.tail(5).round(4))
""")

code("""
# Loss ablation (extension): identical seed, schedule and augmentation;
# only the loss changes.
ablation = train.loss_ablation(losses=('bce', 'dice', 'bce_dice'),
                               epochs=config.EPOCHS)
display(ablation)
""")

code("""
histories = {t: pd.read_csv(config.RESULT_DIR / f'history_{t}.csv')
             for t in ['bce', 'dice', 'bce_dice']}
eda.show(evaluate.training_curves(histories))   # Fig 8
""")

code("""
# Reload the best model and score the held-out validation split.
best_tag = ablation.loc[ablation.best_val_dice.idxmax(), 'loss']
print('Best loss:', best_tag)
model = train.load_model(best_tag)

val_scores = evaluate.per_image_scores(model, 'val')
summary = evaluate.summarise_scores(val_scores)
display(summary)
""")

code("""
# Fig 7: input / ground truth / probability / U-Net mask / Otsu mask.
# Chosen to span the difficulty range rather than to flatter the model.
examples = ['val_000', 'val_005', 'val_012']
eda.show(evaluate.prediction_panels(model, examples, 'val'))
""")

code("""
# Fig 9: where does each method win, at pixel level and at object level?
eda.show(evaluate.unet_vs_otsu_figure(val_scores))
display(pd.DataFrame(evaluate.pick_contrast_examples(val_scores)).T[
    ['image_id', 'density', 'true_n', 'unet_dice', 'otsu_dice',
     'unet_n', 'otsu_n']])
""")

code("""
# Fig 10: localise the errors. Thin rings = boundary error;
# filled blobs = missed or invented objects.
worst = val_scores.loc[val_scores.unet_dice.idxmin(), 'image_id']
print('Worst validation image:', worst)
eda.show(evaluate.error_map_figure(model, worst, 'val'))
""")

# ===========================================================================
md("""
---
## Task 4 — Hybrid pipeline on the unseen test set

`U-Net mask → regionprops → quality gate → LLM JSON + narrative → audited row → CSV`

Two design points that carry most of the trustworthiness argument:

1. **Python computes the record before the LLM is called.** The model is asked
   to echo the values and add prose; it never calculates anything.
2. **Every response is diffed against that record** (`audit_record`). If the
   model alters a number, the mismatch is logged and the Python value wins — a
   hallucination cannot reach the CSV.
""")

code("""
test_df = pipeline.run_pipeline(
    'test', model=model, client=client, use_llm=True,
    segmenter='unet', use_gate=True,
)
display(test_df[['image_id', 'n_objects', 'mean_area', 'density_class',
                 'quality_flag', 'llm_valid', 'n_field_mismatches',
                 'narrative_clean']])
""")

code("""
# One complete per-image record, as written to outputs/results/json_records/.
print(json.dumps(json.loads(
    (config.RESULT_DIR / 'json_records' / 'test_002.json').read_text()), indent=2))
""")

code("""
# Two example narratives.
for iid in ['test_000', 'test_004']:
    row = test_df[test_df.image_id == iid].iloc[0]
    print(f"--- {iid} ({row.density_class}, n={row.n_objects}) ---")
    print(row.narrative, '\\n')
""")

code("""
# How accurate is the *finished record*, scored against metadata.csv?
# This is the number that matters: not "did it segment pixels well" but
# "did the system emit something true about the sample".
display(pipeline.pipeline_accuracy(test_df).T)
""")

# ===========================================================================
md("""
---
## Extension A — Robustness: tracing a corruption through the pipeline

The gate is **disabled** here so the corruption can be watched propagating;
its verdict is reported separately as the mitigation.
""")

code("""
for base in ['test_000', 'test_004']:
    trace = pipeline.trace_corruption(base, model=model, use_llm=False)
    print(f'===== {base} =====')
    display(trace[['variant', 'img_std', 'sharpness', 'mask_pixels',
                   'mask_dice_vs_clean_gt', 'n_objects', 'mean_area',
                   'contrast_ratio', 'density_class', 'quality_flag',
                   'gate_would_pass', 'gate_reason']])
    for _, r in trace.iterrows():
        print(f"[{r.variant}] {r.narrative}\\n")
""")

md("""
### Extension B — Model comparison (optional)

Compares the two vision models pulled in setup. The habit the labs teach is
that the best model for clean JSON is often not the best for readable prose.
""")

code("""
rows = []
for m in ['llava:7b', 'moondream']:
    reps = client.repeat(n=2, model=m, prompt=prompts.OPTIMISED_VISION_PROMPT,
                         prompt_name=f'vision_model_{m.replace(":", "_")}',
                         schema=prompts.VISION_SCHEMA, system=prompts.VISION_SYSTEM,
                         image=vlm_sample.gray, image_id=vlm_sample.image_id)
    rows.append(dict(model=m,
                     schema_valid=llm.validity_rate(reps),
                     mean_latency=round(np.mean([r.latency_s for r in reps]), 1),
                     example=json.dumps(reps[0].parsed or {})[:200],
                     error=reps[0].error))
model_cmp = pd.DataFrame(rows)
model_cmp.to_csv(config.RESULT_DIR / 'vision_model_comparison.csv', index=False)
display(model_cmp)
for r in rows:
    print(f"\\n--- {r['model']} ---\\n{r['example']}")
""")

md("""
---
## Outputs

Everything the report cites is on disk:

* `outputs/figures/` — all figures
* `outputs/results/` — metric CSVs, per-image JSON records, the aggregated CSV
* `outputs/llm_logs/` — every prompt and raw response as JSONL, plus the exact
  images sent to the vision model
* `outputs/models/` — trained weights
""")

code("""
from pathlib import Path
for d in ['figures', 'results', 'llm_logs', 'models']:
    files = sorted(Path(config.OUTPUT_ROOT / d).rglob('*'))
    print(f'{d}: {len([f for f in files if f.is_file()])} files')
""")

# ===========================================================================
nb = {
    "cells": [
        {
            "cell_type": kind,
            "metadata": {},
            "source": src.splitlines(keepends=True),
            **({"outputs": [], "execution_count": None} if kind == "code" else {}),
        }
        for kind, src in C
    ],
    "metadata": {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.10"},
        "colab": {"provenance": [], "toc_visible": True},
        "accelerator": "GPU",
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).parent / "notebooks" / "assignment.ipynb"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, indent=1))
print(f"wrote {out} with {len(C)} cells "
      f"({sum(1 for k, _ in C if k == 'code')} code, "
      f"{sum(1 for k, _ in C if k == 'markdown')} markdown)")
