# Hybrid Biomedical Image-Analysis Pipeline
### Fluorescence microscopy of stained cell nuclei — Assignment 3

A local, auditable pipeline that moves microscopy images through

```
raw image → segmentation → quantitative region features
          → structured JSON record → narrative → aggregated CSV
```

combining a multimodal LLM, classical image processing, and a U-Net, with all
language models run locally via Ollama.

> **Educational use only.** Nothing here is validated for clinical use. The
> dataset is synthetic. Vision-language models hallucinate confidently, and in
> a medical context that can cause harm.

---

## 1. Quick start

### Google Colab (recommended)

Open `notebooks/assignment.ipynb`, set the runtime to **T4 GPU**
(*Runtime → Change runtime type*), and run the four setup cells at the top.
They install Ollama, start the server, pull the models, and download the
dataset. Then *Run all*.

Expect roughly:

| stage | time on a T4 |
|---|---|
| pulling `llava:7b` (~4.7 GB) | 3–8 min |
| pulling `llama3.2` (~2 GB) | 1–3 min |
| pulling `moondream` (~1.7 GB) | 1–2 min |
| U-Net training, 25 epochs | ~2 min |
| loss ablation (3 models) | ~6 min |
| LLM calls across the notebook | 5–10 min |

### Local

```bash
bash setup.sh --ollama      # deps + dataset + Ollama + models
jupyter notebook notebooks/assignment.ipynb
```

Or drive it from Python without a notebook:

```python
from src import config, data, train, pipeline, llm

model, history = train.train_model(loss_name="bce_dice")
df = pipeline.run_pipeline("test", model=model, client=llm.OllamaClient())
```

### Running without Ollama

Every LLM call degrades gracefully: the error is recorded, and the pipeline
falls back to a deterministic template narrative. Tasks 1 (EDA), 2 (classical),
3 (U-Net) and the numeric half of Task 4 run fully without a model server:

```python
df = pipeline.run_pipeline("test", model=model, use_llm=False)
```

---

## 2. Repository layout

```
├── notebooks/
│   └── assignment.ipynb     driver notebook, Tasks 1-4 + extensions
├── src/
│   ├── config.py            paths, constants, seeds, density bands
│   ├── data.py              loading, grayscale, resize, metadata
│   ├── eda.py               Task 1 figures
│   ├── prompts.py           every prompt + JSON schema, versioned
│   ├── llm.py               Ollama client, JSON recovery, validation, logging
│   ├── classical.py         Otsu, morphology, regionprops, summaries
│   ├── unet.py              Lab 4 U-Net, losses, Dice/IoU
│   ├── train.py             training loop, loss ablation
│   ├── evaluate.py          metrics, panels, curves, U-Net vs Otsu
│   └── pipeline.py          end-to-end run, quality gate, auditing
├── outputs/
│   ├── figures/             every figure cited in the report
│   ├── results/             metric CSVs, per-image JSON, aggregated CSV
│   ├── llm_logs/            JSONL of every prompt and raw response
│   └── models/              trained weights
├── build_notebook.py        regenerates the notebook from a cell list
├── setup.sh                 one-shot environment setup
└── requirements.txt
```

Configuration is centralised: to re-point at another dataset or swap models,
edit `src/config.py` or set `BIOMED_DATA_ROOT`, `OLLAMA_HOST`, `VISION_MODEL`,
`TEXT_MODEL` in the environment. Nothing else hard-codes a path.

---

## 3. Design decisions worth knowing before reading the code

**Python computes the record; the LLM only renders it.**
Every number in the final JSON is produced by `classical.aggregate_features`
and `classical.derive_record` *before* any model is called. The LLM is asked to
echo those values and add a paragraph of prose. It never calculates anything.

**Every LLM response is diffed against that record.**
`pipeline.audit_record` compares the five audited fields against the Python
values and logs any mismatch; the Python value always wins in the output CSV.
A hallucinated number cannot reach the aggregated results, and the mismatch
rate becomes a measurable quantity rather than a caveat.

**A deterministic quality gate runs before the LLM.**
`pipeline.quality_gate` rejects images on object count, mean area, contrast
ratio and a Laplacian focus measure. Cheap rule-based checks catch broken
inputs microseconds before an expensive stochastic model is asked to narrate
them, and the rejection reason is a named rule plus the number that tripped it.

**The vision model is `llava:7b`, not `llama3.2-vision`.**
`llama3.2-vision` downloads correctly but current Ollama builds will not load
it, failing with `unknown model architecture: 'mllama'`. Pinning Ollama back to
0.6.8 then fails at the registry with HTTP 412, because the published manifest
now requires a newer client. The revised brief permits a comparable model, so
`llava:7b` is the default, with `moondream` as a second model for the
comparison extension. Note also that Ollama needs `pciutils` and `lshw`
installed *before* it does, or it will not detect the GPU and will run every
model on CPU — which an 11B vision model cannot survive in Colab.

**Two temperatures, for two jobs.**
Record-producing calls use `temperature=0` (greedy, reproducible). The
run-to-run variability demonstration in Task 1 uses the default temperature,
because at 0 the repeated runs would be identical and there would be nothing to
show.

**The U-Net is the Lab 4 architecture, layer for layer.**
`UNet(in_ch=1, out_ch=1, base=16)`, three encoder stages, 128-channel
bottleneck, named `enc1..enc3 / bottleneck / up3..up1 / dec3..dec1 / final` so
that `lab4_unet.pth` loads without modification. It runs at 256×256 instead of
the labs' 128×128 — the network is fully convolutional, so nothing changes, and
the nuclei are small enough (~10–20 px) that downsampling would hurt.

---

## 4. Reproducibility

`config.set_seed()` seeds Python, NumPy and PyTorch. The dataset's own
train/val/test split is honoured exactly and never resampled; the test split is
untouched until Task 4. Model selection is by best validation Dice, not by
final epoch.

The LLM is *not* seeded — that is the point of the Task 1 variability
experiment. Reproducibility of the LLM steps comes from `temperature=0` plus
the JSONL logs in `outputs/llm_logs/`, which archive every prompt, raw
response, parse outcome and validation error, along with the exact images sent
to the vision model.

---

## 5. Known limitations

- The dataset is synthetic, with cleanly bimodal intensity histograms. Otsu is
  therefore an unusually strong baseline and the pixel-level metrics saturate;
  results here should not be read as transferring to real microscopy.
- Connected-component labelling cannot separate touching nuclei, so object
  counts are biased low in the `dense` and `clustered` regimes. A watershed
  option (`otsu_segment(..., watershed_split=True)`) roughly halves the error
  but does not eliminate it.
- The narrative hallucination check (`check_narrative_numbers`) catches
  fabricated *numbers* only. A fabricated qualitative claim would pass; the
  prompt's scope fencing is the only defence against that.
- The focus-measure threshold is calibrated to this dataset's noise floor and
  would need recalibrating on real, denoised images.

---

## 6. References

Otsu, N. (1979) 'A threshold selection method from gray-level histograms',
*IEEE Transactions on Systems, Man, and Cybernetics*, 9(1), pp. 62–66.

Pech-Pacheco, J.L., Cristóbal, G., Chamorro-Martínez, J. and Fernández-Valdivia,
J. (2000) 'Diatom autofocusing in brightfield microscopy: a comparative study',
in *Proceedings of the 15th International Conference on Pattern Recognition*.
Barcelona: IEEE, pp. 314–317.

Ronneberger, O., Fischer, P. and Brox, T. (2015) 'U-Net: convolutional networks
for biomedical image segmentation', in *Medical Image Computing and
Computer-Assisted Intervention (MICCAI 2015)*. Cham: Springer, pp. 234–241.

van der Walt, S., Schönberger, J.L., Nunez-Iglesias, J., Boulogne, F., Warner,
J.D., Yager, N., Gouillart, E. and Yu, T. (2014) 'scikit-image: image processing
in Python', *PeerJ*, 2, e453.
