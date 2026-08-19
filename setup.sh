#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# setup.sh - one-shot environment setup for the biomedical imaging pipeline.
#
#   bash setup.sh            # data + python deps only  (Tasks 1-4 minus LLM)
#   bash setup.sh --ollama   # also install Ollama and pull the models
#
# On Colab, run the equivalent cells in notebooks/assignment.ipynb instead -
# they do the same thing but keep the output visible in the notebook.
# ---------------------------------------------------------------------------
set -euo pipefail

echo "==> Installing Python dependencies"
pip install -q -r requirements.txt

echo "==> Fetching dataset"
if [ ! -d "data/nuclei_dataset" ]; then
  mkdir -p data
  git clone --depth 1 https://github.com/Nickolay-K/Assingnment-3-dataset.git _ds_tmp
  unzip -q _ds_tmp/nuclei_dataset.zip -d data
  rm -rf _ds_tmp
  echo "    dataset -> data/nuclei_dataset"
else
  echo "    dataset already present, skipping"
fi

if [ "${1:-}" == "--ollama" ]; then
  echo "==> Installing Ollama"
  # pciutils/lshw must exist before the installer runs, or Ollama will not
  # detect an NVIDIA GPU and will silently fall back to CPU.
  sudo apt-get -qq update && sudo apt-get -qq install -y pciutils lshw || true
  if ! command -v ollama >/dev/null 2>&1; then
    curl -fsSL https://ollama.com/install.sh | sh
  fi

  echo "==> Starting Ollama server in the background"
  nohup ollama serve > ollama.log 2>&1 &
  sleep 8

  echo "==> Pulling models (this is several GB and takes a while)"
  ollama pull llava:7b          # multimodal, Task 1
  ollama pull llama3.2          # text-only, Tasks 2 and 4
  ollama pull moondream         # second vision model, comparison extension
  ollama list
fi

echo "==> Done. Verify with:  python -c 'from src import data; print(data.summarise_dataset())'"
