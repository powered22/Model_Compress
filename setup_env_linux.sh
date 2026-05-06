#!/usr/bin/env bash
# setup_env_linux.sh
# Staged environment setup untuk Linux + CUDA 12.4
# Menghindari mamba "double free" crash dengan install bertahap
set -euo pipefail

ENV_NAME="llm_experiments"
PYTHON_VER="3.11"

RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
error() { echo -e "${RED}[ERROR]${NC} $*" >&2; }
step()  { echo -e "\n${GREEN}══════════════════════════════════════════${NC}"; \
          echo -e "${GREEN}  STEP $*${NC}"; \
          echo -e "${GREEN}══════════════════════════════════════════${NC}"; }

# ── Cek conda tersedia ────────────────────────────────────────────────────────
if ! command -v conda &>/dev/null; then
    error "conda tidak ditemukan. Pastikan miniforge3/anaconda sudah diinstall."
    exit 1
fi

# ── STEP 1: Buat environment kosong dengan Python saja ───────────────────────
step "1 — Membuat environment kosong: $ENV_NAME (python=$PYTHON_VER)"

if conda env list | grep -q "^${ENV_NAME} "; then
    warn "Environment '$ENV_NAME' sudah ada. Skip pembuatan."
else
    conda create -n "$ENV_NAME" python="$PYTHON_VER" pip -y
    info "Environment '$ENV_NAME' berhasil dibuat."
fi

# ── Aktifkan environment ──────────────────────────────────────────────────────
CONDA_BASE=$(conda info --base)
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
info "Environment aktif: $CONDA_DEFAULT_ENV"

# ── STEP 2: Install conda packages (data & jupyter) ──────────────────────────
step "2 — Install paket data & Jupyter (conda-forge)"
conda install -n "$ENV_NAME" -c conda-forge --override-channels -y \
    numpy=1.26.4 \
    pandas=2.2.2 \
    pyarrow=17.0.0 \
    numexpr \
    bottleneck \
    fsspec \
    dask \
    distributed

# ── STEP 3: Install Jupyter stack ────────────────────────────────────────────
step "3 — Install Jupyter stack (conda-forge)"
conda install -n "$ENV_NAME" -c conda-forge --override-channels -y \
    jupyterlab \
    ipykernel \
    ipython \
    ipywidgets \
    notebook

# ── STEP 4: Install HuggingFace & LLM stack ──────────────────────────────────
step "4 — Install HuggingFace & utilitas (conda-forge)"
conda install -n "$ENV_NAME" -c conda-forge --override-channels -y \
    datasets \
    huggingface_hub \
    tokenizers \
    transformers \
    safetensors \
    openai \
    pydantic \
    requests \
    tqdm \
    nltk \
    pyyaml \
    regex \
    python-dotenv \
    filelock \
    click \
    packaging \
    dill \
    multiprocess \
    pyzmq \
    joblib

# ── STEP 5: Install PyTorch dengan CUDA 12.4 via pip ─────────────────────────
step "5 — Install PyTorch 2.7.1 + CUDA 12.4 (pip)"
pip install torch==2.7.1+cu124 \
    --extra-index-url https://download.pytorch.org/whl/cu124

info "Verifikasi PyTorch + GPU:"
python -c "import torch; print('  CUDA available:', torch.cuda.is_available()); \
           print('  GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A')"

# ── STEP 6: Install scientific computing ─────────────────────────────────────
step "6 — Install scipy, scikit-learn, numba, matplotlib (pip)"
pip install \
    scipy==1.15.3 \
    scikit-learn==1.7.0 \
    matplotlib==3.10.3 \
    plotly==5.24.1 \
    numba==0.61.2 \
    llvmlite==0.44.0 \
    networkx==3.5 \
    sympy==1.14.0 \
    mpmath==1.3.0

# ── STEP 7: Install NLP & embedding ──────────────────────────────────────────
step "7 — Install NLP, embeddings, evaluasi (pip)"
pip install \
    sentence-transformers==4.1.0 \
    bert-score==0.3.13 \
    rouge-score==0.1.2 \
    tiktoken==0.8.0

# ── STEP 8: Install UMAP ──────────────────────────────────────────────────────
step "8 — Install UMAP & dimensionality reduction (pip)"
pip install \
    umap-learn==0.5.7 \
    pynndescent==0.5.13

# ── STEP 9: Install weather/climate data ─────────────────────────────────────
step "9 — Install paket weather & climate (pip)"
pip install \
    xarray==2025.4.0 \
    netcdf4==1.7.2 \
    cftime==1.6.4.post1 \
    cdsapi==0.7.6 \
    ecmwf-datastores-client==0.1.0 \
    multiurl==0.3.5 \
    isodate==0.7.2

# ── STEP 10: Install Ollama client & HTTP utils ───────────────────────────────
step "10 — Install Ollama client & HTTP utils (pip)"
pip install \
    ollama==0.3.3 \
    httpx==0.27.2 \
    httpcore==1.0.5 \
    anyio==4.6.0 \
    sniffio==1.3.1 \
    idna==3.10

# ── STEP 11: Install remaining utilities ──────────────────────────────────────
step "11 — Install utilitas lainnya (pip)"
pip install \
    loguru==0.7.3 \
    tenacity==9.0.0 \
    absl-py==2.3.0 \
    fonttools==4.58.4 \
    kiwisolver==1.4.8 \
    pyparsing==3.2.3 \
    threadpoolctl==3.6.0 \
    typing-extensions==4.15.0 \
    bokeh==3.7.3 \
    xsdata==26.2 \
    xsdata-pydantic==24.5

# ── STEP 12: Register kernel Jupyter ─────────────────────────────────────────
step "12 — Register ipykernel untuk Jupyter"
python -m ipykernel install --user --name "$ENV_NAME" --display-name "LLM Experiments"

# ── Verifikasi akhir ──────────────────────────────────────────────────────────
step "SELESAI — Verifikasi akhir"
python -c "
import torch, pandas, numpy, transformers, ollama, xarray
print('  torch          :', torch.__version__, '| CUDA:', torch.cuda.is_available())
print('  pandas         :', pandas.__version__)
print('  numpy          :', numpy.__version__)
print('  transformers   :', transformers.__version__)
print('  ollama         :', ollama.__version__)
print('  xarray         :', xarray.__version__)
"

info "Setup selesai! Aktifkan dengan: conda activate $ENV_NAME"
