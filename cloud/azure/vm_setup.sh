#!/usr/bin/env bash
# =============================================================================
# Azure VM setup script for ASR training
#
# Run this on a fresh Azure Ubuntu 22.04 Data Science VM (DSVM) or on a
# Standard_NC6s_v3 / Standard_NC4as_T4_v3 GPU virtual machine.
#
# Usage:
#   chmod +x cloud/azure/vm_setup.sh
#   ./cloud/azure/vm_setup.sh
# =============================================================================
set -euo pipefail

REPO_DIR="${HOME}/train-transformer-asr"
CONDA_ENV="asr"
PYTHON_VERSION="3.10"

echo "==> Updating system packages …"
sudo apt-get update -q
sudo apt-get install -y -q git tmux htop libsndfile1

# ---------------------------------------------------------------------------
# Clone repository (skip if already present)
# ---------------------------------------------------------------------------
if [ ! -d "${REPO_DIR}" ]; then
  echo "==> Cloning repository …"
  git clone https://github.com/TrungLHB/train-transformer-asr.git "${REPO_DIR}"
else
  echo "==> Repository already exists, pulling latest …"
  git -C "${REPO_DIR}" pull
fi

# ---------------------------------------------------------------------------
# Conda environment
# ---------------------------------------------------------------------------
if ! command -v conda &>/dev/null; then
  echo "==> Installing Miniconda …"
  wget -q https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O /tmp/miniconda.sh
  bash /tmp/miniconda.sh -b -p "${HOME}/miniconda3"
  export PATH="${HOME}/miniconda3/bin:${PATH}"
  conda init bash
fi

echo "==> Creating conda environment '${CONDA_ENV}' (Python ${PYTHON_VERSION}) …"
conda create -y -n "${CONDA_ENV}" python="${PYTHON_VERSION}" || true

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "==> Installing PyTorch with CUDA 11.8 …"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "==> Installing project dependencies …"
pip install -r "${REPO_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# Convenience launch script
# ---------------------------------------------------------------------------
cat > "${HOME}/run_training.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate asr
cd ~/train-transformer-asr

DECODER="${1:-ctc}"
LANG="${2:-cy}"

if [ "${DECODER}" = "ctc" ]; then
  python scripts/train_ctc.py \
    --config configs/ctc_config.yaml \
    --language "${LANG}" \
    --checkpoint_dir "./checkpoints/ctc_${LANG}"
else
  python scripts/train_transformer.py \
    --config configs/transformer_config.yaml \
    --language "${LANG}" \
    --checkpoint_dir "./checkpoints/transformer_${LANG}"
fi
EOF
chmod +x "${HOME}/run_training.sh"

echo ""
echo "=== Azure VM setup complete! ==="
echo ""
echo "To start training inside a tmux session:"
echo "  tmux new-session -d -s asr 'bash ~/run_training.sh ctc cy 2>&1 | tee ~/train_ctc.log'"
echo "  tmux new-session -d -s asr2 'bash ~/run_training.sh transformer cy 2>&1 | tee ~/train_transformer.log'"
echo ""
echo "Monitor GPU:"
echo "  watch -n1 nvidia-smi"
