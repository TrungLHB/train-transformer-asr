#!/usr/bin/env bash
# =============================================================================
# AWS EC2 setup script for ASR training
#
# Run this script on a fresh Deep Learning AMI (Ubuntu 22.04) instance.
# It installs all dependencies and prepares the training environment.
#
# Recommended instance type: p3.2xlarge (1× V100) or g4dn.xlarge (1× T4)
#
# Usage:
#   chmod +x cloud/aws/ec2_setup.sh
#   ./cloud/aws/ec2_setup.sh
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
  echo "==> Repository already exists, pulling latest changes …"
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

# Activate and install
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"

echo "==> Installing PyTorch with CUDA 11.8 …"
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "==> Installing project dependencies …"
pip install -r "${REPO_DIR}/requirements.txt"

# ---------------------------------------------------------------------------
# Create a convenience launch script
# ---------------------------------------------------------------------------
cat > "${HOME}/run_training.sh" <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate asr
cd ~/train-transformer-asr

DECODER="${1:-ctc}"          # "ctc" or "transformer"
LANG="${2:-cy}"              # language code (default: Welsh)

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
echo "=== Setup complete! ==="
echo ""
echo "To start training inside a tmux session:"
echo "  tmux new-session -d -s asr 'bash ~/run_training.sh ctc cy 2>&1 | tee ~/train_ctc.log'"
echo "  tmux new-session -d -s asr2 'bash ~/run_training.sh transformer cy 2>&1 | tee ~/train_transformer.log'"
echo ""
echo "Monitor GPU:"
echo "  watch -n1 nvidia-smi"
