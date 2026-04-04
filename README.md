# train-transformer-asr

Train an ASR model from scratch for a low resource language, comparing transformer decoding vs CTC.

---

## Overview

This repository provides a complete end-to-end training pipeline for Automatic Speech Recognition (ASR) in low-resource language settings. Two decoder architectures are implemented and can be directly compared:

| Decoder | Architecture | Loss |
|---------|-------------|------|
| **Transformer** | Conformer encoder + autoregressive attention decoder | Cross-entropy |
| **CTC** | Conformer encoder + linear CTC projection | CTC loss |

Both share the same Conformer/Transformer encoder backbone, enabling a fair comparison.

---

## Project Structure

```
train-transformer-asr/
├── configs/
│   ├── base_config.yaml          # shared hyper-parameters
│   ├── transformer_config.yaml   # attention decoder settings
│   └── ctc_config.yaml           # CTC decoder settings
├── src/
│   ├── data/
│   │   ├── dataset.py            # HuggingFace dataset loader & CharTokenizer
│   │   └── preprocessing.py      # LogMelFilterBank & SpecAugment
│   ├── models/
│   │   ├── encoder.py            # Conformer / Transformer encoder
│   │   ├── transformer_decoder.py # Autoregressive Transformer decoder
│   │   ├── ctc_decoder.py        # CTC decoder with prefix beam search
│   │   └── asr_model.py          # CTCASRModel & TransformerASRModel
│   ├── training/
│   │   ├── trainer.py            # Unified training loop (fp16, early stopping)
│   │   └── metrics.py            # WER / CER computation
│   └── utils/
│       └── logger.py             # Logging utilities
├── scripts/
│   ├── train_transformer.py      # Entry point: train Transformer decoder
│   ├── train_ctc.py              # Entry point: train CTC decoder
│   └── evaluate.py               # Evaluate a checkpoint on the test set
├── cloud/
│   ├── aws/
│   │   ├── ec2_setup.sh          # EC2 environment setup script
│   │   └── sagemaker_job.py      # Submit training to SageMaker
│   └── azure/
│       ├── vm_setup.sh           # Azure VM environment setup script
│       └── aml_job.py            # Submit training to Azure ML
├── tests/
│   └── test_asr.py               # Unit tests (no GPU required)
├── requirements.txt
└── setup.py
```

---

## Installation

```bash
# Clone the repository
git clone https://github.com/TrungLHB/train-transformer-asr.git
cd train-transformer-asr

# Create a virtual environment (optional but recommended)
python -m venv .venv && source .venv/bin/activate

# Install PyTorch (adjust CUDA version as needed)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install remaining dependencies
pip install -r requirements.txt
```

---

## Configuration

All hyper-parameters live in the YAML files under `configs/`.  Override anything on the command line via `--<key> <value>`.

Key settings in `configs/base_config.yaml`:

| Section | Key | Default | Description |
|---------|-----|---------|-------------|
| `audio` | `sample_rate` | 16000 | Audio sample rate |
| `audio` | `n_mels` | 80 | Mel filterbank channels |
| `data` | `language` | `cy` | CommonVoice language code (Welsh) |
| `encoder` | `encoder_type` | `conformer` | `conformer` or `transformer` |
| `encoder` | `d_model` | 256 | Model dimension |
| `training` | `batch_size` | 16 | Batch size per device |
| `training` | `max_epochs` | 100 | Maximum training epochs |
| `hardware` | `fp16` | `true` | Mixed-precision training |

---

## Training

### CTC model

```bash
python scripts/train_ctc.py \
    --config configs/ctc_config.yaml \
    --language cy \
    --checkpoint_dir ./checkpoints/ctc_welsh
```

### Transformer decoder model

```bash
python scripts/train_transformer.py \
    --config configs/transformer_config.yaml \
    --language cy \
    --checkpoint_dir ./checkpoints/transformer_welsh
```

Training logs are written to TensorBoard:

```bash
tensorboard --logdir ./logs
```

---

## Evaluation

```bash
python scripts/evaluate.py \
    --checkpoint checkpoints/ctc_welsh/best_model.pt \
    --config configs/ctc_config.yaml \
    --vocab_file checkpoints/ctc_welsh/vocab.txt \
    --language cy \
    --output results/ctc_welsh_test.txt
```

---

## Testing

Run unit tests (no GPU or data download required):

```bash
pip install pytest
python -m pytest tests/ -v
```

---

## Cloud Deployment

### AWS EC2

1. Launch an EC2 instance (recommended: `p3.2xlarge` or `g4dn.xlarge`) from the
   [AWS Deep Learning AMI (Ubuntu 22.04)](https://aws.amazon.com/machine-learning/amis/).
2. SSH into the instance and run:

```bash
chmod +x cloud/aws/ec2_setup.sh
./cloud/aws/ec2_setup.sh
bash ~/run_training.sh ctc cy
```

### AWS SageMaker (Managed Training)

```bash
python cloud/aws/sagemaker_job.py \
    --decoder ctc \
    --language cy \
    --role arn:aws:iam::123456789012:role/SageMakerRole \
    --bucket my-s3-bucket \
    --instance ml.p3.2xlarge
```

### Azure Virtual Machine

1. Create an `Standard_NC6s_v3` (V100) or `Standard_NC4as_T4_v3` (T4) Azure VM.
2. SSH in and run:

```bash
chmod +x cloud/azure/vm_setup.sh
./cloud/azure/vm_setup.sh
bash ~/run_training.sh transformer cy
```

### Azure Machine Learning (Managed Training)

```bash
python cloud/azure/aml_job.py \
    --decoder transformer \
    --language cy \
    --subscription_id <sub_id> \
    --resource_group <rg> \
    --workspace_name <ws> \
    --compute_name gpu-cluster
```

---

## Dataset

By default the scripts use [Common Voice 13](https://huggingface.co/datasets/mozilla-foundation/common_voice_13_0) (**Welsh / cy**) as a low-resource language example. Change `--language` to any [CommonVoice language code](https://commonvoice.mozilla.org/en/languages) (e.g. `br` for Breton, `rm-sursilv` for Romansh).

The dataset is downloaded automatically via the Hugging Face `datasets` library on first run.

---

## Architecture Details

### Encoder (shared)
- Conv sub-sampling (2× stride-2 convolutions, factor-4 time reduction)
- Stack of **Conformer** blocks (default) or Transformer encoder blocks
- Sinusoidal positional encoding

### Transformer Decoder
- Causal masked self-attention + cross-attention over encoder outputs
- Teacher forcing during training; **beam search** at inference

### CTC Decoder
- Linear projection from encoder hidden states to vocabulary logits
- CTC loss with blank token
- **Prefix beam search** at inference

### Training Utilities
- **SpecAugment** (time + frequency masking)
- Noam learning-rate schedule with configurable warm-up
- Mixed-precision training (fp16)
- Early stopping on validation WER
- TensorBoard logging (loss, WER, CER, LR)

