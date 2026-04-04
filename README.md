# train-transformer-asr

Train an ASR model from scratch for a low-resource language, comparing Transformer decoder vs CTC decoding.  
Experiment runs are fully tracked with **Weights & Biases**.

---

## Project structure

```
.
├── config.yaml      # All hyperparameters and W&B settings
├── data.py          # Dataset loading and preprocessing (Hugging Face Datasets)
├── model.py         # Transformer encoder-decoder + CTC head
├── train.py         # Training loop with W&B integration
└── requirements.txt # Python dependencies
```

## Quick start

### 1 – Install dependencies

```bash
pip install -r requirements.txt
```

### 2 – Log in to Weights & Biases

```bash
wandb login          # paste your API key from https://wandb.ai/authorize
```

### 3 – Configure the experiment

Edit `config.yaml` to set your dataset, language, and model size, or pass overrides on the command line.  
Set `wandb.entity` to your W&B username or organization.

### 4 – Train

```bash
# Use defaults from config.yaml
python train.py

# Override individual values
python train.py training.learning_rate=5e-4 wandb.entity=my_org

# Run a Bayesian hyperparameter sweep (20 runs)
python train.py --sweep
```

---

## Weights & Biases integration

| Feature | Where |
|---|---|
| Experiment configuration | `wandb.init(config=…)` in `train.py` |
| Per-step training metrics (loss, CE loss, CTC loss, LR, grad norm) | `wandb.log` every `wandb.log_interval` steps |
| Per-epoch validation metrics (WER / CER for both decoders) | `wandb.log` after each epoch |
| Audio samples with reference & hypothesis transcripts | `wandb.Table` with `wandb.Audio` objects |
| Gradient & parameter histograms | `wandb.watch(model)` |
| Model checkpoints as versioned Artifacts | `wandb.Artifact` after each epoch |
| Best-run summary | `wandb.summary` fields |
| Hyperparameter sweeps | `wandb.sweep` + `wandb.agent` (Bayesian optimisation) |

### Tracked metrics

| Metric | Description |
|---|---|
| `train/loss` | Combined (CE + CTC) training loss |
| `train/ce_loss` | Cross-entropy (attention decoder) loss |
| `train/ctc_loss` | CTC loss |
| `train/learning_rate` | Noam schedule LR |
| `train/grad_norm` | Gradient norm after clipping |
| `val/loss` | Combined validation loss |
| `val/wer_attn` | Word Error Rate – attention decoder |
| `val/cer_attn` | Character Error Rate – attention decoder |
| `val/wer_ctc` | Word Error Rate – CTC decoder |
| `val/cer_ctc` | Character Error Rate – CTC decoder |
| `val/audio_samples` | Audio table with transcription comparisons |
| `model/n_parameters` | Total trainable parameter count |
