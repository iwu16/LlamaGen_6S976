# Watermarking LlamaGen — Setup Guide

## 1. Clone the repo
git clone https://github.com/iwu16/LlamaGen_6S976.git
cd LlamaGen_6S976

## 2. Create conda environment
conda create -n llamagen python=3.10 -y
conda activate llamagen

## 3. Install PyTorch (pick the right CUDA version)
# CUDA 11.8:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
# CUDA 12.1:
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu121

## 4. Install dependencies
pip install -r requirements.txt
pip install huggingface_hub lpips clean-fid

## 5. Download pretrained weights
python watermark/download_weights.py

## 6. Verify everything works
python watermark/verify.py

## 7. Shared evaluation and attack ownership

Shared metric functions live in:

```text
watermark/evaluate.py
watermark/metrics.py
```

The main functions in `watermark/evaluate.py` are:

```text
compute_tpr_fpr(...)
compute_watermark_survival(...)
compute_quality_metrics(...)
```

Attack code lives in separate folders to avoid merge conflicts:

```text
watermark/attack/token_regeneration/run.py
watermark/attack/vqvae_roundtrip/run.py
watermark/attack/diffusion_regeneration/run.py
```

The shared output contract is:

```text
attack_outputs/<attack_name>/images/*.png
attack_outputs/<attack_name>/tokens/*.pt
```

Each attack script should generate its outputs, then import only the metric
functions it needs:

```python
from watermark.evaluate import compute_tpr_fpr
from watermark.evaluate import compute_watermark_survival
from watermark.evaluate import compute_quality_metrics
```

Run an attack as a module, for example:

```bash
python -m watermark.attack.diffusion_regeneration.run
```
