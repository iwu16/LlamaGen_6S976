# Undetectable Watermarking for Autoregressive Image Models: Does the CGZ Impossibility Transfer?

**MIT 6.S976/18.S996 — Cryptography and Machine Learning: Foundations and Frontiers, Spring 2026**

Raymond Bahng · Isabella Wu · Maureen Zhang

---

## Overview

Christ, Gunn, and Zamir (CGZ) prove that any undetectable watermark for autoregressive language models is removable by a polynomial-time adversary via token-by-token regeneration. We ask whether this impossibility transfers to autoregressive image models.

We implement CGZ watermarking on [LlamaGen](https://github.com/FoundationVision/LlamaGen) and red-team it with three attacks:
1. **Token-by-token regeneration**: the direct CGZ attack
2. **VQ-VAE decode→re-encode**: an image-specific attack with no text analog
3. **Diffusion regeneration**: pixel-space removal via Stable Diffusion

Our main finding is that CGZ removability is a latent-level phenomenon whose practical force at the image level depends on the adversary's query access and tolerance for instance-level change. Token regeneration removes the watermark completely. Zero-query image-space attacks behave differently: VQ-VAE roundtripping preserves image quality but mostly fails to remove the watermark, while diffusion regeneration succeeds only after inducing substantial latent disruption and measurable perceptual shift.

---

## Key Results

| Attack | Survival | Mean LPIPS | FID | Tokens changed |
|---|---|---|---|---|
| Baseline (no attack) | 100% | — | — | — |
| Token regeneration | 0.0% | 0.673† | 43.17 | 100%† |
| VQ-VAE roundtrip | 90.8% | 0.026 | 2.91 | 22.3% |
| Diffusion regeneration (s=0.04) | 0.0% | 0.048 | 8.06 | 75.2% |

†Token regeneration produces a fresh independent sample rather than editing the original image. LPIPS and token change reflect instance replacement, not corruption.

**Main finding:** The attacks separate along two axes — query access and instance preservation. Token regeneration uses 576 AR queries, discards the original image, and removes the watermark completely. Zero-query image-space attacks behave differently: VQ-VAE roundtripping preserves the image but mostly fails to remove the watermark; diffusion succeeds only at the cost of larger latent disruption and measurable perceptual shift.

---

## Repository Structure

```
.
├── autoregressive/          # LlamaGen sampling loop (modified for watermarking)
├── watermark/               # CGZ watermark implementation and detection
│   └── run_assumption1.sh   # Empirical verification of decoder distortion assumption
├── results/                 # Attack results and figures
├── paper/                   # Final paper
├── proposal/                # Original project proposal
├── logs/                    # Experiment logs
├── run_generate.sh          # Generate watermarked and clean images
├── run_token_regen.sh       # Run Attack 1 (token-by-token regeneration)
├── run_partial_regen.sh     # Run partial regeneration experiments
└── run_assumption1.sh       # Run decoder distortion lower bound verification
```

---

## Method

### Watermark Implementation

We implement a keyed green-list watermark in the spirit of CGZ/KGW directly in LlamaGen's autoregressive sampling loop for class-conditional ImageNet generation. At each generation step $t$, a secret key, the current position $t$, and the previous $m=4$ tokens are hashed via HMAC-SHA256 to partition the VQ codebook into a green half $G_t \subset [V]$ where $V = 16384$. After top-$k$ filtering ($k=2000$) and temperature scaling, the watermarked sampler zeros out red-token probabilities and samples from the renormalized green distribution. The first 4 tokens are sampled normally; the detector scores $L - m = 572$ positions.

Detection computes $S_k(z) = |\{t \geq m : z_t \in G_t(c_t(z))\}|$ and declares a watermark present if $S_k(z) \geq \tau = 322.3$, derived from the toy-model soundness threshold at $\alpha = 0.01$. The threshold is validated empirically: FPR = 0.000 on 1,000 clean samples.

### Dataset

We generate 1,000 watermarked and 1,000 clean images using LlamaGen-L with classifier-free guidance scale 4.0, temperature 1.0, and top-$k=2000$, cycling through all 1,000 ImageNet classes. All images are 384×384 pixels decoded from 24×24 token grids via the pretrained VQ-VAE tokenizer.

---

## Setup

### Requirements

Follow the original LlamaGen setup:

```bash
pip install -r requirements.txt
```

Download pretrained models and place in `./pretrained_models/`:
- `vq_ds16_c2i.pt` — VQ-VAE tokenizer
- `c2i_L_384.pt` — LlamaGen-L class-conditional model

### Generate Images

```bash
bash run_generate.sh
```

### Run Attacks

```bash
# Attack 1: Token-by-token regeneration
python watermark/attack/token_regeneration/run.py

# Attack 2: VQ-VAE decode→re-encode
python watermark/attack/vqvae_roundtrip/run_attack2.py

# Attack 3: Diffusion regeneration
python watermark/attack/diffusion_regeneration/run.py
```

---

## Theoretical Contributions

We develop a finite-codebook toy model that isolates the decoder/encoder bottleneck distinguishing autoregressive image models from text. Within this model we prove:

- **Theorem 3.1** (One-shot latent undetectability): Averaged over the key, the watermarked token distribution is identical to the clean distribution. Note this does *not* imply fixed-key per-token TV closeness — for any fixed key, TV = 1/2.
- **Theorem 3.2** (Soundness): Clean false-positive rate bounded at $\alpha$ via an Azuma–Hoeffding martingale argument.
- **Theorem 3.3** (Completeness): Perfect detection of watermarked sequences.
- **Theorem 3.4** (Latent closeness under strong undetectability): If a prefix-specifiable scheme has per-token TV closeness $\varepsilon$ for every fixed key, sequential regeneration produces output within $L\varepsilon$ of the clean distribution. Note: our half-green toy watermark does *not* satisfy this — TV = 1/2 per fixed key.
- **Corollary 3.6** (Clean regeneration removal): An adversary with clean model access who discards the original and regenerates fresh tokens achieves detection probability $\leq \alpha$. This is the theoretical prediction confirmed by Attack 1.
- **Theorem 3.8** (Latent-space removal requires image distortion): Under r-robustness and a decoder distortion lower bound, any attacked latent sequence that defeats the detector must incur image distortion $\geq \psi(r)$.
- **Proposition D.1** (Separation): Perceptual undetectability does *not* imply strong latent undetectability — a concrete counterexample is given in Appendix D.

---

## References

- Christ, Gunn. *Pseudorandom Error-Correcting Codes.* CRYPTO 2024.
- Christ, Gunn, Zamir. *Undetectable Watermarks for Language Models.* COLT 2024.
- Gunn, Zhao, Song. *An Undetectable Watermark for Generative Image Models.* ICLR 2025.
- Jovanović, Staab, Vechev. *Watermarking Autoregressive Image Tokens.* ICLR 2025.
- Sun et al. *Autoregressive Model Beats Diffusion: Llama for Scalable Image Generation.* 2024.
- Zhang et al. *The Unreasonable Effectiveness of Deep Features as a Perceptual Metric.* CVPR 2018.
- Zhao et al. *Invisible Image Watermarks Are Provably Removable Using Generative AI.* NeurIPS 2024.

---

## License

This project is built on LlamaGen, which is licensed under the MIT License.
