"""
Generate watermarked and clean image datasets for CGZ evaluation.

Usage:
    python watermark/generate_dataset.py --start 0 --end 332
"""

import torch
import os
import json
import argparse
import sys
sys.path.insert(0, '.')

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from watermark.cgz_watermark import cgz_generate, detect
from torchvision.utils import save_image

SECRET_KEY = b"cgz_llamagen_secret_2024"

# Shared output directory — everyone writes here
OUTPUT_DIR = "/orcd/home/002/isawu888/LlamaGen_6S976/samples"

def load_models(device):
    print("Loading VQ-VAE...")
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    checkpoint = torch.load("pretrained_models/vq_ds16_c2i.pt", map_location=device)
    vq_model.load_state_dict(checkpoint["model"])
    vq_model = vq_model.to(device).eval()
    print("VQ-VAE loaded.")

    print("Loading LlamaGen-L...")
    gpt_model = GPT_models["GPT-L"](
        vocab_size=16384,
        block_size=576,
        num_classes=1000,
        cls_token_num=1,
    ).to(device).eval()
    checkpoint = torch.load("pretrained_models/c2i_L_384.pt", map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    gpt_model.load_state_dict(state_dict, strict=False)
    print("LlamaGen loaded.")

    return vq_model, gpt_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end",   type=int, required=True)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda"

    # Create shared output directories
    os.makedirs(f"{OUTPUT_DIR}/watermarked", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/clean", exist_ok=True)
    os.makedirs(f"{OUTPUT_DIR}/tokens", exist_ok=True)

    vq_model, gpt_model = load_models(device)

    qzshape = [1, 8, 24, 24]

    for i in range(args.start, args.end + 1):
        class_label = i % 1000
        c_indices = torch.tensor([class_label], device=device)

        print(f"[{i}/{args.end}] class {class_label}")

        # --- Watermarked image ---
        with torch.no_grad():
            tokens_wm, generated_list = cgz_generate(
                model=gpt_model,
                cond=c_indices,
                max_new_tokens=576,
                secret_key=SECRET_KEY,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            token_grid_wm = tokens_wm[0].reshape(1, 24, 24)
            pixels_wm = vq_model.decode_code(token_grid_wm, qzshape)
            pixels_wm = (pixels_wm.clamp(-1, 1) + 1) / 2

        save_image(pixels_wm, f"{OUTPUT_DIR}/watermarked/{i:05d}.png")
        torch.save(generated_list, f"{OUTPUT_DIR}/tokens/wm_{i:05d}.pt")

        result = detect(generated_list, SECRET_KEY)
        print(f"  watermark detected: {result['detected']}, "
              f"score: {result['score']}/{result['expected_wm']:.0f}")

        # --- Clean image ---
        with torch.no_grad():
            gpt_model.setup_caches(
                max_batch_size=1,
                max_seq_length=577,
                dtype=gpt_model.tok_embeddings.weight.dtype,
            )
            tokens_clean = generate(
                model=gpt_model,
                cond=c_indices,
                max_new_tokens=576,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
            )
            token_grid_clean = tokens_clean[0].reshape(1, 24, 24)
            pixels_clean = vq_model.decode_code(token_grid_clean, qzshape)
            pixels_clean = (pixels_clean.clamp(-1, 1) + 1) / 2

        save_image(pixels_clean, f"{OUTPUT_DIR}/clean/{i:05d}.png")
        torch.save(tokens_clean[0].tolist(), f"{OUTPUT_DIR}/tokens/clean_{i:05d}.pt")

        print(f"  saved {i:05d}.png")

    print(f"\nDone! Generated images {args.start} to {args.end}")

if __name__ == "__main__":
    main()