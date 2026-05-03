"""Token-by-token regeneration attack.
Regenerates each token from the clean model given prefix,
testing whether watermark survives.

Usage: python watermark/attack_regen.py --start 0 --end 999
"""

import torch
import os
import json
import argparse
import sys
sys.path.insert(0, '.')
import lpips
from torchvision.utils import save_image
from torchvision import transforms
from PIL import Image

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from watermark.cgz_watermark import detect

SECRET_KEY = b"cgz_llamagen_secret_2024"
OUTPUT_DIR = "samples"
TOKENS_DIR = "samples/tokens"

def load_models(device):
    # exact same as generate_dataset.py
    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    checkpoint = torch.load("pretrained_models/vq_ds16_c2i.pt", map_location=device)
    vq_model.load_state_dict(checkpoint["model"])
    vq_model = vq_model.to(device).eval()

    gpt_model = GPT_models["GPT-L"](
        vocab_size=16384,
        block_size=576,
        num_classes=1000,
        cls_token_num=1,
    ).to(device).eval()
    checkpoint = torch.load("pretrained_models/c2i_L_384.pt", map_location=device)
    state_dict = checkpoint["model"] if "model" in checkpoint else checkpoint
    gpt_model.load_state_dict(state_dict, strict=False)

    return vq_model, gpt_model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", type=int, required=True)
    parser.add_argument("--end", type=int, required=True)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=2000)
    args = parser.parse_args()

    device = "cuda"

    os.makedirs(f"{OUTPUT_DIR}/attack1_regen", exist_ok=True)

    vq_model, gpt_model = load_models(device)
    loss_fn = lpips.LPIPS(net='alex').to(device)

    # transform for LPIPS — needs [-1, 1] range
    to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5])
    ])

    qzshape = [1, 8, 24, 24]
    results = []

    for i in range(args.start, args.end + 1):
        class_label = i % 1000
        c_indices = torch.tensor([class_label], device=device)

        wm_token_path = f"{TOKENS_DIR}/wm_{i:05d}.pt"
        if not os.path.exists(wm_token_path):
            print(f"Skipping {i} — no token file found")
            continue

        print(f"[{i}/{args.end}] class {class_label}")

        # load original watermarked tokens
        wm_tokens = torch.load(wm_token_path)  # this is generated_list from cgz_generate

        # --- Regenerate token by token from clean model ---
        with torch.no_grad():
            gpt_model.setup_caches(
                max_batch_size=1,
                max_seq_length=577,
                dtype=gpt_model.tok_embeddings.weight.dtype,
            )
            # generate completely fresh from clean model
            # this is the CGZ Section 6.2 attack:
            # adversary queries clean model autoregressively
            tokens_regen = generate(
                model=gpt_model,
                cond=c_indices,
                max_new_tokens=576,
                cfg_scale=args.cfg_scale,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            # decode regenerated tokens to pixels
            token_grid = tokens_regen[0].reshape(1, 24, 24)
            pixels_regen = vq_model.decode_code(token_grid, qzshape)
            pixels_regen = (pixels_regen.clamp(-1, 1) + 1) / 2

        # save regenerated image
        save_image(pixels_regen, f"{OUTPUT_DIR}/attack1_regen/{i:05d}.png")

        # --- Run detector on regenerated tokens ---
        regen_token_list = tokens_regen[0].tolist()
        detection_result = detect(regen_token_list, SECRET_KEY)

        # --- Compute LPIPS vs original watermarked image ---
        wm_img_path = f"{OUTPUT_DIR}/watermarked/{i:05d}.png"
        wm_img = to_tensor(Image.open(wm_img_path)).unsqueeze(0).to(device)
        regen_img_tensor = (pixels_regen * 2 - 1)  # back to [-1,1] for LPIPS

        with torch.no_grad():
            lpips_score = loss_fn(wm_img, regen_img_tensor).item()

        result = {
            "idx": i,
            "class": class_label,
            "detected": detection_result["detected"],
            "score": detection_result["score"],
            "expected_wm": detection_result["expected_wm"],
            "lpips": lpips_score,
        }
        results.append(result)

        print(f"  detected: {result['detected']}, "
              f"score: {result['score']}/{result['expected_wm']:.0f}, "
              f"lpips: {lpips_score:.4f}")

    # save all results to json
    out_path = f"{OUTPUT_DIR}/attack1_results_{args.start}_{args.end}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved results to {out_path}")

if __name__ == "__main__":
    main()