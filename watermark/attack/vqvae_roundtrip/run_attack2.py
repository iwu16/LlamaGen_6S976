"""
Attack 2: VQ-VAE decode -> re-encode
Runs on all 1000 watermarked images and reports:
- Watermark survival rate
- LPIPS distortion
- Tokens changed
"""
import torch
import os
import json
import sys
import lpips
sys.path.insert(0, '.')

import torchvision.transforms as T
from PIL import Image
from torchvision.utils import save_image
from tokenizer.tokenizer_image.vq_model import VQ_models
from watermark.cgz_watermark import detect

SECRET_KEY = b"cgz_llamagen_secret_2024"
WATERMARKED_DIR = "aggregated_samples/watermarked"
TOKENS_DIR = "aggregated_samples/tokens"
OUTPUT_DIR = "results/attack2"
N = 1000

device = "cuda"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load VQ-VAE
print("Loading VQ-VAE...")
vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
checkpoint = torch.load("pretrained_models/vq_ds16_c2i.pt", map_location=device)
vq_model.load_state_dict(checkpoint["model"])
vq_model = vq_model.to(device).eval()

# Load LPIPS
print("Loading LPIPS...")
loss_fn = lpips.LPIPS(net='alex').to(device)

transform = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

results = []

for i in range(N):
    wm_path = f"{WATERMARKED_DIR}/{i:05d}.png"
    token_path = f"{TOKENS_DIR}/wm_{i:05d}.pt"

    if not os.path.exists(wm_path) or not os.path.exists(token_path):
        print(f"[{i}] missing, skipping")
        continue

    # Load original image and tokens
    img = Image.open(wm_path).convert("RGB")
    pixels_orig = transform(img).unsqueeze(0).to(device)  # [-1, 1]
    orig_tokens = torch.load(token_path)

    with torch.no_grad():
        # Re-encode through VQ-VAE
        quant, emb_loss, info = vq_model.encode(pixels_orig)
        new_indices = info[2]  # integer codebook indices
        new_tokens = new_indices.flatten().tolist()

        # Decode back to pixels
        tokens_grid = new_indices.reshape(1, 24, 24)
        qzshape = [1, 8, 24, 24]
        pixels_attacked = vq_model.decode_code(tokens_grid, qzshape)  # [-1, 1]
        # save attacked image and tokens (add right after pixels_attacked is computed)
        os.makedirs("results/attack2/images", exist_ok=True)
        os.makedirs("results/attack2/tokens", exist_ok=True)
        save_image((pixels_attacked.clamp(-1,1)+1)/2, f"results/attack2/images/{i:05d}.png")
        torch.save(new_tokens, f"results/attack2/tokens/{i:05d}.pt")

        # LPIPS
        lpips_score = loss_fn(pixels_orig, pixels_attacked).item()

    # Detection before and after
    result_before = detect(orig_tokens, SECRET_KEY)
    result_after = detect(new_tokens, SECRET_KEY)

    # Token change rate
    changed = sum(a != b for a, b in zip(orig_tokens, new_tokens))

    results.append({
        "i": i,
        "detected_before": result_before["detected"],
        "detected_after": result_after["detected"],
        "score_before": result_before["score"],
        "score_after": result_after["score"],
        "threshold": result_before["threshold"],
        "lpips": lpips_score,
        "tokens_changed": changed,
        "tokens_changed_pct": round(100 * changed / 576, 2),
    })

    print(f"[{i}] survived={result_after['detected']}, "
          f"score={result_after['score']}/{result_after['expected_wm']:.0f}, "
          f"LPIPS={lpips_score:.4f}, "
          f"tokens changed={changed}/576 ({100*changed/576:.1f}%)")

# Save results
with open(f"{OUTPUT_DIR}/results.json", "w") as f:
    json.dump(results, f, indent=2)

# Summary
n = len(results)
survived = sum(r["detected_after"] for r in results)
mean_lpips = sum(r["lpips"] for r in results) / n
mean_changed = sum(r["tokens_changed_pct"] for r in results) / n

print(f"\n=== Attack 2 Summary ===")
print(f"Images evaluated:        {n}")
print(f"Watermark survival rate: {survived}/{n} ({100*survived/n:.1f}%)")
print(f"Mean LPIPS distortion:   {mean_lpips:.4f}")
print(f"Mean tokens changed:     {mean_changed:.1f}%")
print(f"Results saved to {OUTPUT_DIR}/results.json")