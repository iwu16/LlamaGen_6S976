# test_vqvae_roundtrip.py
import torch
import sys
sys.path.insert(0, '.')
from tokenizer.tokenizer_image.vq_model import VQ_models
from torchvision.utils import save_image
import torchvision.transforms as T
from PIL import Image

device = "cuda"

# Load VQ-VAE
vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
checkpoint = torch.load("pretrained_models/vq_ds16_c2i.pt", map_location=device)
vq_model.load_state_dict(checkpoint["model"])
vq_model = vq_model.to(device).eval()

# Load one watermarked image
img = Image.open("aggregated_samples/watermarked/00000.png").convert("RGB")
transform = T.Compose([
    T.ToTensor(),                        # [0, 1]
    T.Normalize([0.5, 0.5, 0.5],        # → [-1, 1]
                [0.5, 0.5, 0.5])
])
pixels = transform(img).unsqueeze(0).to(device)  # [1, 3, 384, 384]

# Re-encode: pixels → tokens
with torch.no_grad():
    quant, emb_loss, info = vq_model.encode(pixels)
    
    # info = (perplexity, min_encodings, min_encoding_indices)
    min_encoding_indices = info[2]  # these are the integer codebook indices
    print("Indices shape:", min_encoding_indices.shape)
    print("Indices dtype:", min_encoding_indices.dtype)
    print("Indices range:", min_encoding_indices.min().item(), "to", min_encoding_indices.max().item())
    
    # reshape to [1, 24, 24] for decode_code
    tokens = min_encoding_indices.reshape(1, 24, 24)
    
    # decode back to pixels
    qzshape = [1, 8, 24, 24]
    pixels_roundtrip = vq_model.decode_code(tokens, qzshape)
    pixels_roundtrip = (pixels_roundtrip.clamp(-1, 1) + 1) / 2
# with torch.no_grad():
#     tokens, _, _ = vq_model.encode(pixels)   # [1, 24, 24]
#     print("Re-encoded token shape:", tokens.shape)
#     print("Token range:", tokens.min().item(), "to", tokens.max().item())

#     # Decode back to pixels
#     qzshape = [1, 8, 24, 24]
#     pixels_roundtrip = vq_model.decode_code(tokens, qzshape)
#     pixels_roundtrip = (pixels_roundtrip.clamp(-1, 1) + 1) / 2
orig_tokens = torch.load("aggregated_samples/tokens/wm_00000.pt")
new_tokens = min_encoding_indices.flatten().tolist()
changed = sum(a != b for a, b in zip(orig_tokens, new_tokens))
print(f"Tokens changed: {changed}/576 ({100*changed/576:.1f}%)")
from watermark.cgz_watermark import detect
result_before = detect(orig_tokens, b"cgz_llamagen_secret_2024")
result_after = detect(new_tokens, b"cgz_llamagen_secret_2024")
print(f"Watermark BEFORE: detected={result_before['detected']}, score={result_before['score']}/{result_before['expected_wm']:.0f}")
print(f"Watermark AFTER:  detected={result_after['detected']}, score={result_after['score']}/{result_after['expected_wm']:.0f}")

save_image(pixels_roundtrip, "samples/roundtrip_test.png")
print("Saved roundtrip_test.png")