import torch
import sys
import os
sys.path.insert(0, '.')

from tokenizer.tokenizer_image.vq_model import VQ_models
from autoregressive.models.gpt import GPT_models
from autoregressive.models.generate import generate
from torchvision.utils import save_image

os.makedirs("samples", exist_ok=True)
device = "cuda"

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

print("Generating image...")
with torch.no_grad():
    # setup KV cache (required by LlamaGen)
    gpt_model.setup_caches(max_batch_size=1, max_seq_length=576 + 1, dtype=torch.float32)

    # class label 207 = golden retriever
    c_indices = torch.tensor([207], device=device)

    # generate 576 tokens autoregressively
    token_ids = generate(
        model=gpt_model,
        cond=c_indices,
        max_new_tokens=576,
        cfg_scale=4.0,
        temperature=1.0,
        top_k=2000,
    )

    print("Token shape:", token_ids.shape)
    print("Token range:", token_ids.min().item(), "to", token_ids.max().item())

    # decode tokens -> pixels
    token_grid = token_ids[0].reshape(1, 24, 24)
    qzshape = [1, 8, 24, 24]   # [batch, codebook_embed_dim, h, w]
    pixels = vq_model.decode_code(token_grid, qzshape)
    pixels = (pixels.clamp(-1, 1) + 1) / 2
    save_image(pixels, "samples/verify.png")
print("SUCCESS — saved samples/verify.png")
