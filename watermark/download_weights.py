from huggingface_hub import hf_hub_download

print("Downloading VQ-VAE tokenizer...")
hf_hub_download(
    repo_id="FoundationVision/LlamaGen",
    filename="vq_ds16_c2i.pt",
    local_dir="./pretrained_models"
)

print("Downloading LlamaGen-L weights...")
hf_hub_download(
    repo_id="FoundationVision/LlamaGen",
    filename="c2i_L_384.pt",
    local_dir="./pretrained_models"
)

print("Done! Weights saved to ./pretrained_models/")
