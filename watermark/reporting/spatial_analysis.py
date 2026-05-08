"""Spatial distribution of VQ-VAE token changes across the 24x24 token grid.

Usage (from repo root):
    python watermark/reporting/spatial_analysis.py
"""
import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

ORIG_ROOT = Path("aggregated_samples/tokens")
ATTACKED_ROOT = Path("/orcd/home/002/isawu888/LlamaGen_6S976/results/attack2/tokens")
OUTPUT_DIR = Path("watermark/reporting/outputs")

change_maps = []
for attacked_path in sorted(ATTACKED_ROOT.glob("*.pt")):
    i = int(attacked_path.stem)
    orig_path = next(ORIG_ROOT.glob(f"*_wm_{i:05d}.pt"), None)
    if orig_path is None:
        continue
    orig = torch.load(orig_path, map_location="cpu")
    attacked = torch.load(attacked_path, map_location="cpu")
    if isinstance(orig, list):
        orig = torch.tensor(orig)
    if isinstance(attacked, list):
        attacked = torch.tensor(attacked)
    orig = orig.flatten()[:576]
    attacked = attacked.flatten()[:576]
    change_maps.append((orig != attacked).float().numpy().reshape(24, 24))

mean_map = np.stack(change_maps).mean(axis=0)
print(f"Images compared: {len(change_maps)}")
print(f"Mean change rate: {mean_map.mean():.3f}")
print(f"Edge mean:   {np.concatenate([mean_map[0,:], mean_map[-1,:], mean_map[:,0], mean_map[:,-1]]).mean():.3f}")
print(f"Center mean: {mean_map[4:20, 4:20].mean():.3f}")

fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(mean_map, cmap='hot', vmin=0, vmax=1)
plt.colorbar(im, label='Fraction of images changed')
ax.set_title('Spatial distribution of VQ-VAE token changes')
fig.tight_layout()

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_DIR / 'spatial_change_map.pdf', dpi=300, bbox_inches='tight')
fig.savefig(OUTPUT_DIR / 'spatial_change_map.png', dpi=300, bbox_inches='tight')
print("Saved spatial_change_map.pdf/png")
