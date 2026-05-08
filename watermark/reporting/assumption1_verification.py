"""Empirical verification of Assumption 1: larger latent Hamming distance -> larger LPIPS.

Randomly flips k tokens in each watermarked sequence, decodes with VQ-VAE,
and measures LPIPS distortion. Plots mean LPIPS vs k with a power-law fit.

Usage:
    python watermark/reporting/assumption1_verification.py
"""
from __future__ import annotations

import argparse
import json
import numpy as np
import torch
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "watermark" / "reporting" / "outputs"
VOCAB_SIZE = 16384
SEQ_LEN = 576
DEFAULT_K_VALUES = [1, 5, 10, 25, 50, 100, 200, 300, 400, 500, 576]
QZSHAPE = [1, 8, 24, 24]


def load_vqvae(device: str):
    from tokenizer.tokenizer_image.vq_model import VQ_models
    model = VQ_models["VQ-16"](codebook_size=VOCAB_SIZE, codebook_embed_dim=8)
    ckpt = torch.load("pretrained_models/vq_ds16_c2i.pt", map_location=device)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def load_lpips(device: str):
    import lpips
    return lpips.LPIPS(net="alex").to(device).eval()


def decode_tokens(vqvae, tokens: list[int], device: str) -> torch.Tensor:
    idx = torch.tensor(tokens, device=device).reshape(1, 24, 24)
    with torch.no_grad():
        pixels = vqvae.decode_code(idx, QZSHAPE)
    return pixels.clamp(-1, 1)


def wm_token_files(root: Path, limit: int) -> list[Path]:
    files = sorted(root.glob("*_wm_*.pt"))
    return files[:limit]


def run(args: argparse.Namespace) -> None:
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    vqvae = load_vqvae(device)
    lpips_fn = load_lpips(device)

    token_files = wm_token_files(Path(args.root) / "tokens", args.n_images)
    print(f"Processing {len(token_files)} images, k values: {args.k_values}")

    rng = np.random.default_rng(args.seed)
    results: dict[int, list[float]] = {k: [] for k in args.k_values}

    for idx, path in enumerate(token_files):
        raw = torch.load(path, map_location="cpu")
        if isinstance(raw, list):
            raw = torch.tensor(raw)
        tokens = raw.flatten()[:SEQ_LEN].tolist()

        orig_pixels = decode_tokens(vqvae, tokens, device)

        for k in args.k_values:
            flip_pos = rng.choice(SEQ_LEN, k, replace=False)
            perturbed = tokens.copy()
            for p in flip_pos:
                perturbed[p] = int(rng.integers(0, VOCAB_SIZE))

            perturbed_pixels = decode_tokens(vqvae, perturbed, device)
            with torch.no_grad():
                score = float(lpips_fn(orig_pixels, perturbed_pixels).item())
            results[k].append(score)

        if (idx + 1) % 10 == 0:
            print(f"  [{idx+1}/{len(token_files)}] done")

    summary = [
        {"k": k, "mean_lpips": float(np.mean(v)), "std_lpips": float(np.std(v)), "n": len(v)}
        for k, v in sorted(results.items())
    ]
    for row in summary:
        print(f"  k={row['k']:4d}  mean_lpips={row['mean_lpips']:.4f}  std={row['std_lpips']:.4f}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    json_path = OUTPUT_DIR / "assumption1_lpips_vs_k.json"
    json_path.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {json_path}")

    plot(summary)


def plot(summary: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy.optimize import curve_fit

    ks = np.array([r["k"] for r in summary])
    means = np.array([r["mean_lpips"] for r in summary])
    stds = np.array([r["std_lpips"] for r in summary])
    ns = np.array([r["n"] for r in summary])
    sems = stds / np.sqrt(ns)

    # Power-law fit: lpips = a * k^b
    def power_law(k, a, b):
        return a * np.power(k, b)

    try:
        popt, _ = curve_fit(power_law, ks, means, p0=[0.001, 0.5], maxfev=5000)
        a, b = popt
        fit_label = f"Fit: $\\psi(k) \\approx {a:.4f} \\cdot k^{{{b:.3f}}}$"
        fit_ks = np.linspace(ks[0], ks[-1], 200)
        fit_vals = power_law(fit_ks, a, b)
        has_fit = True
    except Exception:
        has_fit = False

    plt.rcParams.update({
        "font.size": 8.5,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.25,
        "figure.dpi": 160,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.errorbar(ks, means, yerr=sems, fmt="o", color="#4c78a8",
                capsize=3, linewidth=1.5, markersize=5, label="Mean LPIPS ± SEM")
    if has_fit:
        ax.plot(fit_ks, fit_vals, color="#d62728", linestyle="--",
                linewidth=1.4, label=fit_label)

    ax.set_xlabel("Tokens randomly flipped ($k$)")
    ax.set_ylabel("LPIPS distortion")
    ax.set_title("Empirical support for Assumption 1")
    ax.legend(frameon=False)
    fig.tight_layout()

    for ext in ("pdf", "png"):
        fig.savefig(OUTPUT_DIR / f"assumption1_lpips_vs_k.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved assumption1_lpips_vs_k.pdf/png to {OUTPUT_DIR}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="aggregated_samples")
    parser.add_argument("--n-images", type=int, default=200)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--plots-only", action="store_true",
                        help="Skip computation, just replot from saved JSON.")
    args = parser.parse_args()

    if args.plots_only:
        json_path = OUTPUT_DIR / "assumption1_lpips_vs_k.json"
        summary = json.loads(json_path.read_text())
        plot(summary)
    else:
        run(args)


if __name__ == "__main__":
    main()
