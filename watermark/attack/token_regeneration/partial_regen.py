"""
Partial token regeneration sweep (Attack 1b).

Keeps the first (576 - k) tokens from each watermarked sequence as a forced
prefix, then regenerates the last k tokens from the clean LlamaGen model
(no CGZ watermarking). Sweeps k across a range to show how many tokens must
be replaced before the watermark is destroyed.

Requires:
  pretrained_models/c2i_L_384.pt   (LlamaGen GPT-L weights)
  aggregated_samples/tokens/        (watermarked token files)

Usage:
    python -m watermark.attack.token_regeneration.partial_regen \
        --root aggregated_samples \
        --output-dir watermark/attack/token_regeneration/partial_regen_outputs
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import List

import torch

from watermark.cgz_watermark import detect, compute_threshold
from watermark.metrics import SECRET_KEY, load_token_sequence

L = 576
T = 1  # class-conditioning token occupies position 0
DEFAULT_K_VALUES = [0, 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 576]
DEFAULT_OUTPUT_DIR = Path("watermark/attack/token_regeneration/partial_regen_outputs")
ALPHA = 0.01


def wm_token_files(tokens_root: Path) -> list[Path]:
    return sorted(tokens_root.glob("*_wm_*.pt")) or sorted(tokens_root.glob("wm_*.pt"))


def class_label_from_path(path: Path) -> int:
    """Extract class label from token filename.

    Convention from generate_dataset.py: class_label = index % 1000.
    Handles both 'isawu888_wm_00050.pt' and 'wm_00050.pt'.
    """
    idx_str = path.stem.rsplit("_", 1)[-1]
    return int(idx_str) % 1000


def load_model(device: str):
    from autoregressive.models.gpt import GPT_models

    checkpoint_path = Path("pretrained_models/c2i_L_384.pt")
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Missing {checkpoint_path}. Run: python watermark/download_weights.py"
        )
    model = GPT_models["GPT-L"](
        vocab_size=16384,
        block_size=576,
        num_classes=1000,
        cls_token_num=1,
    ).to(device).eval()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model", checkpoint)
    model.load_state_dict(state_dict, strict=False)
    return model


@torch.no_grad()
def partial_regenerate(
    model,
    watermarked_tokens: list[int],
    k: int,
    class_label: int,
    cfg_scale: float,
    temperature: float,
    top_k_sampling: int,
    device: str,
) -> list[int]:
    """Return a 576-token sequence: prefix forced, last k generated cleanly.

    k=0  → returns watermarked_tokens unchanged (zero model calls)
    k=576 → fresh clean sample (ignores watermarked_tokens entirely)
    """
    if k == 0:
        return list(watermarked_tokens[:L])

    from autoregressive.models.generate import prefill, sample

    prefix_len = L - k
    prefix = watermarked_tokens[:L]

    cond = torch.tensor([class_label], device=device)
    if cfg_scale > 1.0:
        cond_null = torch.ones_like(cond) * model.num_classes
        cond_combined = torch.cat([cond, cond_null])
    else:
        cond_combined = cond

    max_batch_size = cond.shape[0]
    max_batch_size_cfg = max_batch_size * 2 if cfg_scale > 1.0 else max_batch_size

    with torch.device(device):
        model.setup_caches(
            max_batch_size=max_batch_size_cfg,
            max_seq_length=T + L,
            dtype=model.tok_embeddings.weight.dtype,
        )

    sampling_kwargs = dict(
        temperature=temperature, top_k=top_k_sampling, top_p=1.0, sample_logits=True
    )

    # Prefill: class token → sample image token 0
    input_pos = torch.arange(0, T, device=device)
    next_token = prefill(model, cond_combined, input_pos, cfg_scale, **sampling_kwargs)

    # Override with forced prefix token 0 (or use freely sampled if prefix_len == 0)
    if prefix_len > 0:
        next_token = torch.tensor([[prefix[0]]], device=device, dtype=torch.int)
    generated = [next_token[0, 0].item()]

    # Autoregressive decode: force prefix, then generate freely
    input_pos = torch.tensor([T], device=device, dtype=torch.int)

    for i in range(1, L):
        cur_token = next_token.view(-1, 1)

        with torch.backends.cuda.sdp_kernel(
            enable_flash=False, enable_mem_efficient=False, enable_math=True
        ):
            if cfg_scale > 1.0:
                x_combined = torch.cat([cur_token, cur_token])
                logits, _ = model(x_combined, cond_idx=None, input_pos=input_pos)
                cond_logits, uncond_logits = torch.split(
                    logits, len(logits) // 2, dim=0
                )
                logits = uncond_logits + (cond_logits - uncond_logits) * cfg_scale
            else:
                logits, _ = model(cur_token, cond_idx=None, input_pos=input_pos)

        if i < prefix_len:
            # Force the next prefix token; don't use the model's logits
            next_token = torch.tensor([[prefix[i]]], device=device, dtype=torch.int)
        else:
            next_token, _ = sample(logits, **sampling_kwargs)

        generated.append(next_token[0, 0].item())
        input_pos = input_pos + 1

    return generated


def detection_row(tokens: list[int], path: str) -> dict:
    result = detect(tokens, SECRET_KEY, alpha=ALPHA)
    result["path"] = path
    return result


def write_detection_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["path", "detected", "score", "threshold", "fraction_green", "L", "m", "n"]
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def survival_rate(rows: list[dict]) -> float:
    if not rows:
        return float("nan")
    return sum(1 for r in rows if r["detected"]) / len(rows)


def run_sweep(
    model,
    token_files: list[Path],
    k_values: list[int],
    output_dir: Path,
    cfg_scale: float,
    temperature: float,
    top_k_sampling: int,
    device: str,
) -> list[dict]:
    """Run all k values, with per-k checkpointing. Returns summary rows."""
    summary_rows: list[dict] = []

    for k in k_values:
        csv_path = output_dir / f"k_{k:04d}" / "detection.csv"

        if csv_path.exists():
            print(f"k={k}: loading cached results from {csv_path}")
            with csv_path.open(newline="") as f:
                rows = list(csv.DictReader(f))
            rows = [{**r, "detected": r["detected"] == "True"} for r in rows]
        else:
            print(f"k={k}: regenerating {len(token_files)} images …")
            rows = []
            for path in token_files:
                wm_tokens = load_token_sequence(path)
                class_label = class_label_from_path(path)
                attacked = partial_regenerate(
                    model=model,
                    watermarked_tokens=wm_tokens,
                    k=k,
                    class_label=class_label,
                    cfg_scale=cfg_scale,
                    temperature=temperature,
                    top_k_sampling=top_k_sampling,
                    device=device,
                )
                rows.append(detection_row(attacked, str(path)))
            write_detection_csv(rows, csv_path)

        rate = survival_rate(rows)
        mean_score = sum(float(r["score"]) for r in rows) / len(rows) if rows else float("nan")
        print(f"  survival={rate:.3f}  mean_score={mean_score:.1f}")
        summary_rows.append({"k": k, "survival_rate": rate, "count": len(rows), "mean_score": mean_score})

    return summary_rows


def plot_partial_regen(summary_rows: list[dict], output_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to create plots: pip install matplotlib") from exc

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )

    ks = [r["k"] for r in summary_rows]
    rates = [r["survival_rate"] for r in summary_rows]

    # Find the k where survival drops below 0.5
    removal_k = next((r["k"] for r in summary_rows if r["survival_rate"] < 0.5), None)

    fig, ax = plt.subplots(figsize=(5.2, 3.4))
    ax.plot(ks, rates, marker="o", color="#4c78a8", linewidth=2)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=1, label="50% survival")
    if removal_k is not None:
        ax.axvline(removal_k, color="#d62728", linestyle="--", linewidth=1,
                   label=f"Removal threshold k={removal_k}")
    ax.set_xlabel("Tokens regenerated (k)")
    ax.set_ylabel("Watermark survival rate")
    ax.set_title("Partial token regeneration sweep")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(-10, L + 10)
    ax.legend(frameon=False)
    fig.tight_layout()

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "partial_regen_curve.pdf", bbox_inches="tight")
    fig.savefig(output_dir / "partial_regen_curve.png", bbox_inches="tight")
    plt.close(fig)
    print(f"Saved partial_regen_curve.pdf/png to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("aggregated_samples"),
                        help="Root directory containing tokens/ subfolder.")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--k-values", type=int, nargs="+", default=DEFAULT_K_VALUES,
                        help="Token counts to regenerate (e.g. 0 50 100 ... 576).")
    parser.add_argument("--limit", type=int, default=None,
                        help="Max number of images to process (for fast testing).")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--cfg-scale", type=float, default=4.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-k", type=int, default=2000)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-plots", action="store_true")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    token_files = wm_token_files(args.root / "tokens")
    if not token_files:
        raise FileNotFoundError(f"No watermarked token files found under {args.root / 'tokens'}")

    if args.limit:
        rng = random.Random(args.seed)
        token_files = sorted(rng.sample(token_files, min(args.limit, len(token_files))))

    print(f"Processing {len(token_files)} images on {device}")
    print(f"k values: {args.k_values}")

    # k=0 needs no model — short-circuit before loading weights
    needs_model = any(k > 0 for k in args.k_values)
    model = load_model(device) if needs_model else None

    summary_rows = run_sweep(
        model=model,
        token_files=token_files,
        k_values=sorted(set(args.k_values)),
        output_dir=args.output_dir,
        cfg_scale=args.cfg_scale,
        temperature=args.temperature,
        top_k_sampling=args.top_k,
        device=device,
    )

    summary_path = args.output_dir / "summary.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_rows, indent=2))
    print(f"Saved summary to {summary_path}")

    if not args.skip_plots:
        plot_partial_regen(summary_rows, args.output_dir)


if __name__ == "__main__":
    main()
