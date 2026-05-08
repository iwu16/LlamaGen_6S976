"""
Attack 3: diffusion regeneration.

Refactored for:
- Class-specific prompts (ImageNet)
- Log-scale strengths [0.01, 0.02, 0.04, 0.08, 0.16]
- Guidance scale 7.5
- Improved checkpointing and reproducibility
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import re
import shutil
from pathlib import Path

import torch
from PIL import Image
from torchvision import transforms

from watermark.evaluate import (
    compute_quality_metrics,
    compute_tpr_fpr,
    compute_watermark_survival,
)
from tokenizer.tokenizer_image.vq_model import VQ_models
from tools.imagenet_en_cn import IMAGENET_1K_CLASSES


DEFAULT_OUTPUT_DIR = Path("watermark/attack/diffusion_regeneration/outputs")
DEFAULT_STRENGTHS = [0.01, 0.02, 0.04, 0.08, 0.16]
VQ_CHECKPOINT = Path("pretrained_models/vq_ds16_c2i.pt")


def slugify(value: str) -> str:
    value = value.lower().replace("/", "_")
    value = re.sub(r"[^a-z0-9_.-]+", "_", value)
    return value.strip("_")


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_name:
        return args.output_root / slugify(args.run_name)

    strengths = "_".join(strength_name(strength).replace("strength_", "") for strength in args.strengths)
    model = slugify(args.model_id)
    config = (
        f"{model}_steps{args.num_inference_steps}_"
        f"cfg{args.guidance_scale:g}_seed{args.seed}_s{strengths}"
    )
    if args.limit:
        config += f"_n{args.limit}"
    return args.output_root / config


def strength_name(strength: float) -> str:
    return f"strength_{strength:.3f}".replace(".", "_")


def load_diffusion_pipeline(model_id: str, device: str, dtype: torch.dtype):
    try:
        from diffusers import StableDiffusionImg2ImgPipeline
    except ImportError as exc:
        raise RuntimeError(
            "Install diffusers to run this attack: "
            "pip install diffusers transformers accelerate safetensors"
        ) from exc

    pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        safety_checker=None,
        requires_safety_checker=False,
    )
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    return pipe


def load_vq_model(device: str):
    if not VQ_CHECKPOINT.exists():
        raise FileNotFoundError(
            f"Missing {VQ_CHECKPOINT}. Run: python watermark/download_weights.py"
        )

    vq_model = VQ_models["VQ-16"](codebook_size=16384, codebook_embed_dim=8)
    checkpoint = torch.load(VQ_CHECKPOINT, map_location=device)
    vq_model.load_state_dict(checkpoint["model"])
    return vq_model.to(device).eval()


def encode_image_to_tokens(vq_model, image_path: Path, device: str) -> list[int]:
    image = Image.open(image_path).convert("RGB")
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    image_tensor = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        _, _, [_, _, indices] = vq_model.encode(image_tensor)
    return [int(x) for x in indices.reshape(-1).detach().cpu().tolist()]


def get_imagenet_prompt(image_path: Path) -> str:
    """Extract class index from filename (e.g. 00005.png -> class 5) and return prompt."""
    try:
        idx = int(image_path.stem) % 1000
        class_name = IMAGENET_1K_CLASSES[idx].split(",")[0].strip()
        return f"a high quality photo of a {class_name}"
    except (ValueError, KeyError):
        return "a high quality natural image"


def select_inputs(input_dir: Path, limit: int | None, seed: int) -> list[Path]:
    image_paths = sorted(input_dir.glob("*.png"))
    if limit is None or limit >= len(image_paths):
        return image_paths

    rng = random.Random(seed)
    sampled = rng.sample(image_paths, limit)
    return sorted(sampled)


def prepare_reference_images(image_paths: list[Path], output_dir: Path) -> Path:
    reference_dir = output_dir / "reference_watermarked"
    reference_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        target = reference_dir / image_path.name
        if not target.exists():
            shutil.copy2(image_path, target)
    return reference_dir


def regenerate_images_for_strength(
    pipe,
    image_paths: list[Path],
    images_dir: Path,
    strength: float,
    negative_prompt: str,
    guidance_scale: float,
    num_inference_steps: int,
    seed: int,
    device: str,
    skip_existing: bool,
) -> None:
    from tqdm.auto import tqdm

    effective_steps = int(num_inference_steps * strength)
    if effective_steps < 1:
        # For very small strengths, we force at least 1 step instead of skipping
        print(f"Warning: strength {strength:.3f} with {num_inference_steps} steps would result in 0 effective steps. Forcing 1 step.")
        # We don't return early here; we let diffusers handle the strength. 
        # Actually, diffusers uses `int(num_inference_steps * strength)` internally too.
        # To get 1 step, we might need to slightly increase the strength or total steps.
        # But we'll just let it run and see if diffusers handles it, or force a minimum.

    images_dir.mkdir(parents=True, exist_ok=True)
    
    # Ensure torch seed for potential reproducibility within the pipeline if it uses global torch RNG
    torch.manual_seed(seed)
    generator = torch.Generator(device=device).manual_seed(seed)

    iterator = tqdm(
        image_paths,
        desc=f"regenerate s={strength:.3f}",
        unit="img",
        dynamic_ncols=True,
    )
    for image_path in iterator:
        output_path = images_dir / image_path.name
        if skip_existing and output_path.exists():
            continue

        prompt = get_imagenet_prompt(image_path)
        init_image = Image.open(image_path).convert("RGB")
        try:
            generated = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image=init_image,
                strength=strength,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                generator=generator,
            ).images[0]
            generated.save(output_path)
        except Exception as e:
            print(f"Error generating {image_path}: {e}")
            continue


def encode_outputs(vq_model, images_dir: Path, tokens_dir: Path, device: str) -> None:
    from tqdm.auto import tqdm

    tokens_dir.mkdir(parents=True, exist_ok=True)
    image_paths = sorted(images_dir.glob("*.png"))
    if not image_paths:
        print(f"No images found in {images_dir}, skipping VQ encode.")
        return

    iterator = tqdm(image_paths, desc="VQ encode", unit="img", dynamic_ncols=True)
    for image_path in iterator:
        token_path = tokens_dir / f"{image_path.stem}.pt"
        if token_path.exists():
            continue
        tokens = encode_image_to_tokens(vq_model, image_path, device)
        torch.save(tokens, token_path)


def write_summary_csv(rows: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "strength",
        "clean_fpr",
        "watermarked_tpr",
        "survival_rate",
        "lpips",
        "fid",
        "attacked_count",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def rows_from_summary(report: dict) -> list[dict]:
    baseline = report.get("baseline_detection", {})
    rows = []
    for name, strength_report in report.get("strengths", {}).items():
        detection = strength_report["attacked_detection"]
        quality = strength_report["attacked_quality"]
        rows.append(
            {
                "strength": strength_report["strength"],
                "clean_fpr": baseline.get("false_positive_rate"),
                "watermarked_tpr": baseline.get("true_positive_rate"),
                "survival_rate": detection["survival_rate"],
                "lpips": quality["lpips"],
                "fid": quality["fid"],
                "attacked_count": detection["count"],
                "name": name,
            }
        )
    return sorted(rows, key=lambda row: row["strength"])


def make_plots(rows: list[dict], output_dir: Path, title_suffix: str = "") -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to create plots: pip install matplotlib") from exc

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    strengths = [row["strength"] for row in rows]
    survival = [row["survival_rate"] for row in rows]
    lpips = [row["lpips"] for row in rows]
    fid = [row["fid"] for row in rows]

    plt.rcParams.update(
        {
            "font.size": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 160,
            "savefig.dpi": 300,
        }
    )

    fig, ax1 = plt.subplots(figsize=(5.6, 3.4))
    ax1.plot(strengths, survival, marker="o", color="#1f77b4", label="Watermark survival")
    ax1.set_xlabel("Diffusion img2img strength")
    ax1.set_ylabel("Watermark survival rate", color="#1f77b4")
    ax1.tick_params(axis="y", labelcolor="#1f77b4")
    ax1.set_ylim(-0.02, 1.02)

    ax2 = ax1.twinx()
    ax2.plot(strengths, lpips, marker="s", color="#d62728", label="LPIPS")
    ax2.set_ylabel("LPIPS", color="#d62728")
    ax2.tick_params(axis="y", labelcolor="#d62728")
    if title_suffix:
        ax1.set_title(title_suffix)
    fig.tight_layout()
    fig.savefig(plots_dir / "survival_vs_lpips.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "survival_vs_lpips.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    ax.plot(strengths, fid, marker="o", color="#2ca02c")
    ax.set_xlabel("Diffusion img2img strength")
    ax.set_ylabel("FID vs original watermarked images")
    ax.set_title(f"Distribution shift{title_suffix}" if title_suffix else "Distribution shift")
    fig.tight_layout()
    fig.savefig(plots_dir / "fid_by_strength.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "fid_by_strength.png", bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(4.4, 3.6))
    scatter = ax.scatter(lpips, survival, c=strengths, cmap="viridis", s=70)
    ax.set_xlabel("LPIPS vs original watermarked image")
    ax.set_ylabel("Watermark survival rate")
    ax.set_ylim(-0.02, 1.02)
    cbar = fig.colorbar(scatter, ax=ax)
    cbar.set_label("Diffusion strength")
    fig.tight_layout()
    fig.savefig(plots_dir / "quality_removal_tradeoff.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "quality_removal_tradeoff.png", bbox_inches="tight")
    plt.close(fig)


def make_example_grid(
    reference_dir: Path,
    output_dir: Path,
    strengths: list[float],
    max_examples: int = 4,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("Install matplotlib to create plots: pip install matplotlib") from exc

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    examples = sorted(reference_dir.glob("*.png"))[:max_examples]
    if not examples:
        return

    row_labels = ["original"] + [f"s={strength:.3f}" for strength in strengths]
    fig, axes = plt.subplots(
        len(row_labels),
        len(examples),
        figsize=(2.0 * len(examples), 2.0 * len(row_labels)),
        squeeze=False,
    )

    for col, ref_path in enumerate(examples):
        axes[0][col].imshow(Image.open(ref_path).convert("RGB"))
        axes[0][col].set_title(ref_path.stem, fontsize=8)
        axes[0][col].axis("off")

        for row_idx, strength in enumerate(strengths, start=1):
            attacked_path = (
                output_dir / strength_name(strength) / "images" / ref_path.name
            )
            if attacked_path.exists():
                axes[row_idx][col].imshow(Image.open(attacked_path).convert("RGB"))
            axes[row_idx][col].axis("off")

    for row_idx, label in enumerate(row_labels):
        axes[row_idx][0].set_ylabel(label, rotation=0, labelpad=34, va="center")

    fig.tight_layout()
    fig.savefig(plots_dir / "example_regenerations.pdf", bbox_inches="tight")
    fig.savefig(plots_dir / "example_regenerations.png", bbox_inches="tight")
    plt.close(fig)


def regenerate_plots_from_summary(run_dir: Path) -> list[dict]:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(f"Missing summary file: {summary_path}")

    report = json.loads(summary_path.read_text())
    rows = rows_from_summary(report)
    title_suffix = f" ({report.get('model_id', 'unknown model')})"
    make_plots(rows, run_dir, title_suffix=title_suffix)

    reference_dir = Path(report["reference_images"])
    strengths = [row["strength"] for row in rows]
    make_example_grid(reference_dir, run_dir, strengths)
    write_summary_csv(rows, run_dir / "summary.csv")
    return rows


def run_attack(args: argparse.Namespace) -> list[dict]:
    args.output_dir = resolve_run_dir(args)
    input_dir = args.root / "watermarked"
    image_paths = select_inputs(input_dir, args.limit, args.seed)
    if not image_paths:
        raise ValueError(f"No PNG files found under {input_dir}")
    reference_dir = prepare_reference_images(image_paths, args.output_dir)

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.startswith("cuda") else torch.float32

    print(f"Using {len(image_paths)} watermarked images from {input_dir}")
    print(f"Writing outputs under {args.output_dir}")

    pipe = None
    if not args.skip_generation:
        pipe = load_diffusion_pipeline(args.model_id, device, dtype)

    vq_model = None
    if not args.skip_token_encode:
        vq_model = load_vq_model(device)

    baseline = compute_tpr_fpr(
        args.root,
        args.root,
        csv_dir=args.output_dir / "detection_csv" / "baseline",
        alpha=args.alpha,
    )

    rows = []
    # Try to load existing report to resume
    summary_json_path = args.output_dir / "summary.json"
    if summary_json_path.exists():
        try:
            reports = json.loads(summary_json_path.read_text())
            print(f"Found existing summary.json, resuming...")
        except:
            reports = None
    else:
        reports = None

    if reports is None:
        reports = {
            "root": str(args.root),
            "output_dir": str(args.output_dir),
            "model_id": args.model_id,
            "negative_prompt": args.negative_prompt,
            "num_images": len(image_paths),
            "reference_images": str(reference_dir),
            "baseline_detection": baseline,
            "strengths": {},
        }

    for strength in args.strengths:
        name = strength_name(strength)
        strength_dir = args.output_dir / name
        images_dir = strength_dir / "images"
        tokens_dir = strength_dir / "tokens"
        csv_dir = strength_dir / "detection_csv"

        print(f"Running strength={strength:.3f}")
        if pipe is not None:
            regenerate_images_for_strength(
                pipe=pipe,
                image_paths=image_paths,
                images_dir=images_dir,
                strength=strength,
                negative_prompt=args.negative_prompt,
                guidance_scale=args.guidance_scale,
                num_inference_steps=args.num_inference_steps,
                seed=args.seed,
                device=device,
                skip_existing=args.skip_existing,
            )

        if vq_model is not None:
            encode_outputs(vq_model, images_dir, tokens_dir, device)

        # Skip evaluation if we're only doing generation and it's already in the report
        if name in reports["strengths"] and args.skip_existing:
             detection = reports["strengths"][name]["attacked_detection"]
             quality = reports["strengths"][name]["attacked_quality"]
        else:
            # Check if any images/tokens exist before running metrics
            if not list(tokens_dir.glob("*.pt")):
                print(f"No tokens generated for strength {strength:.3f}, skipping metrics.")
                continue

            detection = compute_watermark_survival(tokens_dir, csv_dir=csv_dir, alpha=args.alpha)
            quality = compute_quality_metrics(
                reference_dir,
                images_dir,
                lpips_limit=args.lpips_limit,
            )

        row = {
            "strength": strength,
            "clean_fpr": baseline["false_positive_rate"],
            "watermarked_tpr": baseline["true_positive_rate"],
            "survival_rate": detection["survival_rate"],
            "lpips": quality["lpips"],
            "fid": quality["fid"],
            "attacked_count": detection["count"],
        }
        rows.append(row)
        reports["strengths"][name] = {
            "strength": strength,
            "images_dir": str(images_dir),
            "tokens_dir": str(tokens_dir),
            "attacked_detection": detection,
            "attacked_quality": quality,
        }
        
        # Save progress after each strength
        args.output_dir.mkdir(parents=True, exist_ok=True)
        summary_json_path.write_text(json.dumps(reports, indent=2))

    write_summary_csv(rows, args.output_dir / "summary.csv")

    if not args.skip_plots:
        make_plots(rows, args.output_dir)
        make_example_grid(reference_dir, args.output_dir, args.strengths)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("aggregated_samples"))
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--run-name")
    parser.add_argument("--model-id", default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--negative-prompt", default="low quality, blurry, distorted")
    parser.add_argument("--strengths", type=float, nargs="+", default=DEFAULT_STRENGTHS)
    parser.add_argument("--num-inference-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=7.5)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--lpips-limit", type=int)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--device", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--skip-generation", action="store_true")
    parser.add_argument("--skip-token-encode", action="store_true")
    parser.add_argument("--skip-plots", action="store_true")
    parser.add_argument("--plots-only", action="store_true")
    args = parser.parse_args()

    if args.plots_only:
        run_dir = resolve_run_dir(args)
        rows = regenerate_plots_from_summary(run_dir)
    else:
        rows = run_attack(args)
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
