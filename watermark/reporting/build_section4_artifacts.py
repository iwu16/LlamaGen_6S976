"""Build unified Section 4 tables and figures from saved attack artifacts.

This script does not rerun any attack. It reads the existing result files for:

  Attack 1: results/token_regeneration.json and detection CSVs
  Attack 2: results/attack2/results.json and eval_summary.json
  Attack 3: watermark/attack/diffusion_regeneration/outputs/sd15_final_optimized

Outputs are written under watermark/reporting/outputs by default.

The default run is lightweight. If PyTorch is available and the original
watermarked tokens exist, the script also computes diffusion token-change
rates so Attack 2 and Attack 3 can share the latent-disruption axis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DIFFUSION_RUN = (
    REPO_ROOT
    / "watermark"
    / "attack"
    / "diffusion_regeneration"
    / "outputs"
    / "sd15_final_optimized"
)

ATTACK_COLORS = {
    "token": "#4c78a8",
    "vqvae": "#f58518",
    "diffusion": "#54a24b",
}
STATUS_COLORS = {
    "removed": "#5f6368",
    "detected": "#b279a2",
}


@dataclass
class AttackSummary:
    attack: str
    setting: str
    count: int
    survival_rate: float
    removal_rate: float
    mean_score: float | None
    mean_threshold: float | None
    lpips: float | None
    fid: float | None
    mean_tokens_changed_pct: float | None
    note: str


def read_json(path: Path) -> Any:
    with path.open() as handle:
        return json.load(handle)


def read_csv_dicts(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def mean(values: list[float]) -> float | None:
    return sum(values) / len(values) if values else None


def pct(value: float | None) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{100.0 * value:.1f}\\%"


def num(value: float | None, digits: int = 3) -> str:
    if value is None or math.isnan(value):
        return "--"
    return f"{value:.{digits}f}"


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def write_latex_table(summaries: list[AttackSummary], path: Path) -> None:
    lines = [
        "\\begin{tabular}{llrrrrr}",
        "\\toprule",
        "Attack & Setting & Survival & LPIPS & FID & Score & Token change \\\\",
        "\\midrule",
    ]
    for row in summaries:
        lines.append(
            f"{row.attack} & {row.setting} & {pct(row.survival_rate)} & "
            f"{num(row.lpips, 3)} & {num(row.fid, 2)} & "
            f"{num(row.mean_score, 1)} & {num(row.mean_tokens_changed_pct, 1)}\\% \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.write_text("\n".join(lines))


def binomial_ci(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """Wilson interval for a binomial proportion."""
    if n == 0:
        return (float("nan"), float("nan"))
    phat = successes / n
    denom = 1.0 + z * z / n
    center = (phat + z * z / (2.0 * n)) / denom
    radius = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * n)) / n) / denom
    return max(0.0, center - radius), min(1.0, center + radius)


def std(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    m = mean(values)
    assert m is not None
    return math.sqrt(sum((x - m) ** 2 for x in values) / (len(values) - 1))


def sem(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    s = std(values)
    return s / math.sqrt(len(values)) if s is not None else None


def representative_diffusion(
    summaries: list[AttackSummary],
    strength: float,
) -> AttackSummary:
    target = f"strength={strength:g}"
    for row in summaries:
        if row.attack == "Diffusion regeneration" and row.setting == target:
            return row
    raise ValueError(f"Could not find diffusion summary for {target}")


def common_summaries(
    summaries: list[AttackSummary],
    representative_strength: float,
) -> list[AttackSummary]:
    return [
        next(row for row in summaries if row.attack == "Token regeneration"),
        next(row for row in summaries if row.attack == "VQ-VAE roundtrip"),
        representative_diffusion(summaries, representative_strength),
    ]


def write_access_table(path: Path) -> None:
    rows = [
        {
            "attack": "Token regeneration",
            "input_required": "class label / generation condition",
            "uses_ar_model": "yes",
            "ar_queries_per_image": 576,
            "uses_vqvae_encoder": "no",
            "uses_diffusion_model": "no",
            "uses_secret_key": "no",
            "preserves_original_image": "no",
            "access_note": "Fresh clean class-conditional sample; exact CGZ-style latent removal.",
        },
        {
            "attack": "VQ-VAE roundtrip",
            "input_required": "watermarked image",
            "uses_ar_model": "no",
            "ar_queries_per_image": 0,
            "uses_vqvae_encoder": "yes",
            "uses_diffusion_model": "no",
            "uses_secret_key": "no",
            "preserves_original_image": "yes",
            "access_note": "Zero AR queries; tests tokenizer bottleneck directly.",
        },
        {
            "attack": "Diffusion regeneration",
            "input_required": "watermarked image",
            "uses_ar_model": "no",
            "ar_queries_per_image": 0,
            "uses_vqvae_encoder": "yes, for detection/evaluation",
            "uses_diffusion_model": "yes",
            "uses_secret_key": "no",
            "preserves_original_image": "strength-dependent",
            "access_note": "Zero AR queries; uses external pixel-space generative model.",
        },
    ]
    write_csv(rows, path.with_suffix(".csv"))
    lines = [
        "\\begin{tabular}{lrrrrl}",
        "\\toprule",
        "Attack & AR queries & AR model & VQ encoder & Diffusion & Preserves original \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['attack']} & {row['ar_queries_per_image']} & "
            f"{row['uses_ar_model']} & {row['uses_vqvae_encoder']} & "
            f"{row['uses_diffusion_model']} & {row['preserves_original_image']} \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    path.with_suffix(".tex").write_text("\n".join(lines))


def load_attack1(results_root: Path) -> tuple[AttackSummary, list[dict[str, str]]]:
    report = read_json(results_root / "token_regeneration.json")
    attacked_rows = read_csv_dicts(results_root / "token_regeneration" / "attacked_detection.csv")
    detection = report["attacked_detection"]
    quality = report["attacked_quality"]
    summary = AttackSummary(
        attack="Token regeneration",
        setting="fresh clean sample",
        count=int(detection["count"]),
        survival_rate=float(detection["survival_rate"]),
        removal_rate=1.0 - float(detection["survival_rate"]),
        mean_score=float(detection["mean_score"]),
        mean_threshold=float(detection["mean_threshold"]),
        lpips=float(quality["lpips"]),
        fid=float(quality["fid"]),
        mean_tokens_changed_pct=100.0,
        note="Fresh class-conditional sample; token change is not a preservation metric.",
    )
    return summary, attacked_rows


def load_attack2(results_root: Path) -> tuple[AttackSummary, list[dict[str, Any]]]:
    rows = read_json(results_root / "attack2" / "results.json")
    eval_summary = read_json(results_root / "attack2" / "eval_summary.json")
    summary = AttackSummary(
        attack="VQ-VAE roundtrip",
        setting="decode/re-encode",
        count=len(rows),
        survival_rate=float(eval_summary["watermark_survival_rate"]),
        removal_rate=1.0 - float(eval_summary["watermark_survival_rate"]),
        mean_score=mean([float(row["score_after"]) for row in rows]),
        mean_threshold=float(rows[0]["threshold"]) if rows else None,
        lpips=mean([float(row["lpips"]) for row in rows]),
        fid=float(eval_summary["fid"]),
        mean_tokens_changed_pct=mean([float(row["tokens_changed_pct"]) for row in rows]),
        note="Per-image LPIPS, token changes, and detection scores are available.",
    )
    return summary, rows


def load_token_sequence(path: Path) -> list[int]:
    try:
        import torch
    except ImportError as exc:
        raise RuntimeError("PyTorch is required to load token .pt files") from exc

    value = torch.load(path, map_location="cpu")
    if hasattr(value, "detach"):
        value = value.detach().cpu().flatten().tolist()
    elif isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def original_wm_token_path(tokens_root: Path, attacked_token_path: Path) -> Path | None:
    stem = attacked_token_path.stem
    parts = stem.rsplit("_", 1)
    if len(parts) != 2:
        return None
    owner, idx = parts
    candidate = tokens_root / f"{owner}_wm_{idx}.pt"
    return candidate if candidate.exists() else None


def compute_diffusion_token_change(
    diffusion_run: Path,
    original_tokens_root: Path,
) -> dict[str, dict[str, Any]]:
    """Return mean token-change statistics by strength name."""
    results: dict[str, dict[str, Any]] = {}
    for strength_dir in sorted(diffusion_run.glob("strength_*")):
        tokens_dir = strength_dir / "tokens"
        if not tokens_dir.exists():
            continue
        rates: list[float] = []
        changed_counts: list[int] = []
        per_image: list[dict[str, Any]] = []
        for attacked_path in sorted(tokens_dir.glob("*.pt")):
            original_path = original_wm_token_path(original_tokens_root, attacked_path)
            if original_path is None:
                continue
            attacked = load_token_sequence(attacked_path)
            original = load_token_sequence(original_path)
            n = min(len(attacked), len(original))
            if n == 0:
                continue
            changed = sum(1 for a, b in zip(attacked[:n], original[:n]) if a != b)
            changed_counts.append(changed)
            rate = 100.0 * changed / n
            rates.append(rate)
            per_image.append(
                {
                    "path": str(attacked_path),
                    "tokens_changed": changed,
                    "tokens_changed_pct": rate,
                }
            )
        results[strength_dir.name] = {
            "count": float(len(rates)),
            "mean_tokens_changed": mean([float(x) for x in changed_counts]) or float("nan"),
            "mean_tokens_changed_pct": mean(rates) or float("nan"),
            "per_image": per_image,
        }
    return results


def load_attack3(
    diffusion_run: Path,
    original_tokens_root: Path,
    compute_token_change: bool,
) -> tuple[list[AttackSummary], dict[str, list[dict[str, str]]], dict[str, dict[str, Any]]]:
    summary_rows = read_csv_dicts(diffusion_run / "summary.csv")
    detection_by_strength: dict[str, list[dict[str, str]]] = {}
    token_change: dict[str, dict[str, Any]] = {}
    if compute_token_change:
        try:
            token_change = compute_diffusion_token_change(diffusion_run, original_tokens_root)
        except Exception as exc:
            print(f"Warning: could not compute diffusion token-change rates: {exc}")
            token_change = {}

    summaries: list[AttackSummary] = []
    for row in summary_rows:
        strength = float(row["strength"])
        strength_name = f"strength_{strength:.3f}".replace(".", "_")
        detection_path = diffusion_run / strength_name / "detection_csv" / "attacked_detection.csv"
        detection_rows = read_csv_dicts(detection_path) if detection_path.exists() else []
        detection_by_strength[strength_name] = detection_rows
        scores = [float(item["score"]) for item in detection_rows]
        thresholds = [float(item["threshold"]) for item in detection_rows]
        change_stats = token_change.get(strength_name, {})
        summaries.append(
            AttackSummary(
                attack="Diffusion regeneration",
                setting=f"strength={strength:g}",
                count=int(row["attacked_count"]),
                survival_rate=float(row["survival_rate"]),
                removal_rate=1.0 - float(row["survival_rate"]),
                mean_score=mean(scores),
                mean_threshold=mean(thresholds),
                lpips=float(row["lpips"]),
                fid=float(row["fid"]),
                mean_tokens_changed_pct=change_stats.get("mean_tokens_changed_pct"),
                note="Per-strength aggregate LPIPS/FID; per-image scores available.",
            )
        )
    return summaries, detection_by_strength, token_change


def setup_matplotlib():
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.labelsize": 8.5,
            "axes.titlesize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "figure.dpi": 160,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    return plt


def paired_lpips_scores(
    reference_dir: Path,
    sample_dir: Path,
    limit: int | None = None,
    device: str | None = None,
) -> list[dict[str, Any]]:
    try:
        import torch
        import lpips
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Install torch, lpips, pillow, and torchvision to compute LPIPS") from exc

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = lpips.LPIPS(net="alex").to(device).eval()
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x * 2.0 - 1.0),
        ]
    )
    names = sorted(path.name for path in reference_dir.glob("*.png"))
    if limit is not None:
        names = names[:limit]

    rows: list[dict[str, Any]] = []
    with torch.no_grad():
        for name in names:
            ref_path = reference_dir / name
            sample_path = sample_dir / name
            if not sample_path.exists():
                continue
            ref = transform(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)
            sample = transform(Image.open(sample_path).convert("RGB")).unsqueeze(0).to(device)
            rows.append({"path": str(sample_path), "lpips": float(model(ref, sample).item())})
    if not rows:
        raise ValueError(f"No matching PNG files found in {reference_dir} and {sample_dir}")
    return rows


def save_figure(fig, output_dir: Path, name: str) -> None:
    fig.savefig(output_dir / f"{name}.png")
    fig.savefig(output_dir / f"{name}.pdf")


def panel_label(ax, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def plot_quality_removal(summaries: list[AttackSummary], output_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.8, 3.8))
    colors = {
        "Token regeneration": ATTACK_COLORS["token"],
        "VQ-VAE roundtrip": ATTACK_COLORS["vqvae"],
        "Diffusion regeneration": ATTACK_COLORS["diffusion"],
    }

    for attack in ["Token regeneration", "VQ-VAE roundtrip"]:
        row = next(item for item in summaries if item.attack == attack)
        ax.scatter(
            row.lpips,
            row.survival_rate,
            s=90,
            color=colors[attack],
            edgecolor="black",
            linewidth=0.7,
            label=attack,
            zorder=3,
        )
        ax.annotate(
            attack.replace(" ", "\n"),
            (row.lpips, row.survival_rate),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )

    diffusion = [row for row in summaries if row.attack == "Diffusion regeneration"]
    diffusion = sorted(diffusion, key=lambda row: float(row.setting.split("=")[1]))
    ax.plot(
        [row.lpips for row in diffusion],
        [row.survival_rate for row in diffusion],
        marker="o",
        color=colors["Diffusion regeneration"],
        label="Diffusion regeneration",
    )
    for row in diffusion:
        ax.annotate(
            row.setting.replace("strength=", "s="),
            (row.lpips, row.survival_rate),
            textcoords="offset points",
            xytext=(4, -12),
            fontsize=8,
        )

    ax.set_xlabel("LPIPS vs original watermarked image")
    ax.set_ylabel("Watermark survival rate")
    ax.set_ylim(-0.04, 1.04)
    ax.set_title("Removal-quality tradeoff across attacks")
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    save_figure(fig, output_dir, "unified_quality_survival_tradeoff")
    plt.close(fig)


def plot_common_quality_removal(
    summaries: list[AttackSummary],
    output_dir: Path,
    representative_strength: float,
) -> None:
    plt = setup_matplotlib()
    rows = common_summaries(summaries, representative_strength)
    colors = [ATTACK_COLORS["token"], ATTACK_COLORS["vqvae"], ATTACK_COLORS["diffusion"]]
    labels = [
        "Token regeneration",
        "VQ-VAE roundtrip",
        f"Diffusion regen, s={representative_strength:g}",
    ]

    fig, ax = plt.subplots(figsize=(4.8, 3.1), constrained_layout=True)
    for row, color, label in zip(rows, colors, labels):
        ax.scatter(
            row.lpips,
            row.survival_rate,
            s=70,
            color=color,
            edgecolor="black",
            linewidth=0.7,
            zorder=3,
            label=label,
        )
    ax.set_xscale("log")
    ax.set_xlim(0.015, 0.9)
    ax.set_xlabel("LPIPS vs original watermarked image")
    ax.set_ylabel("Watermark survival rate")
    ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, loc="center right")
    save_figure(fig, output_dir, "common_quality_survival_tradeoff")
    plt.close(fig)


def plot_common_access_survival(
    summaries: list[AttackSummary],
    output_dir: Path,
    representative_strength: float,
) -> None:
    plt = setup_matplotlib()
    rows = common_summaries(summaries, representative_strength)
    names = ["Token\nregen", "VQ-VAE\nroundtrip", f"Diffusion\ns={representative_strength:g}"]
    ar_queries = [576, 0, 0]
    colors = [ATTACK_COLORS["token"], ATTACK_COLORS["vqvae"], ATTACK_COLORS["diffusion"]]

    fig, ax = plt.subplots(figsize=(4.8, 3.1), constrained_layout=True)
    ax.scatter(ar_queries, [row.survival_rate for row in rows], s=75, c=colors, edgecolor="black")
    for x, row, name in zip(ar_queries, rows, names):
        ax.annotate(
            name,
            (x, row.survival_rate),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7.5,
        )
    ax.set_xlabel("Autoregressive model queries per image")
    ax.set_ylabel("Watermark survival rate")
    ax.set_ylim(-0.05, 1.05)
    ax.set_xscale("symlog", linthresh=1)
    ax.set_xticks([0, 1, 10, 100, 576])
    ax.set_xticklabels(["0", "1", "10", "100", "576"])
    save_figure(fig, output_dir, "common_access_vs_survival")
    plt.close(fig)


def plot_common_comparison_panel(
    summaries: list[AttackSummary],
    output_dir: Path,
    representative_strength: float,
) -> None:
    plt = setup_matplotlib()
    rows = common_summaries(summaries, representative_strength)
    colors = [ATTACK_COLORS["token"], ATTACK_COLORS["vqvae"], ATTACK_COLORS["diffusion"]]
    labels = ["Token regen", "VQ-VAE", f"Diffusion s={representative_strength:g}"]

    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0), constrained_layout=True)
    ax = axes[0]
    for row, color, label in zip(rows, colors, labels):
        ax.scatter(row.lpips, row.survival_rate, s=65, color=color, edgecolor="black", label=label)
    ax.set_xscale("log")
    ax.set_xlim(0.015, 0.9)
    ax.set_xticks([0.02, 0.05, 0.1, 0.2, 0.5])
    ax.set_xticklabels(["0.02", "0.05", "0.10", "0.20", "0.50"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("LPIPS")
    ax.set_ylabel("Watermark survival")
    ax.legend(frameon=False, loc="center right")
    panel_label(ax, "A")

    ax = axes[1]
    for row, color, label in zip(rows, colors, labels):
        ax.scatter(row.fid, row.survival_rate, s=65, color=color, edgecolor="black", label=label)
    ax.set_xlim(0, max(row.fid for row in rows if row.fid is not None) * 1.12)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("FID vs original watermarked images")
    ax.set_ylabel("Watermark survival")
    panel_label(ax, "B")

    save_figure(fig, output_dir, "common_comparison_panel")
    plt.close(fig)


def plot_diffusion_hyperparameter_sweep(
    diffusion_summaries: list[AttackSummary],
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    rows = sorted(diffusion_summaries, key=lambda row: float(row.setting.split("=")[1]))
    strengths = [float(row.setting.split("=")[1]) for row in rows]

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), constrained_layout=True)
    axes[0].plot(strengths, [row.survival_rate for row in rows], marker="o", color=ATTACK_COLORS["diffusion"])
    axes[0].set_xlabel("Diffusion strength")
    axes[0].set_ylabel("Watermark survival")
    axes[0].set_ylim(-0.05, 1.05)
    panel_label(axes[0], "A")

    axes[1].plot(strengths, [row.lpips for row in rows], marker="o", color=ATTACK_COLORS["diffusion"], label="LPIPS")
    axes[1].set_xlabel("Diffusion strength")
    axes[1].set_ylabel("LPIPS")
    panel_label(axes[1], "B")

    axes[2].plot(
        strengths,
        [row.mean_tokens_changed_pct for row in rows],
        marker="o",
        color=ATTACK_COLORS["diffusion"],
        label="Token change",
    )
    axes[2].set_xlabel("Diffusion strength")
    axes[2].set_ylabel("Tokens changed (%)")
    panel_label(axes[2], "C")

    save_figure(fig, output_dir, "diffusion_strength_sweep")
    plt.close(fig)


def plot_diffusion_quality_tradeoff(
    diffusion_summaries: list[AttackSummary],
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    rows = sorted(diffusion_summaries, key=lambda row: float(row.setting.split("=")[1]))
    strengths = [float(row.setting.split("=")[1]) for row in rows]

    fig, ax1 = plt.subplots(figsize=(4.8, 3.1), constrained_layout=True)
    ax1.plot(
        [row.lpips for row in rows],
        [row.survival_rate for row in rows],
        marker="o",
        color=ATTACK_COLORS["diffusion"],
        linewidth=2,
    )
    for strength, row in zip(strengths, rows):
        ax1.annotate(
            f"s={strength:g}",
            (row.lpips, row.survival_rate),
            textcoords="offset points",
            xytext=(5, 5),
            fontsize=8,
        )
    ax1.set_xlabel("LPIPS vs original watermarked image")
    ax1.set_ylabel("Watermark survival rate")
    ax1.set_ylim(-0.05, 1.05)
    save_figure(fig, output_dir, "attack3_quality_removal_tradeoff")
    plt.close(fig)


def plot_diffusion_fid_by_strength(
    diffusion_summaries: list[AttackSummary],
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    rows = sorted(diffusion_summaries, key=lambda row: float(row.setting.split("=")[1]))
    strengths = [float(row.setting.split("=")[1]) for row in rows]

    fig, ax = plt.subplots(figsize=(4.8, 3.1), constrained_layout=True)
    ax.plot(strengths, [row.fid for row in rows], marker="o", color=ATTACK_COLORS["diffusion"], linewidth=2)
    ax.set_xlabel("Diffusion img2img strength")
    ax.set_ylabel("FID vs original watermarked images")
    save_figure(fig, output_dir, "attack3_fid_by_strength")
    plt.close(fig)


def plot_individual_attack1_scores(
    attack1_rows: list[dict[str, str]],
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    scores = [float(row["score"]) for row in attack1_rows]
    threshold = float(attack1_rows[0]["threshold"])
    fig, ax = plt.subplots(figsize=(4.8, 3.0), constrained_layout=True)
    ax.hist(scores, bins=30, color=ATTACK_COLORS["token"], edgecolor="white")
    ax.axvline(286, color="#d62728", linestyle=":", linewidth=1.4, label="Clean mean")
    ax.axvline(threshold, color="black", linestyle="--", linewidth=1.2, label="Threshold")
    ax.set_xlabel("Detection score")
    ax.set_ylabel("Images")
    ax.legend(frameon=False)
    save_figure(fig, output_dir, "individual_attack1_score_histogram")
    save_figure(fig, output_dir, "attack1_scores")
    plt.close(fig)


def plot_individual_attack2_diagnostics(
    attack2_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    lpips_scores = [float(row["lpips"]) for row in attack2_rows]
    tokens_changed = [float(row["tokens_changed_pct"]) for row in attack2_rows]

    fig, axes = plt.subplots(1, 2, figsize=(5.4, 2.6), constrained_layout=True)
    axes[0].hist(lpips_scores, bins=30, color=ATTACK_COLORS["vqvae"], edgecolor="white")
    axes[0].axvline(mean(lpips_scores), color="black", linestyle="--", linewidth=1)
    axes[0].set_xlabel("LPIPS")
    axes[0].set_ylabel("Images")
    panel_label(axes[0], "A")

    axes[1].hist(tokens_changed, bins=30, color=ATTACK_COLORS["vqvae"], edgecolor="white")
    axes[1].axvline(mean(tokens_changed), color="black", linestyle="--", linewidth=1)
    axes[1].set_xlabel("Tokens changed (%)")
    panel_label(axes[1], "B")

    save_figure(fig, output_dir, "individual_attack2_diagnostics")
    save_figure(fig, output_dir, "attack2_plots")
    plt.close(fig)


def plot_individual_attack2_lpips_bins(
    attack2_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, float("inf")]
    labels = ["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04", "0.04-0.05", "0.05-0.07", "0.07-0.10", "0.10+"]
    rates: list[float] = []
    counts: list[int] = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        rows = [row for row in attack2_rows if lo <= float(row["lpips"]) < hi]
        counts.append(len(rows))
        rates.append(sum(bool(row["detected_after"]) for row in rows) / len(rows) if rows else 0.0)

    fig, ax = plt.subplots(figsize=(5.8, 3.0), constrained_layout=True)
    bars = ax.bar(labels, rates, color=ATTACK_COLORS["vqvae"], edgecolor="black", linewidth=0.5)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"n={count}", ha="center", fontsize=8)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1)
    ax.set_ylim(0, 1.08)
    ax.set_xlabel("LPIPS bin")
    ax.set_ylabel("Watermark survival rate")
    ax.tick_params(axis="x", labelrotation=20)
    save_figure(fig, output_dir, "individual_attack2_lpips_bins")
    save_figure(fig, output_dir, "attack2_tradeoff")
    plt.close(fig)


def plot_detection_scores(
    attack1_rows: list[dict[str, str]],
    attack2_rows: list[dict[str, Any]],
    diffusion_detection: dict[str, list[dict[str, str]]],
    output_dir: Path,
    representative_strength: float,
) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True, constrained_layout=True)

    attack1_scores = [float(row["score"]) for row in attack1_rows]
    threshold = float(attack1_rows[0]["threshold"]) if attack1_rows else 322.2916
    axes[0].hist(attack1_scores, bins=30, color=ATTACK_COLORS["token"], edgecolor="white")
    axes[0].axvline(threshold, color="black", linestyle="--", linewidth=1)
    axes[0].axvline(286, color="#d62728", linestyle=":", linewidth=1)
    axes[0].set_title("Token regen")
    axes[0].set_xlabel("Detection score")
    axes[0].set_ylabel("Images")
    panel_label(axes[0], "A")

    attack2_scores = [float(row["score_after"]) for row in attack2_rows]
    threshold2 = float(attack2_rows[0]["threshold"]) if attack2_rows else threshold
    axes[1].hist(attack2_scores, bins=30, color=ATTACK_COLORS["vqvae"], edgecolor="white")
    axes[1].axvline(threshold2, color="black", linestyle="--", linewidth=1)
    axes[1].axvline(286, color="#d62728", linestyle=":", linewidth=1)
    axes[1].set_title("VQ-VAE roundtrip")
    axes[1].set_xlabel("Detection score")
    panel_label(axes[1], "B")

    strength_name = f"strength_{representative_strength:.3f}".replace(".", "_")
    diffusion_scores = [float(row["score"]) for row in diffusion_detection.get(strength_name, [])]
    axes[2].hist(diffusion_scores, bins=30, color=ATTACK_COLORS["diffusion"], edgecolor="white")
    axes[2].axvline(threshold, color="black", linestyle="--", linewidth=1)
    axes[2].axvline(286, color="#d62728", linestyle=":", linewidth=1)
    axes[2].set_title(f"Diffusion s={representative_strength:g}")
    axes[2].set_xlabel("Detection score")
    panel_label(axes[2], "C")

    save_figure(fig, output_dir, "post_attack_detection_scores")
    plt.close(fig)


def plot_latent_disruption(
    attack2_rows: list[dict[str, Any]],
    diffusion_detection: dict[str, list[dict[str, str]]],
    diffusion_token_change: dict[str, dict[str, Any]],
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.6), sharey=True, constrained_layout=True)
    threshold = float(attack2_rows[0]["threshold"]) if attack2_rows else 322.2916

    survived = [bool(row["detected_after"]) for row in attack2_rows]
    colors = [STATUS_COLORS["detected"] if item else STATUS_COLORS["removed"] for item in survived]
    axes[0].scatter(
        [float(row["tokens_changed_pct"]) for row in attack2_rows],
        [float(row["score_after"]) for row in attack2_rows],
        c=colors,
        s=12,
        alpha=0.45,
        linewidth=0,
    )
    axes[0].axhline(threshold, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("VQ-VAE", color=ATTACK_COLORS["vqvae"])
    axes[0].set_xlabel("Tokens changed (%)")
    axes[0].set_ylabel("Detection score")
    panel_label(axes[0], "A")

    for ax, strength, label in zip(axes[1:], [0.02, 0.04], ["B", "C"]):
        strength_name = f"strength_{strength:.3f}".replace(".", "_")
        detection_by_path = {
            Path(row["path"]).name: row
            for row in diffusion_detection.get(strength_name, [])
        }
        xs: list[float] = []
        ys: list[float] = []
        point_colors: list[str] = []
        for item in diffusion_token_change.get(strength_name, {}).get("per_image", []):
            detection = detection_by_path.get(Path(item["path"]).name)
            if detection is None:
                continue
            xs.append(float(item["tokens_changed_pct"]))
            ys.append(float(detection["score"]))
            point_colors.append(
                STATUS_COLORS["detected"]
                if detection["detected"] == "True"
                else STATUS_COLORS["removed"]
            )
        ax.scatter(xs, ys, c=point_colors, s=12, alpha=0.45, linewidth=0)
        ax.axhline(threshold, color="black", linestyle="--", linewidth=1)
        ax.set_title(f"Diffusion s={strength:g}", color=ATTACK_COLORS["diffusion"])
        ax.set_xlabel("Tokens changed (%)")
        ax.set_xlim(0, 100)
        panel_label(ax, label)
    axes[0].set_xlim(0, 100)

    save_figure(fig, output_dir, "latent_disruption_vs_detection")
    plt.close(fig)


def plot_attack_quality_diagnostics(
    attack2_rows: list[dict[str, Any]],
    diffusion_token_change: dict[str, dict[str, Any]],
    diffusion_lpips_rows: list[dict[str, Any]],
    output_dir: Path,
) -> None:
    plt = setup_matplotlib()
    vq_lpips = [float(row["lpips"]) for row in attack2_rows]
    vq_tokens = [float(row["tokens_changed_pct"]) for row in attack2_rows]
    diffusion_lpips = [float(row["lpips"]) for row in diffusion_lpips_rows]
    diffusion_tokens = [
        float(row["tokens_changed_pct"])
        for row in diffusion_token_change.get("strength_0_040", {}).get("per_image", [])
    ]

    fig, axes = plt.subplots(2, 2, figsize=(7.0, 4.8), constrained_layout=True)
    panels = [
        (axes[0, 0], vq_lpips, "VQ-VAE LPIPS", "LPIPS", ATTACK_COLORS["vqvae"], "A"),
        (axes[0, 1], vq_tokens, "VQ-VAE token change", "Tokens changed (%)", ATTACK_COLORS["vqvae"], "B"),
        (axes[1, 0], diffusion_lpips, "Diffusion s=0.04 LPIPS", "LPIPS", ATTACK_COLORS["diffusion"], "C"),
        (axes[1, 1], diffusion_tokens, "Diffusion s=0.04 token change", "Tokens changed (%)", ATTACK_COLORS["diffusion"], "D"),
    ]
    for ax, values, title, xlabel, color, label in panels:
        ax.hist(values, bins=30, color=color, edgecolor="white")
        avg = mean(values)
        if avg is not None:
            ax.axvline(avg, color="black", linestyle="--", linewidth=1)
        ax.set_title(title, color=color)
        ax.set_xlabel(xlabel)
        ax.set_ylabel("Images")
        panel_label(ax, label)
    axes[0, 1].set_xlim(0, 100)
    axes[1, 1].set_xlim(0, 100)

    save_figure(fig, output_dir, "appendix_quality_diagnostics")
    plt.close(fig)


def metric_uncertainty_rows(
    summaries: list[AttackSummary],
    attack1_rows: list[dict[str, str]],
    attack2_rows: list[dict[str, Any]],
    diffusion_detection: dict[str, list[dict[str, str]]],
) -> list[dict[str, Any]]:
    rows_by_label: dict[tuple[str, str], dict[str, Any]] = {}
    for row in summaries:
        successes = int(round(row.survival_rate * row.count))
        lo, hi = binomial_ci(successes, row.count)
        rows_by_label[(row.attack, row.setting)] = {
            "attack": row.attack,
            "setting": row.setting,
            "count": row.count,
            "survival_rate": row.survival_rate,
            "survival_ci95_low": lo,
            "survival_ci95_high": hi,
            "score_sem": None,
            "lpips_sem": None,
            "tokens_changed_pct_sem": None,
        }

    rows_by_label[("Token regeneration", "fresh clean sample")]["score_sem"] = sem(
        [float(row["score"]) for row in attack1_rows]
    )
    rows_by_label[("VQ-VAE roundtrip", "decode/re-encode")]["score_sem"] = sem(
        [float(row["score_after"]) for row in attack2_rows]
    )
    rows_by_label[("VQ-VAE roundtrip", "decode/re-encode")]["lpips_sem"] = sem(
        [float(row["lpips"]) for row in attack2_rows]
    )
    rows_by_label[("VQ-VAE roundtrip", "decode/re-encode")]["tokens_changed_pct_sem"] = sem(
        [float(row["tokens_changed_pct"]) for row in attack2_rows]
    )

    for strength_name, detection_rows in diffusion_detection.items():
        strength = float(strength_name.replace("strength_", "").replace("_", "."))
        key = ("Diffusion regeneration", f"strength={strength:g}")
        if key in rows_by_label:
            rows_by_label[key]["score_sem"] = sem([float(row["score"]) for row in detection_rows])

    return list(rows_by_label.values())


def build(args: argparse.Namespace) -> None:
    results_root = (REPO_ROOT / args.results_root).resolve()
    diffusion_run = (REPO_ROOT / args.diffusion_run).resolve()
    original_tokens_root = (REPO_ROOT / args.original_tokens_root).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    attack1_summary, attack1_rows = load_attack1(results_root)
    attack2_summary, attack2_rows = load_attack2(results_root)
    attack3_summaries, diffusion_detection, diffusion_token_change = load_attack3(
        diffusion_run,
        original_tokens_root,
        compute_token_change=not args.skip_diffusion_token_change,
    )
    summaries = [attack1_summary, attack2_summary, *attack3_summaries]
    common = common_summaries(summaries, args.representative_diffusion_strength)

    summary_dicts = [asdict(row) for row in summaries]
    common_dicts = [asdict(row) for row in common]
    uncertainty_rows = metric_uncertainty_rows(
        summaries,
        attack1_rows,
        attack2_rows,
        diffusion_detection,
    )
    write_csv(summary_dicts, output_dir / "all_attack_settings_summary.csv")
    write_latex_table(summaries, output_dir / "all_attack_settings_summary.tex")
    (output_dir / "all_attack_settings_summary.json").write_text(json.dumps(summary_dicts, indent=2))
    write_csv(common_dicts, output_dir / "common_three_attack_summary.csv")
    write_latex_table(common, output_dir / "common_three_attack_summary.tex")
    (output_dir / "common_three_attack_summary.json").write_text(json.dumps(common_dicts, indent=2))
    write_csv(uncertainty_rows, output_dir / "statistical_uncertainty.csv")
    write_access_table(output_dir / "attack_access_table")
    if diffusion_token_change:
        (output_dir / "diffusion_token_change.json").write_text(
            json.dumps(diffusion_token_change, indent=2)
        )

    if not args.skip_plots:
        try:
            diffusion_lpips_path = output_dir / "diffusion_strength_0_040_lpips.json"
            if diffusion_lpips_path.exists():
                diffusion_lpips_rows = read_json(diffusion_lpips_path)
            else:
                diffusion_lpips_rows = paired_lpips_scores(
                    diffusion_run / "reference_watermarked",
                    diffusion_run / "strength_0_040" / "images",
                )
                diffusion_lpips_path.write_text(json.dumps(diffusion_lpips_rows, indent=2))
            plot_common_quality_removal(
                summaries,
                output_dir,
                representative_strength=args.representative_diffusion_strength,
            )
            plot_common_access_survival(
                summaries,
                output_dir,
                representative_strength=args.representative_diffusion_strength,
            )
            plot_common_comparison_panel(
                summaries,
                output_dir,
                representative_strength=args.representative_diffusion_strength,
            )
            plot_quality_removal(summaries, output_dir)
            plot_diffusion_hyperparameter_sweep(attack3_summaries, output_dir)
            plot_diffusion_quality_tradeoff(attack3_summaries, output_dir)
            plot_diffusion_fid_by_strength(attack3_summaries, output_dir)
            plot_detection_scores(
                attack1_rows,
                attack2_rows,
                diffusion_detection,
                output_dir,
                representative_strength=args.representative_diffusion_strength,
            )
            plot_latent_disruption(
                attack2_rows,
                diffusion_detection,
                diffusion_token_change,
                output_dir,
            )
            plot_attack_quality_diagnostics(
                attack2_rows,
                diffusion_token_change,
                diffusion_lpips_rows,
                output_dir,
            )
            plot_individual_attack1_scores(attack1_rows, output_dir)
            plot_individual_attack2_diagnostics(attack2_rows, output_dir)
            plot_individual_attack2_lpips_bins(attack2_rows, output_dir)
        except ImportError as exc:
            print(
                "Warning: skipping plots because a plotting dependency is missing. "
                f"Install it or run inside the project environment. Details: {exc}"
            )

    print(f"Wrote Section 4 artifacts to {output_dir.relative_to(REPO_ROOT)}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-root", default="results")
    parser.add_argument(
        "--diffusion-run",
        default=str(DEFAULT_DIFFUSION_RUN.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--original-tokens-root", default="aggregated_samples/tokens")
    parser.add_argument("--output-dir", default="watermark/reporting/outputs")
    parser.add_argument("--representative-diffusion-strength", type=float, default=0.04)
    parser.add_argument(
        "--skip-diffusion-token-change",
        action="store_true",
        help="Skip optional token-change computation for diffusion outputs.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Write summary tables/JSON only.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    build(parse_args())
