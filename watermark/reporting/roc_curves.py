"""ROC curves for CGZ watermark detection across all three attacks.

Reads from existing detection CSVs and JSON — no model or GPU required.

Usage:
    python -m watermark.reporting.roc_curves [--output-dir watermark/reporting/outputs]
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[2]

RESULTS_ROOT = REPO_ROOT / "results"
DIFFUSION_RUN = (
    REPO_ROOT
    / "watermark"
    / "attack"
    / "diffusion_regeneration"
    / "outputs"
    / "sd15_final_optimized"
)
ATTACK_COLORS = {
    "no_attack": "#2d6a2d",
    "vqvae": "#f58518",
    "diffusion": "#54a24b",
    "token": "#4c78a8",
}


class Curve(NamedTuple):
    name: str
    fprs: list[float]
    tprs: list[float]
    auc: float
    op_fpr: float | None  # operating point at alpha=0.01 threshold
    op_tpr: float | None
    color: str
    linestyle: str = "-"


def read_csv_scores(path: Path, col: str = "score") -> list[float]:
    with path.open(newline="") as f:
        return [float(row[col]) for row in csv.DictReader(f)]


def read_json_scores(path: Path, col: str) -> list[float]:
    rows = json.loads(path.read_text())
    return [float(row[col]) for row in rows]


def roc_from_scores(
    pos_scores: list[float], neg_scores: list[float]
) -> tuple[list[float], list[float]]:
    """Sweep all unique thresholds; return (fpr_list, tpr_list)."""
    n_pos = len(pos_scores)
    n_neg = len(neg_scores)
    combined = sorted(
        [(s, 1) for s in pos_scores] + [(s, 0) for s in neg_scores],
        key=lambda x: x[0],
        reverse=True,
    )
    tp = fp = 0
    fprs = [0.0]
    tprs = [0.0]
    prev_score = None
    for score, label in combined:
        if prev_score is not None and score != prev_score:
            fprs.append(fp / n_neg)
            tprs.append(tp / n_pos)
        if label == 1:
            tp += 1
        else:
            fp += 1
        prev_score = score
    fprs.append(fp / n_neg)
    tprs.append(tp / n_pos)
    fprs.append(1.0)
    tprs.append(1.0)
    return fprs, tprs


def auc_trapz(fprs: list[float], tprs: list[float]) -> float:
    return sum(
        (fprs[i + 1] - fprs[i]) * (tprs[i + 1] + tprs[i]) / 2
        for i in range(len(fprs) - 1)
    )


def operating_point(
    pos_scores: list[float], neg_scores: list[float], threshold: float
) -> tuple[float, float]:
    tpr = sum(1 for s in pos_scores if s >= threshold) / len(pos_scores)
    fpr = sum(1 for s in neg_scores if s >= threshold) / len(neg_scores)
    return fpr, tpr


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


def plot_roc_curves(curves: list[Curve], output_dir: Path) -> None:
    plt = setup_matplotlib()
    fig, ax = plt.subplots(figsize=(5.2, 4.4))

    ax.plot(
        [0, 1],
        [0, 1],
        color="lightgray",
        linestyle="--",
        linewidth=1,
        label="Random (AUC=0.500)",
        zorder=0,
    )

    for curve in curves:
        ax.plot(
            curve.fprs,
            curve.tprs,
            color=curve.color,
            linestyle=curve.linestyle,
            linewidth=1.6,
            label=f"{curve.name} (AUC={curve.auc:.3f})",
        )
        if curve.op_fpr is not None and curve.op_tpr is not None:
            ax.scatter(
                curve.op_fpr,
                curve.op_tpr,
                color=curve.color,
                s=40,
                zorder=5,
                edgecolor="black",
                linewidth=0.6,
            )

    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate (watermark survival)")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_title("ROC curves — CGZ watermark under attack")
    ax.legend(frameon=False, loc="lower right")

    output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_dir / "roc_curves.pdf")
    fig.savefig(output_dir / "roc_curves.png")
    plt.close(fig)
    print(f"Saved roc_curves.pdf/png to {output_dir}")


def write_auc_csv(curves: list[Curve], output_dir: Path) -> None:
    path = output_dir / "roc_auc_summary.csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["attack", "auc", "op_fpr", "op_tpr"],
        )
        writer.writeheader()
        for c in curves:
            writer.writerow(
                {
                    "attack": c.name,
                    "auc": f"{c.auc:.4f}",
                    "op_fpr": f"{c.op_fpr:.4f}" if c.op_fpr is not None else "",
                    "op_tpr": f"{c.op_tpr:.4f}" if c.op_tpr is not None else "",
                }
            )
    print(f"Saved AUC summary to {path}")


def build_curves(
    clean_csv: Path,
    watermarked_csv: Path,
    attack1_csv: Path,
    attack2_json: Path,
    diffusion_csv: Path,
    diffusion_strength: float = 0.04,
) -> list[Curve]:
    clean_scores = read_csv_scores(clean_csv)
    threshold = 322.2915785436869  # tau at alpha=0.01, L=576, m=4

    # Attack 2 scores come from a JSON (score_after per image)
    attack2_scores: list[float] | None = None
    if attack2_json.exists():
        attack2_scores = read_json_scores(attack2_json, "score_after")

    curves: list[Curve] = []

    # No-attack baseline
    wm_scores = read_csv_scores(watermarked_csv)
    fprs, tprs = roc_from_scores(wm_scores, clean_scores)
    op_fpr, op_tpr = operating_point(wm_scores, clean_scores, threshold)
    curves.append(
        Curve(
            name="No attack",
            fprs=fprs,
            tprs=tprs,
            auc=auc_trapz(fprs, tprs),
            op_fpr=op_fpr,
            op_tpr=op_tpr,
            color=ATTACK_COLORS["no_attack"],
            linestyle="--",
        )
    )

    # Attack 2: VQ-VAE roundtrip
    if attack2_scores is not None:
        fprs, tprs = roc_from_scores(attack2_scores, clean_scores)
        op_fpr, op_tpr = operating_point(attack2_scores, clean_scores, threshold)
        curves.append(
            Curve(
                name="VQ-VAE roundtrip",
                fprs=fprs,
                tprs=tprs,
                auc=auc_trapz(fprs, tprs),
                op_fpr=op_fpr,
                op_tpr=op_tpr,
                color=ATTACK_COLORS["vqvae"],
            )
        )

    # Attack 3: diffusion regeneration at given strength
    if diffusion_csv.exists():
        diff_scores = read_csv_scores(diffusion_csv)
        fprs, tprs = roc_from_scores(diff_scores, clean_scores)
        op_fpr, op_tpr = operating_point(diff_scores, clean_scores, threshold)
        curves.append(
            Curve(
                name=f"Diffusion regen s={diffusion_strength:g}",
                fprs=fprs,
                tprs=tprs,
                auc=auc_trapz(fprs, tprs),
                op_fpr=op_fpr,
                op_tpr=op_tpr,
                color=ATTACK_COLORS["diffusion"],
            )
        )

    # Attack 1: token regeneration
    if attack1_csv.exists():
        tok_scores = read_csv_scores(attack1_csv)
        fprs, tprs = roc_from_scores(tok_scores, clean_scores)
        op_fpr, op_tpr = operating_point(tok_scores, clean_scores, threshold)
        curves.append(
            Curve(
                name="Token regeneration",
                fprs=fprs,
                tprs=tprs,
                auc=auc_trapz(fprs, tprs),
                op_fpr=op_fpr,
                op_tpr=op_tpr,
                color=ATTACK_COLORS["token"],
            )
        )

    return curves


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="watermark/reporting/outputs")
    parser.add_argument(
        "--diffusion-strength",
        type=float,
        default=0.04,
        help="Which diffusion strength to include (must match an existing run dir).",
    )
    parser.add_argument(
        "--diffusion-run",
        default=str(DIFFUSION_RUN.relative_to(REPO_ROOT)),
    )
    parser.add_argument("--results-root", default="results")
    args = parser.parse_args()

    results_root = (REPO_ROOT / args.results_root).resolve()
    diffusion_run = (REPO_ROOT / args.diffusion_run).resolve()
    output_dir = (REPO_ROOT / args.output_dir).resolve()

    strength_name = f"strength_{args.diffusion_strength:.3f}".replace(".", "_")
    diffusion_csv = diffusion_run / strength_name / "detection_csv" / "attacked_detection.csv"

    curves = build_curves(
        clean_csv=results_root / "token_regeneration" / "clean_detection.csv",
        watermarked_csv=results_root / "token_regeneration" / "watermarked_detection.csv",
        attack1_csv=results_root / "token_regeneration" / "attacked_detection.csv",
        attack2_json=results_root / "attack2" / "results.json",
        diffusion_csv=diffusion_csv,
        diffusion_strength=args.diffusion_strength,
    )

    for c in curves:
        print(
            f"  {c.name}: AUC={c.auc:.3f}  "
            f"op=({c.op_fpr:.3f} FPR, {c.op_tpr:.3f} TPR)"
        )

    try:
        plot_roc_curves(curves, output_dir)
    except ImportError as exc:
        print(f"Skipping plot (missing dependency: {exc})")

    write_auc_csv(curves, output_dir)


if __name__ == "__main__":
    main()
