"""
Attack 3: diffusion regeneration.

Only the teammate assigned to this attack should edit this folder.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from watermark.evaluate import (
    compute_quality_metrics,
    compute_tpr_fpr,
    compute_watermark_survival,
)


def run_attack(root: Path, output_dir: Path) -> None:
    images_dir = output_dir / "images"
    tokens_dir = output_dir / "tokens"
    images_dir.mkdir(parents=True, exist_ok=True)
    tokens_dir.mkdir(parents=True, exist_ok=True)

    raise NotImplementedError("Implement diffusion regeneration here.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("aggregated_samples"))
    parser.add_argument("--output-dir", type=Path, default=Path("attack_outputs/diffusion_regeneration"))
    parser.add_argument("--results", type=Path, default=Path("results/diffusion_regeneration.json"))
    args = parser.parse_args()

    run_attack(args.root, args.output_dir)
    report = {
        "baseline_detection": compute_tpr_fpr(
            args.root,
            args.root,
            csv_dir=Path("results/detection_csv/diffusion_regeneration"),
        ),
        "attacked_detection": compute_watermark_survival(
            args.output_dir / "tokens",
            csv_dir=Path("results/detection_csv/diffusion_regeneration"),
        ),
        "attacked_quality": compute_quality_metrics(
            args.root / "watermarked",
            args.output_dir / "images",
        ),
    }
    args.results.parent.mkdir(parents=True, exist_ok=True)
    args.results.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
