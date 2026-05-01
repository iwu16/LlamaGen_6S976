"""Shared evaluation functions for CGZ watermark experiments.

This file intentionally contains small metric-level helpers, not one global
experiment runner. Each attack folder should import the functions it needs and
compose its own report.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path

from watermark.metrics import (
    SECRET_KEY,
    compute_fid,
    compute_lpips,
    detect_token_file,
    detection_accuracy,
    watermark_survival_rate,
    write_detection_csv,
)


def token_files(root: Path, prefix: str) -> list[Path]:
    return sorted((root / "tokens").glob(f"*_{prefix}_*.pt"))


def attacked_token_files(path: Path) -> list[Path]:
    return sorted(path.glob("*.pt"))


def compute_tpr_fpr(
    clean_tokens: Path | list[Path],
    watermarked_tokens: Path | list[Path],
    csv_dir: Path | None = None,
    alpha: float = 0.01,
) -> dict:
    """Compute detector TPR on watermarked tokens and FPR on clean tokens."""
    clean_files = (
        token_files(clean_tokens, "clean")
        if isinstance(clean_tokens, Path) and clean_tokens.is_dir()
        else list(clean_tokens)
    )
    wm_files = (
        token_files(watermarked_tokens, "wm")
        if isinstance(watermarked_tokens, Path) and watermarked_tokens.is_dir()
        else list(watermarked_tokens)
    )

    clean_rows = [
        detect_token_file(path, secret_key=SECRET_KEY, alpha=alpha)
        for path in clean_files
    ]
    wm_rows = [
        detect_token_file(path, secret_key=SECRET_KEY, alpha=alpha)
        for path in wm_files
    ]

    baseline = detection_accuracy(
        clean_files,
        wm_files,
        secret_key=SECRET_KEY,
        alpha=alpha,
    )

    if csv_dir:
        write_detection_csv(clean_rows, csv_dir / "clean_detection.csv")
        write_detection_csv(wm_rows, csv_dir / "watermarked_detection.csv")

    return asdict(baseline)


def compute_watermark_survival(
    attacked_tokens: Path | list[Path],
    csv_dir: Path | None = None,
    alpha: float = 0.01,
) -> dict:
    """Compute the fraction of attacked token files still detected as watermarked."""
    attack_files = (
        attacked_token_files(attacked_tokens)
        if isinstance(attacked_tokens, Path) and attacked_tokens.is_dir()
        else list(attacked_tokens)
    )
    attack_rows = [
        detect_token_file(path, secret_key=SECRET_KEY, alpha=alpha)
        for path in attack_files
    ]
    survival = watermark_survival_rate(
        attack_files,
        secret_key=SECRET_KEY,
        alpha=alpha,
    )
    if csv_dir:
        write_detection_csv(attack_rows, csv_dir / "attacked_detection.csv")

    result = asdict(survival)
    result["survival_rate"] = result["rate"]
    return result


def compute_quality_metrics(
    reference_images: Path,
    attacked_images: Path,
    lpips_limit: int | None = None,
) -> dict:
    """Compute image quality degradation metrics for paired/reference images."""
    return {
        "reference_images": str(reference_images),
        "attacked_images": str(attacked_images),
        "lpips": compute_lpips(reference_images, attacked_images, limit=lpips_limit),
        "fid": compute_fid(reference_images, attacked_images),
    }
