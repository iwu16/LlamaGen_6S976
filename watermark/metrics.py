"""
Shared evaluation metrics for the watermarking experiments.

This module is intentionally independent of any one attack implementation. Each
attack should write images/tokens to its own output folder, then call these
helpers or use watermark/evaluate.py.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import torch

from watermark.cgz_watermark import detect


SECRET_KEY = b"cgz_llamagen_secret_2024"


@dataclass
class DetectionSummary:
    count: int
    detected: int
    rate: float
    mean_score: float
    mean_threshold: float
    mean_fraction_green: float


@dataclass
class DetectionAccuracy:
    clean: DetectionSummary
    watermarked: DetectionSummary
    true_positive_rate: float
    false_positive_rate: float


def load_token_sequence(path: Path) -> List[int]:
    """Load one saved token sequence from generate_dataset.py."""
    value = torch.load(path, map_location="cpu")
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().flatten().tolist()
    elif isinstance(value, tuple):
        value = list(value)
    elif isinstance(value, list) and value and isinstance(value[0], list):
        value = value[0]
    return [int(x) for x in value]


def detect_token_file(
    path: Path,
    secret_key: bytes = SECRET_KEY,
    alpha: float = 0.01,
) -> dict:
    tokens = load_token_sequence(path)
    result = detect(tokens, secret_key, alpha=alpha)
    result["path"] = str(path)
    return result


def summarize_detection(results: Iterable[dict]) -> DetectionSummary:
    rows = list(results)
    count = len(rows)
    if count == 0:
        return DetectionSummary(0, 0, 0.0, 0.0, 0.0, 0.0)

    detected = sum(1 for row in rows if row["detected"])
    return DetectionSummary(
        count=count,
        detected=detected,
        rate=detected / count,
        mean_score=sum(row["score"] for row in rows) / count,
        mean_threshold=sum(row["threshold"] for row in rows) / count,
        mean_fraction_green=sum(row["fraction_green"] for row in rows) / count,
    )


def detection_accuracy(
    clean_token_files: Iterable[Path],
    watermarked_token_files: Iterable[Path],
    secret_key: bytes = SECRET_KEY,
    alpha: float = 0.01,
) -> DetectionAccuracy:
    clean_results = [
        detect_token_file(path, secret_key=secret_key, alpha=alpha)
        for path in sorted(clean_token_files)
    ]
    wm_results = [
        detect_token_file(path, secret_key=secret_key, alpha=alpha)
        for path in sorted(watermarked_token_files)
    ]

    clean_summary = summarize_detection(clean_results)
    wm_summary = summarize_detection(wm_results)
    return DetectionAccuracy(
        clean=clean_summary,
        watermarked=wm_summary,
        true_positive_rate=wm_summary.rate,
        false_positive_rate=clean_summary.rate,
    )


def watermark_survival_rate(
    attacked_token_files: Iterable[Path],
    secret_key: bytes = SECRET_KEY,
    alpha: float = 0.01,
) -> DetectionSummary:
    results = [
        detect_token_file(path, secret_key=secret_key, alpha=alpha)
        for path in sorted(attacked_token_files)
    ]
    return summarize_detection(results)


def compute_fid(reference_dir: Path, sample_dir: Path) -> float:
    """Compute FID between two image folders using clean-fid."""
    try:
        from cleanfid import fid
    except ImportError as exc:
        raise RuntimeError("Install clean-fid to compute FID: pip install clean-fid") from exc

    return float(fid.compute_fid(str(reference_dir), str(sample_dir)))


def compute_lpips(
    reference_dir: Path,
    sample_dir: Path,
    limit: Optional[int] = None,
    device: Optional[str] = None,
) -> float:
    """
    Compute mean paired LPIPS for matching PNG filenames.

    Use this for before/after attack quality, where sample_dir contains the same
    names as reference_dir.
    """
    try:
        import lpips
        from PIL import Image
        from torchvision import transforms
    except ImportError as exc:
        raise RuntimeError("Install lpips, pillow, and torchvision to compute LPIPS") from exc

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

    scores = []
    with torch.no_grad():
        for name in names:
            ref_path = reference_dir / name
            sample_path = sample_dir / name
            if not sample_path.exists():
                continue
            ref = transform(Image.open(ref_path).convert("RGB")).unsqueeze(0).to(device)
            sample = transform(Image.open(sample_path).convert("RGB")).unsqueeze(0).to(device)
            scores.append(float(model(ref, sample).item()))

    if not scores:
        raise ValueError(f"No matching PNG files found in {reference_dir} and {sample_dir}")
    return sum(scores) / len(scores)


def write_detection_csv(rows: Iterable[dict], output_path: Path) -> None:
    rows = list(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "path",
        "detected",
        "score",
        "threshold",
        "fraction_green",
        "L",
        "m",
        "n",
    ]
    with output_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
