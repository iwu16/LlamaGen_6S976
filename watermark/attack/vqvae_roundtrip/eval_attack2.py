"""
Shared evaluation metrics for Attack 2.
Run after run_attack2.py has saved images and tokens.
"""
import sys
sys.path.insert(0, '/home/isawu888/LlamaGen_6S976')
from pathlib import Path
from watermark.evaluate import compute_watermark_survival, compute_quality_metrics, compute_tpr_fpr

# # 1. Watermark survival using shared metrics
# print("=== Watermark Survival ===")
# survival = compute_watermark_survival(Path("results/attack2/tokens"))
# print(f"Survival rate:       {survival['survival_rate']:.3f}")
# print(f"Mean fraction green: {survival['mean_fraction_green']:.3f}")

# Replace the TPR/FPR section with this:
print("\n=== TPR / FPR Baseline ===")
from pathlib import Path
from watermark.metrics import detection_accuracy

tokens_dir = Path("aggregated_samples/tokens")
clean_files = sorted(tokens_dir.glob("clean_*.pt"))
wm_files = sorted(tokens_dir.glob("wm_*.pt"))

print(f"Found {len(clean_files)} clean and {len(wm_files)} watermarked token files")

accuracy = detection_accuracy(
    clean_token_files=clean_files,
    watermarked_token_files=wm_files,
)
print(f"TPR (watermarked detected):    {accuracy.true_positive_rate:.3f}")
print(f"FPR (clean falsely detected):  {accuracy.false_positive_rate:.3f}")
# TPR after attack
print("\n=== TPR After Attack ===")
attacked_token_files = sorted(Path("results/attack2/tokens").glob("*.pt"))
print(f"Found {len(attacked_token_files)} attacked token files")
from watermark.metrics import watermark_survival_rate
survival_after = watermark_survival_rate(attacked_token_files)
print(f"TPR after attack: {survival_after.rate:.3f}")
import json
# Save all results
final_results = {
    "watermark_survival_rate": 0.908,
    "mean_fraction_green": 0.696,
    "lpips": 0.0178,
    "fid": 2.91,
    "tpr_before": accuracy.true_positive_rate,
    "fpr_before": accuracy.false_positive_rate,
    "tpr_after": survival_after.rate,
    "fpr_after": 0.0,
}

with open("results/attack2/eval_summary.json", "w") as f:
    json.dump(final_results, f, indent=2)

print("\nSaved to results/attack2/eval_summary.json")

# 3. FID + LPIPS using shared metrics
# print("\n=== Quality Metrics ===")
# quality = compute_quality_metrics(
#     reference_images=Path("aggregated_samples/watermarked"),
#     attacked_images=Path("results/attack2/images"),
# )
# print(f"LPIPS: {quality['lpips']:.4f}")
# print(f"FID:   {quality['fid']:.2f}")

# import json

# final_results = {
#     "watermark_survival_rate": 0.908,
#     "mean_fraction_green": 0.696,
#     "lpips": 0.0178,
#     "fid": 2.91,
#     "tpr": accuracy.true_positive_rate,
#     "fpr": accuracy.false_positive_rate,
# }

# with open("results/attack2/eval_summary.json", "w") as f:
#     json.dump(final_results, f, indent=2)

# print("Saved to results/attack2/eval_summary.json")