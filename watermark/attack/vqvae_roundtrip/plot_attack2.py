import json
import matplotlib.pyplot as plt
import os

os.makedirs("results/attack2", exist_ok=True)

with open("results/attack2/results.json") as f:
    results = json.load(f)

lpips_scores = [r["lpips"] for r in results]
survived = [r["detected_after"] for r in results]
tokens_changed = [r["tokens_changed_pct"] for r in results]
scores_after = [r["score_after"] for r in results]
threshold = results[0]["threshold"]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Plot 1: LPIPS distribution
axes[0].hist(lpips_scores, bins=30, color="steelblue", edgecolor="black")
axes[0].set_xlabel("LPIPS distortion")
axes[0].set_ylabel("Count")
axes[0].set_title("Attack 2: LPIPS distribution\n(lower = less distortion)")
axes[0].axvline(x=sum(lpips_scores)/len(lpips_scores),
                color="red", linestyle="--", label=f"Mean={sum(lpips_scores)/len(lpips_scores):.4f}")
axes[0].legend()

# Plot 2: tokens changed distribution
axes[1].hist(tokens_changed, bins=30, color="orange", edgecolor="black")
axes[1].set_xlabel("Tokens changed (%)")
axes[1].set_ylabel("Count")
axes[1].set_title("Attack 2: Token change rate\n(% of 576 tokens changed)")

# Plot 3: survival rate vs tokens changed
colors = ["red" if not s else "green" for s in survived]
axes[2].scatter(tokens_changed, scores_after, c=colors, alpha=0.4, s=10)
axes[2].axhline(y=threshold, color="black", linestyle="--", label=f"Threshold={threshold:.1f}")
axes[2].set_xlabel("Tokens changed (%)")
axes[2].set_ylabel("Detection score after attack")
axes[2].set_title("Attack 2: Token changes vs detection score\n(green=detected, red=not detected)")
axes[2].legend()

plt.suptitle("Attack 2: VQ-VAE Decode→Re-encode", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("results/attack2/attack2_plots.png", dpi=150)
print("Saved results/attack2/attack2_plots.png")

# Print summary
n = len(results)
survived_count = sum(survived)
print(f"\n=== Final Summary ===")
print(f"Images evaluated:        {n}")
print(f"Watermark survival rate: {survived_count}/{n} ({100*survived_count/n:.1f}%)")
print(f"Mean LPIPS:              {sum(lpips_scores)/n:.4f}")
print(f"Mean tokens changed:     {sum(tokens_changed)/n:.1f}%")
print(f"Detection threshold:     {threshold:.1f}")