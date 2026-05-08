import json
import matplotlib.pyplot as plt
import numpy as np

with open("results/attack2/results.json") as f:
    results = json.load(f)

lpips_scores = [r["lpips"] for r in results]
survived = [r["detected_after"] for r in results]

# Bin by LPIPS
bins = [0, 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15]
bin_labels = ["0-0.01", "0.01-0.02", "0.02-0.03", "0.03-0.04",
              "0.04-0.05", "0.05-0.07", "0.07-0.10", "0.10+"]

bin_survival = []
bin_counts = []

for i in range(len(bins)-1):
    lo, hi = bins[i], bins[i+1]
    in_bin = [j for j, l in enumerate(lpips_scores) if lo <= l < hi]
    if len(in_bin) == 0:
        bin_survival.append(0)
        bin_counts.append(0)
    else:
        rate = sum(survived[j] for j in in_bin) / len(in_bin)
        bin_survival.append(rate * 100)
        bin_counts.append(len(in_bin))

# handle last bin
in_bin = [j for j, l in enumerate(lpips_scores) if l >= bins[-1]]
if in_bin:
    bin_survival[-1] = sum(survived[j] for j in in_bin) / len(in_bin) * 100
    bin_counts[-1] = len(in_bin)

fig, ax = plt.subplots(figsize=(10, 5))

bars = ax.bar(bin_labels, bin_survival, color="steelblue", edgecolor="black")

# add count labels on top of each bar
for bar, count in zip(bars, bin_counts):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
            f"n={count}", ha="center", va="bottom", fontsize=9)

ax.set_xlabel("LPIPS distortion (binned)", fontsize=12)
ax.set_ylabel("Watermark survival rate (%)", fontsize=12)
ax.set_title("Attack 2: Watermark Survival vs Quality Degradation\n(VQ-VAE Decode→Re-encode)", fontsize=13)
ax.set_ylim(0, 110)
ax.axhline(y=50, color="red", linestyle="--", alpha=0.5, label="Random chance (50%)")
ax.legend()

plt.xticks(rotation=15)
plt.tight_layout()
plt.savefig("results/attack2/tradeoff_plot.png", dpi=150)
print("Saved results/attack2/tradeoff_plot.png")

# print the table
print("\nLPIPS bin | Count | Survival rate")
print("-" * 40)
for label, count, rate in zip(bin_labels, bin_counts, bin_survival):
    print(f"{label:12s} | {count:5d} | {rate:.1f}%")