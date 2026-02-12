#!/usr/bin/env python3
"""
Generate model comparison chart for the ENSF612 Vehicle Price Prediction report.
Saves: model_comparison_updated.png
"""

import matplotlib.pyplot as plt

# Model performance data from the notebook results
models_display = ["Linear\nRegression", "Random\nForest", "Gradient\nBoosting"]
rmse_values = [29605, 21023, 16399]
mae_values = [13592, 8864, 6553]
r2_values = [0.7034, 0.8504, 0.9090]

# Colors for each model
colors = ["#66b3ff", "#99ff99", "#00cc99"]

fig, axes = plt.subplots(1, 3, figsize=(14, 5))

# RMSE Chart
ax1 = axes[0]
bars1 = ax1.bar(
    models_display, rmse_values, color=colors, edgecolor="black", linewidth=1.2
)
ax1.set_ylabel("RMSE ($)", fontsize=12)
ax1.set_title("Root Mean Squared Error", fontsize=14, fontweight="bold")
ax1.set_ylim(0, max(rmse_values) * 1.15)
for bar, val in zip(bars1, rmse_values):
    ax1.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 500,
        f"${val:,}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# MAE Chart
ax2 = axes[1]
bars2 = ax2.bar(
    models_display, mae_values, color=colors, edgecolor="black", linewidth=1.2
)
ax2.set_ylabel("MAE ($)", fontsize=12)
ax2.set_title("Mean Absolute Error", fontsize=14, fontweight="bold")
ax2.set_ylim(0, max(mae_values) * 1.15)
for bar, val in zip(bars2, mae_values):
    ax2.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 300,
        f"${val:,}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

# R² Chart
ax3 = axes[2]
bars3 = ax3.bar(
    models_display, r2_values, color=colors, edgecolor="black", linewidth=1.2
)
ax3.set_ylabel("R² Score", fontsize=12)
ax3.set_title("Coefficient of Determination", fontsize=14, fontweight="bold")
ax3.set_ylim(0, 1.0)
for bar, val in zip(bars3, r2_values):
    ax3.text(
        bar.get_x() + bar.get_width() / 2,
        bar.get_height() + 0.02,
        f"{val:.4f}",
        ha="center",
        va="bottom",
        fontsize=11,
        fontweight="bold",
    )

plt.tight_layout()
plt.savefig(
    "model_comparison_updated.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.close()

print("Model comparison chart saved to model_comparison_updated.png")
