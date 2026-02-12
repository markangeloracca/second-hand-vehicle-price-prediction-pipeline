#!/usr/bin/env python3
"""
Generate architecture diagram for the ENSF612 Vehicle Price Prediction report.
Saves: architecture_diagram_updated.png
"""

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(1, 1, figsize=(14, 8))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis("off")

# Title
ax.text(
    7,
    9.5,
    "Vehicle Price Prediction Pipeline - Medallion Architecture",
    fontsize=16,
    fontweight="bold",
    ha="center",
)

# Data Source Box
data_source = FancyBboxPatch(
    (0.5, 5.5),
    2.5,
    1.5,
    boxstyle="round,pad=0.05",
    facecolor="#f0f0f0",
    edgecolor="black",
    linewidth=2,
)
ax.add_patch(data_source)
ax.text(1.75, 6.5, "Data Source", fontsize=11, fontweight="bold", ha="center")
ax.text(1.75, 6.0, "CSV: 24,198 records", fontsize=9, ha="center", style="italic")

# Bronze Layer Box
bronze = FancyBboxPatch(
    (4, 6.5),
    2.5,
    1.5,
    boxstyle="round,pad=0.05",
    facecolor="#cd7f32",
    edgecolor="black",
    linewidth=2,
)
ax.add_patch(bronze)
ax.text(
    5.25,
    7.5,
    "Bronze Layer",
    fontsize=11,
    fontweight="bold",
    ha="center",
    color="white",
)
ax.text(
    5.25, 7.0, "Raw + Metadata", fontsize=9, ha="center", style="italic", color="white"
)

# Silver Layer Box
silver = FancyBboxPatch(
    (4, 4),
    2.5,
    1.5,
    boxstyle="round,pad=0.05",
    facecolor="#c0c0c0",
    edgecolor="black",
    linewidth=2,
)
ax.add_patch(silver)
ax.text(5.25, 5.0, "Silver Layer", fontsize=11, fontweight="bold", ha="center")
ax.text(5.25, 4.5, "Cleaned: 19,461 rows", fontsize=9, ha="center", style="italic")

# Gold Layer Box
gold = FancyBboxPatch(
    (4, 1.5),
    2.5,
    1.5,
    boxstyle="round,pad=0.05",
    facecolor="#ffd700",
    edgecolor="black",
    linewidth=2,
)
ax.add_patch(gold)
ax.text(5.25, 2.5, "Gold Layer", fontsize=11, fontweight="bold", ha="center")
ax.text(5.25, 2.0, "Feature Engineering", fontsize=9, ha="center", style="italic")

# ML Models Box
ml_models = FancyBboxPatch(
    (8, 5),
    2.5,
    2.5,
    boxstyle="round,pad=0.05",
    facecolor="#40e0d0",
    edgecolor="black",
    linewidth=2,
)
ax.add_patch(ml_models)
ax.text(9.25, 6.7, "ML Models", fontsize=11, fontweight="bold", ha="center")
ax.text(9.25, 6.2, "LR | RF | GB", fontsize=9, ha="center", style="italic")

# Predictions Box
predictions = FancyBboxPatch(
    (8, 1.5),
    2.5,
    2,
    boxstyle="round,pad=0.05",
    facecolor="#87ceeb",
    edgecolor="black",
    linewidth=2,
)
ax.add_patch(predictions)
ax.text(9.25, 2.8, "Predictions", fontsize=11, fontweight="bold", ha="center")
ax.text(
    9.25,
    2.2,
    "R² = 0.9090",
    fontsize=10,
    ha="center",
    style="italic",
    fontweight="bold",
)

# Arrows
arrow_props = dict(arrowstyle="->", color="black", lw=2)

# Data Source -> Bronze
ax.annotate("", xy=(4, 7.25), xytext=(3, 6.5), arrowprops=arrow_props)
ax.text(3.3, 7.0, "Ingest", fontsize=9, ha="center")

# Bronze -> Silver
ax.annotate("", xy=(5.25, 5.5), xytext=(5.25, 6.5), arrowprops=arrow_props)
ax.text(5.7, 6.0, "Clean", fontsize=9, ha="center")

# Silver -> Gold
ax.annotate("", xy=(5.25, 3), xytext=(5.25, 4), arrowprops=arrow_props)
ax.text(5.9, 3.5, "Transform", fontsize=9, ha="center")

# Gold -> ML Models
ax.annotate("", xy=(8, 5.5), xytext=(6.5, 2.75), arrowprops=arrow_props)
ax.text(7.0, 4.5, "Train", fontsize=9, ha="center")

# ML Models -> Predictions
ax.annotate("", xy=(9.25, 3.5), xytext=(9.25, 5), arrowprops=arrow_props)
ax.text(9.7, 4.2, "Predict", fontsize=9, ha="center")

# Legend boxes
legend_box1 = FancyBboxPatch(
    (0.3, 3),
    1.8,
    1.2,
    boxstyle="round,pad=0.05",
    facecolor="white",
    edgecolor="black",
    linewidth=1,
)
ax.add_patch(legend_box1)
ax.text(1.2, 3.9, "• PySpark I/O", fontsize=8, ha="left")
ax.text(1.2, 3.5, "• Delta Lake", fontsize=8, ha="left")
ax.text(1.2, 3.1, "• Unity Catalog", fontsize=8, ha="left")

legend_box2 = FancyBboxPatch(
    (11, 6.5),
    2.2,
    1.2,
    boxstyle="round,pad=0.05",
    facecolor="white",
    edgecolor="black",
    linewidth=1,
)
ax.add_patch(legend_box2)
ax.text(12.1, 7.4, "• Scikit-learn", fontsize=8, ha="left")
ax.text(12.1, 7.0, "• GridSearchCV", fontsize=8, ha="left")
ax.text(12.1, 6.6, "• 3-fold CV", fontsize=8, ha="left")

plt.tight_layout()
plt.savefig(
    "architecture_diagram_updated.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.close()

print("Architecture diagram saved to architecture_diagram_updated.png")
