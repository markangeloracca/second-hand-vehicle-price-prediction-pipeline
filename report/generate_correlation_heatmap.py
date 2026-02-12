#!/usr/bin/env python3
"""
Generate correlation heatmap for the ENSF612 Vehicle Price Prediction report.
Saves: correlation_heatmap.png
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Sample correlation matrix data based on the analysis results
# These values come from the Gold layer feature analysis
correlation_data = {
    "Price": [1.00, -0.41, 0.26, -0.26, 0.44, 0.44, 0.45, 0.32, 0.15],
    "Kilometres": [-0.41, 1.00, -0.35, 0.35, -0.15, -0.12, -0.14, -0.05, 0.02],
    "Year": [0.26, -0.35, 1.00, -1.00, 0.10, 0.08, 0.09, 0.05, 0.03],
    "vehicle_age": [-0.26, 0.35, -1.00, 1.00, -0.10, -0.08, -0.09, -0.05, -0.03],
    "City": [0.44, -0.15, 0.10, -0.10, 1.00, 0.92, 0.98, 0.65, 0.12],
    "Highway": [0.44, -0.12, 0.08, -0.08, 0.92, 1.00, 0.97, 0.62, 0.10],
    "avg_fuel_efficiency": [0.45, -0.14, 0.09, -0.09, 0.98, 0.97, 1.00, 0.65, 0.11],
    "cylinder_count": [0.32, -0.05, 0.05, -0.05, 0.65, 0.62, 0.65, 1.00, 0.15],
    "model_frequency_log": [0.15, 0.02, 0.03, -0.03, 0.12, 0.10, 0.11, 0.15, 1.00],
}

feature_names = [
    "Price",
    "Kilometres",
    "Year",
    "vehicle_age",
    "City",
    "Highway",
    "avg_fuel_efficiency",
    "cylinder_count",
    "model_frequency_log",
]

corr_matrix = pd.DataFrame(correlation_data, index=feature_names)

# Create figure
fig, ax = plt.subplots(figsize=(10, 8))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)  # Upper triangle mask

# Custom colormap
cmap = sns.diverging_palette(250, 10, as_cmap=True)

# Plot heatmap
heatmap = sns.heatmap(
    corr_matrix,
    mask=mask,
    cmap=cmap,
    vmin=-1,
    vmax=1,
    center=0,
    annot=True,
    fmt=".2f",
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8},
    annot_kws={"size": 10},
)

ax.set_title(
    "Pearson Correlation Matrix - Numerical Features",
    fontsize=14,
    fontweight="bold",
    pad=20,
)
plt.xticks(rotation=45, ha="right")
plt.yticks(rotation=0)

plt.tight_layout()
plt.savefig(
    "correlation_heatmap.png",
    dpi=150,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="none",
)
plt.close()

print("Correlation heatmap saved to correlation_heatmap.png")
