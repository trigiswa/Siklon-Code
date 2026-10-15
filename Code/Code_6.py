import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

labels = [
    "Latitude",
    "SST",
    "Total\nPrecipitation",
    "Vertical\nVelocity",
    "Relative\nVorticity",
    "Wind",
    "Pressure"
]

R = np.array([
    [ 1.0000,  0.1103, -0.1171,  0.0477,  0.3836,  0.0807, -0.1463],
    [ 0.1103,  1.0000,  0.0288,  0.0001, -0.0520, -0.0978,  0.0337],
    [-0.1171,  0.0288,  1.0000, -0.4014,  0.1457,  0.1339,  0.0553],
    [ 0.0477,  0.0001, -0.4014,  1.0000, -0.1123, -0.0463, -0.0007],
    [ 0.3836, -0.0520,  0.1457, -0.1123,  1.0000,  0.2315, -0.1421],
    [ 0.0807, -0.0978,  0.1339, -0.0463,  0.2315,  1.0000, -0.4794],
    [-0.1463,  0.0337,  0.0553, -0.0007, -0.1421, -0.4794,  1.0000]
], dtype=float)

corr_df = pd.DataFrame(R, index=labels, columns=labels)

fig, ax = plt.subplots(figsize=(9, 7))

im = ax.imshow(corr_df.values, vmin=-1, vmax=1)

ax.set_xticks(np.arange(len(labels)))
ax.set_yticks(np.arange(len(labels)))
ax.set_xticklabels(labels, fontsize=11)
ax.set_yticklabels(labels, fontsize=11)
plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

# Thin white grid lines
ax.set_xticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.set_yticks(np.arange(-0.5, len(labels), 1), minor=True)
ax.grid(which="minor", color="white", linestyle="-", linewidth=1)
ax.tick_params(which="minor", bottom=False, left=False)

# Values inside each cell
for i in range(len(labels)):
    for j in range(len(labels)):
        value = corr_df.iloc[i, j]
        ax.text(
            j, i,
            f"{value:.3f}",
            ha="center",
            va="center",
            fontsize=9,
            fontweight="bold" if i == j else "normal"
        )

cbar = fig.colorbar(im, ax=ax)
cbar.set_label("Correlation Coefficient", fontsize=11)

ax.set_title(
    "Heatmap of the Gaussian Copula Correlation Matrix Between Model Variables",
    fontsize=13,
    pad=15
)

plt.tight_layout()
plt.savefig("Table_3_9_Gaussian_Copula_Correlation_Heatmap.png", dpi=400, bbox_inches="tight")
plt.show()