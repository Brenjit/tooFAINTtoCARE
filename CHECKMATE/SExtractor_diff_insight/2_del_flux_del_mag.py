import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------------
# Read catalog
# --------------------------
cat = pd.read_csv("catalog_samepos_diff_flux.txt", sep='\t')

# --------------------------
# Calculate absolute differences
# --------------------------
cat["DELTA_FLUX"] = np.abs(cat["FLUX_AUTO_mine"] - cat["FLUX_AUTO_romeo"])
cat["DELTA_MAG"] = np.abs(cat["MAG_AUTO_mine"] - cat["MAG_AUTO_romeo"])

# Avoid log(0) by adding small epsilon if needed (optional)
epsilon = 1e-5
cat["DELTA_FLUX"] = cat["DELTA_FLUX"].replace(0, epsilon)
cat["DELTA_MAG"] = cat["DELTA_MAG"].replace(0, epsilon)

# --------------------------
# X-axis: source index
# --------------------------
x_idx = np.arange(1, len(cat) + 1)

# --------------------------
# Plot ΔFlux
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(x_idx, cat["DELTA_FLUX"], 'o', markersize=4, alpha=0.7, label='|ΔFlux|')
plt.yscale('log')
plt.xlabel("Source index (with same positions)")
plt.ylabel("|ΔFlux| [counts] (log scale)")
plt.title("Absolute Flux Difference vs Source Index")
plt.grid(True, which="both", ls='--')
plt.tight_layout()
plt.savefig("delta_flux_vs_index_log.png", dpi=300)
plt.show()

# --------------------------
# Plot ΔMag
# --------------------------
plt.figure(figsize=(10, 6))
plt.plot(x_idx, cat["DELTA_MAG"], 'o', markersize=4, color='green', alpha=0.7, label='|ΔMag|')
plt.yscale('log')
plt.xlabel("Source index (with same positions)")
plt.ylabel("|ΔMag| [mag] (log scale)")
plt.title("Absolute Mag Difference vs Source Index")
plt.grid(True, which="both", ls='--')
plt.tight_layout()
plt.savefig("delta_mag_vs_index_log.png", dpi=300)
plt.show()
