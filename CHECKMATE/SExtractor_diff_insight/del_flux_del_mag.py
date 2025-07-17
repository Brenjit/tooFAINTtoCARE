import pandas as pd
import matplotlib.pyplot as plt

# --------------------------
# Read the catalog
# --------------------------
cat = pd.read_csv("catalog_samepos_diff_flux.txt", sep='\t')

# --------------------------
# Calculate differences
# --------------------------
cat["DEL_FLUX"] = cat["FLUX_AUTO_mine"] - cat["FLUX_AUTO_romeo"]
cat["DEL_MAG"] = cat["MAG_AUTO_mine"] - cat["MAG_AUTO_romeo"]

index = range(1, len(cat) + 1)

# --------------------------
# ΔFlux plot
# --------------------------
plt.figure(figsize=(12, 6))
plt.errorbar(index, cat["DEL_FLUX"], 
             yerr=(cat["FLUXERR_AUTO_mine"] + cat["FLUXERR_AUTO_romeo"]), 
             fmt='o', color='purple', alpha=0.7)
plt.axhline(0, color='k', linestyle='--', label='Zero line')
plt.xlabel("Source Index")
plt.ylabel("ΔFlux (mine - romeo) [counts]")
plt.title("ΔFlux per Source Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("delta_flux_vs_index.png", dpi=300)
plt.show()

# --------------------------
# ΔMag plot
# --------------------------
plt.figure(figsize=(12, 6))
plt.errorbar(index, cat["DEL_MAG"], 
             yerr=(cat["MAGERR_AUTO_mine"] + cat["MAGERR_AUTO_romeo"]), 
             fmt='o', color='orange', alpha=0.7)
plt.axhline(0, color='k', linestyle='--', label='Zero line')
plt.xlabel("Source Index")
plt.ylabel("ΔMag (mine - romeo) [mag]")
plt.title("ΔMag per Source Index")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("delta_mag_vs_index.png", dpi=300)
plt.show()

# --------------------------
# Save updated catalog (optional)
# --------------------------
cat.to_csv("catalog_with_deltas.txt", sep='\t', index=False)
